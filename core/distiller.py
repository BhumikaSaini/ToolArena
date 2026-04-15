"""
Two knowledge distillation training strategies for ToolArena, both using
KDLoss from kd_loss.py and InstructionFormatter from instruction_formatter.py.
The difference is where teacher logits come from:

SelfDistillationTrainer
  The same model acts as both teacher and student. A dropout-disabled forward
  pass produces soft targets; a dropout-enabled pass treats them as supervision.
  Produces: self-distilled-base-small

KnowledgeDistillationTrainer
  A frozen 7B teacher (4-bit quantized) provides soft targets for a 0.5B student
  trained with LoRA. Produces: distilled-base-big-to-base-small

Both trainers use a custom training loop (not SFTTrainer) because SFTTrainer
doesn't support injecting custom loss functions that depend on teacher logits.
The teacher is always 4-bit in KD to keep two models simultaneously in T4 VRAM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import wandb
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from core.instruction_formatter import InstructionFormatter
from core.kd_loss import KDLoss


DEFAULT_ADAPTER_OUTPUT_DIR: str = "adapters"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN_ID: int = 0


@dataclass
class DistillationConfig:
    """
    Shared hyperparameters for both distillation trainers.

    Attributes
    ----------
    alpha : float
        CE weight in KD loss. (1-alpha) is the KL weight.
    temperature : float
        Softmax temperature for teacher soft-label smoothing.
        Higher T → softer distribution → more signal from near-misses.
    num_train_epochs : int
    learning_rate : float
    warmup_ratio : float
    per_device_train_batch_size : int
        Keep at 1 for Colab free tier.
    gradient_accumulation_steps : int
        Effective batch = 1 × 8 = 8.
    max_seq_length : int
    logging_steps : int
    lora_r : int
    lora_alpha : int
    lora_dropout : float
    lora_target_modules : list[str]
    seed : int
    fp16 : bool
    """

    alpha: float = 0.5
    temperature: float = 2.0
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    logging_steps: int = 5
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    seed: int = 42
    fp16: bool = DEVICE == "cuda"


def _load_model_and_tokenizer(model_name: str, load_in_4bit: bool = False) -> tuple:
    """Load a HuggingFace causal LM and tokenizer.

    Parameters
    ----------
    model_name : str
    load_in_4bit : bool
        Apply 4-bit NF4 quantization via bitsandbytes (for the teacher model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = (
            torch.float16 if DEVICE == "cuda" else torch.float32
        )
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.use_cache = False
    return tokenizer, model


def _attach_lora(model: Any, cfg: DistillationConfig) -> Any:
    """Attach a LoRA adapter using config's LoRA settings."""
    peft_config = PeftLoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config)


def _tokenize_batch(
    texts: List[str],
    tokenizer: Any,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise a list of text strings into a padded batch dict on DEVICE.

    Labels have -100 at all padding positions. Completion-only masking is
    applied: everything before the first occurrence of '{"reasoning":' is
    also set to -100 so loss is computed on the assistant turn only.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Find the response marker and mask all prompt tokens before it.
    response_marker = tokenizer.encode('{"reasoning":', add_special_tokens=False)
    mlen = len(response_marker)
    marker_t = torch.tensor(response_marker, device=DEVICE)
    for b in range(input_ids.size(0)):
        found = False
        for i in range(input_ids.size(1) - mlen + 1):
            if (input_ids[b, i : i + mlen] == marker_t).all():
                labels[b, :i] = -100
                found = True
                break
        if not found:
            labels[b] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


class SelfDistillationTrainer:
    """
    Self-knowledge distillation — the model is its own teacher.

    Each training step:
    1. model.eval() forward (dropout off) → teacher logits
    2. model.train() forward (dropout on) → student logits
    3. KDLoss(student, teacher, labels) → backward

    Parameters
    ----------
    model_name : str
    variant_name : str
    config : DistillationConfig
    wandb_project : str
    output_dir : str
    """

    def __init__(
        self,
        model_name: str,
        variant_name: str,
        config: Optional[DistillationConfig] = None,
        wandb_project: str = "ToolArena",
        output_dir: str = DEFAULT_ADAPTER_OUTPUT_DIR,
    ) -> None:
        self.model_name    = model_name
        self.variant_name  = variant_name
        self.config        = config or DistillationConfig()
        self.wandb_project = wandb_project
        self._save_dir     = Path(output_dir) / variant_name
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples:   List[Dict[str, Any]],
    ) -> str:
        """Run self-distillation training and save the adapter.

        Returns the path to the saved LoRA adapter directory.
        """
        cfg = self.config
        print(
            f"\n[SelfDistillationTrainer] Starting | variant={self.variant_name} | "
            f"train={len(train_samples)} | val={len(val_samples)}"
        )

        wandb.init(
            project=self.wandb_project,
            name=f"{self.variant_name}-run",
            group=self.variant_name,
            tags=["distillation", "self-kd", self.variant_name],
            config={"model": self.model_name, **vars(cfg)},
            reinit=True,
        )

        tokenizer, model = _load_model_and_tokenizer(self.model_name)
        model = _attach_lora(model, cfg)
        model.print_trainable_parameters()

        formatter = InstructionFormatter(tokenizer, cfg.max_seq_length)
        tr_texts  = formatter.format_list(train_samples)
        vl_texts  = formatter.format_list(val_samples)

        loss_fn   = KDLoss(alpha=cfg.alpha, temperature=cfg.temperature)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )

        total_steps = (
            (len(tr_texts) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps))
            * cfg.num_train_epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(cfg.warmup_ratio * total_steps)),
            num_training_steps=total_steps,
        )

        scaler = torch.amp.GradScaler("cuda") if cfg.fp16 else None
        global_step = 0
        optimizer.zero_grad()

        for epoch in range(cfg.num_train_epochs):
            model.train()
            epoch_loss = 0.0

            for i in range(0, len(tr_texts), cfg.per_device_train_batch_size):
                batch_texts = tr_texts[i : i + cfg.per_device_train_batch_size]
                batch = _tokenize_batch(batch_texts, tokenizer, cfg.max_seq_length)

                model.eval()
                with torch.no_grad():
                    teacher_out = model(**batch)
                teacher_logits = teacher_out.logits.detach()

                model.train()
                if cfg.fp16:
                    with torch.amp.autocast("cuda"):
                        student_out = model(**batch)
                        loss = loss_fn(
                            student_out.logits, teacher_logits, batch["labels"]
                        ) / cfg.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    student_out = model(**batch)
                    loss = loss_fn(
                        student_out.logits, teacher_logits, batch["labels"]
                    ) / cfg.gradient_accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * cfg.gradient_accumulation_steps

                step_in_accum = (i // cfg.per_device_train_batch_size) + 1
                if step_in_accum % cfg.gradient_accumulation_steps == 0:
                    if cfg.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        wandb.log({"train/loss": epoch_loss / step_in_accum,
                                   "train/epoch": epoch + 1,
                                   "train/step": global_step})

            val_loss = self._eval_loss(model, tokenizer, vl_texts, loss_fn, cfg)
            print(
                f"[SelfDistillationTrainer] Epoch {epoch+1}/{cfg.num_train_epochs} "
                f"| train_loss: {epoch_loss/max(len(tr_texts),1):.4f} "
                f"| val_loss: {val_loss:.4f}"
            )
            wandb.log({"val/loss": val_loss, "train/epoch": epoch + 1})

        adapter_path = str(self._save_dir / "final_adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        wandb.finish()
        print(f"[SelfDistillationTrainer] Adapter saved → {adapter_path}")
        return adapter_path

    def _eval_loss(
        self,
        model: Any,
        tokenizer: Any,
        val_texts: List[str],
        loss_fn: KDLoss,
        cfg: DistillationConfig,
    ) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(val_texts), cfg.per_device_train_batch_size):
                batch_texts = val_texts[i : i + cfg.per_device_train_batch_size]
                batch = _tokenize_batch(batch_texts, tokenizer, cfg.max_seq_length)

                teacher_out    = model(**batch)
                teacher_logits = teacher_out.logits.detach()

                student_out = model(**batch)
                loss = loss_fn(student_out.logits, teacher_logits, batch["labels"])
                total_loss += loss.item()
                n += 1
        model.train()
        return total_loss / max(n, 1)


class KnowledgeDistillationTrainer:
    """
    Knowledge distillation — frozen large teacher guides a small student.

    Teacher: Qwen2.5-7B-Instruct loaded in 4-bit (~4 GB VRAM, frozen).
    Student: Qwen2.5-0.5B-Instruct with LoRA (~1 GB VRAM, trainable).
    Both fit simultaneously on a Colab T4 (15 GB).

    Parameters
    ----------
    teacher_model_name : str
    student_model_name : str
    variant_name : str
    config : DistillationConfig
    wandb_project : str
    output_dir : str
    """

    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        variant_name: str,
        config: Optional[DistillationConfig] = None,
        wandb_project: str = "ToolArena",
        output_dir: str = DEFAULT_ADAPTER_OUTPUT_DIR,
    ) -> None:
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.variant_name       = variant_name
        self.config             = config or DistillationConfig()
        self.wandb_project      = wandb_project
        self._save_dir          = Path(output_dir) / variant_name
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples:   List[Dict[str, Any]],
    ) -> str:
        """Run KD training and save the student adapter.

        Teacher is loaded first in 4-bit, then student at float16.
        Returns the path to the saved student LoRA adapter directory.
        """
        cfg = self.config
        print(
            f"\n[KnowledgeDistillationTrainer] Starting | "
            f"variant={self.variant_name} | "
            f"train={len(train_samples)} | val={len(val_samples)}"
        )

        wandb.init(
            project=self.wandb_project,
            name=f"{self.variant_name}-run",
            group=self.variant_name,
            tags=["distillation", "kd", self.variant_name],
            config={
                "teacher": self.teacher_model_name,
                "student": self.student_model_name,
                **vars(cfg),
            },
            reinit=True,
        )

        print("[KnowledgeDistillationTrainer] Loading teacher (4-bit) ...")
        _, teacher = _load_model_and_tokenizer(self.teacher_model_name, load_in_4bit=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        print("[KnowledgeDistillationTrainer] Loading student ...")
        tokenizer, student = _load_model_and_tokenizer(self.student_model_name)
        student = _attach_lora(student, cfg)
        student.print_trainable_parameters()

        formatter   = InstructionFormatter(tokenizer, cfg.max_seq_length)
        tr_texts    = formatter.format_list(train_samples)
        vl_texts    = formatter.format_list(val_samples)

        loss_fn   = KDLoss(alpha=cfg.alpha, temperature=cfg.temperature)
        optimizer = AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )

        total_steps = (
            (len(tr_texts) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps))
            * cfg.num_train_epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(cfg.warmup_ratio * total_steps)),
            num_training_steps=total_steps,
        )

        scaler = torch.amp.GradScaler("cuda") if cfg.fp16 else None
        global_step = 0
        optimizer.zero_grad()

        for epoch in range(cfg.num_train_epochs):
            student.train()
            epoch_loss = 0.0

            for i in range(0, len(tr_texts), cfg.per_device_train_batch_size):
                batch_texts = tr_texts[i : i + cfg.per_device_train_batch_size]
                batch = _tokenize_batch(batch_texts, tokenizer, cfg.max_seq_length)

                with torch.no_grad():
                    teacher_out    = teacher(**batch)
                    teacher_logits = teacher_out.logits.detach()

                if cfg.fp16:
                    with torch.amp.autocast("cuda"):
                        student_out = student(**batch)
                        loss = loss_fn(
                            student_out.logits, teacher_logits, batch["labels"]
                        ) / cfg.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    student_out = student(**batch)
                    loss = loss_fn(
                        student_out.logits, teacher_logits, batch["labels"]
                    ) / cfg.gradient_accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * cfg.gradient_accumulation_steps

                step_in_accum = (i // cfg.per_device_train_batch_size) + 1
                if step_in_accum % cfg.gradient_accumulation_steps == 0:
                    if cfg.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        wandb.log({
                            "train/loss":  epoch_loss / step_in_accum,
                            "train/epoch": epoch + 1,
                            "train/step":  global_step,
                        })

            val_loss = self._eval_loss(student, teacher, tokenizer, vl_texts, loss_fn, cfg)
            print(
                f"[KnowledgeDistillationTrainer] Epoch {epoch+1}/{cfg.num_train_epochs} "
                f"| train_loss: {epoch_loss/max(len(tr_texts),1):.4f} "
                f"| val_loss: {val_loss:.4f}"
            )
            wandb.log({"val/loss": val_loss, "train/epoch": epoch + 1})

        adapter_path = str(self._save_dir / "final_adapter")
        student.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        wandb.finish()
        print(f"[KnowledgeDistillationTrainer] Student adapter saved → {adapter_path}")
        return adapter_path

    def _eval_loss(
        self,
        student: Any,
        teacher: Any,
        tokenizer: Any,
        val_texts: List[str],
        loss_fn: KDLoss,
        cfg: DistillationConfig,
    ) -> float:
        student.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(val_texts), cfg.per_device_train_batch_size):
                batch_texts = val_texts[i : i + cfg.per_device_train_batch_size]
                batch = _tokenize_batch(batch_texts, tokenizer, cfg.max_seq_length)

                teacher_out    = teacher(**batch)
                teacher_logits = teacher_out.logits.detach()

                student_out = student(**batch)
                loss = loss_fn(student_out.logits, teacher_logits, batch["labels"])
                total_loss += loss.item()
                n += 1
        student.train()
        return total_loss / max(n, 1)
