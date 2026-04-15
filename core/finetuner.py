"""
SFT of Qwen model variants using LoRA adapters and TRL's SFTTrainer.

The same LoRAFinetuner class handles both the 7B (finetuned-big) and
0.5B (finetuned-small) variants — the difference is captured in LoRAConfig
and TrainingConfig:
  - 7B requires 4-bit quantization (load_in_4bit=True).
  - 0.5B trains at full precision.

Loss is computed on the assistant turn only: we locate the '{"reasoning":'
token sequence and set labels=-100 for everything before it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from core.instruction_formatter import InstructionFormatter, FORMATTED_TEXT_COLUMN


DEFAULT_ADAPTER_OUTPUT_DIR: str = "adapters"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LoRAConfig:
    """
    LoRA adapter hyperparameters.

    Attributes
    ----------
    r : int
        LoRA rank. r=8 is a well-established default for instruction-tuning.
    lora_alpha : int
        Scaling factor. Conventionally 2×r (alpha=16 for r=8).
    lora_dropout : float
    target_modules : list[str]
        q_proj and v_proj are the standard targets for Qwen — best
        accuracy/parameter tradeoff.
    bias : str
    task_type : str
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """
    SFT training hyperparameters.

    Attributes
    ----------
    num_train_epochs : int
    per_device_train_batch_size : int
    gradient_accumulation_steps : int
        Effective batch = 1 × 8 = 8.
    learning_rate : float
        2e-4 is the standard LoRA fine-tuning starting point.
    warmup_ratio : float
    lr_scheduler_type : str
    logging_steps : int
    save_strategy : str
    fp16 : bool
    max_seq_length : int
    seed : int
    report_to : str
    """

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 5
    save_strategy: str = "epoch"
    fp16: bool = DEVICE == "cuda"
    max_seq_length: int = 1024
    seed: int = 42
    report_to: str = "wandb"


class LoRAFinetuner:
    """
    SFT + LoRA fine-tuning for a single ToolArena model variant.

    The same class is used for finetuned-big (7B, 4-bit) and finetuned-small
    (0.5B, full precision) — the difference is in constructor arguments.

    Parameters
    ----------
    base_model_name : str
    variant_name : str
    lora_config : LoRAConfig
    training_config : TrainingConfig
    load_in_4bit : bool
        Required for the 7B model on Colab T4/A100 free tier.
    wandb_project : str
    output_dir : str
    """

    def __init__(
        self,
        base_model_name: str,
        variant_name: str,
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        load_in_4bit: bool = False,
        wandb_project: str = "ToolArena",
        output_dir: str = DEFAULT_ADAPTER_OUTPUT_DIR,
    ) -> None:
        self.base_model_name = base_model_name
        self.variant_name    = variant_name
        self.lora_config     = lora_config or LoRAConfig()
        self.training_config = training_config or TrainingConfig()
        self.load_in_4bit    = load_in_4bit
        self.wandb_project   = wandb_project

        self._adapter_output_dir = Path(output_dir) / variant_name
        self._adapter_output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
    ) -> str:
        """
        Load base model, attach LoRA, run SFT training, save adapter.

        Returns the path to the saved LoRA adapter directory. Pass this to
        ModelPredictor.from_pretrained() as peft_adapter_path.
        """
        print(
            f"\n[LoRAFinetuner] Starting SFT | variant={self.variant_name} | "
            f"train={len(train_samples)} | val={len(val_samples)}"
        )

        wandb.init(
            project=self.wandb_project,
            name=f"{self.variant_name}-run",
            group=self.variant_name,
            tags=["finetuning", "lora", "sft", self.variant_name],
            config={
                "base_model":   self.base_model_name,
                "variant_name": self.variant_name,
                "load_in_4bit": self.load_in_4bit,
                **vars(self.lora_config),
                **vars(self.training_config),
            },
            reinit=True,
        )

        tokenizer, model = self._load_base_model()
        model = self._attach_lora(model)

        cfg = self.training_config
        formatter   = InstructionFormatter(tokenizer, cfg.max_seq_length)
        tr_texts    = formatter.format_list(train_samples)
        vl_texts    = formatter.format_list(val_samples)

        # Completion-only masking: loss is computed on the assistant turn only.
        # We locate '{"reasoning":' and set labels=-100 for everything before it.
        response_marker = tokenizer.encode('{"reasoning":', add_special_tokens=False)

        def tokenize_and_mask(texts):
            result = {"input_ids": [], "attention_mask": [], "labels": []}
            for text in texts:
                enc    = tokenizer(text, truncation=True, max_length=cfg.max_seq_length)
                ids    = enc["input_ids"]
                labels = [-100] * len(ids)
                mlen   = len(response_marker)
                for i in range(len(ids) - mlen + 1):
                    if ids[i : i + mlen] == response_marker:
                        for j in range(i, len(ids)):
                            labels[j] = ids[j]
                        break
                result["input_ids"].append(ids)
                result["attention_mask"].append(enc["attention_mask"])
                result["labels"].append(labels)
            return result

        train_ds = Dataset.from_dict(tokenize_and_mask(tr_texts))
        val_ds   = Dataset.from_dict(tokenize_and_mask(vl_texts))

        sft_config = SFTConfig(
            output_dir=str(self._adapter_output_dir / "checkpoints"),
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            fp16=cfg.fp16 and not self.load_in_4bit,
            bf16=self.load_in_4bit and torch.cuda.is_available(),
            seed=cfg.seed,
            report_to=cfg.report_to,
            run_name=f"{self.variant_name}-run",
            eval_strategy=cfg.save_strategy,
            load_best_model_at_end=False,
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        print(
            f"[LoRAFinetuner] Trainable parameters: "
            f"{self._count_trainable_params(model):,}"
        )

        trainer.train()

        adapter_save_path = str(self._adapter_output_dir / "final_adapter")
        trainer.model.save_pretrained(adapter_save_path)
        tokenizer.save_pretrained(adapter_save_path)

        wandb.finish()
        print(f"[LoRAFinetuner] Training complete. Adapter saved → {adapter_save_path}")
        return adapter_save_path

    def _load_base_model(self):
        """Load tokenizer and base model. Applies 4-bit quantization if load_in_4bit."""
        print(f"[LoRAFinetuner] Loading base model: {self.base_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.load_in_4bit:
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

        model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **load_kwargs)
        model.config.use_cache = False  # required for gradient checkpointing
        return tokenizer, model

    def _attach_lora(self, model: Any) -> Any:
        """Attach LoRA adapter via PEFT. Freezes all base weights."""
        cfg = self.lora_config
        peft_config = PeftLoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias=cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def _count_trainable_params(model: Any) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
