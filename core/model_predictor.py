"""
Runs inference for any HuggingFace causal LM and parses the structured
{"reasoning": ..., "selected_tool": ...} JSON output. JSON parse failures
return sentinel values rather than raising, so one bad sample doesn't abort
an evaluation run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


INFERENCE_SYSTEM_PROMPT: str = """\
You are a tool-selection assistant. Given a user query and a list of \
available tools, select the single most appropriate tool to call and \
explain your reasoning.

Available tools (JSON array):
{tool_list_json}

Respond ONLY with a valid JSON object in this exact format — no preamble, \
no markdown, no extra text:
{{
  "reasoning": "<your step-by-step reasoning for choosing the tool>",
  "selected_tool": "<exact tool name from the list above>"
}}
"""

MAX_NEW_TOKENS: int = 256
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
PARSE_FAILURE_TOOL: str = "__parse_error__"
PARSE_FAILURE_REASONING: str = "__parse_error__"


class ModelPredictor:
    """
    Wraps a HuggingFace causal LM for ToolArena inference.

    Use the ``from_pretrained`` class method rather than calling __init__
    directly — it handles device placement and 4-bit quantization.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    variant_name : str
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        variant_name: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.variant_name = variant_name

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        variant_name: str,
        load_in_4bit: bool = False,
        peft_adapter_path: Optional[str] = None,
    ) -> "ModelPredictor":
        """
        Load model + tokenizer and return a ready-to-use ModelPredictor.

        Parameters
        ----------
        model_name_or_path : str
            HuggingFace hub ID or local path.
        variant_name : str
        load_in_4bit : bool
            Use 4-bit NF4 quantization (required for 7B on Colab T4).
        peft_adapter_path : str, optional
            Path to a saved PEFT/LoRA adapter directory.
        """
        print(f"[ModelPredictor] Loading '{variant_name}' from {model_name_or_path} ...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            # Pin to cuda:0 — some transformers/bitsandbytes combos raise ValueError
            # when device_map="auto" tries a post-quantization .to() dispatch.
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

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

        if peft_adapter_path is not None:
            from peft import PeftModel
            print(f"[ModelPredictor] Attaching PEFT adapter from {peft_adapter_path}")
            model = PeftModel.from_pretrained(model, peft_adapter_path)

        model.eval()
        print(f"[ModelPredictor] '{variant_name}' ready on {DEVICE}.")
        return cls(model=model, tokenizer=tokenizer, variant_name=variant_name)

    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single inference pass on one dataset sample.

        Parameters
        ----------
        sample : dict
            Must contain: id, query, candidate_tools, correct_tool,
            confusion_group, reference_reasoning.

        Returns
        -------
        dict
            Keys: id, query, correct_tool, confusion_group,
            reference_reasoning, predicted_tool, reasoning,
            is_correct, raw_output, parse_success.
        """
        prompt = self._build_prompt(
            query=sample["query"],
            candidate_tools=sample["candidate_tools"],
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw_output = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        predicted_tool, reasoning, parse_success = self._parse_output(raw_output)

        return {
            "id":                  sample.get("id", ""),
            "query":               sample["query"],
            "correct_tool":        sample["correct_tool"],
            "confusion_group":     sample.get("confusion_group", ""),
            "reference_reasoning": sample.get("reference_reasoning", ""),
            "predicted_tool":      predicted_tool,
            "reasoning":           reasoning,
            "is_correct":          predicted_tool == sample["correct_tool"],
            "raw_output":          raw_output,
            "parse_success":       parse_success,
        }

    def predict_batch(self, samples: list, verbose: bool = True) -> list:
        """
        Run inference on a list of samples, one at a time.

        Single-sample calls are used (rather than batched generation) because
        prompts vary in length — padding artefacts are avoided and memory usage
        stays predictable on Colab.
        """
        results = []
        n = len(samples)
        for i, sample in enumerate(samples):
            result = self.predict(sample)
            results.append(result)
            if verbose and (i + 1) % 10 == 0:
                correct = sum(r["is_correct"] for r in results)
                print(
                    f"[{self.variant_name}] {i+1}/{n}  "
                    f"running accuracy: {correct/(i+1):.3f}"
                )
        return results

    def _build_prompt(self, query: str, candidate_tools: list) -> str:
        """
        Build the full inference prompt. Uses the tokenizer's chat template
        when available, falls back to plain concatenation.
        """
        tool_list_json = json.dumps(
            [{"name": t["name"], "description": t["description"]} for t in candidate_tools],
            indent=2,
        )
        system_content = INFERENCE_SYSTEM_PROMPT.format(tool_list_json=tool_list_json)

        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": query},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return (
            f"### System\n{system_content}\n\n"
            f"### User\n{query}\n\n"
            f"### Assistant\n"
        )

    @staticmethod
    def _parse_output(raw: str) -> tuple:
        """
        Parse raw model output into (predicted_tool, reasoning, parse_success).

        Three strategies in order of strictness:
        1. Direct json.loads on the full string.
        2. Regex extraction of the first {...} block.
        3. Regex extraction of the selected_tool value only.
        """
        try:
            obj = json.loads(raw)
            tool   = (obj.get("selected_tool") or PARSE_FAILURE_TOOL).strip()
            reason = (obj.get("reasoning") or PARSE_FAILURE_REASONING).strip()
            return tool, reason, True
        except (json.JSONDecodeError, AttributeError):
            pass

        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                tool   = (obj.get("selected_tool") or PARSE_FAILURE_TOOL).strip()
                reason = (obj.get("reasoning") or PARSE_FAILURE_REASONING).strip()
                return tool, reason, True
            except json.JSONDecodeError:
                pass

        tool_match = re.search(r'"selected_tool"\s*:\s*"([^"]+)"', raw)
        if tool_match:
            tool = tool_match.group(1).strip()
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw)
            reason = reason_match.group(1).strip() if reason_match else PARSE_FAILURE_REASONING
            return tool, reason, True

        return PARSE_FAILURE_TOOL, PARSE_FAILURE_REASONING, False
