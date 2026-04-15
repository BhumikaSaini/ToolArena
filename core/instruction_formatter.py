"""
Converts ToolArena dataset samples into the chat/instruction format used
for SFT with TRL's SFTTrainer. Keeping formatting separate from the training
loop lets both LoRAFinetuner and the distillation trainers share the same code.

The assistant turn always contains the ground-truth structured JSON response:
  {"reasoning": "<reference_reasoning>", "selected_tool": "<correct_tool>"}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase


FINETUNE_SYSTEM_PROMPT: str = """\
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

FORMATTED_TEXT_COLUMN: str = "text"


class InstructionFormatter:
    """
    Converts ToolArena dataset samples into SFT-ready text strings.

    Uses the tokenizer's chat template when available, falling back to a
    plain instruction format for tokenizers without one.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
    max_seq_length : int
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_sample(self, sample: Dict[str, Any]) -> str:
        """
        Format one dataset sample as a complete chat/instruction string.

        The string contains three turns: system (task instructions + tool list),
        user (the query), and assistant (ground-truth JSON response).

        Parameters
        ----------
        sample : dict
            Must contain: query, candidate_tools, correct_tool,
            reference_reasoning.
        """
        system_content = self._build_system_content(sample["candidate_tools"])
        user_content   = sample["query"]
        assistant_content = self._build_assistant_content(
            reasoning=sample.get("reference_reasoning", ""),
            selected_tool=sample["correct_tool"],
        )

        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            messages = [
                {"role": "system",    "content": system_content},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        return (
            f"### System\n{system_content}\n\n"
            f"### User\n{user_content}\n\n"
            f"### Assistant\n{assistant_content}"
        )

    def format_dataset(self, dataset: Any) -> Any:
        """
        Apply format_sample to every row and add the result as a ``"text"``
        column. Original columns are preserved for downstream evaluation.
        """
        return dataset.map(
            lambda sample: {FORMATTED_TEXT_COLUMN: self.format_sample(sample)},
            desc="Formatting samples",
        )

    def format_list(self, samples: List[Dict[str, Any]]) -> List[str]:
        """Convenience wrapper for distillation trainers that work with lists."""
        return [self.format_sample(s) for s in samples]

    @staticmethod
    def _build_system_content(candidate_tools: List[Dict[str, Any]]) -> str:
        """Inject the candidate tool list into the system prompt template.
        Only name and description are included — the model doesn't need args
        to select a tool."""
        tool_list_json = json.dumps(
            [{"name": t["name"], "description": t["description"]} for t in candidate_tools],
            indent=2,
        )
        return FINETUNE_SYSTEM_PROMPT.format(tool_list_json=tool_list_json)

    @staticmethod
    def _build_assistant_content(reasoning: str, selected_tool: str) -> str:
        """Ground-truth assistant response as valid JSON.
        json.dumps ensures special characters in reasoning don't break parsing."""
        return json.dumps(
            {"reasoning": reasoning, "selected_tool": selected_tool},
            ensure_ascii=False,
        )
