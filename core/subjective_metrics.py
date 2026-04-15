"""
Computes subjective evaluation metrics for a ToolArena model run:
  - LLM-as-Judge Score (1–5) ↔ RAG Answer Relevance
  - BERTScore F1             ↔ RAG Answer Semantic Similarity
  - Reasoning-Tool Consistency ↔ RAG Faithfulness

Judge model is preferably a completely different family
from the contenders to reduce architectural overlap bias scoring.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import torch
from bert_score import score as bert_score_fn
from transformers import AutoModelForCausalLM, AutoTokenizer


JUDGE_SYSTEM_PROMPT: str = """\
You are an expert evaluator assessing the quality of tool-selection reasoning \
produced by an AI assistant.

You will be given:
1. A user query.
2. The correct tool that should have been selected.
3. The tool the AI actually selected.
4. The reasoning the AI produced for its selection.

Score the reasoning on a scale of 1 to 5 using this rubric:

  5 — Reasoning is correct, clearly identifies the disambiguating signal in \
the query, and is fully consistent with the tool selected.
  4 — Reasoning is mostly correct with minor gaps but reaches the right \
conclusion.
  3 — Reasoning identifies some relevant signals but misses key \
disambiguation points or contains minor logical errors.
  2 — Reasoning is partially relevant but contains significant logical errors \
or misidentifies the key signal.
  1 — Reasoning is incorrect, irrelevant, or contradicts the tool selected.

Respond ONLY with a valid JSON object — no preamble, no markdown:
{{
  "score": <integer 1–5>,
  "explanation": "<one sentence explaining the score>"
}}
"""

MAX_NEW_TOKENS_JUDGE: int = 128
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
BERTSCORE_MODEL: str = "distilbert-base-uncased"
JUDGE_PARSE_FAILURE_SCORE: float = 0.0

# Keywords used by the rule-based consistency checker.
CONSISTENCY_POSITIVE_PATTERNS: List[str] = [
    r"\bcorrect tool\b",
    r"\bselect\w*\b.*\btool\b",
    r"\bchoose\b.*\bbecause\b",
    r"\bthis tool\b",
    r"\baggregate\b",
    r"\brolling\b",
    r"\bcumulative\b",
    r"\bfilter\b",
    r"\bsegment\b",
    r"\bcohort\b",
    r"\btrend\b",
    r"\bforecast\b",
    r"\bseasonality\b",
    r"\brank\b",
    r"\btop.n\b",
    r"\bpercentile\b",
    r"\bcompare.*period\b",
    r"\bbenchmark\b",
    r"\bbaseline\b",
    r"\bexport\b",
    r"\breport\b",
    r"\bdashboard\b",
    r"\bschema\b",
    r"\bstatistic\b",
    r"\bmetric\b",
    r"\bcorrelation\b",
    r"\bregression\b",
    r"\banomaly\b",
]


class SubjectiveMetrics:
    """
    Computes LLM-as-Judge score, BERTScore F1, and reasoning-tool consistency.

    The judge model is loaded lazily on first compute() call to avoid
    allocating GPU memory when subjective evaluation is skipped.

    Parameters
    ----------
    judge_model_path : str
    bertscore_model : str
    verbose : bool
    """

    def __init__(
        self,
        judge_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        bertscore_model: str = BERTSCORE_MODEL,
        verbose: bool = True,
    ) -> None:
        self.judge_model_path = judge_model_path
        self.bertscore_model = bertscore_model
        self.verbose = verbose

        self._judge_model: Any = None
        self._judge_tokenizer: Any = None

    def compute(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute all three subjective metrics for a list of predictions.

        Samples with parse_success=False receive JUDGE_PARSE_FAILURE_SCORE and
        is_consistent=False without going through the judge.
        """
        self._ensure_judge_loaded()

        judge_scores: List[float] = []
        consistency_flags: List[bool] = []
        gen_texts: List[str] = []
        ref_texts: List[str] = []

        for i, pred in enumerate(predictions):
            if not pred.get("parse_success", True):
                judge_scores.append(JUDGE_PARSE_FAILURE_SCORE)
                consistency_flags.append(False)
                gen_texts.append("")
                ref_texts.append(pred.get("reference_reasoning", ""))
                continue

            judge_scores.append(self._judge_one(pred))
            consistency_flags.append(self._check_consistency(
                reasoning=pred.get("reasoning", ""),
                selected_tool=pred.get("predicted_tool", ""),
                tool_description=pred.get("tool_description", ""),
            ))
            gen_texts.append(pred.get("reasoning", ""))
            ref_texts.append(pred.get("reference_reasoning", ""))

            if self.verbose and (i + 1) % 10 == 0:
                print(f"[SubjectiveMetrics] {i+1}/{len(predictions)} samples evaluated.")

        bertscore_f1s = self._compute_bertscore(gen_texts, ref_texts)

        mean_judge = float(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0
        mean_bert  = float(sum(bertscore_f1s) / len(bertscore_f1s)) if bertscore_f1s else 0.0
        consistency_rate = (
            sum(consistency_flags) / len(consistency_flags) if consistency_flags else 0.0
        )

        return {
            "mean_judge_score":           mean_judge,
            "judge_scores":               judge_scores,
            "mean_bertscore_f1":          mean_bert,
            "bertscore_f1s":              bertscore_f1s,
            "reasoning_consistency_rate": consistency_rate,
            "consistency_flags":          consistency_flags,
        }

    def _ensure_judge_loaded(self) -> None:
        """Lazily load judge model and tokenizer. Judge is never fine-tuned —
        it must remain a fixed, independent evaluator across all six runs."""
        if self._judge_model is not None:
            return

        print(f"[SubjectiveMetrics] Loading judge model: {self.judge_model_path}")
        self._judge_tokenizer = AutoTokenizer.from_pretrained(
            self.judge_model_path, trust_remote_code=True
        )
        self._judge_model = AutoModelForCausalLM.from_pretrained(
            self.judge_model_path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self._judge_model.eval()

        if self._judge_tokenizer.pad_token is None:
            self._judge_tokenizer.pad_token = self._judge_tokenizer.eos_token

        print("[SubjectiveMetrics] Judge model ready.")

    def _judge_one(self, pred: Dict[str, Any]) -> float:
        """Score one prediction's reasoning via the judge model."""
        user_content = (
            f"Query: {pred['query']}\n"
            f"Correct tool: {pred['correct_tool']}\n"
            f"Selected tool: {pred['predicted_tool']}\n"
            f"Reasoning: {pred['reasoning']}"
        )

        if (
            hasattr(self._judge_tokenizer, "apply_chat_template")
            and self._judge_tokenizer.chat_template is not None
        ):
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ]
            prompt = self._judge_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                f"### System\n{JUDGE_SYSTEM_PROMPT}\n\n"
                f"### User\n{user_content}\n\n"
                f"### Assistant\n"
            )

        inputs = self._judge_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=768
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = self._judge_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_JUDGE,
                do_sample=False,
                pad_token_id=self._judge_tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self._judge_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return self._parse_judge_score(raw)

    @staticmethod
    def _parse_judge_score(raw: str) -> float:
        """Parse judge output to an integer score 1–5. Falls back through three
        increasingly lenient strategies before returning JUDGE_PARSE_FAILURE_SCORE."""
        try:
            obj = json.loads(raw)
            score = int(obj.get("score", 0))
            if 1 <= score <= 5:
                return float(score)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                score = int(obj.get("score", 0))
                if 1 <= score <= 5:
                    return float(score)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        score_match = re.search(r'"score"\s*:\s*([1-5])', raw)
        if score_match:
            return float(score_match.group(1))

        return JUDGE_PARSE_FAILURE_SCORE

    def _compute_bertscore(
        self, generated: List[str], references: List[str]
    ) -> List[float]:
        """
        BERTScore F1 between generated and reference reasoning strings,
        batched in one call for efficiency. Empty strings (parse failures)
        skip the scorer and receive 0.0.
        """
        if not generated:
            return []

        valid_indices = [
            i for i, (g, r) in enumerate(zip(generated, references))
            if g.strip() and r.strip()
        ]
        f1s = [0.0] * len(generated)

        if valid_indices:
            valid_gen = [generated[i] for i in valid_indices]
            valid_ref = [references[i] for i in valid_indices]

            _, _, f1_tensor = bert_score_fn(
                valid_gen,
                valid_ref,
                model_type=self.bertscore_model,
                lang="en",
                verbose=False,
                device=DEVICE,
            )

            for idx, score in zip(valid_indices, f1_tensor.tolist()):
                f1s[idx] = float(score)

        return f1s

    @staticmethod
    def _check_consistency(
        reasoning: str,
        selected_tool: str,
        tool_description: str = "",
    ) -> bool:
        """
        Rule-based check: does the stated reasoning support the selected tool?

        Analogous to RAG Faithfulness — checks whether reasoning is grounded
        in the tool's semantics rather than just coincidentally naming the
        right tool.

        Three checks (any one passing → True):
        1. At least 2 of the tool's name words appear in the reasoning.
        2. At least 3 words from the tool description appear in the reasoning.
        3. At least one domain-level consistency pattern matches.
        """
        if not reasoning or not selected_tool:
            return False

        reasoning_lower = reasoning.lower()

        tool_keywords = selected_tool.replace("_", " ").lower().split()
        keyword_hits = sum(1 for kw in tool_keywords if kw in reasoning_lower)
        if keyword_hits >= min(2, len(tool_keywords)):
            return True

        if tool_description:
            desc_words = set(
                w.lower() for w in tool_description.split() if len(w) > 4
            )
            reasoning_words = set(reasoning_lower.split())
            if len(desc_words & reasoning_words) >= 3:
                return True

        for pattern in CONSISTENCY_POSITIVE_PATTERNS:
            if re.search(pattern, reasoning_lower):
                return True

        return False
