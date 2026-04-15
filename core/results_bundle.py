"""
ResultsBundle holds every evaluation metric for a single model variant run.
Kept in its own file to avoid circular imports — other evaluator modules can
import it without pulling in inference or metric-computation machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ResultsBundle:
    """
    All evaluation results for a single ToolArena model variant run.

    Produced by ``EvalRunner.run()`` and consumed by the notebook comparison
    cells and W&B logging.

    Attributes
    ----------
    variant_name : str
        e.g. ``"base-small"`` or ``"finetuned-big"``.
    n_samples : int
        Number of test samples evaluated.
    tool_accuracy : float
        Fraction of samples where selected_tool == correct_tool (0–1).
        RAG analogue: Context Recall.
    macro_f1 : float
        Macro-averaged F1 across all 28 tool classes (0–1).
    per_group_accuracy : dict[str, float]
        Accuracy per confusion group — the most project-specific metric,
        showing exactly where each model struggles.
    confusion_matrix : list[list[int]]
        sklearn-style confusion matrix; row = true, col = predicted.
    cm_labels : list[str]
        Tool names corresponding to confusion matrix row/column indices.
    mean_judge_score : float
        Mean LLM-as-Judge score across all samples (1–5).
        RAG analogue: Answer Relevance.
    judge_scores : list[float]
        Per-sample judge scores.
    mean_bertscore_f1 : float
        Mean BERTScore F1 between generated and reference reasoning (0–1).
        RAG analogue: Answer Semantic Similarity.
    bertscore_f1s : list[float]
        Per-sample BERTScore F1 values.
    reasoning_consistency_rate : float
        Fraction of samples where reasoning supports the selected tool (0–1).
        RAG analogue: Faithfulness.
    consistency_flags : list[bool]
        Per-sample consistency flag.
    predictions : list[dict]
        Full per-sample prediction records.
    inference_time_s : float
        Wall-clock seconds spent on inference.
    eval_time_s : float
        Wall-clock seconds spent computing metrics.
    """

    variant_name: str
    n_samples: int

    tool_accuracy: float = 0.0
    macro_f1: float = 0.0
    per_group_accuracy: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    cm_labels: List[str] = field(default_factory=list)

    mean_judge_score: float = 0.0
    judge_scores: List[float] = field(default_factory=list)
    mean_bertscore_f1: float = 0.0
    bertscore_f1s: List[float] = field(default_factory=list)
    reasoning_consistency_rate: float = 0.0
    consistency_flags: List[bool] = field(default_factory=list)

    predictions: List[Dict[str, Any]] = field(default_factory=list)

    inference_time_s: float = 0.0
    eval_time_s: float = 0.0

    def summary(self) -> str:
        """Compact human-readable summary of all scalar metrics."""
        lines = [
            f"ResultsBundle — {self.variant_name}  ({self.n_samples:,} samples)",
            "",
            "  ── Objective Metrics ───────────────────────────────────────",
            f"    Tool Accuracy          : {self.tool_accuracy:.4f}",
            f"    Macro F1               : {self.macro_f1:.4f}",
            "    Per-Group Accuracy     :",
        ]
        for group, acc in sorted(self.per_group_accuracy.items()):
            lines.append(f"      {group:<42} {acc:.4f}")
        lines += [
            "",
            "  ── Subjective Metrics ──────────────────────────────────────",
            f"    Mean Judge Score       : {self.mean_judge_score:.4f}  (scale 1–5)",
            f"    Mean BERTScore F1      : {self.mean_bertscore_f1:.4f}",
            f"    Reasoning Consistency  : {self.reasoning_consistency_rate:.4f}",
            "",
            "  ── Timing ──────────────────────────────────────────────────",
            f"    Inference              : {self.inference_time_s:.1f}s",
            f"    Metric evaluation      : {self.eval_time_s:.1f}s",
        ]
        return "\n".join(lines)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Serialise scalar metrics to a flat dict for wandb.run.summary or a
        comparison DataFrame. Excludes per-sample lists and the raw confusion
        matrix.
        """
        flat: Dict[str, Any] = {
            "variant_name":               self.variant_name,
            "n_samples":                  self.n_samples,
            "tool_accuracy":              self.tool_accuracy,
            "macro_f1":                   self.macro_f1,
            "mean_judge_score":           self.mean_judge_score,
            "mean_bertscore_f1":          self.mean_bertscore_f1,
            "reasoning_consistency_rate": self.reasoning_consistency_rate,
            "inference_time_s":           self.inference_time_s,
            "eval_time_s":                self.eval_time_s,
        }
        for group, acc in self.per_group_accuracy.items():
            flat[f"group_acc__{group}"] = acc
        return flat
