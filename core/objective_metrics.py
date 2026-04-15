"""
Computes objective evaluation metrics for a ToolArena model run — tool
accuracy, Macro F1, per-confusion-group accuracy, and the sklearn confusion
matrix. All metrics are deterministic; no model calls needed.

RAG parallel: Tool Selection Accuracy ↔ Context Recall (binary single-selection accuracy vs. partial recall across multiple retrieved documents).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class ObjectiveMetrics:
    """
    Computes objective tool-selection metrics from a list of prediction dicts.

    Parameters
    ----------
    ignore_parse_failures : bool
        If True, samples where parse_success=False are excluded. If False,
        they count as incorrect predictions.
    """

    def __init__(self, ignore_parse_failures: bool = True) -> None:
        self.ignore_parse_failures = ignore_parse_failures

    def compute(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute all objective metrics from prediction records.

        Each dict must contain: correct_tool, predicted_tool, confusion_group,
        parse_success.
        """
        evaluated, n_failures = self._filter_predictions(predictions)

        if not evaluated:
            return self._empty_result(n_failures)

        y_true, y_pred = self._extract_labels(evaluated)

        tool_accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_group_accuracy = self._per_group_accuracy(evaluated)
        cm, cm_labels = self._confusion_matrix(y_true, y_pred)

        return {
            "tool_accuracy":      tool_accuracy,
            "macro_f1":           macro_f1,
            "per_group_accuracy": per_group_accuracy,
            "confusion_matrix":   cm,
            "cm_labels":          cm_labels,
            "n_evaluated":        len(evaluated),
            "n_parse_failures":   n_failures,
        }

    def _filter_predictions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        n_failures = sum(1 for p in predictions if not p.get("parse_success", True))
        if self.ignore_parse_failures:
            evaluated = [p for p in predictions if p.get("parse_success", True)]
        else:
            evaluated = predictions
        return evaluated, n_failures

    @staticmethod
    def _extract_labels(
        predictions: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        y_true = [p["correct_tool"] for p in predictions]
        y_pred = [p["predicted_tool"] for p in predictions]
        return y_true, y_pred

    @staticmethod
    def _per_group_accuracy(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Accuracy per confusion group — reveals which groups remain hard for a
        given model rather than masking differences behind a single aggregate.
        """
        g_correct: Dict[str, int] = defaultdict(int)
        g_total: Dict[str, int] = defaultdict(int)

        for pred in predictions:
            group = pred.get("confusion_group", "unknown")
            g_total[group] += 1
            if pred.get("is_correct", False):
                g_correct[group] += 1

        return {
            group: g_correct[group] / g_total[group]
            for group in sorted(g_total.keys())
        }

    @staticmethod
    def _confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
    ) -> Tuple[List[List[int]], List[str]]:
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        return cm, labels

    @staticmethod
    def _empty_result(n_failures: int) -> Dict[str, Any]:
        return {
            "tool_accuracy":      0.0,
            "macro_f1":           0.0,
            "per_group_accuracy": {},
            "confusion_matrix":   [],
            "cm_labels":          [],
            "n_evaluated":        0,
            "n_parse_failures":   n_failures,
        }
