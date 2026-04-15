"""
Orchestrates the full evaluation pipeline for any ToolArena model variant by
composing ModelPredictor, ObjectiveMetrics, SubjectiveMetrics, and ResultsBundle.
EvalRunner is intentionally thin — all inference and metric logic lives in the imported modules.

A single EvalRunner instance is reused across all six model variant evaluations
to ensure identical conditions: same judge model, same test split, same W&B
project.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import wandb

from core.model_predictor import ModelPredictor
from core.objective_metrics import ObjectiveMetrics
from core.results_bundle import ResultsBundle
from core.subjective_metrics import SubjectiveMetrics
from core.tool_registry import ToolRegistry


DEFAULT_JUDGE_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_WANDB_PROJECT: str = "ToolArena"


class EvalRunner:
    """
    Coordinates the full ToolArena evaluation pipeline for one model variant.

    Create one instance and reuse it across all six variant evaluations —
    this ensures the same judge model scores every run and timing is measured
    uniformly.

    Parameters
    ----------
    judge_model_path : str
        Must be from a different model family than the six Qwen contenders.
    wandb_project : str
    domain : str
    log_predictions_to_wandb : bool
    """

    def __init__(
        self,
        judge_model_path: str = DEFAULT_JUDGE_MODEL,
        wandb_project: str = DEFAULT_WANDB_PROJECT,
        domain: str = "bi",
        log_predictions_to_wandb: bool = True,
    ) -> None:
        self.judge_model_path = judge_model_path
        self.wandb_project = wandb_project
        self.log_predictions_to_wandb = log_predictions_to_wandb

        self._registry = ToolRegistry(domain=domain)
        self._obj_metrics = ObjectiveMetrics(ignore_parse_failures=True)
        self._subj_metrics = SubjectiveMetrics(
            judge_model_path=judge_model_path,
            verbose=True,
        )

    def run(
        self,
        predictor: ModelPredictor,
        test_samples: List[Dict[str, Any]],
        variant_name: str,
        wandb_run_name: Optional[str] = None,
    ) -> ResultsBundle:
        """
        Run the full evaluation pipeline for one model variant.

        Steps: inference → attach tool descriptions → objective metrics →
        subjective metrics → assemble bundle → W&B logging.

        Parameters
        ----------
        predictor : ModelPredictor
        test_samples : list[dict]
            Must be identical across all six variant evaluations.
        variant_name : str
        wandb_run_name : str, optional
            Defaults to "{variant_name}-run-{timestamp}".
        """
        run_name = wandb_run_name or f"{variant_name}-run-{int(time.time())}"
        print(f"\n[EvalRunner] Starting evaluation: {variant_name}  ({len(test_samples)} samples)")

        t0 = time.time()
        predictions = predictor.predict_batch(test_samples, verbose=True)
        inference_time = time.time() - t0

        predictions = self._attach_tool_descriptions(predictions)

        # The T4 (15 GB) can't hold a 4-bit 7B contender (~4 GB) and the fp16
        # Phi-3.5-mini judge (~3 GB) simultaneously with activations. Offloading
        # the contender first frees ~4-5 GB before subjective metrics run.
        self._offload_predictor(predictor)

        t1 = time.time()
        obj_results  = self._obj_metrics.compute(predictions)
        subj_results = self._subj_metrics.compute(predictions)
        eval_time = time.time() - t1

        predictions = self._merge_sample_scores(predictions, subj_results)

        bundle = ResultsBundle(
            variant_name=variant_name,
            n_samples=len(test_samples),
            tool_accuracy=obj_results["tool_accuracy"],
            macro_f1=obj_results["macro_f1"],
            per_group_accuracy=obj_results["per_group_accuracy"],
            confusion_matrix=obj_results["confusion_matrix"],
            cm_labels=obj_results["cm_labels"],
            mean_judge_score=subj_results["mean_judge_score"],
            judge_scores=subj_results["judge_scores"],
            mean_bertscore_f1=subj_results["mean_bertscore_f1"],
            bertscore_f1s=subj_results["bertscore_f1s"],
            reasoning_consistency_rate=subj_results["reasoning_consistency_rate"],
            consistency_flags=subj_results["consistency_flags"],
            predictions=predictions,
            inference_time_s=inference_time,
            eval_time_s=eval_time,
        )

        print(bundle.summary())
        self._log_to_wandb(bundle, run_name, variant_name)

        return bundle

    def _attach_tool_descriptions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add tool_description to each prediction for the consistency checker."""
        for pred in predictions:
            tool_name = pred.get("predicted_tool", "")
            if self._registry.tool_exists(tool_name):
                pred["tool_description"] = self._registry.get_tool(tool_name).get("description", "")
            else:
                pred["tool_description"] = ""
        return predictions

    @staticmethod
    def _merge_sample_scores(
        predictions: List[Dict[str, Any]],
        subj_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Merge per-sample subjective scores into prediction records so each
        record is fully self-contained for W&B table logging."""
        judge_scores  = subj_results.get("judge_scores", [])
        bertscore_f1s = subj_results.get("bertscore_f1s", [])
        consistency   = subj_results.get("consistency_flags", [])

        for i, pred in enumerate(predictions):
            pred["judge_score"]   = judge_scores[i]  if i < len(judge_scores)  else 0.0
            pred["bertscore_f1"]  = bertscore_f1s[i] if i < len(bertscore_f1s) else 0.0
            pred["is_consistent"] = consistency[i]   if i < len(consistency)   else False

        return predictions

    def _offload_predictor(self, predictor: "ModelPredictor") -> None:
        """Move contender model to CPU and clear VRAM before the judge loads."""
        import gc
        import torch
        try:
            predictor.model.cpu()
            del predictor.model
            gc.collect()
            torch.cuda.empty_cache()
            print("[EvalRunner] Contender model offloaded from GPU — VRAM freed for judge.")
        except Exception as exc:
            print(f"[EvalRunner] Offload warning (non-fatal): {exc}")

    def _log_to_wandb(
        self,
        bundle: ResultsBundle,
        run_name: str,
        variant_name: str,
    ) -> None:
        """Log bundle to W&B. Run is opened and immediately finished so it
        doesn't interfere with training runs that manage their own W&B context."""
        try:
            run = wandb.init(
                project=self.wandb_project,
                name=run_name,
                group=variant_name,
                tags=["evaluation", variant_name],
                reinit=True,
            )

            run.summary.update(bundle.to_flat_dict())

            for group, acc in bundle.per_group_accuracy.items():
                wandb.log({f"group_accuracy/{group}": acc})

            if self.log_predictions_to_wandb and bundle.predictions:
                columns = [
                    "id", "query", "correct_tool", "predicted_tool",
                    "is_correct", "reasoning", "judge_score",
                    "bertscore_f1", "is_consistent", "confusion_group",
                ]
                table_data = [
                    [p.get(c, "") for c in columns]
                    for p in bundle.predictions
                ]
                wandb.log({"predictions": wandb.Table(columns=columns, data=table_data)})

            run.finish()
            print(
                f"[EvalRunner] W&B run '{run_name}' logged to "
                f"project '{self.wandb_project}'."
            )

        except Exception as exc:  # noqa: BLE001
            # W&B logging is non-critical — results are returned in the bundle
            print(
                f"[EvalRunner] W&B logging failed (non-fatal): {exc}\n"
                f"Results are available in the returned ResultsBundle."
            )
