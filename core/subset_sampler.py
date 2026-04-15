"""
Samples a representative demo subset from the full ToolArena dataset via
stratified sampling over (confusion_group, difficulty) strata. Stratification guarantees the demo distribution mirrors the full dataset — simple random sampling on 25k rows could accidentally under-represent rare strata.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DEMO_N: int = 200
DEFAULT_SEED: int = 42
DEFAULT_INPUT_PATH: str = "datasets/bi/full_dataset.jsonl"
DEFAULT_OUTPUT_PATH: str = "datasets/bi/demo_subset.jsonl"

TEST_SPLIT_RATIO: float = 0.20
VAL_SPLIT_RATIO: float = 0.15


class StratifiedSampler:
    """
    Samples proportionally from each (confusion_group, difficulty) stratum.

    Parameters
    ----------
    seed : int
    """

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def sample(
        self,
        full_dataset_path: str,
        n: int = DEFAULT_DEMO_N,
        project_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load the full dataset and sample ``n`` rows proportionally from each
        (confusion_group, difficulty) stratum.

        Raises FileNotFoundError if the dataset file doesn't exist, or
        ValueError if n > total dataset size.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        resolved_path = Path(full_dataset_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(project_root) / resolved_path

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Full dataset not found at: {resolved_path}\n"
                f"Run core/dataset_generator.py first to generate it."
            )

        full_dataset = self._load_jsonl(resolved_path)

        if n > len(full_dataset):
            raise ValueError(
                f"Requested subset size n={n} exceeds the full dataset "
                f"size of {len(full_dataset):,} rows."
            )

        strata = self._build_strata(full_dataset)
        subset = self._proportional_sample(strata, n)
        self._rng.shuffle(subset)

        print(
            f"[StratifiedSampler] Sampled {len(subset):,} rows from "
            f"{len(full_dataset):,} (seed={self.seed})"
        )
        return subset

    def save(
        self,
        subset: List[Dict[str, Any]],
        output_path: str,
        project_root: Optional[Path] = None,
    ) -> Path:
        """Write the subset to a JSONL file. Parent directories are created if needed."""
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        resolved = Path(output_path)
        if not resolved.is_absolute():
            resolved = Path(project_root) / resolved

        resolved.parent.mkdir(parents=True, exist_ok=True)

        with resolved.open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[StratifiedSampler] Saved {len(subset):,} rows → {resolved}")
        return resolved

    def compute_splits(
        self,
        subset: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split the subset into train, val, test partitions.

        Uses a fresh RNG derived from self.seed so splits are reproducible
        independently of how many times sample() was called.
        """
        n = len(subset)
        n_test  = max(1, math.floor(n * TEST_SPLIT_RATIO))
        n_val   = max(1, math.floor((n - n_test) * VAL_SPLIT_RATIO))
        n_train = n - n_test - n_val

        split_rng = random.Random(self.seed + 1)
        shuffled = list(subset)
        split_rng.shuffle(shuffled)

        train = shuffled[:n_train]
        val   = shuffled[n_train : n_train + n_val]
        test  = shuffled[n_train + n_val :]

        print(
            f"[StratifiedSampler] Splits — "
            f"train: {len(train):,} | val: {len(val):,} | test: {len(test):,}"
        )
        return train, val, test

    def stratum_report(
        self,
        samples: List[Dict[str, Any]],
        label: str = "dataset",
    ) -> str:
        """Distribution report per (confusion_group, difficulty) stratum."""
        strata = self._build_strata(samples)
        total = len(samples)

        lines = [
            f"Stratum report — {label} ({total:,} samples)",
            f"{'Confusion Group':<35} {'Difficulty':>10} {'Count':>8} {'%':>7}",
            "-" * 65,
        ]

        for (group, diff), rows in sorted(strata.items()):
            pct = 100.0 * len(rows) / total if total > 0 else 0.0
            lines.append(f"{group:<35} {diff:>10} {len(rows):>8,} {pct:>6.1f}%")

        lines.append("-" * 65)
        lines.append(f"{'TOTAL':<35} {'':>10} {total:>8,} {'100.0%':>7}")
        return "\n".join(lines)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        samples = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _build_strata(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
        strata: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for row in samples:
            key = (row.get("confusion_group", "unknown"), row.get("difficulty", 0))
            strata[key].append(row)
        return dict(strata)

    def _proportional_sample(
        self,
        strata: Dict[Tuple[str, int], List[Dict[str, Any]]],
        n: int,
    ) -> List[Dict[str, Any]]:
        """
        Floor-division proportional allocation across strata. Leftover slots
        go to the largest strata first until the target n is reached.
        """
        total = sum(len(v) for v in strata.values())
        result: List[Dict[str, Any]] = []

        allocs: Dict[Tuple[str, int], int] = {}
        allocated = 0
        for key, rows in strata.items():
            count = math.floor(n * len(rows) / total)
            allocs[key] = count
            allocated += count

        remainder = n - allocated
        for key in sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True):
            if remainder <= 0:
                break
            allocs[key] += 1
            remainder -= 1

        for key, count in allocs.items():
            rows    = strata[key]
            count   = min(count, len(rows))
            sampled = self._rng.sample(rows, count)
            result.extend(sampled)

        return result


def main(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    n: int = DEFAULT_DEMO_N,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Sample the demo subset from the full dataset and save it."""
    print(
        f"[StratifiedSampler] Sampling {n:,} rows | "
        f"input='{input_path}' | seed={seed}"
    )
    sampler = StratifiedSampler(seed=seed)
    subset  = sampler.sample(full_dataset_path=input_path, n=n)

    print(sampler.stratum_report(subset, label="demo subset"))

    train, val, test = sampler.compute_splits(subset)
    return sampler.save(subset, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample a stratified demo subset from the full ToolArena dataset."
    )
    parser.add_argument("--input_path",  type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n",           type=int, default=DEFAULT_DEMO_N)
    parser.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        n=args.n,
        seed=args.seed,
    )
