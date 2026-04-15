"""
Orchestrates generation of the full ToolArena confusion-attack dataset by
composing ToolRegistry, QueryTemplateEngine, ArgGenerator, and
ConfusionAttackSampler. DatasetBuilder handles stratified allocation, sample
assembly, shuffling, UUID assignment, and JSONL serialisation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.confusion_sampler import ConfusionAttackSampler
from core.query_template_engine import ArgGenerator, QueryTemplateEngine
from core.tool_registry import DEFAULT_DOMAIN, ToolRegistry


DEFAULT_SEED: int = 100
DEFAULT_FULL_N: int = 25_000
DEFAULT_OUTPUT_DIR: str = "datasets/bi"
NUM_CANDIDATE_TOOLS: int = 5

# D1=easy, D2=medium, D3=hard
DIFFICULTY_WEIGHTS: Dict[int, float] = {1: 0.25, 2: 0.35, 3: 0.40}


@dataclass
class DatasetSample:
    """A single row in the ToolArena benchmark dataset."""

    id: str
    query: str
    correct_tool: str
    correct_tool_args: dict
    candidate_tools: list
    confusion_group: str
    difficulty: int
    distractor_tools: list
    reference_reasoning: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                  self.id,
            "query":               self.query,
            "correct_tool":        self.correct_tool,
            "correct_tool_args":   self.correct_tool_args,
            "candidate_tools":     self.candidate_tools,
            "confusion_group":     self.confusion_group,
            "difficulty":          self.difficulty,
            "distractor_tools":    self.distractor_tools,
            "reference_reasoning": self.reference_reasoning,
            "rationale":           self.rationale,
        }


class DatasetBuilder:
    """
    Orchestrates end-to-end generation of the ToolArena dataset.

    Total samples are distributed proportionally across all
    (tool_name, difficulty) cells. A single master seed is used; each cell
    derives its own seed so adding/removing tools doesn't disturb other cells.

    Parameters
    ----------
    domain : str
    seed : int
    project_root : Path, optional
    """

    def __init__(
        self,
        domain: str = DEFAULT_DOMAIN,
        seed: int = DEFAULT_SEED,
        project_root: Optional[Path] = None,
    ) -> None:
        self.domain = domain
        self.seed = seed

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent
        self._project_root = Path(project_root)

        self._registry = ToolRegistry(domain=domain, project_root=self._project_root)
        self._engine = QueryTemplateEngine(domain=domain, project_root=self._project_root)
        self._arg_gen = ArgGenerator(
            registry=self._registry,
            vocabulary=self._engine.vocabulary,
        )
        self._sampler = ConfusionAttackSampler(registry=self._registry)

    def build(self, n: int = DEFAULT_FULL_N) -> List[DatasetSample]:
        """
        Generate ``n`` dataset samples stratified by tool and difficulty.

        Returns a globally shuffled list with UUID identifiers assigned
        after the shuffle.
        """
        master_rng = random.Random(self.seed)
        tool_names = self._registry.get_all_tool_names()
        n_distractors = NUM_CANDIDATE_TOOLS - 1
        allocs = self._compute_cell_allocations(n, tool_names)

        all_samples: List[DatasetSample] = []

        for tool_name in tool_names:
            tool_def = self._registry.get_tool(tool_name)

            for difficulty in sorted(DIFFICULTY_WEIGHTS.keys()):
                count = allocs.get((tool_name, difficulty), 0)
                if count == 0:
                    continue

                # Each cell gets its own seed so adding/removing a tool
                # doesn't shift samples in unrelated cells.
                cell_seed = master_rng.randint(0, 2**31 - 1)
                cell_rng = random.Random(cell_seed)

                for _ in range(count):
                    sample = self._build_one_sample(
                        tool_name=tool_name,
                        tool_def=tool_def,
                        difficulty=difficulty,
                        n_distractors=n_distractors,
                        rng=cell_rng,
                    )
                    all_samples.append(sample)

        master_rng.shuffle(all_samples)
        for sample in all_samples:
            sample.id = uuid.UUID(int=master_rng.getrandbits(128), version=4).hex

        return all_samples

    def save(
        self,
        samples: List[DatasetSample],
        output_dir: str,
        filename: str = "full_dataset.jsonl",
    ) -> Path:
        """Serialise samples to a JSONL file. Creates output_dir if needed."""
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = self._project_root / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / filename
        with out_path.open("w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

        print(f"[DatasetBuilder] Saved {len(samples):,} samples → {out_path}")
        return out_path

    def _compute_cell_allocations(
        self,
        n: int,
        tool_names: List[str],
    ) -> Dict[Tuple[str, int], int]:
        """
        Compute sample count per (tool_name, difficulty) cell via floor-division
        proportional allocation. Remainders are absorbed by the hardest difficulty
        level so the total is exactly ``n``.
        """
        n_tools = len(tool_names)
        total_weight = sum(DIFFICULTY_WEIGHTS.values())
        allocs: Dict[Tuple[str, int], int] = {}
        base_per_tool, tool_remainder = divmod(n, n_tools)
        diff_items = sorted(DIFFICULTY_WEIGHTS.items())

        for i, tool_name in enumerate(tool_names):
            tool_n = base_per_tool + (1 if i < tool_remainder else 0)
            remaining = tool_n

            for j, (diff, weight) in enumerate(diff_items):
                if j == len(diff_items) - 1:
                    allocs[(tool_name, diff)] = remaining
                else:
                    count = math.floor(tool_n * weight / total_weight)
                    allocs[(tool_name, diff)] = count
                    remaining -= count

        return allocs

    def _build_one_sample(
        self,
        tool_name: str,
        tool_def: Dict[str, Any],
        difficulty: int,
        n_distractors: int,
        rng: random.Random,
    ) -> DatasetSample:
        """Assemble a single DatasetSample. ``id`` is a placeholder until UUIDs
        are assigned in ``build()``."""
        query, slots = self._engine.fill_query(tool_name, difficulty, rng)
        correct_tool_args = self._arg_gen.generate(tool_name, slots, rng)

        distractor_names = self._sampler.sample_distractors(
            correct_tool_name=tool_name,
            n_distractors=n_distractors,
            rng=rng,
        )
        candidate_defs = [tool_def] + [
            self._registry.get_tool(d) for d in distractor_names
        ]
        rng.shuffle(candidate_defs)
        candidate_tools_slim = [self._slim_tool_def(t) for t in candidate_defs]

        reference_reasoning = self._engine.fill_reasoning(
            tool_name=tool_name,
            slots=slots,
            rng=rng,
            disambiguation_hint=tool_def.get("disambiguation_hint", ""),
        )
        rationale = self._build_rationale(tool_def, slots)

        return DatasetSample(
            id="",
            query=query,
            correct_tool=tool_name,
            correct_tool_args=correct_tool_args,
            candidate_tools=candidate_tools_slim,
            confusion_group=tool_def["confusion_group"],
            difficulty=difficulty,
            distractor_tools=distractor_names,
            reference_reasoning=reference_reasoning,
            rationale=rationale,
        )

    @staticmethod
    def _slim_tool_def(tool_def: Dict[str, Any]) -> Dict[str, Any]:
        """Strip internal fields (e.g. confusion_group) — only expose what the
        model needs during inference."""
        return {
            "name":                tool_def["name"],
            "display_name":        tool_def.get("display_name", tool_def["name"]),
            "description":         tool_def["description"],
            "disambiguation_hint": tool_def.get("disambiguation_hint", ""),
            "args":                tool_def.get("args", {}),
        }

    @staticmethod
    def _build_rationale(tool_def: Dict[str, Any], slots: Dict[str, Any]) -> str:
        """One-sentence rationale from the tool's disambiguation_hint, slot-filled."""
        hint: str = tool_def.get("disambiguation_hint", "")
        first_sentence = hint.split(".")[0].strip() + "."
        try:
            return first_sentence.format(**{k: str(v) for k, v in slots.items()})
        except (KeyError, ValueError):
            return first_sentence


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load a saved JSONL dataset into a list of dicts."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    samples = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main(
    domain: str = DEFAULT_DOMAIN,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n: int = DEFAULT_FULL_N,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Generate the full ToolArena dataset and save it as JSONL."""
    print(
        f"[DatasetBuilder] Generating {n:,} samples | "
        f"domain='{domain}' | seed={seed}"
    )
    builder = DatasetBuilder(domain=domain, seed=seed)
    print(builder._registry.summary())

    samples = builder.build(n=n)
    print(f"[DatasetBuilder] Generation complete — {len(samples):,} samples.")
    return builder.save(samples, output_dir=output_dir, filename="full_dataset.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the ToolArena confusion-attack benchmark dataset."
    )
    parser.add_argument("--domain",     type=str, default=DEFAULT_DOMAIN)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n",          type=int, default=DEFAULT_FULL_N)
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    main(
        domain=args.domain,
        output_dir=args.output_dir,
        n=args.n,
        seed=args.seed,
    )
