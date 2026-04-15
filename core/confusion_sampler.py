"""
Selects distractor tools for each dataset sample to maximise within-group
ambiguity. For each sample the candidate set is 1 correct tool +
NUM_CANDIDATE_TOOLS-1 distractors, with ~75% of distractors drawn from the
same confusion group as the correct tool.
"""

from __future__ import annotations

import math
import random
from typing import List

from core.tool_registry import ToolRegistry


# Fraction of distractor slots filled from the *same* confusion group.
INTRA_GROUP_DISTRACTOR_RATIO: float = 0.75


class ConfusionAttackSampler:
    """
    Selects distractor tools to maximise within-group confusion in the
    candidate set presented to the model.

    Parameters
    ----------
    registry : ToolRegistry
    intra_group_ratio : float
        Fraction of distractor slots to fill from the same confusion group.
        Defaults to 0.75. Must be in [0.0, 1.0].
    """

    def __init__(
        self,
        registry: ToolRegistry,
        intra_group_ratio: float = INTRA_GROUP_DISTRACTOR_RATIO,
    ) -> None:
        if not 0.0 <= intra_group_ratio <= 1.0:
            raise ValueError(
                f"intra_group_ratio must be in [0.0, 1.0], got {intra_group_ratio}."
            )
        self._registry = registry
        self._intra_group_ratio = intra_group_ratio

    def sample_distractors(
        self,
        correct_tool_name: str,
        n_distractors: int,
        rng: random.Random,
    ) -> List[str]:
        """
        Sample ``n_distractors`` distractor tool names for a given correct tool.

        Returns exactly ``n_distractors`` names, shuffled, with no duplicates
        and never including ``correct_tool_name`` itself.

        Raises
        ------
        KeyError
            If ``correct_tool_name`` is not in the registry.
        ValueError
            If ``n_distractors`` exceeds the number of other available tools.
        """
        correct_tool = self._registry.get_tool(correct_tool_name)
        group_id: str = correct_tool["confusion_group"]

        total_other = len(self._registry.tools) - 1
        if n_distractors > total_other:
            raise ValueError(
                f"Requested {n_distractors} distractors but only {total_other} "
                f"other tools exist in the registry."
            )

        intra: List[str] = [
            t["name"]
            for t in self._registry.get_tools_in_group(group_id)
            if t["name"] != correct_tool_name
        ]
        inter: List[str] = [
            t["name"]
            for t in self._registry.tools
            if t["confusion_group"] != group_id
        ]

        n_intra = min(
            math.ceil(self._intra_group_ratio * n_distractors),
            len(intra),
        )
        n_inter = n_distractors - n_intra

        if n_inter > len(inter):
            n_inter = len(inter)
            n_intra = min(n_distractors - n_inter, len(intra))

        selected: List[str] = []
        if n_intra > 0:
            selected.extend(rng.sample(intra, n_intra))
        if n_inter > 0:
            selected.extend(rng.sample(inter, n_inter))

        # Edge-case: both pools exhausted (very small registries).
        if len(selected) < n_distractors:
            all_others = [
                t["name"]
                for t in self._registry.tools
                if t["name"] != correct_tool_name and t["name"] not in selected
            ]
            needed = n_distractors - len(selected)
            selected.extend(rng.sample(all_others, min(needed, len(all_others))))

        rng.shuffle(selected)
        return selected[:n_distractors]
