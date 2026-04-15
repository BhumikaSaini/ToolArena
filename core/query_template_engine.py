"""
Fills query and reasoning templates with domain-specific vocabulary to produce diverse, realistic natural-language queries and reference reasoning strings.

QueryTemplateEngine and ArgGenerator live in the same file because they're
tightly coupled: both consume the same vocabulary dict from query_templates.json, and ArgGenerator depends on slot values produced by QueryTemplateEngine.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.tool_registry import DEFAULT_DOMAIN, ToolRegistry


class QueryTemplateEngine:
    """
    Fills query templates with domain-specific vocabulary via slot substitution.

    Templates live in ``domains/<domain>/templates/query_templates.json`` as
    Python str.format()-style strings with named slots (e.g. {metric},
    {window_size}). Slot values are sampled uniformly from the vocabulary
    lists in the same file.

    Parameters
    ----------
    domain : str
    project_root : Path, optional

    Attributes
    ----------
    vocabulary : dict
    tool_templates : dict   tool_name -> {difficulty_str -> [template, ...]}
    reference_reasoning_templates : dict   tool_name -> [template, ...]
    """

    def __init__(
        self,
        domain: str = DEFAULT_DOMAIN,
        project_root: Optional[Path] = None,
    ) -> None:
        self.domain = domain

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        templates_path: Path = (
            Path(project_root)
            / "domains"
            / domain
            / "templates"
            / "query_templates.json"
        )

        if not templates_path.exists():
            raise FileNotFoundError(
                f"Query templates file not found at: {templates_path}"
            )

        with templates_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)

        self.vocabulary: Dict[str, Any] = raw.get("vocabulary", {})
        self.tool_templates: Dict[str, Dict[str, List[str]]] = raw.get("tool_templates", {})
        self.reference_reasoning_templates: Dict[str, List[str]] = raw.get(
            "reference_reasoning_templates", {}
        )

    def fill_query(
        self,
        tool_name: str,
        difficulty: int,
        rng: "random.Random",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Pick and fill a query template for a given tool and difficulty level.

        Returns (query, slots) where slots maps placeholder names to the values
        substituted into the template. Slots are passed downstream to
        ArgGenerator and fill_reasoning to maintain internal consistency.

        Raises KeyError if the tool has no templates or a slot is absent from
        the vocabulary.
        """
        if tool_name not in self.tool_templates:
            raise KeyError(
                f"No query templates defined for tool '{tool_name}'. "
                f"Available tools: {sorted(self.tool_templates.keys())}"
            )

        difficulty_key = str(difficulty)
        bucket: List[str] = self.tool_templates[tool_name].get(difficulty_key, [])

        if not bucket:
            # Fall back to medium difficulty if the requested bucket is empty.
            bucket = self.tool_templates[tool_name].get("2", [])

        template: str = rng.choice(bucket)
        slots: Dict[str, Any] = self._sample_slots(template, rng)

        try:
            query = template.format(**slots)
        except KeyError as exc:
            raise KeyError(
                f"Template slot {exc} for tool '{tool_name}' (difficulty {difficulty}) "
                f"is not present in the vocabulary."
            ) from exc

        return query, slots

    def fill_reasoning(
        self,
        tool_name: str,
        slots: Dict[str, Any],
        rng: "random.Random",
        disambiguation_hint: str,
    ) -> str:
        """
        Generate a reference reasoning string for a dataset sample.

        Uses the same slot values as fill_query to ensure internal consistency.
        Falls back to disambiguation_hint when no reasoning template exists.
        """
        templates: List[str] = self.reference_reasoning_templates.get(tool_name, [])

        if not templates:
            return disambiguation_hint

        template: str = rng.choice(templates)
        safe_slots = {k: str(v) for k, v in slots.items()}
        try:
            return template.format(**safe_slots)
        except KeyError:
            # Return the unfilled template if slot names don't fully align.
            return template

    def _sample_slots(
        self,
        template: str,
        rng: "random.Random",
    ) -> Dict[str, Any]:
        """
        Identify every {slot} in a template and sample a value for each from
        the vocabulary. Unknown slots receive an empty string rather than
        raising, so templates degrade gracefully on incomplete vocabulary.
        """
        formatter = string.Formatter()
        slot_names = {
            field_name
            for _, field_name, _, _ in formatter.parse(template)
            if field_name is not None
        }

        slots: Dict[str, Any] = {}
        voc = self.vocabulary

        for slot in slot_names:
            if slot in voc and isinstance(voc[slot], list) and voc[slot]:
                slots[slot] = rng.choice(voc[slot])
            elif slot == "aggregation_function":
                slots[slot] = rng.choice(["sum", "mean", "count", "median"])
            else:
                slots[slot] = ""

        return slots


class ArgGenerator:
    """
    Generates tool-argument dicts that are consistent with the slot values
    used in the filled query. Optional args are included stochastically to
    add variety across samples.

    Parameters
    ----------
    registry : ToolRegistry
    vocabulary : dict
        From query_templates.json — used as fallback value source.
    """

    OPTIONAL_ARG_INCLUSION_PROB: float = 0.40

    def __init__(self, registry: ToolRegistry, vocabulary: Dict[str, Any]) -> None:
        self._registry = registry
        self._voc = vocabulary

    def generate(
        self,
        tool_name: str,
        slots: Dict[str, Any],
        rng: "random.Random",
    ) -> Dict[str, Any]:
        """
        Generate a correct_tool_args dict for a tool call.

        Required args are always populated; optional args are included with
        probability OPTIONAL_ARG_INCLUSION_PROB.
        """
        tool_def = self._registry.get_tool(tool_name)
        args_schema: Dict[str, Any] = tool_def.get("args", {})
        result: Dict[str, Any] = {}

        for arg_name, arg_spec in args_schema.items():
            required: bool = arg_spec.get("required", False)
            if not required and rng.random() > self.OPTIONAL_ARG_INCLUSION_PROB:
                continue
            value = self._resolve_arg(arg_name, arg_spec, slots, rng)
            if value is not None:
                result[arg_name] = value

        return result

    def _resolve_arg(
        self,
        arg_name: str,
        arg_spec: Dict[str, Any],
        slots: Dict[str, Any],
        rng: "random.Random",
    ) -> Any:
        """
        Resolve a single argument value via a priority cascade:
        1. Reuse a matching slot value (keeps args consistent with the query).
        2. Pick from enum if defined in the schema.
        3. Delegate to type-specific helpers.
        """
        arg_type: str = arg_spec.get("type", "string")
        enum_values: Optional[List] = arg_spec.get("enum")

        # Slot reuse: arg names that map to a differently-named slot.
        aliases: Dict[str, str] = {
            "metric_column":    "metric",
            "date_column":      "time_period",
            "group_by_columns": "dimension",
            "entity_column":    "entity_singular",
            "sort_column":      "metric",
            "target_column":    "target_names",
        }
        canonical = aliases.get(arg_name, arg_name)
        if canonical in slots:
            raw = slots[canonical]
            if arg_type == "array":
                return [raw] if not isinstance(raw, list) else raw
            return raw

        if enum_values:
            return rng.choice(enum_values)

        if arg_type == "boolean":
            return rng.choice([True, False])
        if arg_type == "integer":
            return self._resolve_integer(arg_name, rng)
        if arg_type == "array":
            return self._resolve_array(arg_name, rng)

        return self._resolve_string(arg_name, rng)

    def _resolve_integer(self, arg_name: str, rng: "random.Random") -> int:
        """Sample an integer within a name-appropriate range."""
        int_bounds: Dict[str, Tuple[int, int]] = {
            "window_size":          (7, 90),
            "n":                    (3, 20),
            "polynomial_degree":    (2, 4),
            "forecast_horizon":     (1, 12),
            "baseline_window_days": (7, 90),
        }
        lo, hi = int_bounds.get(arg_name, (1, 30))
        return rng.randint(lo, hi)

    def _resolve_array(self, arg_name: str, rng: "random.Random") -> List[str]:
        """Return 1–3 items sampled from the appropriate vocabulary list."""
        arr_src: Dict[str, List[str]] = {
            "metric_columns":    self._voc.get("metrics", ["revenue"]),
            "predictor_columns": self._voc.get("column_names", ["ad_spend"]),
            "columns":           self._voc.get("column_names", ["revenue"]),
            "group_by_columns":  self._voc.get("dimensions", ["region"]),
            "columns_to_return": self._voc.get("column_names", ["revenue"]),
        }
        source = arr_src.get(arg_name, self._voc.get("column_names", ["revenue"]))
        k = rng.randint(1, min(3, len(source)))
        return rng.sample(source, k)

    def _resolve_string(self, arg_name: str, rng: "random.Random") -> Optional[str]:
        """Return a vocabulary-sourced string for common string argument names."""
        str_src: Dict[str, Any] = {
            "metric_column":     self._voc.get("metrics"),
            "date_column":       ["date", "created_at", "event_date", "order_date"],
            "filter_condition":  self._build_filter_condition(rng),
            "dataset_name":      self._voc.get("dataset_names"),
            "entity_column":     ["user_id", "customer_id", "product_id", "rep_id"],
            "target_column":     ["target_revenue", "quota", "goal_value"],
            "filename":          ["export", "data_export", "analysis_output"],
            "report_title":      ["Performance Summary", "Monthly Report", "Q-Report"],
            "dashboard_id":      ["main_dashboard", "sales_dash", "exec_overview"],
            "cohort_definition": ["users who signed up and converted within 30 days"],
            "cohort_start_date": ["2024-01-01", "2024-04-01", "2024-07-01"],
            "cohort_end_date":   ["2024-03-31", "2024-06-30", "2024-09-30"],
            "period_a_start":    ["2024-01-01", "2024-04-01", "2024-07-01"],
            "period_a_end":      ["2024-03-31", "2024-06-30", "2024-09-30"],
            "period_b_start":    ["2024-04-01", "2024-07-01", "2024-10-01"],
            "period_b_end":      ["2024-06-30", "2024-09-30", "2024-12-31"],
            "period_start":      ["2024-01-01", "2024-04-01"],
            "period_end":        ["2024-03-31", "2024-12-31"],
            "query_or_table":    self._voc.get("dataset_names"),
        }

        source = str_src.get(arg_name)
        if source is None:
            return None
        if isinstance(source, str):
            return source
        if isinstance(source, list) and source:
            return rng.choice(source)
        return None

    def _build_filter_condition(self, rng: "random.Random") -> str:
        """Build a simple SQL-style filter condition from vocabulary."""
        dims = self._voc.get("dimensions", ["region"])
        vals = self._voc.get("dimension_values", ["North"])
        dim  = rng.choice(dims).replace(" ", "_")
        val  = rng.choice(vals)
        return f'{dim} = "{val}"'
