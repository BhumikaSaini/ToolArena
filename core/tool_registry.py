"""
Loads, validates, and exposes tool definitions from the domain-specific
``tools.json`` config file
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DOMAIN: str = "bi"


class ToolRegistry:
    """
    Loads, validates, and exposes tool definitions from
    ``domains/<domain>/config/tools.json``.

    Parameters
    ----------
    domain : str
        Domain identifier (e.g. ``"bi"``).
    project_root : Path, optional
        Repo root. Defaults to two levels above this file.

    Raises
    ------
    FileNotFoundError
        If ``tools.json`` can't be found.
    ValueError
        If a tool is missing required fields, references an unknown group,
        or has a duplicate name.
    """

    REQUIRED_TOOL_FIELDS: Tuple[str, ...] = (
        "name",
        "display_name",
        "confusion_group",
        "description",
        "disambiguation_hint",
        "args",
        "returns",
    )

    def __init__(
        self,
        domain: str = DEFAULT_DOMAIN,
        project_root: Optional[Path] = None,
    ) -> None:
        self.domain: str = domain

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        self._project_root: Path = Path(project_root)

        tools_path: Path = (
            self._project_root / "domains" / domain / "config" / "tools.json"
        )

        if not tools_path.exists():
            raise FileNotFoundError(
                f"Tools definition file not found at: {tools_path}\n"
                f"Ensure 'domains/{domain}/config/tools.json' exists under "
                f"the project root '{self._project_root}'."
            )

        with tools_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)

        self.confusion_groups: List[Dict[str, Any]] = raw.get("confusion_groups", [])
        self.tools: List[Dict[str, Any]] = raw.get("tools", [])

        self._validate()

        self._tool_by_name: Dict[str, Dict[str, Any]] = {
            t["name"]: t for t in self.tools
        }
        self._group_ids: set = {g["id"] for g in self.confusion_groups}

        self._tools_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for tool in self.tools:
            self._tools_by_group[tool["confusion_group"]].append(tool)

    def _validate(self) -> None:
        known_groups = {g["id"] for g in self.confusion_groups}
        seen: set = set()

        for tool in self.tools:
            name: str = tool.get("name", "<unnamed>")

            for field_name in self.REQUIRED_TOOL_FIELDS:
                if field_name not in tool:
                    raise ValueError(
                        f"Tool '{name}' is missing required field '{field_name}'."
                    )

            if name in seen:
                raise ValueError(f"Duplicate tool name detected: '{name}'.")
            seen.add(name)

            group = tool["confusion_group"]
            if group not in known_groups:
                raise ValueError(
                    f"Tool '{name}' references unknown confusion group '{group}'. "
                    f"Known groups: {sorted(known_groups)}."
                )

    def get_tool(self, name: str) -> Dict[str, Any]:
        """Return the full tool definition dict for ``name``."""
        if name not in self._tool_by_name:
            raise KeyError(
                f"Tool '{name}' not found in registry. "
                f"Available tools: {self.get_all_tool_names()}"
            )
        return self._tool_by_name[name]

    def get_tools_in_group(self, group_id: str) -> List[Dict[str, Any]]:
        """Return all tool dicts in a confusion group. Empty list if unknown."""
        return list(self._tools_by_group.get(group_id, []))

    def get_group_meta(self, group_id: str) -> Dict[str, Any]:
        """Return metadata dict for a confusion group.

        Raises KeyError if the group_id is not found.
        """
        for g in self.confusion_groups:
            if g["id"] == group_id:
                return g
        raise KeyError(
            f"Confusion group '{group_id}' not found. "
            f"Available groups: {self.get_all_group_ids()}"
        )

    def get_all_tool_names(self) -> List[str]:
        """Sorted list of all tool names."""
        return sorted(self._tool_by_name.keys())

    def get_all_group_ids(self) -> List[str]:
        """Sorted list of all confusion-group IDs."""
        return sorted(self._group_ids)

    def tool_exists(self, name: str) -> bool:
        return name in self._tool_by_name

    def summary(self) -> str:
        """Human-readable summary listing every confusion group and its tools."""
        lines: List[str] = [
            f"ToolRegistry — domain : {self.domain}",
            f"  Total tools         : {len(self.tools)}",
            f"  Confusion groups    : {len(self.confusion_groups)}",
            "",
        ]
        for group in self.confusion_groups:
            gid = group["id"]
            display = group.get("display_name", gid)
            tool_names = [t["name"] for t in self._tools_by_group.get(gid, [])]
            lines.append(f"  [{display}]")
            lines.append(f"    ID    : {gid}")
            lines.append(f"    Tools : {', '.join(tool_names)}")
            lines.append("")
        return "\n".join(lines)
