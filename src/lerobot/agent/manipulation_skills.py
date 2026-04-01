# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DimOS-inspired skill metadata for SO arm + stereo (OAK-D) manipulation MCP.

Dimensional exposes @skill methods with mandatory docstrings and JSON-serializable
parameters; the MCP server mirrors those as tools. See:
https://github.com/dimensionalOS/dimos/blob/main/docs/capabilities/agents/readme.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ManipulationSkillMeta:
    """Static description of one skill (tool) for list-tools and system instructions."""

    name: str
    description: str


# Docstrings follow DimOS style: imperative summary + Args block (verbatim in MCP tool description).
MANIPULATION_SKILL_METAS: tuple[ManipulationSkillMeta, ...] = (
    ManipulationSkillMeta(
        name="observe",
        description="""Capture RGB + depth from the robot camera (e.g. OAK-D), run VLM detection and 3D fusion, then return a JSON scene (task, summary text, objects with indices and poses).

        Args:
            (none) Uses the current manipulation task from set_task; call set_task first if the goal changed.

        Returns:
            JSON string with ok, summary, objects[], and task.
        """,
    ),
    ManipulationSkillMeta(
        name="plan_next_action",
        description="""Ask the LLM planner for the next action given the last observe() result.

        Args:
            (none) Requires a prior successful observe() in this session.

        Returns:
            JSON string with ok and action: pick | place | done | fail | retry (with indices or place_xyz as applicable).
        """,
    ),
    ManipulationSkillMeta(
        name="pick_object",
        description="""Grasp the object at the given index from the last observe() output (0-based).

        Args:
            object_index: Index from the objects list in the last observe() JSON.

        Returns:
            JSON string with ok, label, and optional placed_after_pick if CLI place_target is set.
        """,
    ),
    ManipulationSkillMeta(
        name="place_at",
        description="""Place the held object at a position in the robot base frame (meters).

        Args:
            x: X position in meters (forward).
            y: Y position in meters.
            z: Z position in meters (up).

        Returns:
            JSON string with ok and place_xyz.
        """,
    ),
    ManipulationSkillMeta(
        name="set_task",
        description="""Set the natural-language manipulation goal used for VLM detection and the LLM planner.

        Args:
            task: Goal description, e.g. "pick up the red cube".

        Returns:
            JSON string confirming the task.
        """,
    ),
    ManipulationSkillMeta(
        name="wait",
        description="""Pause execution for a fixed time (useful between motions or after grasp).

        Args:
            seconds: Sleep duration in seconds.

        Returns:
            Short confirmation string.
        """,
    ),
    ManipulationSkillMeta(
        name="mcp_status",
        description="""Report MCP server binding, current task, and connection state (DimOS-style status).

        Args:
            (none)

        Returns:
            JSON string with transport, host, port (if HTTP/SSE), task, connected flag, and skill names.
        """,
    ),
    ManipulationSkillMeta(
        name="manipulation_step",
        description="""Run one closed-loop step: observe(), plan_next_action(), then execute pick or place if the planner requests it (done/fail/retry return without motion).

        Args:
            (none)

        Returns:
            JSON string describing the step outcome.
        """,
    ),
    ManipulationSkillMeta(
        name="list_registered_skills",
        description="""Return JSON metadata for all manipulation skills (like dimos mcp list-tools).

        Args:
            (none)

        Returns:
            JSON array of {name, description} for each skill.
        """,
    ),
)


def list_skills_json() -> str:
    """JSON for CLI `lerobot-agentic-mcp mcp list-tools` (DimOS-style)."""
    payload = [{"name": m.name, "description": m.description.strip()} for m in MANIPULATION_SKILL_METAS]
    return json.dumps(payload, indent=2)


def build_skills_system_appendix() -> str:
    """Block to append to MCP server instructions (similar to DimOS # AVAILABLE SKILLS)."""
    lines = [
        "# AVAILABLE SKILLS",
        "The host LLM (Cursor, Claude, etc.) should call these tools to manipulate the arm.",
        "Typical flow: set_task → observe → plan_next_action → pick_object or place_at.",
        "Coordinates for place_at are in the robot base frame (meters).",
        "After the scene changes, call observe again before pick_object.",
        "",
    ]
    for m in MANIPULATION_SKILL_METAS:
        lines.append(f"- {m.name}: {m.description.strip().splitlines()[0]}")
    return "\n".join(lines)


def skill_meta_by_name() -> dict[str, ManipulationSkillMeta]:
    return {m.name: m for m in MANIPULATION_SKILL_METAS}


def validate_tool_args(tool: str, args: dict[str, Any]) -> str | None:
    """Return error message if args are invalid for tool, else None."""
    valid = {m.name for m in MANIPULATION_SKILL_METAS}
    if tool not in valid:
        return f"Unknown skill {tool!r}. Try: mcp list-tools"
    if tool == "pick_object" and "object_index" not in args:
        return "pick_object requires object_index in --json-args"
    if tool == "place_at":
        for k in ("x", "y", "z"):
            if k not in args:
                return f"place_at requires x, y, z in --json-args (missing {k})"
    if tool == "set_task" and "task" not in args:
        return "set_task requires task in --json-args"
    if tool == "wait" and "seconds" not in args:
        return "wait requires seconds in --json-args"
    return None
