# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
"""Structured text inside G05's existing Subtask CoT slot, with causal feedback."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_PATTERN = re.compile(
    r"\s*<memory>(.*?)</memory>\s*<plan>(.*?)</plan>\s*"
    r"<grasp_point>(.*?)</grasp_point>\s*<subtask>(.*?)</subtask>\s*",
    re.DOTALL,
)


@dataclass(frozen=True)
class Scratchpad:
    memory: str = ""
    plan: str = ""
    grasp_point: str = ""
    subtask: str = ""

    @property
    def done(self) -> bool:
        return self.subtask == "done"

    @classmethod
    def parse(cls, text: str) -> Scratchpad:
        match = _PATTERN.fullmatch(text)
        if match is None:
            raise ValueError("Expected memory, plan, grasp_point and subtask in G05's Subtask field")
        fields = [s.strip() for s in match.groups()]
        if any(re.search(r"</?(?:memory|plan|grasp_point|subtask)>", s) or "|" in s for s in fields):
            raise ValueError("Nested scratchpad fields or CoT separators are invalid")
        state = cls(*fields)
        if not state.memory or not state.subtask or (state.done and state.plan):
            raise ValueError("Memory/subtask are required; done requires an empty plan")
        if state.grasp_point:
            point = json.loads(state.grasp_point)
            if not isinstance(point, dict):
                raise ValueError("Grasp point must be an object with camera and point fields")
            xy = point.get("point")
            if (
                point.get("camera") != "observation.images.front"
                or not isinstance(xy, list)
                or len(xy) != 2
                or not all(
                    not isinstance(v, bool) and isinstance(v, (float, int)) and 0 <= v <= 1 for v in xy
                )
            ):
                raise ValueError("Grasp point must identify the front camera and a [0,1] xy pair")
        return state


def grasp_target(messages: list[dict[str, Any]], target_indices: Iterable[int], camera: str) -> str:
    """Ignore other VQA answers; only an explicitly labelled grasp point is a target."""
    for index in target_indices:
        content = messages[int(index)].get("content", "")
        if not content.startswith("GraspJSON:"):
            continue
        try:
            value = json.loads(content.removeprefix("GraspJSON:").strip())
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict) or not str(value.get("label", "")).startswith("grasp point of "):
            continue
        xy = value.get("point")
        if value.get("point_format", "xy") != "xy" or not isinstance(xy, list) or len(xy) != 2:
            raise ValueError("Expected a normalized xy grasp point")
        payload = json.dumps({"camera": camera, "label": value["label"], "point": xy}, separators=(",", ":"))
        Scratchpad.parse(
            f"<memory>validate</memory><plan></plan><grasp_point>{payload}</grasp_point><subtask>pick</subtask>"
        )
        return payload
    return ""


def runtime_command(recipe: dict[str, Any], task: str, state: Scratchpad) -> str:
    from lerobot.datasets.recipe import TrainingRecipe, render_message_turns

    recipe = TrainingRecipe.from_dict(recipe)
    rendered = render_message_turns(
        recipe.prompt_turns("subtask"),
        {
            "task": task,
            "previous_memory": state.memory,
            "previous_plan": state.plan,
            "previous_subtask": state.subtask,
        },
    )
    return rendered["messages_rendered"][-1]["content"]
