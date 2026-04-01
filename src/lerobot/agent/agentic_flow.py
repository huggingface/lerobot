# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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
Agentic flow: combine stereo camera, VLM detections, and 3D state with an LLM
to reason and choose actions (e.g. which object to pick, or to retry).

The flow is: Observe (stereo + VLM + depth → scene) → Reason (LLM) → Act.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SceneObject:
    """One detected object with 3D state for the reasoning agent."""

    index: int
    label: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    distance_m: float


@dataclass
class SceneObservation:
    """Full scene from stereo + VLM + depth, for the agent to reason over."""

    objects: list[SceneObject] = field(default_factory=list)
    task: str = ""

    def is_empty(self) -> bool:
        return len(self.objects) == 0


def build_scene_summary(observation: SceneObservation) -> str:
    """Format scene for the LLM: task + list of objects with index, label, position, size, distance."""
    lines = [f"Task: {observation.task}", ""]
    if not observation.objects:
        lines.append("No objects detected in the scene.")
        return "\n".join(lines)
    lines.append("Detected objects (use index to refer to them):")
    for obj in observation.objects:
        cx, cy, cz = obj.center_xyz
        sx, sy, sz = obj.size_xyz
        lines.append(
            f"  [{obj.index}] {obj.label}: center=({cx:.3f}, {cy:.3f}, {cz:.3f}) m, "
            f"size=({sx:.3f}, {sy:.3f}, {sz:.3f}) m, distance={obj.distance_m:.3f} m"
        )
    return "\n".join(lines)


@dataclass
class AgentAction:
    """Structured action produced by the reasoning agent."""

    action: str  # "pick" | "place" | "done" | "fail" | "retry"
    object_index: int | None = None
    place_xyz: tuple[float, float, float] | None = None
    success: bool | None = None
    reason: str | None = None

    def is_terminal(self) -> bool:
        return self.action in ("done", "fail")

    def is_pick(self) -> bool:
        return self.action == "pick" and self.object_index is not None

    def is_place(self) -> bool:
        return self.action == "place" and self.place_xyz is not None


# Default system prompt for the reasoning agent
REASONING_SYSTEM_PROMPT = """You are a robot manipulation assistant. You receive a scene description (list of detected objects with 3D positions and sizes) and a task. You must output exactly one action as a single line of JSON.

Available actions:
- pick: pick the object at the given index (0-based). Use when the user asked to pick/grasp something and you can identify which object.
- place: place the currently held object at (x, y, z) in meters in camera frame. Use when the user asked to place something and you have a target location.
- done: task is complete (include "success": true or false).
- fail: cannot complete (include "reason").
- retry: need to try again, e.g. different detection (include "reason").

Output format (one line, valid JSON only):
{"action": "pick", "object_index": 0}
{"action": "place", "place_xyz": [0.1, 0.0, 0.05]}
{"action": "done", "success": true}
{"action": "fail", "reason": "no red cube in scene"}
{"action": "retry", "reason": "ambiguous which cube to pick"}

Rules:
- Prefer "pick" with the correct object_index when the task is to pick something and you see a matching object.
- Use "done" only when the task is finished (e.g. after a successful pick, or if nothing to do).
- Use "fail" when the task is impossible (e.g. object not found).
- Use "retry" when you want the system to re-detect or try again.
- Reply with ONLY the JSON line, no other text."""


class ReasoningAgent:
    """LLM-based agent that reasons over scene observations and returns structured actions.

    Uses an OpenAI-compatible API (Gemini, GPT-4o, etc.) to choose the next action
    given the current task and scene summary.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gemini-2.5-flash",
        is_gemini: bool = True,
        system_prompt: str | None = None,
    ):
        if is_gemini:
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            self.model = model or "gemini-2.5-flash"
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.base_url = base_url
            self.model = model or "gpt-4o"
        self.is_gemini = is_gemini
        self.system_prompt = system_prompt or REASONING_SYSTEM_PROMPT
        # Debug hooks for logging / introspection (populated on each call to `reason()`).
        self.last_scene_text: str | None = None
        self.last_raw_response_text: str | None = None
        if not self.api_key and self.is_gemini:
            logger.warning("GEMINI_API_KEY not set; reasoning agent calls will fail.")

    def reason(
        self,
        observation: SceneObservation,
        *,
        scene_text_override: str | None = None,
    ) -> AgentAction:
        """Produce the next action from the current scene observation."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        scene_text = (
            scene_text_override if scene_text_override is not None else build_scene_summary(observation)
        )
        self.last_scene_text = scene_text
        self.last_raw_response_text = None

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": scene_text},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        self.last_raw_response_text = text
        return self._parse_action(text, observation)

    def _parse_action(self, text: str, observation: SceneObservation) -> AgentAction:
        """Extract a single JSON object from the response and map to AgentAction."""
        # Try to find a JSON object in the response
        json_match = re.search(r"\{[^{}]*\}", text)
        if not json_match:
            logger.warning("No JSON found in agent response: %s", text[:200])
            return AgentAction(action="fail", reason="Invalid response: no JSON")
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse agent JSON: %s", e)
            return AgentAction(action="fail", reason=f"Invalid JSON: {e}")
        action_name = data.get("action", "").lower()
        if action_name not in ("pick", "place", "done", "fail", "retry"):
            return AgentAction(action="fail", reason=f"Unknown action: {action_name}")

        obj_index = data.get("object_index")
        if obj_index is not None:
            obj_index = int(obj_index)
        place_xyz = data.get("place_xyz")
        if place_xyz is not None:
            place_xyz = tuple(float(x) for x in place_xyz[:3])
        success = data.get("success")
        reason = data.get("reason")

        return AgentAction(
            action=action_name,
            object_index=obj_index,
            place_xyz=place_xyz,
            success=success,
            reason=reason,
        )
