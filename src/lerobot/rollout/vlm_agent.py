# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct visual agent: observe, request one bounded motion, inspect measured feedback.

Inspired by inspect-robots-agent's motion/observation tools:
https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent
This implementation uses LeRobot's JSON planner transport, IK and motor drivers.
"""

import json
from dataclasses import asdict

from .hybrid import HybridPlanner, PlannerDecision
from .inference.base import InferenceEngine, QueryKind
from .planner import VlmPlanner, text_block


class AgentToolError(ValueError):
    """A model tool request was malformed; no motion was authorized."""


class NoPolicyEngine(InferenceEngine):
    """Lifecycle delegate with no checkpoint, inference worker or action generator."""

    @property
    def control_thread_owns_policy(self):
        return True

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass

    def discard_actions(self):
        pass

    def get_action(self, obs_frame):
        raise RuntimeError("VLM-only control cannot request VLA actions")


class VlmAgentPlanner(HybridPlanner):
    """One JSON tool request per observation, with controller-owned motion duration."""

    def __call__(self, obs_processed, query, task):
        # The shared planner transport logs raw replies and keeps bounded image history.
        # Tool-format errors are returned to the next observation, never interpreted as motion.
        return VlmPlanner.__call__(self, obs_processed, query, task)

    def request_text(self, query, task):
        if query.kind is QueryKind.VQA:
            return VlmPlanner.request_text(self, query, task)
        contracts = {
            name: {k: v for k, v in asdict(ee).items() if k not in {"model_path", "joint_names"}}
            for name, ee in self.hybrid.end_effectors.items()
        }
        return (
            f"You directly control a real {self.robot_type}. Goal: {query.text}\n"
            "There is no VLA and no other policy to start or give subtasks to. Achieve the goal by "
            "issuing one small deliberate action, then examining fresh images and measured feedback. "
            "Every observation includes current joint positions, end-effector FK and labeled camera images. "
            "Always explain what you see and why the next action helps in a brief note. "
            "Break grasping into approach, align, lower, close, verify, lift, transport, release and verify. "
            "Judge success from observations, not from your own prior commands. Use measured residuals to "
            "check whether the last move happened; a rejected command executed nothing. Revise rejected "
            "targets instead of repeating them unchanged. An incomplete task is not a reason to give up. "
            "No arbitrary attempt or time limit is imposed. Operator instructions update the goal.\n"
            "Return one JSON object with exactly: tool, scene, note, ee_targets, targets. "
            "scene and note are nonempty strings. Choose one tool:\n"
            "- move_to: ee_targets maps EXACTLY ONE configured arm name to "
            "{position_m:[x,y,z],quaternion_wxyz:[w,x,y,z]}; targets may include a gripper opening.\n"
            "- move_gripper: ee_targets={}, targets maps gripper action keys to absolute openings.\n"
            "- look: both target maps empty; request another observation without motion.\n"
            "- done: both maps empty; only when images show the whole goal is complete.\n"
            "- give_up: both maps empty; describe a concrete safety concern or required operator help.\n"
            "No code, action lists, joint-angle guesses, velocities or duration fields. Unmentioned axes "
            "keep holding. The controller interpolates each motion over "
            f"{self.hybrid.max_intervention_s}s, enforcing the bounds below. "
            "Use absolute poses in each arm's OWN base frame (metres, wxyz quaternion), not pixels or a "
            "shared world frame. Start from measured FK; preserve orientation unless a rotation is justified. "
            "Local IK computes the joints. Joint interpolation is not a straight Cartesian path and has "
            "no collision planner. Choose motions with visible clearance from the table, objects and other "
            "arm. Camera mounts marked estimated provide approximate directions, not calibrated object "
            "coordinates; pixels require depth and intrinsics for metric 3D. Use small visually justified "
            "adjustments and reobserve instead of inventing a cube's XYZ. Do not mistake the stationary "
            "hold while awaiting your reply for failed execution.\n"
            f"End-effector contracts: {json.dumps(contracts)}\n"
            f"Action bounds and units: {json.dumps({k: asdict(v) for k, v in self.hybrid.limits.items()})}"
        )

    def observation_blocks(self, label, obs_processed):
        blocks = super().observation_blocks(label, obs_processed)
        if "_vlm_feedback" in obs_processed:
            blocks.append(text_block(f"Last tool result: {json.dumps(obs_processed['_vlm_feedback'])}"))
        return blocks

    def parse_reply(self, reply, query, task):
        if query.kind is QueryKind.VQA:
            return VlmPlanner.parse_reply(self, reply, query, task)
        try:
            return self._parse_tool(reply)
        except (ValueError, TypeError) as exc:
            raise AgentToolError(str(exc)) from exc

    def _parse_tool(self, reply):
        fields = {"tool", "scene", "note", "ee_targets", "targets"}
        if not isinstance(reply, dict) or set(reply) != fields:
            raise ValueError(f"Return exactly these tool fields: {sorted(fields)}")
        modes = {
            "move_to": "end_effector",
            "move_gripper": "intervention",
            "look": "observe",
            "done": "done",
            "give_up": "hold",
        }
        if not isinstance(reply["tool"], str) or reply["tool"] not in modes:
            raise ValueError("Unknown VLM tool")
        mode = modes[reply["tool"]]
        return PlannerDecision.parse(
            {
                "mode": mode,
                "scene": reply["scene"],
                "reason": reply["note"],
                "instruction": "",
                "targets": reply["targets"],
                "ee_targets": reply["ee_targets"],
                "duration_s": self.hybrid.max_intervention_s
                if mode in {"end_effector", "intervention"}
                else 0,
            }
        )
