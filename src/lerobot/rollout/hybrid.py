# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Bounded, absolute-position interventions for an external VLM planner.

All values use robot action units, AFTER policy unnormalization. This is not an
generated-code executor. Optional end-effector corrections use local bounded IK. The operator supplies the action contract;
the model cannot change it. The robot driver remains responsible for hardware limits.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

from .end_effector import EndEffectorConfig, EndEffectorKinematics, parse_pose
from .inference import PolicyQuery, QueryKind
from .planner import VlmPlanner, normalize_instruction, text_block


@dataclass
class InterventionLimit:
    """Bounds for one absolute-position action/observation key, in robot units."""

    minimum: float
    maximum: float
    max_delta: float
    max_speed: float
    tolerance: float
    description: str = ""

    def __post_init__(self):
        values = (self.minimum, self.maximum, self.max_delta, self.max_speed, self.tolerance)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Intervention limits must be finite")
        if self.minimum >= self.maximum or self.max_delta < 0 or min(values[3:]) <= 0:
            raise ValueError("Invalid intervention range, delta, speed, or tolerance")


@dataclass
class HybridConfig:
    """Opt-in supervisor. A complete explicit action contract is required."""

    limits: dict[str, InterventionLimit] = field(default_factory=dict)
    end_effectors: dict[str, EndEffectorConfig] = field(default_factory=dict)
    policy_window_s: float = 5.0
    max_intervention_s: float = 2.0
    settle_s: float = 0.5
    review_timeout_s: float = 60.0
    max_observation_age_s: float = 0.2
    # Cap consecutive direct corrections; a policy step resets this counter.
    max_consecutive_interventions: int = 3

    def __post_init__(self):
        for value in (
            self.policy_window_s,
            self.max_intervention_s,
            self.settle_s,
            self.review_timeout_s,
            self.max_observation_age_s,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Hybrid durations must be finite and positive")
        mapped = [key for ee in self.end_effectors.values() for key in ee.action_keys]
        if len(set(mapped)) != len(mapped) or not set(mapped) <= set(self.limits):
            raise ValueError("End-effector mappings must be disjoint subsets of the action contract")
        if self.max_consecutive_interventions < 1:
            raise ValueError("max_consecutive_interventions must be positive")


@dataclass(frozen=True)
class PlannerDecision:
    mode: str
    scene: str
    reason: str
    instruction: str
    targets: dict[str, float]
    duration_s: float
    ee_targets: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    @classmethod
    def parse(cls, reply: object) -> PlannerDecision:
        fields = {"mode", "scene", "reason", "instruction", "targets", "duration_s"}
        if not isinstance(reply, dict) or set(reply) not in (fields, fields | {"ee_targets"}):
            raise ValueError(f"Hybrid reply must contain exactly {sorted(fields)}")
        if reply["mode"] not in {"policy", "intervention", "end_effector", "hold", "done"}:
            raise ValueError("Unknown hybrid mode")
        for key in ("scene", "reason", "instruction"):
            if not isinstance(reply[key], str):
                raise ValueError(f"{key} must be text")
        if not reply["scene"].strip() or not reply["reason"].strip():
            raise ValueError("scene and reason must not be empty")
        targets = reply["targets"]
        if not isinstance(targets, dict) or not all(isinstance(k, str) for k in targets):
            raise ValueError("targets must be a mapping of action keys to absolute positions")
        for value in [reply["duration_s"], *targets.values()]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Targets and duration must be finite numbers")
        ee_targets = reply.get("ee_targets", {})
        if not isinstance(ee_targets, dict) or not all(isinstance(k, str) for k in ee_targets):
            raise ValueError("ee_targets must map configured end-effector names to poses")
        for value in ee_targets.values():
            parse_pose(value)
        if reply["mode"] == "end_effector":
            if not ee_targets or reply["duration_s"] <= 0 or reply["instruction"]:
                raise ValueError("An end-effector correction needs poses and duration, with no instruction")
        elif ee_targets:
            raise ValueError("Only end_effector mode may supply ee_targets")
        if reply["mode"] == "intervention":
            if not targets or reply["duration_s"] <= 0 or reply["instruction"]:
                raise ValueError("An intervention needs targets and duration, with an empty instruction")
        elif reply["mode"] != "end_effector" and (targets or reply["duration_s"] != 0):
            raise ValueError("Only an intervention may supply targets or duration")
        if (reply["mode"] == "policy") != bool(reply["instruction"].strip()):
            raise ValueError("Only policy mode must supply a non-empty instruction")
        return cls(**reply)

    def validate_motion(self, config: HybridConfig, pose: dict[str, float]) -> dict[str, float]:
        """Validate the entire proposal before returning a complete target pose."""
        if self.duration_s > config.max_intervention_s:
            raise ValueError("Intervention exceeds maximum duration")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("Intervention duration must be positive")
        target = dict(pose)
        ee_keys = {key for ee in config.end_effectors.values() for key in ee.action_keys}
        for key, value in self.targets.items():
            if self.mode == "intervention" and key in ee_keys:
                raise ValueError("Use end_effector mode for configured arm joints")
            if key not in config.limits or key not in pose:
                raise ValueError(f"Unknown intervention action key: {key}")
            limit = config.limits[key]
            delta = abs(value - pose[key])
            if not limit.minimum <= value <= limit.maximum:
                raise ValueError(f"Intervention target outside range: {key}")
            if delta > limit.max_delta or delta / self.duration_s > limit.max_speed:
                raise ValueError(f"Intervention exceeds delta/speed limit: {key}")
            target[key] = value
        return target

    def resolve_motion(self, config, pose, kinematics):
        """Resolve all IK targets atomically, then reuse native joint/gripper guards."""
        if self.mode != "end_effector":
            return self.validate_motion(config, pose)
        if not self.ee_targets or len(self.ee_targets) != 1:
            raise ValueError("Correct exactly one end effector at a time")
        if self.duration_s <= 0 or self.duration_s > config.max_intervention_s:
            raise ValueError("Invalid end-effector duration")
        ee_keys = {key for ee in config.end_effectors.values() for key in ee.action_keys}
        if ee_keys.intersection(self.targets):
            raise ValueError("Do not mix end-effector targets with raw arm-joint targets")
        resolved = dict(self.targets)
        for name, requested in self.ee_targets.items():
            if name not in kinematics:
                raise ValueError(f"Unknown end effector: {name}")
            resolved.update(kinematics[name].solve(requested, pose, config.limits, self.duration_s))
        converted = PlannerDecision(self.mode, self.scene, self.reason, "", resolved, self.duration_s)
        return converted.validate_motion(config, pose)


class HybridPlanner(VlmPlanner):
    """Reuse the external planner's images/history/client with a typed action contract."""

    def __init__(self, *args, hybrid: HybridConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.hybrid = hybrid
        self.kinematics = {name: EndEffectorKinematics(ee) for name, ee in hybrid.end_effectors.items()}

    def recipe_turns(self, query: PolicyQuery) -> list[dict]:
        return []

    def request_text(self, query: PolicyQuery, task: str) -> str:
        if query.kind is QueryKind.VQA:
            return super().request_text(query, task)
        return (
            f"Robot: {self.robot_type}\nOverall goal: {query.text}\nLast policy instruction: {task}\n"
            "Review the current images and measured positions. Commands in history are not evidence of success. "
            "The robot is holding while you review. Choose one decision, then inspect the next observation. "
            "Prefer a concrete, object-specific policy instruction; do not repeat a broad multi-object goal. "
            "Use intervention only for a small correction whose direction and outcome you can justify from "
            "the supplied action descriptions and observations. Never guess coordinate frames, IK, or joint "
            "directions. Use hold if uncertain or operator help is needed. Use done only when the images show "
            "the entire goal is complete. hold ends this segment for operator input.\n"
            "Return exactly one JSON object with mode (policy, intervention, hold, done), scene (brief visible "
            "evidence), reason (brief rationale), instruction (non-empty only for policy), targets (absolute "
            "robot-unit positions for interventions), duration_s (positive only for motion, "
            "otherwise 0). Unmentioned joints are held. No code, velocities, normalized policy actions, "
            "or simultaneous policy and intervention commands.\n"
            + self.end_effector_contract()
            + f"Policy instructions: {self.config.instructions or 'free-form concrete subtasks'}\n"
            f"Policy execution window: {self.hybrid.policy_window_s}s. "
            f"Maximum intervention duration: {self.hybrid.max_intervention_s}s.\n"
            f"Action contract: {json.dumps({k: asdict(v) for k, v in self.hybrid.limits.items()})}"
        )

    def observation_blocks(self, label: str, obs_processed: dict) -> list[dict]:
        blocks = super().observation_blocks(label, obs_processed)
        pose = {k: float(obs_processed[k]) for k in self.hybrid.limits}
        blocks.append(text_block(f"Measured absolute positions in robot units: {json.dumps(pose)}"))
        if self.kinematics:
            measured = {name: solver.forward(pose) for name, solver in self.kinematics.items()}
            blocks.append(text_block(f"Measured end-effector poses from FK: {json.dumps(measured)}"))
        return blocks

    def end_effector_contract(self):
        if not self.hybrid.end_effectors:
            return "End-effector commands are disabled.\n"
        contracts = {}
        for name, ee in self.hybrid.end_effectors.items():
            contracts[name] = {k: v for k, v in asdict(ee).items() if k not in {"model_path", "joint_names"}}
        return (
            "You may also choose mode=end_effector. Supply ee_targets with EXACTLY ONE arm name mapped to "
            "{position_m:[x,y,z], quaternion_wxyz:[w,x,y,z]}. These are absolute poses in that arm's "
            "documented model base frame, not camera pixels or a shared world frame. Use current FK as "
            "the reference for small corrections; preserve its quaternion unless intentionally rotating. "
            "Local IK converts the pose to radians; never calculate joint angles yourself. Raw joint "
            "targets for these arms are forbidden. targets may still contain a gripper command; other "
            "joints/grippers hold. instruction must be empty and duration_s positive. For all other modes "
            "ee_targets must be {}. IK rejects unreachable or excessive moves. Joint interpolation follows "
            "IK; it is not a straight Cartesian path or collision-aware plan. Do not propose motions near "
            "obstacles, the other arm, or the table without visible clearance. Do not infer a camera-to-base "
            "transform from this contract. If direction is uncertain use the policy or hold.\n"
            f"End-effector contracts: {json.dumps(contracts)}\n"
        )

    def parse_reply(self, reply: object, query: PolicyQuery, task: str) -> PlannerDecision | str:
        if query.kind is QueryKind.VQA:
            return super().parse_reply(reply, query, task)
        decision = PlannerDecision.parse(reply)
        if decision.mode == "policy" and self.config.instructions:
            allowed = {normalize_instruction(s) for s in self.config.instructions}
            if normalize_instruction(decision.instruction) not in allowed:
                raise ValueError("Hybrid policy instruction is outside the allowed vocabulary")
        return decision
