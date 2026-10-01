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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    review_policy_chunks: bool = False
    proposal_execution_steps: int = 15
    proposal_timeout_s: float = 30.0
    max_intervention_s: float = 2.0
    settle_s: float = 0.5
    review_timeout_s: float = 60.0
    max_observation_age_s: float = 0.2
    # Cap consecutive direct corrections; a policy step resets this counter.
    max_consecutive_interventions: int = 3

    def __post_init__(self):
        for value in (
            self.policy_window_s,
            self.proposal_timeout_s,
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
        if (
            isinstance(self.proposal_execution_steps, bool)
            or not isinstance(self.proposal_execution_steps, int)
            or self.proposal_execution_steps < 1
        ):
            raise ValueError("proposal_execution_steps must be a positive integer")


@dataclass(frozen=True)
class PlannerDecision:
    mode: str
    scene: str
    reason: str
    instruction: str
    targets: dict[str, float]
    duration_s: float
    ee_targets: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    execution_status: str = ""
    intent_status: str = ""

    @classmethod
    def parse(cls, reply: object) -> PlannerDecision:
        fields = {"mode", "scene", "reason", "instruction", "targets", "duration_s"}
        optional = {"ee_targets", "execution_status", "intent_status"}
        if not isinstance(reply, dict) or not fields <= set(reply) or set(reply) - fields - optional:
            raise ValueError(f"Hybrid reply must contain exactly {sorted(fields)}")
        if reply["mode"] not in {"accept", "policy", "intervention", "end_effector", "hold", "done"}:
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
            raise ValueError(
                f"{reply['mode']} mode requires targets={{}}, ee_targets={{}}, and duration_s=0; "
                "the controller sets the policy execution window"
            )
        if (reply["mode"] == "policy") != bool(reply["instruction"].strip()):
            raise ValueError("Only policy mode must supply a non-empty instruction")
        if reply.get("execution_status", "") not in {
            "",
            "not_started",
            "progressing",
            "failed",
            "uncertain",
            "recovered",
        }:
            raise ValueError("Invalid execution_status")
        if reply.get("intent_status", "") not in {"", "aligned", "misaligned", "uncertain"}:
            raise ValueError("Invalid intent_status")
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

    def __call__(self, obs_processed: dict, query: PolicyQuery, task: str) -> PlannerDecision | str:
        if query.kind is QueryKind.VQA:
            return super().__call__(obs_processed, query, task)
        messages = self.build_messages(obs_processed, query, task)
        if self.config.log_path and "_hybrid_proposal" in obs_processed:
            path = Path(self.config.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(
                    json.dumps({"event": "proposal_review", "proposal": self.proposal_context(obs_processed)})
                    + "\n"
                )
        for attempt in range(2):
            started = time.perf_counter()
            try:
                reply = self.client.generate_json([messages])[0]
            except Exception as exc:
                self._log_exchange(query, task, started, reply=None, returned=None, error=exc)
                raise
            try:
                decision = self.parse_reply(reply, query, task)
            except (ValueError, TypeError) as exc:
                self._log_exchange(query, task, started, reply=reply, returned=None, error=exc)
                if attempt:
                    raise
                # Repair the reply contract once, never relax or clip a motion proposal.
                # The engine's original review deadline/epoch still covers both requests.
                messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(reply)},
                    {
                        "role": "user",
                        "content": (
                            f"Your reply was rejected before execution: {exc}. "
                            "Return one corrected decision using the original observation and contract. "
                            "For accept, policy, hold or done: targets={}, ee_targets={}, duration_s=0. "
                            "The controller owns policy_window_s; do not copy it into duration_s. "
                            "Only intervention/end_effector use a positive duration. "
                            "Do not change mode to authorize motion just to satisfy the format."
                        ),
                    },
                ]
                continue
            self._log_exchange(query, task, started, reply=reply, returned=decision, error=None)
            return decision
        raise RuntimeError("Unreachable hybrid reply loop")

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
            "Be patient with the policy: startup, reaching, grasp retries, and recovery can take time. "
            "An unfinished subtask, small motion, or unchanged images across reviews is not by itself a "
            "reason to stop. Continue the same policy instruction while it remains appropriate and safe; "
            "do not impose a fixed attempt count or declare controller failure from slow visual progress. "
            "When repeated policy windows show little movement or the same failed approach, actively look "
            "for a small end-effector correction that can help the policy recover. If end-effector control "
            "is available and the images, measured FK, and frame contract justify a correction with clear "
            "space, choose end_effector rather than endlessly repeating an ineffective instruction. "
            "Move one arm by a small bounded amount from its measured pose, preserve orientation unless "
            "a rotation is justified, inspect the result, then return control to the policy. Lack of "
            "movement alone must not produce hold. Judge progress across policy execution windows; "
            "the intentional stationary hold during your API review is not a policy failure. "
            "Use intervention only for a small correction whose direction and outcome you can justify from "
            "the supplied action descriptions and observations. Never guess coordinate frames, IK, or joint "
            "directions. Uncertainty about a direct correction alone is a reason to continue a safe policy "
            "subtask, not to hold. Use hold for a concrete safety concern or a problem requiring operator "
            "help; explain the observed evidence. Use done only when the images show "
            "the entire goal is complete. hold ends this segment for operator input.\n"
            "Return exactly one JSON object with mode, scene, reason, instruction, targets, ee_targets, "
            "duration_s. scene and reason must be non-empty text. Choose one mode:\n"
            "- policy: instruction is a concrete subtask; targets={}, ee_targets={}, duration_s=0. "
            "The controller sets the execution window; NEVER put that window in duration_s.\n"
            "- intervention: instruction is empty, targets contains absolute robot-unit positions, "
            "ee_targets={}, and duration_s is positive and bounded.\n"
            "- end_effector: available only with the contract below; instruction is empty, ee_targets "
            "contains one tool pose, targets may contain a gripper command, and duration_s is positive.\n"
            "- hold or done: instruction is empty, targets={}, ee_targets={}, duration_s=0.\n"
            "Unmentioned joints are held. No code, velocities, normalized policy actions, "
            "or simultaneous policy and intervention commands.\n"
            + self.end_effector_contract()
            + self.proposal_contract()
            + f"Policy instructions: {self.config.instructions or 'free-form concrete subtasks'}\n"
            f"Controller-owned policy_window_s: {self.hybrid.policy_window_s}s (policy replies still use duration_s=0). "
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
            cameras = {
                name: solver.camera_poses(pose)
                for name, solver in self.kinematics.items()
                if solver.config.camera_mounts
            }
            if cameras:
                blocks.append(
                    text_block(f"Tool-mounted camera poses in each arm's base: {json.dumps(cameras)}")
                )
        if "_hybrid_proposal" in obs_processed:
            blocks.append(
                text_block(
                    f"Unexecuted policy proposal (robot-only FK, not object predictions): {json.dumps(self.proposal_context(obs_processed))}"
                )
            )
        return blocks

    def proposal_context(self, obs):
        proposal = dict(obs["_hybrid_proposal"])
        proposal["end_effector_trajectory"] = [
            {
                name: solver.forward(dict(zip(proposal["action_keys"], row, strict=True)))
                for name, solver in self.kinematics.items()
            }
            for row in proposal["actions"]
        ]
        return proposal

    def proposal_contract(self):
        if not self.hybrid.review_policy_chunks:
            return "The accept mode is disabled; future policy chunks are not supplied.\n"
        return (
            "PROPOSAL REVIEW: Review the supplied unexecuted policy chunk before any of it moves the robot. "
            "Add execution_status (not_started/progressing/failed/uncertain/recovered) and intent_status "
            "(aligned/misaligned/uncertain) to your JSON. In reason separately explain the observed result "
            "of the previous execution and whether this proposed trajectory pursues the right next subgoal. "
            "Use previous/current RGB and measured poses for outcomes, and the proposed FK trajectory and "
            "gripper sequence for predicted motion. Robot FK does not predict object motion, contact or success. "
            "Uncertainty or slow startup is not failure; permit self-recovery. "
            "Choose mode=accept with empty instruction/targets/ee_targets and duration_s=0 to execute only "
            "the advertised prefix of this exact proposal. The remainder is discarded, followed by fresh "
            "inference and review. Prefer accepting an appropriate proposal under the current instruction. "
            "mode=policy changes the instruction and requests another proposal; it does NOT authorize that "
            "new proposal to execute without review. An intervention/end_effector takeover requires "
            "execution_status=failed or intent_status=misaligned, supported by evidence in reason. "
            "After recovery, accept a suitable policy proposal to hand control back.\n"
        )

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
            "obstacles, the other arm, or the table without visible clearance. Use a camera-to-base "
            "transform only when explicitly supplied. An estimated mount gives approximate directions, "
            "not calibrated object coordinates. Honour its provenance and assumptions; a pixel still "
            "needs intrinsics and depth to become a metric 3D point. Camera axes rotate with the wrist. "
            "Missing camera calibration does not forbid every correction: "
            "use documented base axes, current measured FK, and supplied policy FK trajectories to reason "
            "about a small recovery in that known frame when clearance and benefit are evident. For example, "
            "a small lift along a documented upward axis may be justified by a failed grasp and visible "
            "clearance; preserve orientation. Do not assume that image left/right equals base X/Y, or "
            "invent a metric cube location. Proposed FK is a motion hypothesis, not a camera calibration. "
            "Explain the geometric evidence for the correction; if its direction is unknown, continue an "
            "appropriate policy rather than inventing coordinates.\n"
            f"End-effector contracts: {json.dumps(contracts)}\n"
        )

    def parse_reply(self, reply: object, query: PolicyQuery, task: str) -> PlannerDecision | str:
        if query.kind is QueryKind.VQA:
            return super().parse_reply(reply, query, task)
        decision = PlannerDecision.parse(reply)
        if decision.mode == "accept" and not self.hybrid.review_policy_chunks:
            raise ValueError("accept requires review_policy_chunks")
        if self.hybrid.review_policy_chunks:
            if not decision.execution_status or not decision.intent_status:
                raise ValueError("Proposal review requires execution_status and intent_status")
            if decision.mode == "accept" and decision.intent_status == "misaligned":
                raise ValueError("Cannot accept a proposal assessed as misaligned")
            if (
                decision.mode in {"intervention", "end_effector"}
                and decision.execution_status != "failed"
                and decision.intent_status != "misaligned"
            ):
                raise ValueError(
                    "A correction requires observed execution failure or misaligned proposal intent"
                )
        if decision.mode == "policy" and self.config.instructions:
            allowed = {normalize_instruction(s) for s in self.config.instructions}
            if normalize_instruction(decision.instruction) not in allowed:
                raise ValueError("Hybrid policy instruction is outside the allowed vocabulary")
        return decision
