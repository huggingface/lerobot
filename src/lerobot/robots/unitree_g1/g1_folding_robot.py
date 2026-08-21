#!/usr/bin/env python

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

"""Present a Unitree G1 to an OpenArm policy as though it were an OpenArm.

The folding checkpoint only knows the OpenArm, so rather than teach it a second embodiment
this wraps the G1 at the robot boundary and retargets in both directions:

    policy (16-D OpenArm, degrees)
        ^  observation.state                 |  action
        |  G1 joints -> FK -> IK -> OpenArm  v  OpenArm -> FK -> IK -> G1 joints
    G1FoldingRobot
        |
    UnitreeG1 + SonicLowerBodyController (legs and waist; arms left free)

Because the policy emits *relative* actions, the state handed back is the anchor its deltas
are applied to, so the reverse map matters as much as the forward one and cannot be stubbed
with a constant.

Shared by the sync (``g1_folding_rollout.py``) and async (``g1_folding_async.py``) launchers.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from lerobot.robots.openarm_follower.openarm_kinematics import (
    GRIP_FULL_DEG,
    GRIPPER_IDX,
    POLICY_ORDER,
)

from .g1_openarm_retarget import IK_ITERS, REVERSE_IK_ITERS, G1OpenArmRetargeter

logger = logging.getLogger(__name__)

SIDES = ("right", "left")

# Past this the IK is returning the closest pose it can rather than the one asked for, which
# on a folding task usually means the target crossed the G1's shorter forward reach.
REACH_WARN_M = 0.05

# The analogue of the OpenArm follower's `max_relative_target`: the G1 has no per-step joint
# clamp of its own, and the arms track hard enough that a single bad solve is worth guarding.
# This rate-limits the *commanded* trajectory, one tick to the next.
DEFAULT_MAX_STEP_DEG = 8.0

# How far the command may get ahead of where the arms actually are. This is the safety half
# of the clamp, and it has to be much looser than the per-tick rate: a position controller
# only pulls as hard as its error, so pinning the command near the measurement caps the
# torque and the arm crawls instead of tracking. Small enough to bound what a jammed joint
# can wind up, large enough to leave normal following error alone.
DEFAULT_MAX_LEAD_DEG = 25.0


def policy_keys() -> list[str]:
    """The 16 OpenArm feature keys the checkpoint was trained on, in POLICY_ORDER."""
    return [f"{name}.pos" for name in POLICY_ORDER]


class G1FoldingRobot:
    """A robot-shaped adapter around ``UnitreeG1`` speaking the OpenArm's 16-D interface.

    Anything not overridden proxies to the wrapped robot, so callers treat this exactly like
    the robot it wraps.
    """

    def __init__(
        self,
        robot,
        ik_iters: int = IK_ITERS,
        reverse_ik_iters: int = REVERSE_IK_ITERS,
        use_waist: bool = False,
        max_step_deg: float = DEFAULT_MAX_STEP_DEG,
        max_lead_deg: float = DEFAULT_MAX_LEAD_DEG,
    ) -> None:
        self._robot = robot
        # get_observation and send_action are called from different threads by the async
        # client, and the retargeter carries warm-start seeds and a MuJoCo data buffer that
        # would race.
        self._lock = threading.Lock()
        self._retarget = G1OpenArmRetargeter(
            use_waist=use_waist, iters=ik_iters, reverse_iters=reverse_ik_iters
        )
        self._policy_keys = policy_keys()
        self._g1_arm_keys = self._retarget.action_keys[:14]
        self._max_step = np.deg2rad(max_step_deg)
        self._max_lead = np.deg2rad(max_lead_deg)
        # The rate limit is measured from the last command, not from the measurement, so the
        # trajectory keeps advancing at a bounded speed whatever the arms are doing.
        self._q_cmd: np.ndarray | None = None
        # The bridge's hands are write-only, so the gripper state reported back is the last
        # command rather than a measurement.
        self._gripper_cmd = dict.fromkeys(SIDES, 0.0)
        self._arm_now: np.ndarray | None = None

    def __getattr__(self, name):
        # Only reached for attributes this class does not define.
        return getattr(self._robot, name)

    @property
    def observation_features(self) -> dict:
        cameras = {k: v for k, v in self._robot.observation_features.items() if not k.endswith(".q")}
        return {**dict.fromkeys(self._policy_keys, float), **cameras}

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(self._policy_keys, float)

    def get_observation(self) -> dict:
        obs = self._robot.get_observation()
        missing = [k for k in self._g1_arm_keys if k not in obs]
        if missing:
            raise RuntimeError(f"observation is missing G1 arm joints: {missing}")

        with self._lock:
            self._arm_now = np.array([float(obs[k]) for k in self._g1_arm_keys])
            state16 = self._retarget.to_openarm(self._arm_now, grippers=self._gripper_cmd)

        out = {k: v for k, v in obs.items() if not k.endswith(".q")}
        out.update(dict(zip(self._policy_keys, (float(v) for v in state16), strict=True)))
        return out

    def send_action(self, action: dict) -> dict:
        missing = [k for k in self._policy_keys if k not in action]
        if missing:
            # A partial action cannot be retargeted: the IK needs whole wrist poses.
            logger.warning(f"skipping action missing {len(missing)} OpenArm keys, e.g. {missing[:2]}")
            return action

        state16 = np.array([float(action[k]) for k in self._policy_keys])
        with self._lock:
            q, err = self._retarget.solve(state16)
            arm_now = self._arm_now
        if err.max() > REACH_WARN_M:
            logger.warning(f"wrist target out of reach by {err.max() * 1e3:.0f} mm")

        q_arm = np.asarray(q[:14], float)
        # Rate limit against the previous command; fall back to the measurement on the first
        # tick so the arms are not asked to jump from wherever they happen to be.
        ref = self._q_cmd if self._q_cmd is not None else arm_now
        if ref is not None:
            q_arm = np.clip(q_arm, ref - self._max_step, ref + self._max_step)
        if arm_now is not None:
            q_arm = np.clip(q_arm, arm_now - self._max_lead, arm_now + self._max_lead)
        self._q_cmd = q_arm

        g1_action = dict(zip(self._g1_arm_keys, (float(v) for v in q_arm), strict=True))
        # Waist joints the IK owns pass through unclamped: they are slow, and the clamp is
        # there to guard the fast high-authority arm joints.
        for key, value in zip(self._retarget.action_keys[14:], q[14:], strict=True):
            g1_action[key] = float(value)

        for side in SIDES:
            # The recorded follower encodes the gripper as negative degrees, 0 closed down to
            # -GRIP_FULL_DEG fully open, while the bridge wants closedness in 0..1.
            opening = min(1.0, abs(float(state16[GRIPPER_IDX[side]])) / GRIP_FULL_DEG)
            self._gripper_cmd[side] = 1.0 - opening
            g1_action[f"{side}_gripper.pos"] = self._gripper_cmd[side]

        self._robot.send_action(g1_action)
        return action

    def reset(self, *args, **kwargs):
        self._retarget.reset()
        self._q_cmd = None
        return self._robot.reset(*args, **kwargs)
