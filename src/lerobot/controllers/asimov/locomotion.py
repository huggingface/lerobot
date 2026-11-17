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

"""Menlo's lower-body Asimov 1 locomotion policies, run from LeRobot instead of onboard.

The policy drives the 12 leg joints from a velocity command; the waist, arms and neck take
position targets from elsewhere (teleop), as in training. Joint vectors here are the 25
firmware slots of ``menlo.asimov.robots.ASIMOV_1_BIPED_JOINTS`` in joint space: each ankle's
two slots hold its pitch and roll, not the A/B motors the firmware reports.
"""

import json
import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.asimov.asimov import ASIMOV_JOINTS

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Menlo/asimov1-locomotion-26-10-26-lower-body-27dof"

NUM_JOINTS = len(ASIMOV_JOINTS)
NUM_LEGS = 12
WAIST = ASIMOV_JOINTS.index("Waist_Yaw")

# The policy's action order is the firmware's leg slot order, ankles as (pitch, roll).
POLICY_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
POLICY_OBSERVATIONS = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos_slot01",
    "joint_pos_slot23",
    "joint_pos_slot45",
    "joint_vel_slot01",
    "joint_vel_slot23",
    "joint_vel_slot45",
    "actions",
)
# Leg joints grouped by motor slot, as indices into POLICY_JOINTS.
SLOT_ORDER = np.array([0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11])
ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.1

# Upper-body home pose and PD gains from the checkpoint's env.yaml, which the ONNX metadata
# does not carry. Not observed by the policy: teleop may move these joints freely.
UPPER_BODY_DEFAULTS = {
    "L_Shoulder_Pitch": -0.25,
    "R_Shoulder_Pitch": 0.25,
    "L_Shoulder_Roll": -0.05,
    "R_Shoulder_Roll": 0.05,
    "L_Elbow": 0.4,
    "R_Elbow": -0.4,
}
UPPER_BODY_GAINS = {
    "Shoulder_Pitch": (57.0, 5.0),
    "Shoulder_Roll": (86.0, 5.0),
    "Shoulder_Yaw": (96.0, 5.0),
    "Elbow": (40.0, 2.0),
    "Wrist_Yaw": (40.0, 2.0),
    "Waist_Yaw": (65.0, 5.0),
    "Neck_Yaw": (40.0, 2.0),
    "Neck_Pitch": (40.0, 2.0),
}


def _upper_body_gain(name: str) -> tuple[float, float]:
    for suffix, gains in UPPER_BODY_GAINS.items():
        if name.endswith(suffix):
            return gains
    raise KeyError(f"No PD gains for Asimov joint {name!r}")


def _metadata(session: ort.InferenceSession) -> dict:
    meta = {}
    for key, value in session.get_modelmeta().custom_metadata_map.items():
        try:
            meta[key] = json.loads(value)
        except json.JSONDecodeError:
            meta[key] = value
    return meta


def _torso_frame(vector: np.ndarray, waist_yaw: float) -> np.ndarray:
    """Rotate a pelvis-frame vector into the torso frame, which turns with the waist yaw."""
    c, s = np.cos(waist_yaw), np.sin(waist_yaw)
    x, y, z = vector
    return np.array([c * x + s * y, -s * x + c * y, z], dtype=np.float32)


class AsimovLocomotionController:
    """Velocity-commanded leg policy for Asimov 1, with upper-body targets passed through."""

    def __init__(self, repo_id: str = DEFAULT_REPO_ID):
        path = hf_hub_download(repo_id=repo_id, filename="policy.onnx")
        options = ort.SessionOptions()
        # A small MLP at 50 Hz in a background thread: one thread is enough and avoids
        # starving teleop IK and cameras.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        meta = _metadata(self.session)
        expected = {
            "joint_names": list(POLICY_JOINTS),
            "observation_names": list(POLICY_OBSERVATIONS),
            "obs_dim": 45,
            "action_dim": NUM_LEGS,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{repo_id} has {key}={meta.get(key)!r}; this controller expects {value!r}")
        self.control_dt = 1.0 / float(meta["policy_rate_hz"])
        self.input_name = self.session.get_inputs()[0].name

        self.action_scale = np.asarray(meta["action_scale"], dtype=np.float32)
        self.leg_defaults = np.asarray(meta["action_offset"], dtype=np.float32)
        clip = meta["action_clip"]
        self.clip_low = np.array([-np.inf if lo is None else lo for lo, _ in clip], dtype=np.float32)
        self.clip_high = np.array([np.inf if hi is None else hi for _, hi in clip], dtype=np.float32)

        self.default_pose: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.default_pose[:NUM_LEGS] = self.leg_defaults
        self.kp: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.kd: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.kp[:NUM_LEGS] = meta["joint_stiffness"]
        self.kd[:NUM_LEGS] = meta["joint_damping"]
        for i, name in enumerate(ASIMOV_JOINTS[NUM_LEGS:], start=NUM_LEGS):
            self.default_pose[i] = UPPER_BODY_DEFAULTS.get(name, 0.0)
            self.kp[i], self.kd[i] = _upper_body_gain(name)

        self.last_action: np.ndarray = np.zeros(NUM_LEGS, dtype=np.float32)
        logger.info(f"Asimov locomotion policy loaded from {repo_id} ({1 / self.control_dt:.0f} Hz)")

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def observation(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        """Build the 45-value actor input from pelvis-IMU state in joint space."""
        waist, waist_vel = float(joint_pos[WAIST]), float(joint_vel[WAIST])
        ang_vel = _torso_frame(np.asarray(gyro, dtype=np.float32) + [0.0, 0.0, waist_vel], waist)
        legs_pos = (np.asarray(joint_pos[:NUM_LEGS], dtype=np.float32) - self.leg_defaults)[SLOT_ORDER]
        legs_vel = np.asarray(joint_vel[:NUM_LEGS], dtype=np.float32)[SLOT_ORDER]
        return np.concatenate(
            [
                ang_vel * ANG_VEL_SCALE,
                _torso_frame(np.asarray(gravity, dtype=np.float32), waist),
                np.asarray(command, dtype=np.float32),
                legs_pos,
                legs_vel * JOINT_VEL_SCALE,
                self.last_action,
            ]
        ).astype(np.float32)

    def run_step(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        command: np.ndarray,
        upper_body: np.ndarray,
    ) -> np.ndarray:
        """One policy tick: ``command`` is (vx m/s, vy m/s, wz rad/s); returns 25 joint targets.

        Slots ``12:`` of the result are ``upper_body[12:]`` unchanged.
        """
        obs = self.observation(joint_pos, joint_vel, gyro, gravity, command)
        action = self.session.run(None, {self.input_name: obs[None]})[0][0].astype(np.float32)
        self.last_action = action
        targets = np.asarray(upper_body, dtype=np.float32).copy()
        targets[:NUM_LEGS] = np.clip(
            self.leg_defaults + self.action_scale * action, self.clip_low, self.clip_high
        )
        return targets
