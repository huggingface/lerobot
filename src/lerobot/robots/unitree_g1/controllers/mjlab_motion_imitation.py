#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""mjlab BeyondMimic motion-imitation controller for the Unitree G1.

Deploys any mjlab BeyondMimic motion-tracking policy (trained without state
estimation and exported to ONNX) together with its reference clip. The pipeline
is motion-agnostic: point it at a different ``(onnx, motion.npz)`` pair and it
will track that clip. The bundled default is the "double spin kick".

The exported policy is a pure function ``actions = policy(obs)`` of a 154-dim
observation. The observation terms (in concatenation order) are::

    command            (58)  ref joint_pos[t] (29) + ref joint_vel[t] (29)
    motion_anchor_ori_b (6)  first two columns of R(q_robot_torso^-1 * q_ref_torso[t])
    base_ang_vel        (3)  pelvis IMU angular velocity (body frame)
    joint_pos          (29)  q - default_q
    joint_vel          (29)  dq
    actions            (29)  previous raw policy output

The reference trajectory (``command`` + reference torso quaternion) is read from
``motion.npz``. Joint order matches ``G1_29_JointIndex`` exactly, so no remapping
is needed. Per-joint action scale and PD gains are read from the ONNX metadata so
the controller always stays consistent with the exported policy.

Note: the policy's anchor body is ``torso_link`` while ``base_ang_vel`` comes from
the pelvis IMU (mjlab's sensors live on ``imu_in_pelvis``). This sim publishes the
pelvis as ``imu_state``, so the torso orientation used for ``motion_anchor_ori_b``
is reconstructed from the pelvis quaternion + the three waist joints.

Override the deployed policy/clip without code changes via env vars:
    MJLAB_ONNX_PATH=/path/to/policy.onnx
    MJLAB_MOTION_PATH=/path/to/motion.npz
"""

import logging
import os

import numpy as np
import onnxruntime as ort

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Policies + reference clips live in examples/unitree_g1/motions/.
_MOTIONS_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "..", "examples", "unitree_g1", "motions")
)
# Default bundled policy + clip (the double spin kick). Override with env vars or
# the onnx_path / motion_path constructor args to deploy any other mjlab policy.
DEFAULT_ONNX_PATH = os.environ.get(
    "MJLAB_ONNX_PATH", os.path.join(_MOTIONS_DIR, "2026-06-26_14-03-33.onnx")
)
DEFAULT_MOTION_PATH = os.environ.get("MJLAB_MOTION_PATH", os.path.join(_MOTIONS_DIR, "motion.npz"))

CONTROL_DT = 0.02  # 50 Hz, matches mjlab decimation=4 * timestep=0.005

# Index of ``torso_link`` in the full 30-body mjlab G1 body ordering (the layout
# stored in motion.npz body_*_w arrays).
TORSO_BODY_INDEX = 15

# Hold the first motion frame for a short settle window before playing the clip,
# so the robot can reach the start pose before the motion begins.
START_HOLD_STEPS = 50


def _parse_floats(meta: dict, key: str) -> np.ndarray:
    return np.array([float(x) for x in meta[key].split(",")], dtype=np.float32)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (w, x, y, z)."""
    q = q / (np.linalg.norm(q) + 1e-8)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _yaw_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only component of a quaternion (w, x, y, z)."""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _pelvis_to_torso_quat(
    pelvis_quat: np.ndarray, waist_yaw: float, waist_roll: float, waist_pitch: float
) -> np.ndarray:
    """Torso-link world orientation from the pelvis quaternion + waist joints.

    The kinematic chain is pelvis -> waist_yaw(z) -> waist_roll(x) ->
    torso_link/waist_pitch(y), and every intermediate body frame is identity, so::

        q_torso = q_pelvis (x) Rz(waist_yaw) (x) Rx(waist_roll) (x) Ry(waist_pitch)
    """
    hy, hr, hp = waist_yaw / 2.0, waist_roll / 2.0, waist_pitch / 2.0
    qz = np.array([np.cos(hy), 0.0, 0.0, np.sin(hy)], dtype=np.float32)
    qx = np.array([np.cos(hr), np.sin(hr), 0.0, 0.0], dtype=np.float32)
    qy = np.array([np.cos(hp), 0.0, np.sin(hp), 0.0], dtype=np.float32)
    return _quat_mul(_quat_mul(_quat_mul(pelvis_quat, qz), qx), qy)


def _ori_6d(q_rel: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix of ``q_rel`` (w, x, y, z).

    Matches mjlab ``matrix_from_quat(...)[..., :2].reshape(-1)``: row-major
    flatten of the first two columns -> [R00, R01, R10, R11, R20, R21].
    """
    q = q_rel / (np.linalg.norm(q_rel) + 1e-8)
    w, x, y, z = q
    two = 2.0
    return np.array(
        [
            1.0 - two * (y * y + z * z),  # R00
            two * (x * y - z * w),  # R01
            two * (x * y + z * w),  # R10
            1.0 - two * (x * x + z * z),  # R11
            two * (x * z - y * w),  # R20
            two * (y * z + x * w),  # R21
        ],
        dtype=np.float32,
    )


class MjlabMotionImitationController:
    """Full-body mjlab BeyondMimic motion-imitation controller for UnitreeG1.

    Motion-agnostic: pass any exported ``(onnx_path, motion_path)`` pair (or set
    the ``MJLAB_ONNX_PATH`` / ``MJLAB_MOTION_PATH`` env vars) to deploy a policy
    trained on a different reference clip.
    """

    control_dt = CONTROL_DT
    full_body = True

    def __init__(
        self,
        onnx_path: str = DEFAULT_ONNX_PATH,
        motion_path: str = DEFAULT_MOTION_PATH,
        imu_is_pelvis: bool = True,
    ):
        logger.info("Loading mjlab motion-imitation controller (%s)...", os.path.basename(onnx_path))
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"mjlab ONNX policy not found: {onnx_path}")
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"mjlab motion file not found: {motion_path}")

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

        meta = dict(self.session.get_modelmeta().custom_metadata_map)

        self.default_q = _parse_floats(meta, "default_joint_pos")
        self.action_scale = _parse_floats(meta, "action_scale")
        self.kp = _parse_floats(meta, "joint_stiffness")
        self.kd = _parse_floats(meta, "joint_damping")
        if not (len(self.default_q) == len(self.action_scale) == len(self.kp) == len(self.kd) == 29):
            raise ValueError("mjlab policy metadata must define 29 values per joint array")

        # ONNX input names: obs (float32 [1, 154]) + time_step ([1, 1]).
        self._inputs = {i.name: i for i in self.session.get_inputs()}
        self._obs_name = next(n for n in self._inputs if n != "time_step")
        self._ts_name = "time_step" if "time_step" in self._inputs else None
        self._ts_dtype = (
            np.int64 if (self._ts_name and "int" in self._inputs[self._ts_name].type) else np.float32
        )

        motion = np.load(motion_path)
        self.ref_joint_pos = motion["joint_pos"].astype(np.float32)  # (T, 29)
        self.ref_joint_vel = motion["joint_vel"].astype(np.float32)  # (T, 29)
        self.ref_torso_quat = motion["body_quat_w"][:, TORSO_BODY_INDEX, :].astype(np.float32)  # (T, 4)
        self.n_frames = int(self.ref_joint_pos.shape[0])

        # The MuJoCo sim publishes the PELVIS (floating base) as imu_state, but the
        # policy's anchor body is torso_link. When True, reconstruct the torso
        # orientation from the pelvis quat + waist joints. On hardware where the IMU
        # already reports the torso, set this False.
        self.imu_is_pelvis = imu_is_pelvis

        self.step = 0
        self.last_action = np.zeros(29, dtype=np.float32)
        # Yaw alignment between the reference clip and the robot's actual heading,
        # captured lazily on the first run_step after a reset.
        self._align_quat: np.ndarray | None = None
        logger.info("mjlab motion-imitation ready: %d frames @ %dHz", self.n_frames, int(1.0 / CONTROL_DT))

    def reset(self) -> None:
        self.step = 0
        self.last_action[:] = 0.0
        self._align_quat = None

    def _motion_index(self) -> int:
        """Hold frame 0 during the settle window, then advance and clamp at the end."""
        idx = self.step - START_HOLD_STEPS
        if idx < 0:
            idx = 0
        return int(min(idx, self.n_frames - 1))

    def run_step(self, action: dict, lowstate) -> dict:
        if lowstate is None:
            return {}

        t = self._motion_index()

        # Robot joint state (native G1_29 order == policy order).
        q = np.empty(29, dtype=np.float32)
        dq = np.empty(29, dtype=np.float32)
        for motor in G1_29_JointIndex:
            i = motor.value
            q[i] = lowstate.motor_state[i].q
            dq[i] = lowstate.motor_state[i].dq

        # base_ang_vel is the pelvis gyro (mjlab's IMU sensors live on imu_in_pelvis),
        # so the raw gyro is already correct. The anchor orientation, however, is the
        # torso_link (anchor_body); reconstruct it from the pelvis quat + waist joints
        # since imu_state reports the pelvis in this sim.
        pelvis_quat = np.array(lowstate.imu_state.quaternion, dtype=np.float32)  # (w, x, y, z)
        gyro = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
        if self.imu_is_pelvis:
            quat = _pelvis_to_torso_quat(
                pelvis_quat,
                float(q[G1_29_JointIndex.kWaistYaw.value]),
                float(q[G1_29_JointIndex.kWaistRoll.value]),
                float(q[G1_29_JointIndex.kWaistPitch.value]),
            )
        else:
            quat = pelvis_quat

        # Heading-align the reference clip to the robot's actual yaw at start.
        # The anchor_ori_b term is NOT yaw-invariant, so a clip that starts at a
        # nonzero world yaw would make the policy see a huge heading error at t=0
        # and spin. Yaw-only, so true roll/pitch tracking is preserved.
        if self._align_quat is None:
            self._align_quat = _quat_mul(_yaw_quat(quat), _quat_inv(_yaw_quat(self.ref_torso_quat[0])))
        ref_torso = _quat_mul(self._align_quat, self.ref_torso_quat[t])

        # motion_anchor_ori_b = first-2-cols of R(q_robot_torso^-1 * q_ref_torso[t]).
        q_rel = _quat_mul(_quat_inv(quat), ref_torso)
        anchor_ori_b = _ori_6d(q_rel)

        obs = np.concatenate(
            [
                self.ref_joint_pos[t],  # command: ref joint pos (29)
                self.ref_joint_vel[t],  # command: ref joint vel (29)
                anchor_ori_b,  # motion_anchor_ori_b (6)
                gyro,  # base_ang_vel (3)
                q - self.default_q,  # joint_pos (29)
                dq,  # joint_vel (29)
                self.last_action,  # actions (29)
            ]
        ).astype(np.float32)

        feeds: dict[str, np.ndarray] = {self._obs_name: obs[None, :]}
        if self._ts_name is not None:
            feeds[self._ts_name] = np.array([[t]], dtype=self._ts_dtype)
        raw_action = self.session.run(["actions"], feeds)[0].reshape(-1).astype(np.float32)
        self.last_action = raw_action

        target_q = self.default_q + raw_action * self.action_scale

        # Advance the motion cursor: hold frame 0 for the settle window, play the
        # clip, then clamp at the final frame (see _motion_index).
        self.step += 1

        return {f"{motor.name}.q": float(target_q[motor.value]) for motor in G1_29_JointIndex}

    def shutdown(self) -> None:
        pass
