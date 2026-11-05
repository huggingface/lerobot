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

"""Reference-motion processing for SuperMimic: clip loading, resampling and goal features.

Mirrors Uni-SuperMimic's ``MotionLib`` (https://github.com/Motphys/Uni-SuperMimic,
Apache-2.0) operation for operation, so goals built here match the ones the actor was
trained on. A clip is a CSV of ``[root_pos(3), root_quat_xyzw(4), joint_pos(J)]`` rows at
``source_fps``; it is resampled to the control rate with linear/slerp interpolation and
differentiated with central differences.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .geometry import State, mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

GOAL_FRAME_DIM = 73  # displacement(3) + orientation 6D(6) + lin vel(3) + ang vel(3) + q(29) + 0.1 qd(29)
GOAL_JOINT_VELOCITY_SCALE = 0.1


def load_motion_csv(path: str | os.PathLike, num_joints: int) -> torch.Tensor:
    """Read a ``(frames, 7 + num_joints)`` clip; ``#`` lines are comments."""
    rows = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    if rows.shape[0] < 2 or rows.shape[1] != 7 + num_joints:
        raise ValueError(f"{path} must have at least two rows and {7 + num_joints} columns, got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError(f"{path} contains non-finite values")
    return torch.from_numpy(rows.astype(np.float32))


def _gradient(value: torch.Tensor, dt: float) -> torch.Tensor:
    gradient = torch.zeros_like(value)
    gradient[1:-1] = (value[2:] - value[:-2]) / (2 * dt)
    gradient[0] = (value[1] - value[0]) / dt
    gradient[-1] = (value[-1] - value[-2]) / dt
    return gradient


def _angular_velocity(rotation: torch.Tensor, dt: float) -> torch.Tensor:
    angular_velocity = torch.zeros(rotation.shape[:-2] + (3,), dtype=rotation.dtype)
    angular_velocity[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (2 * dt)
    angular_velocity[0] = so3_log(rotation[1] @ rotation[0].transpose(-1, -2)) / dt
    angular_velocity[-1] = so3_log(rotation[-1] @ rotation[-2].transpose(-1, -2)) / dt
    return angular_velocity


class MotionReference:
    """Concatenated reference clips at the control rate, indexed by global frame.

    Root quantities are world-frame: ``root_p`` (F, 3), ``root_R`` (F, 3, 3), ``root_v``
    and ``root_w`` (F, 3). Joint quantities ``q`` and ``qd`` are (F, J) in actor joint order.
    """

    def __init__(
        self,
        clips: Sequence[torch.Tensor],
        names: Sequence[str],
        *,
        num_joints: int,
        control_dt: float = 0.02,
        source_fps: float = 30.0,
    ):
        if not clips or len(clips) != len(names):
            raise ValueError("MotionReference needs one name per clip and at least one clip")
        self.names = list(names)
        self.num_joints = num_joints
        self.control_dt = control_dt
        self.source_fps = source_fps

        root_p, root_rot, root_v, root_w, joint_q, joint_qd, lengths = [], [], [], [], [], [], []
        for raw in clips:
            if raw.ndim != 2 or raw.shape[1] != 7 + num_joints or raw.shape[0] < 2:
                raise ValueError(f"Clip has shape {tuple(raw.shape)}; expected (>=2, {7 + num_joints})")
            position, quat_xyzw, joints = raw[:, :3], raw[:, 3:7], raw[:, 7:]
            quaternion = torch.cat((quat_xyzw[:, 3:4], quat_xyzw[:, :3]), -1)
            source_frames = raw.shape[0]
            duration = (source_frames - 1) / source_fps
            num_frames = math.floor(duration / control_dt) + 1
            sample = torch.arange(num_frames, dtype=torch.float32) * control_dt * source_fps
            lower = torch.clamp(sample.floor().long(), max=source_frames - 2)
            alpha = (sample - lower.float()).clamp(0, 1)
            position = position[lower] * (1 - alpha[:, None]) + position[lower + 1] * alpha[:, None]
            joints = joints[lower] * (1 - alpha[:, None]) + joints[lower + 1] * alpha[:, None]
            quaternion = quat_slerp(quaternion[lower], quaternion[lower + 1], alpha)

            rotation = quat_to_mat(quaternion)
            root_p.append(position)
            root_rot.append(rotation)
            root_v.append(_gradient(position, control_dt))
            root_w.append(_angular_velocity(rotation, control_dt))
            joint_q.append(joints)
            joint_qd.append(_gradient(joints, control_dt))
            lengths.append(num_frames)

        self.root_p = torch.cat(root_p)
        self.root_R = torch.cat(root_rot)
        self.root_v = torch.cat(root_v)
        self.root_w = torch.cat(root_w)
        self.q = torch.cat(joint_q)
        self.qd = torch.cat(joint_qd)

        self.clip_len = torch.tensor(lengths, dtype=torch.long)
        self.clip_start = torch.cumsum(torch.tensor([0, *lengths[:-1]], dtype=torch.long), 0)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = int(sum(lengths))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lengths)])

        root_rotation_t = self.root_R.transpose(-1, -2)
        self.root_v_local = (root_rotation_t @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (root_rotation_t @ self.root_w.unsqueeze(-1)).squeeze(-1)

    @classmethod
    def from_csv(
        cls,
        files: Sequence[str | os.PathLike],
        *,
        num_joints: int,
        control_dt: float = 0.02,
        source_fps: float = 30.0,
    ) -> MotionReference:
        clips = [load_motion_csv(path, num_joints) for path in files]
        names = [Path(path).stem for path in files]
        return cls(clips, names, num_joints=num_joints, control_dt=control_dt, source_fps=source_fps)

    @property
    def num_clips(self) -> int:
        return len(self.names)

    def clip_frames(self, clip: int | str) -> tuple[int, int]:
        """First and last global frame of ``clip`` (index or name), inclusive."""
        index = self.names.index(clip) if isinstance(clip, str) else clip
        return int(self.clip_start[index]), int(self.clip_end[index])

    def future_frames(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Frames ``f+1 … f+K``, clamped at each clip's last frame."""
        end = self.clip_end[self.clip_id[frames]]
        future = frames.unsqueeze(-1) + torch.arange(1, num_frames + 1, device=frames.device)
        return torch.minimum(future, end.unsqueeze(-1))

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """``(B, K * 73)`` future reference features in the current reference heading frame."""
        future = self.future_frames(frames, num_frames)
        heading = yaw_mat(self.root_R[frames])
        heading_t = heading.transpose(-1, -2).unsqueeze(1)
        position_delta = (
            heading_t @ (self.root_p[future] - self.root_p[frames].unsqueeze(1)).unsqueeze(-1)
        ).squeeze(-1)
        rotation = mat_to_6d(heading_t @ self.root_R[future])
        features = torch.cat(
            (
                position_delta,
                rotation,
                self.root_v_local[future],
                self.root_w_local[future],
                self.q[future],
                self.qd[future] * GOAL_JOINT_VELOCITY_SCALE,
            ),
            -1,
        )
        return features.reshape(frames.shape[0], -1)

    def static_goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Goal of a reference standing still at ``frames``: every future frame is the current
        pose, with zero root and joint velocities."""
        heading_t = yaw_mat(self.root_R[frames]).transpose(-1, -2)
        frame = torch.cat(
            (
                torch.zeros(frames.shape[0], 3),
                mat_to_6d(heading_t @ self.root_R[frames]),
                torch.zeros(frames.shape[0], 6),
                self.q[frames],
                torch.zeros(frames.shape[0], self.num_joints),
            ),
            -1,
        )
        return frame.repeat(1, num_frames)

    def ref_state(self, frames: torch.Tensor) -> State:
        return (
            self.root_p[frames],
            self.root_R[frames],
            self.root_v[frames],
            self.root_w[frames],
            self.q[frames],
            self.qd[frames],
        )
