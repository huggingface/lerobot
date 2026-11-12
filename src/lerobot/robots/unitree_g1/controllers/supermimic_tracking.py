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

"""SuperMimic motion-tracking controller for the Unitree G1 (29 DoF).

Plays reference clips with a SuperMimic actor exported from Uni-SuperMimic
(https://github.com/Motphys/Uni-SuperMimic). Each 50 Hz tick builds the actor's inputs exactly
as the upstream evaluator does — a 10-frame history of 99-D proprioceptive observations and
the next 10 reference frames as goal — and maps the raw action to joint targets
``default_q + offset + action_scale * action`` tracked by the manifest's PD gains.

One actor tracks every loaded clip. R1 / L1 on the Unitree remote (or the simulator's
gamepad) select the next / previous clip, START plays the selected clip or replays a
finished one.

Phases: ``hold`` balances at the selected clip's first frame (the goal is that pose standing
still) until START or ``controller.play`` is pressed, or right away with ``autoplay``;
``track`` advances one reference frame per tick; ``done`` balances at the clip's last frame.
The robot eases into the first clip's first frame (``default_angles``) before the controller
takes over. Switching clips mid-motion is not part of the actor's training and may fail.

The reference orientation fed to the actor is rotated about the vertical so the selected
clip's first frame faces the robot's heading; the goal itself is expressed in the reference
heading frame and needs no alignment.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.policies.supermimic import (
    GOAL_FRAME_DIM,
    MotionReference,
    ObservationHistory,
    SuperMimicExport,
    load_export,
    proprio_observation,
)
from lerobot.policies.supermimic.geometry import mat_to_quat, quat_to_mat, yaw_angle, z_rotation

from ..g1_utils import NUM_MOTORS, G1_29_JointIndex

if TYPE_CHECKING:
    from ..config_unitree_g1 import UnitreeG1Config

logger = logging.getLogger(__name__)

PLAY_KEY = "controller.play"
CLIP_KEY = "controller.clip"
FRAME_KEY = "controller.frame"
PHASE_KEY = "controller.phase"
DONE_KEY = "controller.done"
INFERENCE_MS_KEY = "controller.inference_ms"

# Bits of the Unitree remote key field (bytes 2-3 of ``wireless_remote``, little-endian).
REMOTE_R1 = 1 << 0
REMOTE_L1 = 1 << 1
REMOTE_START = 1 << 2

_LATENCY_WINDOW = 20_000


class Phase(IntEnum):
    HOLD = 0
    TRACK = 1
    DONE = 2


def dds_joint_name(joint_name: str) -> str:
    """``left_hip_pitch_joint`` -> ``kLeftHipPitch``, the ``G1_29_JointIndex`` naming."""
    return "k" + "".join(part.capitalize() for part in joint_name.removesuffix("_joint").split("_"))


def remote_keys(wireless_remote) -> int:
    """Pressed-key bit field of a raw Unitree ``wireless_remote`` buffer (0 when absent)."""
    if wireless_remote is None or len(wireless_remote) < 4:
        return 0
    return int(wireless_remote[2]) | int(wireless_remote[3]) << 8


def motion_files(path: str | Path) -> list[Path]:
    """``path`` itself if it is a CSV file, else the CSVs in the directory sorted by name."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"No reference motion CSVs at {path}")
    return files


class SuperMimicController:
    """Tracks the clips of a ``MotionReference`` with a SuperMimic actor at the export's control rate."""

    def __init__(
        self,
        export: SuperMimicExport,
        reference: MotionReference,
        *,
        start_clip: int | str = 0,
        autoplay: bool = False,
        auto_advance: bool = False,
    ):
        spec = export.spec
        if reference.num_joints != spec.num_joints or abs(reference.control_dt - spec.control_dt) > 1e-9:
            raise ValueError("Reference motion must be resampled to the actor's joints and control period")
        if isinstance(start_clip, str):
            if start_clip not in reference.names:
                raise ValueError(f"Unknown clip {start_clip!r}; loaded clips are {reference.names}")
            start_clip = reference.names.index(start_clip)
        if not 0 <= start_clip < reference.num_clips:
            raise ValueError(f"Clip index {start_clip} is outside the {reference.num_clips} loaded clips")
        if bool((reference.clip_len < 2).any()):
            raise ValueError("Every clip needs at least two control frames")

        self.export = export
        self.spec = spec
        self.reference = reference
        self.control_dt = spec.control_dt
        self.start_clip = start_clip
        self.autoplay = autoplay
        self.auto_advance = auto_advance
        self.goal_frames = export.goal_frames

        self.slots = np.array([G1_29_JointIndex[dds_joint_name(name)].value for name in spec.joint_names])
        if sorted(self.slots.tolist()) != list(range(NUM_MOTORS)):
            raise ValueError("The actor's joints must cover every G1 29-DoF motor exactly once")
        self._joint_keys = [f"{G1_29_JointIndex(slot).name}.q" for slot in self.slots]

        first_frame, _ = reference.clip_frames(start_clip)
        self.kp = self._to_slots(spec.array("kp"))
        self.kd = self._to_slots(spec.array("kd"))
        self.default_angles = self._to_slots(reference.q[first_frame].numpy())
        self.action_ft = {PLAY_KEY: float}

        self._default_q = torch.from_numpy(spec.array("default_q"))
        self._joint_offset = torch.from_numpy(export.joint_offset)
        self._history = ObservationHistory(export.history_length)
        self._latencies: deque[float] = deque(maxlen=_LATENCY_WINDOW)
        self._warm_up()
        self.reset()

    @classmethod
    def from_robot_config(cls, config: UnitreeG1Config) -> SuperMimicController:
        settings = config.supermimic
        if settings.export is None or settings.motions is None:
            raise ValueError("SuperMimicController needs supermimic.export and supermimic.motions")
        export = load_export(
            settings.export, revision=settings.export_revision, num_threads=settings.num_threads
        )
        if settings.check_golden_vectors:
            error = export.check_golden_vectors()
            logger.info("SuperMimic golden vectors reproduced (max abs error %.2e)", error)
        reference = MotionReference.from_csv(
            motion_files(settings.motions),
            num_joints=export.spec.num_joints,
            control_dt=export.spec.control_dt,
            source_fps=settings.motion_fps,
        )
        logger.info("SuperMimic loaded %d clips: %s", reference.num_clips, ", ".join(reference.names))
        return cls(
            export,
            reference,
            start_clip=settings.start_clip if settings.start_clip is not None else 0,
            autoplay=settings.autoplay,
            auto_advance=settings.auto_advance,
        )

    def _to_slots(self, values: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.zeros(NUM_MOTORS, dtype=np.float32)
        out[self.slots] = values
        return out

    @property
    def default_base_pose(self) -> np.ndarray:
        """Pelvis ``[x, y, z, qw, qx, qy, qz]`` of the start clip's first frame, at the world origin."""
        frame, _ = self.reference.clip_frames(self.start_clip)
        position = self.reference.root_p[frame].numpy().copy()
        position[:2] = 0.0
        return np.concatenate((position, mat_to_quat(self.reference.root_R[frame]).numpy())).astype(
            np.float32
        )

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def clip(self) -> int:
        return self._clip

    @property
    def clip_name(self) -> str:
        return self.reference.names[self._clip]

    @property
    def frame(self) -> int:
        """Global reference frame of the current tick."""
        return self._frame

    def latency_stats(self) -> dict[str, float]:
        """Actor-tick compute time in milliseconds (observation, goal, inference, action mapping)."""
        if not self._latencies:
            return {}
        values = np.asarray(self._latencies) * 1e3
        return {
            "count": float(values.size),
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        }

    @torch.inference_mode()
    def _warm_up(self) -> None:
        # TorchScript's profiling executor optimizes the graph over the first calls.
        obs = torch.zeros(1, self.export.history_length, self._history.obs_dim)
        goal = torch.zeros(1, self.goal_frames * GOAL_FRAME_DIM)
        for _ in range(3):
            self.export.actor(obs, goal)

    def reset(self) -> None:
        self._history.reset()
        self._previous_action = torch.zeros(self.spec.num_joints)
        self._previous_keys = 0
        self._previous_play = False
        self._last_targets: dict[str, float] = {}
        self._last_inference_s = 0.0
        self.select_clip(self.start_clip)

    def select_clip(self, clip: int) -> None:
        """Rewind to the first frame of ``clip`` (wrapped into range), playing it with ``autoplay``."""
        self._clip = clip % self.reference.num_clips
        self._frame, self._last_frame = self.reference.clip_frames(self._clip)
        self._heading_alignment: torch.Tensor | None = None
        self._phase = Phase.TRACK if self.autoplay else Phase.HOLD
        logger.info(
            "SuperMimic clip %d/%d: %s (%s)",
            self._clip + 1,
            self.reference.num_clips,
            self.clip_name,
            "playing" if self._phase == Phase.TRACK else "press START to play",
        )

    def observation_state(self) -> dict[str, float]:
        first_frame, _ = self.reference.clip_frames(self._clip)
        return {
            CLIP_KEY: float(self._clip),
            FRAME_KEY: float(self._frame - first_frame),
            PHASE_KEY: float(self._phase),
            DONE_KEY: float(self._phase == Phase.DONE),
            INFERENCE_MS_KEY: self._last_inference_s * 1e3,
        }

    def _handle_input(self, action: dict, lowstate) -> None:
        keys = remote_keys(getattr(lowstate, "wireless_remote", None))
        pressed = keys & ~self._previous_keys
        self._previous_keys = keys
        play = float(action.get(PLAY_KEY, 0.0)) > 0.5
        play_pressed = play and not self._previous_play
        self._previous_play = play

        if pressed & REMOTE_R1:
            self.select_clip(self._clip + 1)
        elif pressed & REMOTE_L1:
            self.select_clip(self._clip - 1)
        elif pressed & REMOTE_START or play_pressed:
            if self._phase == Phase.DONE:
                self.select_clip(self._clip)
            if self._phase == Phase.HOLD:
                self._phase = Phase.TRACK
                logger.info("SuperMimic playing %s", self.clip_name)

    def _read_state(self, lowstate) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motors = lowstate.motor_state
        q = torch.tensor([motors[slot].q for slot in self.slots], dtype=torch.float32)
        dq = torch.tensor([motors[slot].dq for slot in self.slots], dtype=torch.float32)
        rotation = quat_to_mat(torch.tensor(lowstate.imu_state.quaternion, dtype=torch.float32))
        gyro = torch.tensor(lowstate.imu_state.gyroscope, dtype=torch.float32)
        return q, dq, rotation, gyro

    @torch.inference_mode()
    def run_step(self, action: dict, lowstate) -> dict:
        started = time.perf_counter()
        self._handle_input(action, lowstate)

        q, dq, rotation, gyro = self._read_state(lowstate)
        frame = torch.tensor([self._frame])
        if self._heading_alignment is None:
            first_frame, _ = self.reference.clip_frames(self._clip)
            self._heading_alignment = z_rotation(
                yaw_angle(rotation) - yaw_angle(self.reference.root_R[first_frame])
            )
        reference_rotation = self._heading_alignment @ self.reference.root_R[frame[0]]

        obs = proprio_observation(
            rotation,
            gyro,
            q,
            dq,
            self._previous_action,
            reference_rotation,
            self._default_q,
            self._joint_offset,
        )
        history = self._history.push(obs)
        if self._phase == Phase.TRACK:
            goal = self.reference.goal(frame, self.goal_frames)
        else:
            goal = self.reference.static_goal(frame, self.goal_frames)
        raw_action = self.export.actor(history.unsqueeze(0), goal)[0]
        self._previous_action = raw_action
        targets = self.spec.action_to_target(raw_action.numpy(), self.export.joint_offset)

        if self._phase == Phase.TRACK:
            if self._frame < self._last_frame:
                self._frame += 1
            elif self.auto_advance:
                self.select_clip(self._clip + 1)
            else:
                self._phase = Phase.DONE
                logger.info(
                    "SuperMimic finished %s; press START to replay or R1/L1 to switch", self.clip_name
                )

        self._last_targets = {key: float(value) for key, value in zip(self._joint_keys, targets, strict=True)}
        self._last_inference_s = time.perf_counter() - started
        self._latencies.append(self._last_inference_s)
        return self._last_targets
