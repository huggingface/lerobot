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

"""Training windows of the single-arm checkpoint: the window structure and the pose helpers.

Vendored from runwayml/praxis-1-oss at commit 8dad1a4 (``praxis/train/data.py``), the subset
that does not read files. The DROID file reader of the original (``DroidWindowDataset``) is
replaced by ``LeRobotDataset`` with the window of ``PraxisConfig.observation_delta_indices``;
``PraxisPolicy.forward`` builds the windows from the batch with the functions here.

The recipe of the checkpoint, which the LeRobot path keeps:

* Window: 93 consecutive frames at 15 Hz in 4 chunks of 21, 24, 24 and 24 frames
  (``chunking`` of the config).
* Start: uniform over the frames that hold a window plus a margin of 0.5 s
  (:func:`draw_anchor`; in LeRobot the sampler with ``drop_n_last_frames``).
* Poses: DROID gives the flange pose ``[x, y, z, roll, pitch, yaw]`` with extrinsic x-y-z
  Euler angles. :func:`droid_pose8` makes the quaternion with the operations of
  ``scipy.spatial.transform.Rotation``, so the bits are those of the original data code. The
  gripper is clipped to [0, 1].
* Actions: the commanded pose of each frame relative to the observed pose at the first frame
  of its chunk (``xyz - xyz_anchor``, ``R @ R_anchor^T``), the gripper absolute
  (``robot_io.pose8_to_10d``). The proprio is the observed absolute pose at each chunk start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.utils.rotation import Rotation

__all__ = [
    "LAST_FRAME_S",
    "MARGIN_S",
    "TrainingWindow",
    "draw_anchor",
    "droid_pose8",
    "episode_duration",
    "window_fits",
]

# The window plus this much video must fit in an episode.
MARGIN_S = 0.5
# The last frame of an episode lasts this long (the 60 fps raw file).
LAST_FRAME_S = 1.0 / 60.0
# Tolerance of the float comparisons of times.
TIME_EPS = 1e-9


@dataclass
class TrainingWindow:
    """One training window: the canvas frames, the prompt, the actions and the proprio.

    ``frames`` is ``(T, 3, H, W)`` uint8: the tile canvas of each window
    frame, as the video encoder gets it. ``actions_rel`` is ``(T, A)``: the
    commanded pose of each frame relative to the observed pose at the start
    of its chunk. ``proprio_abs`` is ``(K, A)``: the observed absolute pose at
    each chunk start. Both are float32 and not normalized. ``meta`` holds
    ``episode``, ``anchor`` (the first frame) and ``source``.
    """

    frames: torch.Tensor
    prompt: str
    actions_rel: np.ndarray
    proprio_abs: np.ndarray
    meta: dict


# Poses
# ---------------------------------------------------------------------------


def droid_pose8(cartesian: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert DROID poses to pose8 rows ``[xyz3, quat_wxyz4, grip1]`` (float32).

    ``cartesian`` is ``(T, 6)``: the flange pose ``[x, y, z, roll, pitch,
    yaw]`` with extrinsic x-y-z Euler angles. ``gripper`` is ``(T,)``. The
    pose stays in the flange frame (``pose_frame: "flange"``). The quaternion
    comes from the Euler angles, the matrix and back
    (:meth:`lerobot.utils.rotation.Rotation.matrix_to_quat_xyzw`), with the operations and
    the sign of ``scipy.spatial.transform``. The function clips the gripper
    to [0, 1].
    """
    pose6 = np.asarray(cartesian, dtype=np.float64).reshape(-1, 6)
    rot = Rotation.euler_xyz_to_matrix(pose6[:, 3:6])
    quat = Rotation.matrix_to_quat_xyzw(rot)[:, [3, 0, 1, 2]]
    grip = np.clip(np.asarray(gripper, dtype=np.float64).reshape(-1), 0.0, 1.0)
    return np.concatenate([pose6[:, :3], quat, grip[:, None]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Window start
# ---------------------------------------------------------------------------


def episode_duration(n_frames: int, fps: float) -> float:
    """Return the video duration (s) of an episode with ``n_frames`` frames.

    The frames are ``1 / fps`` apart, and the last frame lasts
    :data:`LAST_FRAME_S`.
    """
    return (int(n_frames) - 1) / float(fps) + LAST_FRAME_S


def window_fits(n_frames: int, window_frames: int, fps: float) -> bool:
    """Return True when an episode holds a window plus :data:`MARGIN_S` of video."""
    need = window_frames / float(fps) + MARGIN_S
    return episode_duration(n_frames, fps) >= need - TIME_EPS


def draw_anchor(n_frames: int, window_frames: int, fps: float, rng: np.random.Generator) -> int:
    """Draw the first frame of a window of an episode with ``n_frames`` frames.

    The start time is uniform in ``[0, duration - window]`` (s). It then
    snaps down to the frame grid, and the window stays inside the episode.
    Raises ``ValueError`` when the episode does not hold the window plus
    :data:`MARGIN_S`.
    """
    if not window_fits(n_frames, window_frames, fps):
        raise ValueError(f"{n_frames} frames do not hold a window of {window_frames}")
    span = max(episode_duration(n_frames, fps) - window_frames / float(fps), 0.0)
    start = float(rng.uniform(0.0, span))
    anchor = math.floor(start * float(fps) + TIME_EPS)
    return int(min(max(anchor, 0), int(n_frames) - int(window_frames)))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
