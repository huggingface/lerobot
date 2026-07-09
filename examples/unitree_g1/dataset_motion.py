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

"""Load a 29-DoF joint trajectory from a LeRobot dataset episode for SONIC mode 0.

SONIC's locomotion/tracking mode (``encode_mode == 0``) references the robot in
**29-DoF joint space** (see ``build_encoder_obs`` -> ``motion_joint_positions``).
Humanoid teleop datasets like ``BitRobot/HIW-500-lerobot`` store exactly that under
``observation.state`` (29 joints, same G1 index order as ``G1_29_JointIndex``), so
we can feed a recorded episode straight in as the reference and let SONIC try to
track it.

Note the dataset's ``action`` feature is a 23-dim whole-body command (pivot
velocities + EE poses), *not* joint targets -- so we deliberately read
``observation.state`` (the measured 29-DoF joints), not ``action``.

The dataset runs at 30 fps; SONIC ticks at 50 Hz and consumes one reference frame
per tick, so we resample to 50 fps to preserve real-time speed.

Example:
    python examples/unitree_g1/dataset_motion.py \
        --repo-id BitRobot/HIW-500-lerobot --episode 0
"""

from __future__ import annotations

import argparse

import numpy as np

STATE_KEY = "observation.state"
N_JOINTS = 29
SONIC_FPS = 50.0


def _resample(traj: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly resample a (T, D) trajectory from src_fps to dst_fps."""
    if abs(src_fps - dst_fps) < 1e-6:
        return traj.astype(np.float32)
    t_in = np.arange(traj.shape[0]) / src_fps
    dur = t_in[-1] if traj.shape[0] > 1 else 0.0
    t_out = np.arange(0.0, dur + 1e-9, 1.0 / dst_fps)
    out = np.empty((t_out.shape[0], traj.shape[1]), np.float32)
    for j in range(traj.shape[1]):
        out[:, j] = np.interp(t_out, t_in, traj[:, j])
    return out


class DatasetJointMotion:
    """A recorded 29-DoF joint episode, resampled to SONIC's 50 Hz tick.

    Attributes:
        joints:     (T, 29) float32 reference joint positions at ``fps``.
        velocities: (T, 29) float32 finite-difference joint velocities.
        fps:        output rate (50 Hz).
        src_fps:    original dataset rate.
    """

    def __init__(
        self,
        repo_id: str,
        episode: int = 0,
        max_frames: int | None = None,
        root: str | None = None,
        revision: str = "main",
    ):
        # Imported lazily so the heavy datasets stack is only pulled in on demand.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Pin the branch (default "main"): many community datasets aren't tagged with a
        # LeRobot codebase_version, and the version-resolution path crashes on them.
        # A non-PEP440 revision like "main" skips that resolution entirely.
        ds = LeRobotDataset(
            repo_id,
            root=root,
            episodes=[episode],
            revision=revision,
            download_videos=False,  # we only need observation.state, skip ~TB of video
        )
        self.src_fps = float(ds.fps)

        # Read the joint column straight from the underlying table. Going through
        # ds[i] would trigger video decoding (the dataset has camera features) and
        # fail because we intentionally skipped the mp4 download.
        raw = np.asarray(ds.hf_dataset[STATE_KEY], np.float32)  # (T_src, 29)
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise ValueError(f"Episode {episode} of {repo_id} has no usable {STATE_KEY}")
        if raw.shape[1] != N_JOINTS:
            raise ValueError(f"{STATE_KEY} must be (T, {N_JOINTS}), got {raw.shape}")

        self.joints = _resample(raw, self.src_fps, SONIC_FPS)
        if max_frames is not None:
            self.joints = self.joints[:max_frames]
        self.fps = SONIC_FPS

        # Finite-difference velocities (rad/s) at the resampled rate.
        self.velocities = np.gradient(self.joints, axis=0).astype(np.float32) * self.fps
        self.num_frames = self.joints.shape[0]
        self.repo_id = repo_id
        self.episode = episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="BitRobot/HIW-500-lerobot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--revision", default="main", help="Repo branch/tag (default: main)")
    args = parser.parse_args()

    m = DatasetJointMotion(
        args.repo_id, episode=args.episode, max_frames=args.max_frames, revision=args.revision
    )
    dur = m.num_frames / m.fps
    print(f"Loaded {args.repo_id} episode {args.episode}")
    print(f"  src_fps={m.src_fps:.1f} -> {m.fps:.1f}  frames={m.num_frames}  duration={dur:.1f}s")
    print(f"  joints={m.joints.shape} range=[{m.joints.min():.3f}, {m.joints.max():.3f}]")
    print(f"  |velocity| max={np.abs(m.velocities).max():.3f} rad/s")


if __name__ == "__main__":
    main()
