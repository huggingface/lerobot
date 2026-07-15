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

"""Convert SMPL ``.npz`` motion clips into a LeRobotDataset for SONIC replay.

Each dataset frame's ``action`` is the 720-dim ``smpl.*`` window that the
``pico_headset`` teleoperator emits and ``SonicWholeBodyController`` reassembles
into ``encode_mode == 2``. So the resulting dataset can be pushed straight
through ``lerobot-replay`` to drive SONIC whole-body tracking with no headset:

    lerobot-replay \
        --robot.type=unitree_g1 \
        --robot.controller=SonicWholeBodyController \
        --dataset.repo_id=<user>/<clip> --dataset.episode=0

The 10-frame window is built exactly like the live ``SmplStream`` (oldest->newest,
the first frame repeated to pre-fill), so replayed actions match a live session.

Usage:
    # One clip -> one-episode dataset
    python -m lerobot.teleoperators.pico_headset.smpl_to_dataset \
        --motion-file examples/unitree_g1/motions/walk_forward.npz \
        --repo-id me/sonic_walk_forward

    # Every clip in a dir -> one episode each
    python -m lerobot.teleoperators.pico_headset.smpl_to_dataset \
        --motion-dir examples/unitree_g1/motions --repo-id me/sonic_motions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.teleoperators.pico_headset.smpl_constants import (
    ACTION_DIM,
    JOINT_DIM,
    N_JOINTS,
    ROOT_ACTION_DIM as ROOT_DIM,
    ROOT_ACTION_PREFIX,
    SMPL_ACTION_PREFIX,
    SMPL_OBS_DIM,
    WINDOW,
)
from lerobot.teleoperators.pico_headset.smpl_fk import canonicalize_smpl_joints, root_quats_from_aa


def _load_canonical_joints(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load an SMPL clip -> (canonical (T,24,3) joints, (T,4) root wxyz, fps)."""
    data = np.load(path)
    joints = data["smpl_joints"].astype(np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (N_JOINTS, JOINT_DIM):
        raise ValueError(f"{path}: expected smpl_joints (T, 24, 3), got {joints.shape}")
    t = joints.shape[0]
    if "pose_aa" in data.files:
        root_aa = data["pose_aa"].astype(np.float32)[:, :3]
        joints = canonicalize_smpl_joints(joints, root_aa)
        root_quat = root_quats_from_aa(root_aa)  # (T, 4) wxyz, matches live stream
    else:
        # No global orient available: identity root (anchor falls back to standing).
        root_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (t, 1))
    fps = float(data["fps"]) if "fps" in data.files else 50.0
    return joints, root_quat, fps


def _windows(joints: np.ndarray) -> np.ndarray:
    """(T, 24, 3) -> (T, 720): rolling 10-frame window, matching SmplStream.

    Window t = frames [t-9 .. t], clamped to 0 at the start (first frame repeated).
    """
    t = joints.shape[0]
    idx = np.clip(np.arange(t)[:, None] + np.arange(-WINDOW + 1, 1)[None, :], 0, t - 1)
    return joints[idx].reshape(t, -1).astype(np.float32)


def _action_features() -> dict:
    names = [f"{SMPL_ACTION_PREFIX}{i}" for i in range(SMPL_OBS_DIM)]
    names += [f"{ROOT_ACTION_PREFIX}{i}" for i in range(ROOT_DIM)]
    return {"action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": names}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--motion-file", type=str, help="Single SMPL .npz clip")
    src.add_argument("--motion-dir", type=str, help="Directory of .npz clips (one episode each)")
    p.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. user/name")
    p.add_argument("--root", type=str, default=None, help="Local dataset root (default HF cache)")
    p.add_argument("--fps", type=int, default=None, help="Override fps (default: clip fps)")
    p.add_argument("--task", type=str, default="sonic whole-body SMPL replay")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.motion_dir:
        clips = sorted(str(pth) for pth in Path(args.motion_dir).glob("*.npz"))
        if not clips:
            raise SystemExit(f"No .npz clips found in {args.motion_dir}")
    else:
        clips = [args.motion_file]

    first_joints, first_root, first_fps = _load_canonical_joints(clips[0])
    fps = args.fps or int(round(first_fps))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=_action_features(),
        root=args.root,
        robot_type="unitree_g1",
        use_videos=False,
    )

    for clip_i, clip in enumerate(clips):
        if clip_i == 0:
            joints, root_quat = first_joints, first_root
        else:
            joints, root_quat, _ = _load_canonical_joints(clip)
        windows = _windows(joints)  # (T, 720)
        # action = [720 joint window | 4 root wxyz] per frame -> (T, 724)
        actions = np.concatenate([windows, root_quat.astype(np.float32)], axis=1)
        for a in actions:
            dataset.add_frame({"action": a, "task": args.task})
        dataset.save_episode()
        print(f"[smpl_to_dataset] episode {clip_i}: {Path(clip).name} ({actions.shape[0]} frames)")

    dataset.finalize()
    print(f"[smpl_to_dataset] wrote {len(clips)} episode(s) to {dataset.root}")


if __name__ == "__main__":
    main()
