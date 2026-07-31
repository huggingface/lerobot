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

"""Replay a recorded bimanual-OpenArm episode into an mp4, in simulation.

This drives the official OpenArm MuJoCo model (``enactic/openarm_mujoco``, v1) directly
from a LeRobot dataset's recorded ``observation.state`` and renders a headless video. It is
a visual sanity check that a recorded/commanded joint trajectory is what you think it is --
the same forward-kinematics path used to expose end-effector poses (see the OpenArm
end-effector kinematics helper ``OpenArmFollower.make_kinematics``).

The ``observation.state`` is expected in the 16-D bimanual layout (degrees):
    right_joint_1..7, right_gripper, left_joint_1..7, left_gripper

Model (Apache-2.0): https://github.com/enactic/openarm_mujoco  (use the **v1** revision;
v2 is a different wrist hardware revision and will look sign-flipped on v1 recordings).

Examples:
    # replay episode 1 of a local LeRobot v3.0 dataset
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl python -m examples.openarm.render_episode \
        --dataset data/folding_src_meta --episode 1 --out openarm_ep1.mp4

    # smoke test with a synthetic wave (no dataset / model joints only)
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl python -m examples.openarm.render_episode \
        --demo --out openarm_demo.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering

import numpy as np

# Policy/dataset state layout: 16-D, degrees.
POLICY_ORDER = [
    *(f"right_joint_{i}" for i in range(1, 8)),
    "right_gripper",
    *(f"left_joint_{i}" for i in range(1, 8)),
    "left_gripper",
]
# MuJoCo joint name for each of the 14 arm entries (grippers handled separately).
ARM_MAP = {
    **{f"right_joint_{i}": f"openarm_right_joint{i}" for i in range(1, 8)},
    **{f"left_joint_{i}": f"openarm_left_joint{i}" for i in range(1, 8)},
}
RIGHT_GRIPPER_IDX, LEFT_GRIPPER_IDX = 7, 15
GRIP_FULL_DEG = 65.0  # follower gripper limit magnitude -> fully open


def locate_mjcf(explicit: str | None) -> str:
    """Resolve the OpenArm v1 bimanual MJCF path.

    Priority: --mjcf arg, then $OPENARM_MJCF, then the file installed by the
    ``openarm_mujoco`` pip package under ``<prefix>/share/openarm_mujoco/v1``.
    """
    if explicit:
        return explicit
    if os.environ.get("OPENARM_MJCF"):
        return os.environ["OPENARM_MJCF"]
    for prefix in (sys.prefix, os.environ.get("CONDA_PREFIX", "")):
        if not prefix:
            continue
        cand = Path(prefix) / "share" / "openarm_mujoco" / "v1" / "openarm_bimanual.xml"
        if cand.exists():
            return str(cand)
    raise SystemExit(
        "Could not find the OpenArm v1 MJCF. Pass --mjcf /path/to/v1/openarm_bimanual.xml, "
        "set $OPENARM_MJCF, or clone https://github.com/enactic/openarm_mujoco."
    )


def load_state_from_dataset(root: str, ep: int) -> np.ndarray:
    """Read one episode's recorded observation.state (N, 16; degrees) from a LeRobot v3.0 root."""
    import pandas as pd

    root = Path(root)
    ep_meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    row = ep_meta[ep_meta["episode_index"] == ep].iloc[0]
    a, b = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    dchunk, dfile = int(row["data/chunk_index"]), int(row["data/file_index"])
    df = pd.read_parquet(root / "data" / f"chunk-{dchunk:03d}" / f"file-{dfile:03d}.parquet")
    df = df[(df["index"] >= a) & (df["index"] < b)].sort_values("frame_index")
    return np.stack(df["observation.state"].to_numpy()).astype(np.float32)


def encode_mp4(frames: list[np.ndarray], path: str, fps: int) -> None:
    import av

    h, w = frames[0].shape[:2]
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for f in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default=None, help="LeRobot v3.0 dataset root (meta/ + data/)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="drive a synthetic wave (no dataset)")
    ap.add_argument("--mjcf", default=None, help="path to v1/openarm_bimanual.xml (see locate_mjcf)")
    ap.add_argument("--out", default="openarm_episode.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(locate_mjcf(args.mjcf))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)

    qadr = {
        pk: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj)])
        for pk, mj in ARM_MAP.items()
    }

    def finger_adr(names):
        out = []
        for nm in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            out.append((int(model.jnt_qposadr[jid]), model.jnt_range[jid].copy(), int(model.jnt_type[jid])))
        return out

    right_fingers = finger_adr(["openarm_right_finger_joint1", "openarm_right_finger_joint2"])
    left_fingers = finger_adr(["openarm_left_finger_joint1", "openarm_left_finger_joint2"])
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)

    def finger_target(gripper_deg, rng, jtype):
        opening = min(1.0, abs(gripper_deg) / GRIP_FULL_DEG)  # 0=closed .. 1=open
        if jtype == hinge_type:  # hinge in radians; sign encodes side via range direction
            lo, hi = rng
            mag = np.deg2rad(min(abs(gripper_deg), GRIP_FULL_DEG))
            return np.clip(-mag if lo < 0 else mag, lo, hi)
        return rng[0] + opening * (rng[1] - rng[0])  # slide (v1): lo=closed .. hi=open

    if args.demo:
        n_frames = 120
        traj = np.zeros((n_frames, 16), np.float32)
        wave = 40.0 * np.sin(np.linspace(0, 2 * np.pi, n_frames))
        for i, pk in enumerate(POLICY_ORDER):
            if pk in qadr:
                traj[:, i] = wave * (0.5 + 0.5 * (i % 3))
    elif args.dataset:
        traj = load_state_from_dataset(args.dataset, args.episode)
    else:
        raise SystemExit("provide --dataset <root> (with --episode) or --demo")
    n_frames = traj.shape[0]
    print(f"driving {n_frames} frames")

    # Auto-frame the arms (exclude pedestal/world) from body positions at the mid pose.
    mid = n_frames // 2
    for i, pk in enumerate(POLICY_ORDER):
        if pk in qadr:
            data.qpos[qadr[pk]] = np.deg2rad(traj[mid, i])
    mujoco.mj_forward(model, data)
    arm_pts = [
        data.xpos[b].copy()
        for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("openarm")
        and "base" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
    ]
    arm_pts = np.array(arm_pts) if arm_pts else data.xpos[1:]
    lo, hi = arm_pts.min(0), arm_pts.max(0)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = 150.0, -20.0
    cam.distance = max(0.8, float(np.linalg.norm(hi - lo)) * 1.3)
    cam.lookat[:] = (lo + hi) / 2.0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames = []
    for t in range(n_frames):
        for i, pk in enumerate(POLICY_ORDER):
            if pk in qadr:
                data.qpos[qadr[pk]] = np.deg2rad(traj[t, i])
        for adr, rng, jt in right_fingers:
            data.qpos[adr] = finger_target(float(traj[t, RIGHT_GRIPPER_IDX]), rng, jt)
        for adr, rng, jt in left_fingers:
            data.qpos[adr] = finger_target(float(traj[t, LEFT_GRIPPER_IDX]), rng, jt)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    renderer.close()

    encode_mp4(frames, args.out, args.fps)
    print(f"wrote {args.out}  ({n_frames} frames @ {args.fps} fps, {args.width}x{args.height})")


if __name__ == "__main__":
    main()
