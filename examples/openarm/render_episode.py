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

This drives the official OpenArm MuJoCo model (``enactic/openarm_mujoco``, v1) from a
LeRobot dataset's recorded ``observation.state`` and renders a headless video.

By default the replay goes *through end-effector (Cartesian) space* rather than pushing the
recorded joint angles straight into the simulator: for every frame and every arm it runs
forward kinematics (recorded joints -> EE pose) and then inverse kinematics (EE pose ->
joints), and drives MuJoCo with the IK-recovered joints. This exercises the exact
``lerobot.model.RobotKinematics`` solver that ``OpenArmFollower.make_kinematics()`` builds,
so the rendered video is a visual sanity check of the OpenArm end-effector kinematics -- not
just of the raw recording. Pass ``--joint-space`` to bypass kinematics and replay the raw
recorded joints directly (the previous behaviour).

The ``observation.state`` is expected in the 16-D bimanual layout (degrees):
    right_joint_1..7, right_gripper, left_joint_1..7, left_gripper

The single published OpenArm URDF is *bimanual*, so end-effector kinematics build one solver
per arm (right/left), each keyed to that arm's joints (``openarm_<side>_joint1..7``) and
tool-center frame (``openarm_<side>_hand_tcp``). Solving in the full bimanual frame is what
keeps both arms at their correct, matching heights.

Assets (both Apache-2.0, nothing vendored into LeRobot):
    MuJoCo MJCF: https://github.com/enactic/openarm_mujoco  (use the **v1** revision; v2 is a
        different wrist hardware revision and will look sign-flipped on v1 recordings).
    URDF (for RobotKinematics): https://github.com/enactic/openarm_description

Examples:
    # replay episode 1 of a local LeRobot v3.0 dataset through EE kinematics (bimanual URDF)
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl python -m examples.openarm.render_episode \
        --dataset data/folding_src_meta --episode 1 \
        --urdf /path/to/openarm_bimanual.urdf \
        --out openarm_ep1.mp4

    # smoke test with a synthetic wave (no dataset; add --urdf to also exercise the kinematics)
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl python -m examples.openarm.render_episode \
        --demo --out openarm_demo.mp4

    # bypass kinematics and replay the raw recorded joints directly
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib MUJOCO_GL=egl python -m examples.openarm.render_episode \
        --dataset data/folding_src_meta --episode 1 --joint-space --out openarm_ep1_raw.mp4
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

# Per-arm slices into the 16-D state (7 arm joints each; grippers handled separately).
ARM_JOINT_SLICES = {"right": slice(0, 7), "left": slice(8, 15)}
# The only published OpenArm URDF is *bimanual*, so each arm has its own joint names and
# end-effector (tool-center-point) frame. Using the bimanual URDF -- rather than a single
# arm placed at the origin -- is what makes FK/IK return correct world poses for each arm
# (it encodes the shoulder-mount transform), so the two arms land at the right heights.
ARM_SIDE_JOINTS = {
    side: [f"openarm_{side}_joint{i}" for i in range(1, 8)] for side in ("right", "left")
}
DEFAULT_EE_FRAMES = {side: f"openarm_{side}_hand_tcp" for side in ("right", "left")}


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


def locate_urdf(explicit: str | None) -> str | None:
    """Resolve the single-arm OpenArm URDF path for ``RobotKinematics`` (may be None)."""
    if explicit:
        return explicit
    if os.environ.get("OPENARM_URDF"):
        return os.environ["OPENARM_URDF"]
    return None


def make_kinematics(urdf_path: str, ee_frames: dict[str, str]) -> dict:
    """Build one ``RobotKinematics`` EE solver per arm from the bimanual OpenArm URDF.

    This mirrors what ``OpenArmFollower.make_kinematics()`` does per arm, but we construct
    ``RobotKinematics`` directly (instead of instantiating an ``OpenArmFollower``) so this
    render-only example does not pull in the CAN/motor hardware stack. Each side gets its own
    solver keyed to that arm's URDF joint names (``openarm_<side>_joint1..7``) and tool-center
    frame, so forward/inverse kinematics resolve in the full bimanual (world) frame -- that is
    what keeps the two arms at their correct, matching heights.
    """
    from lerobot.model import RobotKinematics

    return {
        side: RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=ee_frames[side],
            joint_names=ARM_SIDE_JOINTS[side],
        )
        for side in ("right", "left")
    }


def ee_roundtrip(kins: dict, traj: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Route a recorded joint trajectory through end-effector space, per arm.

    For each frame and each arm, run forward kinematics (recorded joints -> EE pose) then
    inverse kinematics (EE pose -> joints) with that arm's own solver, replacing the arm joints
    with the IK-recovered ones. Grippers pass through unchanged. Returns the new trajectory plus
    the mean joint and EE-position round-trip errors (a sanity check that the solver tracks the
    recording).
    """
    out = traj.copy()
    joint_err = []
    pos_err = []
    n_arm = len(ARM_SIDE_JOINTS["right"])
    for t in range(traj.shape[0]):
        for side, sl in ARM_JOINT_SLICES.items():
            kin = kins[side]
            recorded = traj[t, sl].astype(np.float64)
            ee_pose = kin.forward_kinematics(recorded)
            recovered = kin.inverse_kinematics(recorded, ee_pose)[:n_arm]
            out[t, sl] = recovered
            joint_err.append(np.abs(recovered - recorded).mean())
            # EE position after IK vs. the FK target, to catch non-converged solves.
            pos_err.append(np.linalg.norm(kin.forward_kinematics(recovered)[:3, 3] - ee_pose[:3, 3]))
    return out, float(np.mean(joint_err)), float(np.mean(pos_err))


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
    ap.add_argument(
        "--urdf",
        default=None,
        help="bimanual OpenArm URDF for end-effector kinematics (or set $OPENARM_URDF)",
    )
    ap.add_argument(
        "--right-ee-frame",
        default=os.environ.get("OPENARM_RIGHT_EE_FRAME", DEFAULT_EE_FRAMES["right"]),
        help="right-arm end-effector link name in the URDF (or set $OPENARM_RIGHT_EE_FRAME)",
    )
    ap.add_argument(
        "--left-ee-frame",
        default=os.environ.get("OPENARM_LEFT_EE_FRAME", DEFAULT_EE_FRAMES["left"]),
        help="left-arm end-effector link name in the URDF (or set $OPENARM_LEFT_EE_FRAME)",
    )
    ap.add_argument(
        "--joint-space",
        action="store_true",
        help="bypass kinematics and replay the raw recorded joints directly",
    )
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

    # Route the recorded joints through end-effector (Cartesian) space via the same solver
    # OpenArmFollower.make_kinematics() builds, unless the user opted for raw joint replay.
    urdf_path = None if args.joint_space else locate_urdf(args.urdf)
    if args.joint_space:
        print(f"driving {n_frames} frames (raw joint space)")
    elif urdf_path is None:
        raise SystemExit(
            "End-effector replay needs the OpenArm URDF: pass --urdf /path/to/openarm.urdf "
            "(and --ee-frame if the tip link differs), set $OPENARM_URDF, or use --joint-space "
            "to replay the raw recorded joints. URDF: https://github.com/enactic/openarm_description"
        )
    else:
        ee_frames = {"right": args.right_ee_frame, "left": args.left_ee_frame}
        kins = make_kinematics(urdf_path, ee_frames)
        traj, joint_err, pos_err = ee_roundtrip(kins, traj)
        print(
            f"driving {n_frames} frames (end-effector kinematics via frames "
            f"right='{ee_frames['right']}', left='{ee_frames['left']}'; "
            f"FK->IK round-trip: {joint_err:.3f}° mean joint error, {pos_err * 1e3:.2f} mm mean EE error)"
        )

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
