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

"""Standalone ``rt/smpl`` publisher for the PICO headset (no gear_sonic / torch).

Reads 24 body-joint poses from the XRoboToolkit SDK, runs pure-numpy SMPL forward
kinematics + canonicalization (``smpl_fk.py``), and publishes one canonical
``(24, 3)`` SMPL frame per tick over ZMQ on the ``rt/smpl`` topic — the exact
message ``lerobot.teleoperators.pico_headset.smpl_stream.SmplStream`` consumes.

This makes the LeRobot side self-contained: the only runtime dependency to drive
SONIC whole-body teleop from the headset is the ``xrobotoolkit_sdk`` Python package
(plus numpy/scipy/pyzmq), not the full ``gear_sonic`` stack.

Usage:
    # Real headset (XRoboToolkit PC Service must be running and connected):
    python -m lerobot.teleoperators.pico_headset.pico_publisher --fps 50 --port 5560

    # No hardware — emit a synthetic waving motion to test the consumer end-to-end:
    python -m lerobot.teleoperators.pico_headset.pico_publisher --fake

    # Replay a canned SMPL clip to the robot through the same rt/smpl -> SONIC path:
    python -m lerobot.teleoperators.pico_headset.pico_publisher \
        --motion-file examples/unitree_g1/motions/walk_forward.npz
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np
import zmq

from lerobot.teleoperators.pico_headset.smpl_fk import (
    SmplForwardKinematics,
    canonicalize_smpl_joints,
    root_quats_from_aa,
)

SMPL_TOPIC = "rt/smpl"
DEFAULT_SMPL_PORT = 5560


def pack_message(
    smpl_joints_local: np.ndarray,
    frame_index: int,
    stamp_ns: int,
    root_quat: np.ndarray | None = None,
    root_transl: np.ndarray | None = None,
) -> bytes:
    """Build the rt/smpl JSON message (single frame, topic embedded in payload)."""
    data = {
        "smpl_joints_local": np.asarray(smpl_joints_local, np.float32).reshape(-1).tolist(),
        "frame_index": int(frame_index),
        "stamp_ns": int(stamp_ns),
    }
    if root_quat is not None:
        data["root_quat"] = np.asarray(root_quat, np.float32).reshape(-1).tolist()
    if root_transl is not None:
        data["root_transl"] = np.asarray(root_transl, np.float32).reshape(-1).tolist()
    return json.dumps({"topic": SMPL_TOPIC, "data": data}).encode("utf-8")


def _fake_body_poses(t: float) -> np.ndarray:
    """Synthetic (24, 7) body poses: identity rotations + a gently waving right arm."""
    poses = np.zeros((24, 7), np.float32)
    poses[:, 6] = 1.0  # unit quaternion (qw = 1), scalar-last
    poses[:, 1] = 1.0  # ~1 m pelvis height (positions only matter for root_transl)
    # Wave the right shoulder (SMPL body joint 17) about Z.
    ang = 0.5 * np.sin(2.0 * np.pi * 0.5 * t)
    poses[17, 3:7] = [0.0, 0.0, np.sin(ang / 2), np.cos(ang / 2)]
    return poses


def _load_motion_clip(path: str) -> dict:
    """Load an SMPL ``.npz`` clip and canonicalize it for rt/smpl streaming.

    Expects the same keys as ``motion_loader.SmplMotion``:
        smpl_joints (T, 24, 3), pose_aa (T, 72) optional, transl (T, 3) optional.
    Returns per-frame joints already in the encoder's root-removed convention,
    plus optional per-frame root quat/translation.
    """
    data = np.load(path)
    joints = data["smpl_joints"].astype(np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"Expected smpl_joints (T, 24, 3), got {joints.shape}")

    pose_aa = data["pose_aa"].astype(np.float32) if "pose_aa" in data.files else None
    root_quat = None
    if pose_aa is not None:
        joints = canonicalize_smpl_joints(joints, pose_aa[:, :3])
        root_quat = root_quats_from_aa(pose_aa[:, :3])
    transl = data["transl"].astype(np.float32) if "transl" in data.files else None
    return {"joints": joints, "root_quat": root_quat, "transl": transl}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=DEFAULT_SMPL_PORT, help="ZMQ PUB port for rt/smpl")
    p.add_argument("--fps", type=float, default=50.0, help="Target publish rate (Hz)")
    p.add_argument("--skeleton", type=str, default=None, help="Path to smpl_skeleton.npz")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--fake", action="store_true", help="Publish synthetic motion (no headset)")
    src.add_argument("--motion-file", type=str, default=None, help="Replay an SMPL .npz clip over rt/smpl")
    p.add_argument("--no-loop", action="store_true", help="Play a --motion-file once, then stop")
    args = p.parse_args()

    clip = _load_motion_clip(args.motion_file) if args.motion_file else None

    # FK is only needed for live/synthetic (24,7) body poses; clips are pre-canonical.
    fk = None
    if clip is None:
        fk = SmplForwardKinematics(args.skeleton) if args.skeleton else SmplForwardKinematics()

    xrt = None
    if clip is None and not args.fake:
        try:
            import xrobotoolkit_sdk as xrt  # noqa: PLC0415
        except ImportError as e:
            raise SystemExit(
                "xrobotoolkit_sdk not available. Install it, or run with --fake / --motion-file "
                "to test the pipeline without a headset."
            ) from e
        xrt.init()
        print("[pico_publisher] XRoboToolkit initialized")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    src_desc = f"motion-file {args.motion_file}" if clip else ("fake" if args.fake else "headset")
    print(
        f"[pico_publisher] '{SMPL_TOPIC}' bound to tcp://*:{args.port} @ {args.fps:.0f} Hz "
        f"[source: {src_desc}]"
    )
    if clip is not None:
        print(f"[pico_publisher] clip frames={clip['joints'].shape[0]} loop={not args.no_loop}")

    period = 1.0 / max(1.0, args.fps)
    frame_index = 0
    t0 = time.time()
    try:
        while True:
            loop_start = time.time()
            if clip is not None:
                n = clip["joints"].shape[0]
                if args.no_loop and frame_index >= n:
                    print("\n[pico_publisher] clip finished")
                    break
                i = frame_index % n
                joints = clip["joints"][i]
                root_quat = None if clip["root_quat"] is None else clip["root_quat"][i]
                root_transl = None if clip["transl"] is None else clip["transl"][i]
                stamp_ns = time.time_ns()
            else:
                if args.fake:
                    body_poses = _fake_body_poses(loop_start - t0)
                    stamp_ns = time.time_ns()
                else:
                    body_poses = np.asarray(xrt.get_body_joints_pose(), np.float32)
                    stamp_ns = int(xrt.get_time_stamp_ns())
                    if body_poses.shape != (24, 7):
                        time.sleep(0.005)
                        continue
                out = fk.compute(body_poses)
                joints = out["smpl_joints_local"]
                root_quat = out["root_quat"]
                root_transl = out["root_transl"]

            sock.send(
                pack_message(joints, frame_index, stamp_ns, root_quat=root_quat, root_transl=root_transl)
            )
            frame_index += 1
            if frame_index % int(max(1, args.fps)) == 0:
                print(f"[pico_publisher] sent {frame_index} frames", end="\r")

            dt = time.time() - loop_start
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[pico_publisher] stopping")
    finally:
        sock.close(0)
        if xrt is not None:
            with contextlib.suppress(Exception):
                xrt.close()


if __name__ == "__main__":
    main()
