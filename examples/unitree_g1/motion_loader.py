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

"""Load an SMPL motion clip and expose it in SONIC's encoder format.

SONIC's whole-body tracking mode (``encode_mode == 2``) consumes a flat
720-vector ``smpl_joints_10frame_step1`` = 10 consecutive frames x 24 SMPL
joints x 3 (xyz) at 50 Hz.

IMPORTANT - frame convention: the encoder expects each frame's joints with the
body's *root orientation removed* (per-frame canonical), exactly like the live
deploy stream's ``smpl_joints_local`` (see ``process_smpl_joints`` in the GEAR
PICO teleop and ``smpl_joints_multi_future_local`` in training). The reference
``smpl_filtered`` clips instead store **world-frame** joints (heading retained),
so feeding them raw makes the robot move but mis-track / never face-forward.
This loader therefore canonicalizes on load using the clip's per-frame root
orientation (``pose_aa[:, :3]``):

    A     = Rx(+90deg) * rotvec(pose_aa[:, :3])        # y-up -> z-up root quat
    local = base120 * A^-1 * joints                    # remove root orient

with ``base120 = quat(0.5,0.5,0.5,0.5)`` (SMPL base rotation). This reproduces
the deployed transform (verified: per-frame hip-heading std -> 0).

Clip is read from a numpy ``.npz``. Expected keys:
    smpl_joints : (T, 24, 3) float32  -- world-frame joint positions, 50 fps
    pose_aa     : (T, 72)    float32  -- SMPL axis-angle (root = [:, :3])
    transl      : (T, 3)     float32  -- global root translation (optional)
    fps         : scalar

Example:
    python examples/unitree_g1/motion_loader.py \
        --motion examples/unitree_g1/motions/walk_forward.npz
"""

import argparse

import numpy as np

WINDOW = 10  # frames per encoder window (smpl_joints_10frame_step1)
N_JOINTS = 24
JOINT_DIM = 3
SMPL_OBS_DIM = WINDOW * N_JOINTS * JOINT_DIM  # 720


def canonicalize_smpl_joints(smpl_joints: np.ndarray, root_aa: np.ndarray) -> np.ndarray:
    """Remove per-frame root orientation -> SONIC ``smpl_joints_local`` format.

    Args:
        smpl_joints: (T, 24, 3) world-frame (z-up) SMPL joint positions.
        root_aa: (T, 3) SMPL global-orient axis-angle (y-up convention).

    Returns:
        (T, 24, 3) per-frame root-orientation-removed joints.
    """
    from scipy.spatial.transform import Rotation as R

    rx90 = R.from_euler("x", 90, degrees=True)        # smpl_root_ytoz_up
    base120 = R.from_quat([0.5, 0.5, 0.5, 0.5])       # remove_smpl_base_rot
    a = rx90 * R.from_rotvec(root_aa)                 # z-up root quat (left-mult)
    b_inv = base120 * a.inv()                         # inv(remove_smpl_base_rot(a))
    return np.einsum("tij,tkj->tki", b_inv.as_matrix(), smpl_joints).astype(np.float32)


class SmplMotion:
    """A single SMPL clip with SONIC-format windowing."""

    def __init__(self, path: str, loop: bool = True, canonicalize: bool = True):
        data = np.load(path)
        smpl_joints = data["smpl_joints"].astype(np.float32)  # (T, 24, 3)
        self.pose_aa = data["pose_aa"].astype(np.float32) if "pose_aa" in data.files else None
        self.transl = data["transl"].astype(np.float32) if "transl" in data.files else None
        self.fps = float(data["fps"]) if "fps" in data.files else 50.0
        self.loop = loop

        if smpl_joints.ndim != 3 or smpl_joints.shape[1:] != (N_JOINTS, JOINT_DIM):
            raise ValueError(
                f"Expected smpl_joints (T, {N_JOINTS}, {JOINT_DIM}), got {smpl_joints.shape}"
            )

        # Reference clips store world-frame joints; the encoder wants per-frame
        # root-orientation-removed joints. Canonicalize when we have the root pose.
        self.canonicalized = False
        if canonicalize and self.pose_aa is not None:
            smpl_joints = canonicalize_smpl_joints(smpl_joints, self.pose_aa[:, :3])
            self.canonicalized = True
        self.smpl_joints = smpl_joints

        self.num_frames = self.smpl_joints.shape[0]
        self._cursor = 0

    def window(self, start: int) -> np.ndarray:
        """Return the 720-vector for the 10-frame window beginning at ``start``.

        Frames are laid out oldest->newest, joint-major within a frame:
        [f0_j0_xyz, f0_j1_xyz, ..., f9_j23_xyz].
        """
        idx = np.arange(start, start + WINDOW)
        if self.loop:
            idx = np.mod(idx, self.num_frames)
        else:
            idx = np.clip(idx, 0, self.num_frames - 1)
        return self.smpl_joints[idx].reshape(-1).astype(np.float32)

    def reset(self):
        self._cursor = 0

    def step(self) -> np.ndarray:
        """Advance one frame and return the current 720-vector window."""
        w = self.window(self._cursor)
        self._cursor += 1
        if self.loop:
            self._cursor %= self.num_frames
        return w

    @property
    def done(self) -> bool:
        return (not self.loop) and (self._cursor + WINDOW >= self.num_frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="Path to motion .npz")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--no-canon", action="store_true",
                        help="Skip canonicalization (feed raw stored joints)")
    args = parser.parse_args()

    m = SmplMotion(args.motion, loop=not args.no_loop, canonicalize=not args.no_canon)
    duration = m.num_frames / m.fps
    print(f"Loaded '{args.motion}'")
    print(f"  frames={m.num_frames} fps={m.fps:.1f} duration={duration:.1f}s")
    print(f"  smpl_joints={m.smpl_joints.shape} canonicalized={m.canonicalized} "
          f"pose_aa={None if m.pose_aa is None else m.pose_aa.shape} "
          f"transl={None if m.transl is None else m.transl.shape}")

    # Sanity: after canonicalization the per-frame body heading should be fixed.
    j = m.smpl_joints
    v = (j[:, 2, :2] - j[:, 1, :2])  # R_hip - L_hip, horizontal
    a = np.arctan2(v[:, 1], v[:, 0])
    rlen = np.clip(np.hypot(np.cos(a).mean(), np.sin(a).mean()), 1e-9, 1.0)
    circ_std = np.degrees(np.sqrt(-2 * np.log(rlen)))
    print(f"  hip-heading circ-std={circ_std:.1f} deg "
          f"(~0 => orientation removed; large => world-frame)")

    w0 = m.window(0)
    print(f"  window(0): shape={w0.shape} (expected {SMPL_OBS_DIM}) "
          f"min={w0.min():.3f} max={w0.max():.3f}")
    assert w0.shape == (SMPL_OBS_DIM,), "window must be 720-dim for obs[922:1642]"

    # Simulate a few control ticks.
    print("  stepping 5 ticks:")
    for t in range(5):
        w = m.step()
        print(f"    t={t} cursor={m._cursor} window_norm={np.linalg.norm(w):.2f}")

    print("OK: motion loads and yields SONIC-format 720-vec windows.")


if __name__ == "__main__":
    main()
