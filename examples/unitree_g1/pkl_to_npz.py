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

"""Convert a GEAR-SONIC / BONES-SEED ``smpl_filtered`` clip (.pkl) to .npz.

The reference clips are zlib-compressed joblib pickles holding a dict with
``pose_aa`` (T, 72), ``transl`` (T, 3), ``smpl_joints`` (T, 24, 3), ``fps``.
``motion_loader.SmplMotion`` consumes the .npz form so the runtime needs no
joblib dependency. Canonicalization (root-orientation removal) happens at load
time in ``motion_loader``, so this converter just repackages the raw arrays.

Run this in an environment that has ``joblib`` (e.g. the sonic teleop venv):

    python examples/unitree_g1/pkl_to_npz.py \
        --pkl sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl \
        --out examples/unitree_g1/motions/walk_forward.npz
"""

import argparse
from pathlib import Path

import numpy as np


def load_pkl(path: str) -> dict:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        # joblib clips are zlib-compressed pickles; fall back to manual inflate.
        import pickle
        import zlib

        with open(path, "rb") as f:
            raw = f.read()
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        return pickle.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True, help="Input smpl_filtered .pkl")
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()

    d = load_pkl(args.pkl)
    if not isinstance(d, dict) or "smpl_joints" not in d:
        raise ValueError(f"Unexpected pkl structure; keys={list(d) if isinstance(d, dict) else type(d)}")

    smpl_joints = np.asarray(d["smpl_joints"], np.float32)
    if smpl_joints.ndim != 3 or smpl_joints.shape[1:] != (24, 3):
        raise ValueError(f"smpl_joints must be (T,24,3), got {smpl_joints.shape}")

    out = {"smpl_joints": smpl_joints, "fps": np.float32(d.get("fps", 50.0))}
    if "pose_aa" in d:
        out["pose_aa"] = np.asarray(d["pose_aa"], np.float32)
    else:
        print("[warn] no pose_aa -> loader cannot canonicalize (will feed raw)")
    if "transl" in d:
        out["transl"] = np.asarray(d["transl"], np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    dur = smpl_joints.shape[0] / float(out["fps"])
    print(f"Wrote {args.out}")
    print(f"  frames={smpl_joints.shape[0]} fps={float(out['fps']):.1f} duration={dur:.1f}s "
          f"keys={sorted(out)}")


if __name__ == "__main__":
    main()
