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

"""Render a recorded OpenArm episode and its G1 retargeting overlaid in one scene.

The OpenArm (cyan) plays the recorded joint angles; the G1 (orange) plays what
``G1OpenArmRetargeter`` solved to put its hands on the same poses. Both robots are already
in one MuJoCo model, so this is just a render of the retargeter's own state -- if the hands
track and the elbows stay sane, the retargeting is working.

    python examples/openarm/render_g1_retarget.py --episode 1 --out /tmp/g1_retarget.mp4
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from lerobot.robots.unitree_g1.g1_openarm_retarget import (  # noqa: E402
    OA_PREFIX,
    G1OpenArmRetargeter,
)

PEDESTAL_BODY = f"{OA_PREFIX}openarm_body_link0"


def load_episode(root: str, episode: int) -> np.ndarray:
    """Read one episode's ``observation.state`` (16-D, degrees) out of a LeRobot dataset."""
    import pandas as pd

    root = Path(root)
    meta = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    row = meta[meta["episode_index"] == episode].iloc[0]
    lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    chunk, file = int(row["data/chunk_index"]), int(row["data/file_index"])
    frame = pd.read_parquet(root / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet")
    frame = frame[(frame["index"] >= lo) & (frame["index"] < hi)].sort_values("frame_index")
    return np.stack(frame["observation.state"].to_numpy()).astype(np.float64)


def paint(model, g1_rgba, oa_rgba, show_pedestal: bool) -> None:
    """Recolour every geom by owner: G1 one colour, OpenArm the other."""
    import mujoco

    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        is_openarm = body.startswith(OA_PREFIX)
        model.geom_matid[gid] = -1  # drop materials/textures so the tint is what shows
        model.geom_rgba[gid] = oa_rgba if is_openarm else g1_rgba
        if is_openarm and body.startswith(PEDESTAL_BODY) and not show_pedestal:
            model.geom_rgba[gid, 3] = 0.0  # the rig's pedestal is not part of the comparison


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="data/folding_src_meta")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--stride", type=int, default=4, help="render every k-th recorded frame")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--waist", action="store_true", help="let the IK use the waist too")
    ap.add_argument("--show-pedestal", action="store_true")
    ap.add_argument("--az", type=float, default=170.0)
    ap.add_argument("--el", type=float, default=-20.0)
    ap.add_argument("--dist", type=float, default=1.35)
    ap.add_argument("--lookat", type=float, nargs=3, default=[0.25, 0.0, 0.05])
    ap.add_argument("--out", default="g1_retarget.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco

    traj = load_episode(args.dataset, args.episode)
    keep = list(range(0, len(traj), max(1, args.stride)))[: args.max_frames or None]
    print(f"episode {args.episode}: {len(traj)} frames, rendering {len(keep)}")

    retarget = G1OpenArmRetargeter(use_waist=args.waist)
    model, data = retarget.model, retarget.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]  # the meshes are dark by default
    model.vis.headlight.diffuse[:] = [0.5, 0.5, 0.5]
    paint(
        model,
        g1_rgba=np.array([1.00, 0.55, 0.15, 1.0]),
        oa_rgba=np.array([0.20, 0.75, 1.00, 0.55]),
        show_pedestal=args.show_pedestal,
    )

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = args.az, args.el, args.dist
    cam.lookat[:] = args.lookat

    errors = []
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=None)
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        for i, t in enumerate(keep):
            # solve() poses the OpenArm from the recording and the G1 from its own IK, both
            # in this model, so the render below needs no extra bookkeeping.
            _, err = retarget.solve(traj[t])
            errors.append(err)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            writer.append_data(renderer.render())
            if i % 50 == 0:
                print(f"  frame {i}/{len(keep)}  wrist err {err.mean() * 1e3:.1f} mm", flush=True)
    writer.close()

    errors = np.array(errors)
    print(f"wrist position error: mean {errors.mean() * 1e3:.2f} mm, max {errors.max() * 1e3:.2f} mm")
    print(f"wrote {args.out} ({len(keep)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
