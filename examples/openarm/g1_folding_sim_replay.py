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

"""Replay a recorded OpenArm episode onto a simulated G1, hands included.

Unlike ``render_g1_retarget.py``, which renders the retargeter's own scratch state, this
drives a MuJoCo body through the real :class:`G1FoldingRobot` -- the same object the hardware
rollout uses. So the replay exercises the whole boundary and not just the IK:

* the reverse projection, because ``get_observation`` reads the simulated arms back and
  reprojects them into the 16-D OpenArm state the policy would be shown;
* the ``max_step_deg`` clamp, because the arms are commanded from where the sim actually is;
* the **gripper channel**, because ``send_action`` emits ``{side}_gripper.pos`` closedness and
  the sim drives the G1's fingers from it. On hardware those same values go out over ZMQ to
  the CAN hands, so a clench here means the channel is wired end to end.

The G1 (orange) is the simulated body; the OpenArm (cyan, translucent) is the recording it is
chasing. Playback is kinematic -- targets are written to ``qpos`` rather than tracked by a PD
loop -- because the question here is whether the retarget is right, not whether SONIC balances.

    python examples/openarm/g1_folding_sim_replay.py --episode 1 --out g1_sim_replay.mp4
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from render_g1_retarget import load_episode, paint  # noqa: E402

from lerobot.robots.unitree_g1.g1_folding_robot import G1FoldingRobot, policy_keys  # noqa: E402
from lerobot.robots.unitree_g1.g1_openarm_retarget import (  # noqa: E402
    G1_ARM_JOINTS,
    build_scene,
    openarm_qadr,
    set_openarm,
)
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex  # noqa: E402

# Fingers whose curl shows a clench. thumb_0 is abduction, not curl, and its range is
# symmetric, so there is no "closed" end to drive it to -- it stays neutral.
CURL_JOINTS = ("thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


class SimG1:
    """A robot-shaped MuJoCo stand-in, so ``G1FoldingRobot`` can drive a simulated body.

    Implements just the surface the wrapper touches: ``observation_features``,
    ``get_observation`` and ``send_action``. Commands land in ``qpos`` directly.
    """

    def __init__(self) -> None:
        import mujoco

        self._mj = mujoco
        self.model = build_scene()
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]  # pelvis at the origin

        def qadr(name: str) -> int:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} not in the scene")
            return int(self.model.jnt_qposadr[jid])

        # G1_ARM_JOINTS is MJCF naming; the action keys are the motor enum's. Same 14 joints
        # in the same order, which is what lets these two be zipped together.
        self._arm_qadr = [qadr(j) for j in G1_ARM_JOINTS]
        self._arm_keys = [f"{m.name}.q" for m in G1_29_JointArmIndex]

        # Per finger joint: where "open" and "closed" sit. The hands mirror -- the left curls
        # toward negative, the right toward positive -- so closed is whichever end of the
        # range is further from zero, and open is the other.
        self._fingers: list[tuple[int, float, float]] = []
        for side in ("left", "right"):
            for finger in CURL_JOINTS:
                name = f"{side}_hand_{finger}_joint"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    continue
                lo, hi = self.model.jnt_range[jid]
                closed, opened = (hi, lo) if abs(hi) >= abs(lo) else (lo, hi)
                self._fingers.append(
                    (qadr(name), float(np.clip(opened, min(lo, hi), max(lo, hi))), float(closed))
                )

        self.oa_arms, self.oa_fingers = openarm_qadr(self.model)
        self.gripper_cmd = {"left": 0.0, "right": 0.0}

    @property
    def observation_features(self) -> dict[str, type]:
        return dict.fromkeys(self._arm_keys, float)

    def get_observation(self) -> dict[str, float]:
        return {
            key: float(self.data.qpos[adr]) for key, adr in zip(self._arm_keys, self._arm_qadr, strict=True)
        }

    def send_action(self, action: dict) -> dict:
        for key, adr in zip(self._arm_keys, self._arm_qadr, strict=True):
            if key in action:
                self.data.qpos[adr] = float(action[key])

        n = len(self._fingers) // 2
        for i, (adr, opened, closed) in enumerate(self._fingers):
            side = "left" if i < n else "right"
            closedness = float(np.clip(action.get(f"{side}_gripper.pos", 0.0), 0.0, 1.0))
            self.gripper_cmd[side] = closedness
            self.data.qpos[adr] = opened + closedness * (closed - opened)
        return action

    def reset(self, *args, **kwargs) -> None:
        return None

    def pose_reference(self, state16: np.ndarray) -> None:
        """Pose the translucent OpenArm from the recording, as the thing being chased."""
        set_openarm(self.data, self.oa_arms, self.oa_fingers, np.asarray(state16, np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="data/folding_src_meta")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--stride", type=int, default=4, help="replay every k-th recorded frame")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-step-deg", type=float, default=8.0)
    ap.add_argument("--show-pedestal", action="store_true")
    ap.add_argument("--hide-reference", action="store_true", help="G1 only, no OpenArm ghost")
    ap.add_argument("--az", type=float, default=170.0)
    ap.add_argument("--el", type=float, default=-20.0)
    ap.add_argument("--dist", type=float, default=1.35)
    ap.add_argument("--lookat", type=float, nargs=3, default=[0.25, 0.0, 0.05])
    ap.add_argument("--out", default="g1_sim_replay.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco

    traj = load_episode(args.dataset, args.episode)
    keep = list(range(0, len(traj), max(1, args.stride)))[: args.max_frames or None]
    print(f"episode {args.episode}: {len(traj)} frames, replaying {len(keep)}")

    sim = SimG1()
    robot = G1FoldingRobot(sim, max_step_deg=args.max_step_deg)
    keys = policy_keys()

    model, data = sim.model, sim.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    model.vis.headlight.diffuse[:] = [0.5, 0.5, 0.5]
    paint(
        model,
        g1_rgba=np.array([1.00, 0.55, 0.15, 1.0]),
        oa_rgba=np.array([0.20, 0.75, 1.00, 0.0 if args.hide_reference else 0.45]),
        show_pedestal=args.show_pedestal,
    )

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = args.az, args.el, args.dist
    cam.lookat[:] = args.lookat

    grip_trace: list[tuple[float, float]] = []
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=None)
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        for i, t in enumerate(keep):
            state16 = traj[t]
            # Round trip through the boundary the hardware run uses: read the simulated arms
            # back as an OpenArm state, then command the recorded pose as if it were a policy
            # action. The reference pose comes last so the ghost is not clobbered by the IK.
            robot.get_observation()
            robot.send_action(dict(zip(keys, (float(v) for v in state16), strict=True)))
            if not args.hide_reference:
                sim.pose_reference(state16)

            grip_trace.append((sim.gripper_cmd["left"], sim.gripper_cmd["right"]))
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            writer.append_data(renderer.render())
            if i % 50 == 0:
                left, right = grip_trace[-1]
                print(f"  frame {i}/{len(keep)}  grip L {left:.2f} R {right:.2f}", flush=True)
    writer.close()

    grip = np.array(grip_trace)
    print(f"\ngripper closedness (0 open .. 1 closed), over {len(grip)} frames")
    for k, side in enumerate(("left", "right")):
        print(
            f"  {side:5s} min {grip[:, k].min():.2f}  max {grip[:, k].max():.2f}  travel {np.ptp(grip[:, k]):.2f}"
        )
    if np.ptp(grip, axis=0).max() < 0.05:
        print("  WARNING: the hands barely moved -- check the gripper channel, not the episode")
    print(f"\nwrote {args.out} ({len(keep)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
