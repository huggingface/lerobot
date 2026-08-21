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

"""Replay a folding episode on the G1 in the MuJoCo viewer, with SONIC holding the legs.

Where ``g1_folding_sim_replay.py`` writes joint angles straight into ``qpos``, this runs the
real thing: full physics, the SONIC decoder stepping at 50 Hz on legs and waist, and the arms
tracked by the same PD controller the hardware uses. So the robot has to actually stay
standing while the arms move, and if the retargeted arm motion disturbs the balance, you see
it here rather than on the robot.

It is the repo's own sim stack, unmodified -- ``UnitreeG1(is_simulation=True)`` bridges lowcmd
and lowstate over loopback DDS into the hub MuJoCo env, whose state thread steps physics and
drives the viewer. The only thing layered on top is ``G1FoldingRobot``, exactly as on hardware.
Simulation is the one case where the controller legitimately runs off-robot, because there is
no robot.

    python examples/openarm/g1_folding_sim_live.py --episode 1 --speed 1.0

Requires a display. Set ``MUJOCO_GL=glfw`` (the default here) and run it on the desktop, not
over a plain ssh session.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")


from render_g1_retarget import load_episode  # noqa: E402

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config  # noqa: E402
from lerobot.robots.unitree_g1.g1_folding_robot import G1FoldingRobot, policy_keys  # noqa: E402
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="/home/yope/Documents/sonic/data/folding_src_meta")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--fps", type=float, default=20.0, help="rate the recorded frames are fed at")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = the whole episode")
    ap.add_argument("--settle-s", type=float, default=3.0, help="stand under SONIC before moving")
    ap.add_argument("--max-step-deg", type=float, default=8.0)
    ap.add_argument(
        "--controller",
        default="SonicLowerBodyController",
        help="use SonicWholeBodyController to watch SONIC drive the arms too (ignores the replay)",
    )
    args = ap.parse_args()

    traj = load_episode(args.dataset, args.episode)
    if args.max_frames:
        traj = traj[: args.max_frames]
    print(f"episode {args.episode}: {len(traj)} frames at {args.fps} Hz x{args.speed}")

    cfg = UnitreeG1Config(
        is_simulation=True,
        controller=args.controller,
        # In sim the closedness drives the model's Dex3 hands over DDS, so this reaches
        # nothing real -- there is no CAN bridge here to forward it to.
        grippers=True,
        cameras={},
    )
    robot = UnitreeG1(cfg)
    wrapped = G1FoldingRobot(robot, max_step_deg=args.max_step_deg)
    keys = policy_keys()

    print("Connecting sim (this eases the robot to SONIC's default pose first)...")
    robot.connect()

    try:
        # Let the decoder settle into a stand before the arms start moving, so a wobble in the
        # first seconds is clearly the arms and not the controller still converging.
        print(f"Standing for {args.settle_s:.1f}s under {args.controller}...")
        time.sleep(args.settle_s)

        period = 1.0 / (args.fps * args.speed)
        t_start = time.perf_counter()
        late = 0
        for i, state16 in enumerate(traj):
            deadline = t_start + (i + 1) * period
            wrapped.get_observation()
            wrapped.send_action(dict(zip(keys, (float(v) for v in state16), strict=True)))

            slack = deadline - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                late += 1
            if i % int(max(1, args.fps * 5)) == 0:
                height = _pelvis_height(robot)
                print(f"  frame {i}/{len(traj)}  pelvis z {height:.3f} m  late {late}", flush=True)

        print(f"\ndone: {len(traj)} frames, {late} missed the {period * 1e3:.0f} ms deadline")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        robot.disconnect()


def _pelvis_height(robot) -> float:
    """Pelvis z out of the sim, as the cheapest 'is it still standing' signal."""
    try:
        return float(robot.sim_env.sim_env.mj_data.qpos[2])
    except Exception:  # noqa: BLE001
        return float("nan")


if __name__ == "__main__":
    main()
