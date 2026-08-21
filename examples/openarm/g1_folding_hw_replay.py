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

"""Replay a recorded OpenArm folding episode onto the real G1.

The hardware counterpart of ``g1_folding_sim_replay.py``, and the step before letting the
policy drive: the trajectory is fixed and repeatable, so anything that goes wrong is the
retarget, the arms or the hands, never the policy. No cameras are opened, because a replay
has nothing to look at -- one less thing to be wrong.

Everything downstream of the recorded state is exactly what the rollout does: the same
:class:`G1FoldingRobot`, the same IK, the same step clamp, the same gripper channel.

Topology is unchanged -- SONIC holds the stand onboard, so start the robot first::

    python src/lerobot/robots/unitree_g1/run_g1_server.py --onboard \\
        --controller SonicLowerBodyController --grippers

Then here::

    python examples/openarm/g1_folding_hw_replay.py --robot-ip 172.18.130.111 \\
        --episode 1 --speed 0.5

The arms are ramped from wherever they are into the episode's first pose over
``--ramp-s``, and ramped back on the way out, including on Ctrl-C. Without that the first
command would be a step to an arbitrary pose, and the only thing between the two would be
the per-tick clamp saturating for however long it takes.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time

import numpy as np
from render_g1_retarget import load_episode

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_folding_robot import G1FoldingRobot, policy_keys
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logger = logging.getLogger("g1_folding_hw_replay")

RECORDED_FPS = 30.0


def ramp(robot, keys, start16: np.ndarray, end16: np.ndarray, seconds: float, fps: float) -> None:
    """Walk the arms from one OpenArm state to another over `seconds`, in step with the clock."""
    steps = max(1, int(seconds * fps))
    period = 1.0 / fps
    for i in range(1, steps + 1):
        # Smoothstep rather than linear: no velocity discontinuity at either end, which is
        # what you feel as a jolt when a ramp hands over to playback.
        u = i / steps
        blend = u * u * (3.0 - 2.0 * u)
        state16 = start16 + blend * (end16 - start16)
        t0 = time.perf_counter()
        robot.send_action(dict(zip(keys, (float(v) for v in state16), strict=True)))
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="data/folding_src_meta")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--robot-ip", default="172.18.130.111")
    ap.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="fraction of recorded speed; 1.0 replays at the 30 fps it was captured at",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 replays the whole episode")
    ap.add_argument("--max-step-deg", type=float, default=3.0, help="per-tick joint clamp")
    ap.add_argument("--ramp-s", type=float, default=3.0, help="ease in/out of the episode")
    ap.add_argument("--no-grippers", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="retarget every frame and report, without connecting to the robot",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    traj = load_episode(args.dataset, args.episode)
    if args.max_frames:
        traj = traj[: args.max_frames]
    fps = RECORDED_FPS * args.speed
    print(
        f"episode {args.episode}: {len(traj)} frames, replaying at {fps:.1f} Hz "
        f"({args.speed:g}x) -- {len(traj) / fps:.0f}s"
    )

    keys = policy_keys()

    if args.dry_run:
        from g1_folding_sim_replay import SimG1

        robot = G1FoldingRobot(SimG1(), max_step_deg=args.max_step_deg)
        t0 = time.perf_counter()
        for state16 in traj:
            robot.get_observation()
            robot.send_action(dict(zip(keys, (float(v) for v in state16), strict=True)))
        dt = (time.perf_counter() - t0) / len(traj)
        print(f"dry run OK: {dt * 1e3:.1f} ms/frame both solves, budget {1e3 / fps:.0f} ms")
        return

    cfg = UnitreeG1Config(
        id="g1",
        is_simulation=False,
        robot_ip=args.robot_ip,
        controller="SonicLowerBodyController",
        grippers=not args.no_grippers,
        cameras={},
    )
    robot = G1FoldingRobot(UnitreeG1(cfg), max_step_deg=args.max_step_deg)
    robot.connect()

    stop = False

    def request_stop(*_) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)

    try:
        # Read the arms back through the same path the policy would, so "home" is expressed
        # in the space the ramp interpolates in.
        obs = robot.get_observation()
        home16 = np.array([float(obs[k]) for k in keys])

        print(f"ramping into the episode over {args.ramp_s:g}s ...")
        ramp(robot, keys, home16, traj[0], args.ramp_s, fps)

        print("replaying (Ctrl-C to ease out) ...")
        period = 1.0 / fps
        late = 0
        t_start = time.perf_counter()
        for i, state16 in enumerate(traj):
            if stop:
                break
            t0 = time.perf_counter()
            robot.get_observation()
            robot.send_action(dict(zip(keys, (float(v) for v in state16), strict=True)))
            spare = period - (time.perf_counter() - t0)
            if spare < 0:
                late += 1
            time.sleep(max(0.0, spare))
            if i % int(fps * 5) == 0:
                print(f"  {i}/{len(traj)}  t={time.perf_counter() - t_start:.0f}s  late={late}")

        if late:
            print(f"{late}/{len(traj)} ticks ran over budget -- lower --speed for a clean replay")

        obs = robot.get_observation()
        here16 = np.array([float(obs[k]) for k in keys])
        print(f"easing back to the start pose over {args.ramp_s:g}s ...")
        ramp(robot, keys, here16, home16, args.ramp_s, fps)
    finally:
        robot.disconnect()
        print("done -- SONIC still has the stand")


if __name__ == "__main__":
    main()
