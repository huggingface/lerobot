#!/usr/bin/env python
"""Replay a precomputed retargeted SHORT-arm trajectory on the real bimanual OpenArm.

The trajectory (N,16 degrees, POLICY order: right_joint_1..7,right_gripper,left_joint_1..7,
left_gripper) is produced offline by .precompute_short_traj.py as:
    FK on the LONG model (recorded data) -> gripper-tip pose -> IK on the SHORT model,
clamped to the real per-arm joint limits. This is the deterministic, no-policy validation of
the morphology retarget before running the live policy rollout.

Run in the deployment env (lerobot312), with CAN up (can0/can1):

    python examples/openarm/retarget_replay.py \
        --traj openarm_ep1_short_retargeted.npy \
        --left-port can1 --right-port can0 --id openarms \
        --max-relative-target 8.0 --fps 30 --ramp-seconds 4.0

Use --dry-run first to print the ramp/first/last targets without touching the robot.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

# POLICY column -> robot action key (bimanual, ".pos", degrees)
JMAP: list[tuple[str, int]] = (
    [(f"right_joint_{i}.pos", i - 1) for i in range(1, 8)]
    + [("right_gripper.pos", 7)]
    + [(f"left_joint_{i}.pos", 8 + i - 1) for i in range(1, 8)]
    + [("left_gripper.pos", 15)]
)


def action_of(row: np.ndarray) -> dict[str, float]:
    return {k: float(row[c]) for k, c in JMAP}


def present_row(obs: dict) -> np.ndarray:
    row = np.zeros(16, dtype=np.float64)
    for k, c in JMAP:
        row[c] = float(obs[k])
    return row


def build_robot(args):
    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

    common = dict(
        can_interface="socketcan",
        disable_torque_on_disconnect=True,
        max_relative_target=args.max_relative_target,
        cameras={},
    )
    cfg = BiOpenArmFollowerConfig(
        id=args.id,
        left_arm_config=OpenArmFollowerConfigBase(port=args.left_port, side="left", **common),
        right_arm_config=OpenArmFollowerConfigBase(port=args.right_port, side="right", **common),
        cameras={},
    )
    return BiOpenArmFollower(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="(N,16) npy in degrees, POLICY order")
    ap.add_argument("--left-port", default="can1")
    ap.add_argument("--right-port", default="can0")
    ap.add_argument("--id", default="openarms")
    ap.add_argument("--max-relative-target", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ramp-seconds", type=float, default=4.0,
                    help="time to smoothly move from current pose to the first frame")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true", help="print plan, do not connect")
    args = ap.parse_args()

    traj = np.load(args.traj).astype(np.float64)
    assert traj.ndim == 2 and traj.shape[1] == 16, f"expected (N,16), got {traj.shape}"
    end = traj.shape[0] if args.end_index < 0 else args.end_index
    traj = traj[args.start_index:end]
    n = traj.shape[0]
    dt = 1.0 / args.fps
    print(f"loaded {args.traj}: {n} frames @ {args.fps} fps (~{n*dt:.1f}s)")
    print("first target:", np.round(traj[0], 1))
    print("last  target:", np.round(traj[-1], 1))

    if args.dry_run:
        print("[dry-run] not connecting.")
        return

    robot = build_robot(args)
    print("connecting... (ensure CAN is up and arms are clear)")
    robot.connect()
    try:
        # --- gentle ramp from current pose to first frame ---
        present = present_row(robot.get_observation())
        n_ramp = max(1, int(round(args.ramp_seconds * args.fps)))
        print(f"ramping to first frame over {args.ramp_seconds:.1f}s ({n_ramp} steps)...")
        for k in range(1, n_ramp + 1):
            a = k / n_ramp
            row = (1.0 - a) * present + a * traj[0]
            t0 = time.perf_counter()
            robot.send_action(action_of(row))
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

        # --- stream the trajectory ---
        print("replaying...")
        start = time.perf_counter()
        for t in range(n):
            robot.send_action(action_of(traj[t]))
            target = start + (t + 1) * dt
            time.sleep(max(0.0, target - time.perf_counter()))
            if t % 60 == 0:
                print(f"  frame {t}/{n}  ({t/args.fps:.1f}s)")
        print("done.")
    except KeyboardInterrupt:
        print("\ninterrupted by user.")
    finally:
        print("disconnecting (torque off)...")
        robot.disconnect()


if __name__ == "__main__":
    main()
