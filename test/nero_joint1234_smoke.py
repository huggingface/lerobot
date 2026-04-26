#!/usr/bin/env python3
"""Real-hardware smoke test for NERO joints 1-4.

This script executes tiny per-joint movements on joint1..joint4 and prints
observed deltas from robot feedback.

Safety behavior:
- Low speed by default.
- No gripper command is sent.
- joint4 is clamped away from known SDK limits.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from lerobot.robots.nero_follower import NEOFollower, NEOFollowerRobotConfig

TEST_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
ALL_JOINTS = [f"joint{i}" for i in range(1, 8)]

# Observed SDK limits for joint4 from live warnings.
J4_MIN = -1.012291
J4_MAX = 2.146755


def clamp_joint4(value: float, margin: float) -> float:
    low = J4_MIN + margin
    high = J4_MAX - margin
    return max(low, min(high, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NERO joint1-4 smoke test")
    parser.add_argument("--channel", default="can0", help="CAN channel, default: can0")
    parser.add_argument("--interface", default="socketcan", help="CAN interface, default: socketcan")
    parser.add_argument(
        "--firmeware-version",
        default="default",
        help="NERO firmware selector used by pyAgxArm (default or v111)",
    )
    parser.add_argument("--speed", type=int, default=5, help="Speed percent, default: 5")
    parser.add_argument("--delta", type=float, default=0.02, help="Per-axis command delta in rad")
    parser.add_argument("--settle", type=float, default=1.0, help="Settle seconds after each command")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=0.08,
        help="Safety cap passed to NEOFollower config",
    )
    parser.add_argument(
        "--joint4-margin",
        type=float,
        default=0.01,
        help="Keep joint4 this far from hard limits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = NEOFollowerRobotConfig(
        channel=args.channel,
        interface=args.interface,
        firmeware_version=args.firmeware_version,
        speed_percent=args.speed,
        max_relative_target=args.max_relative_target,
        cameras={},
    )

    robot = NEOFollower(cfg)
    print("joint1234_smoke_start")

    try:
        robot.connect()
        print("connect_ok", robot.is_connected)

        # Move once to a safer baseline around current pose.
        obs = robot.get_observation()
        baseline = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in ALL_JOINTS}
        baseline["joint4.pos"] = clamp_joint4(baseline["joint4.pos"], args.joint4_margin)
        robot.send_action(baseline)
        time.sleep(args.settle)

        results: list[dict[str, object]] = []
        for joint in TEST_JOINTS:
            pre = robot.get_observation()
            cmd = {f"{j}.pos": float(pre[f"{j}.pos"]) for j in ALL_JOINTS}
            cmd["joint4.pos"] = clamp_joint4(cmd["joint4.pos"], args.joint4_margin)

            step = args.delta
            if joint == "joint4":
                # Pick direction away from nearest limit.
                current = float(pre["joint4.pos"])
                midpoint = (J4_MIN + J4_MAX) / 2.0
                step = -args.delta if current > midpoint else args.delta

            cmd[f"{joint}.pos"] = float(cmd[f"{joint}.pos"]) + step

            robot.send_action(cmd)
            time.sleep(args.settle)
            post = robot.get_observation()

            axis_delta = float(post[f"{joint}.pos"]) - float(pre[f"{joint}.pos"])
            all_delta = {
                j: float(post[f"{j}.pos"]) - float(pre[f"{j}.pos"])
                for j in ALL_JOINTS
            }

            # Return this joint toward original position.
            cmd_back = {f"{j}.pos": float(post[f"{j}.pos"]) for j in ALL_JOINTS}
            cmd_back["joint4.pos"] = clamp_joint4(cmd_back["joint4.pos"], args.joint4_margin)
            cmd_back[f"{joint}.pos"] = float(cmd_back[f"{joint}.pos"]) - step
            robot.send_action(cmd_back)
            time.sleep(args.settle)

            results.append(
                {
                    "joint": joint,
                    "requested_step": step,
                    "observed_axis_delta": axis_delta,
                    "all_joint_deltas": all_delta,
                }
            )

        print("joint1234_results_begin")
        print(json.dumps(results, ensure_ascii=True))
        print("joint1234_results_end")
        print("joint1234_smoke_ok")
        return 0
    except Exception as exc:
        print("joint1234_smoke_failed", type(exc).__name__, str(exc))
        print(traceback.format_exc())
        return 1
    finally:
        try:
            robot.disconnect()
            print("disconnect_ok")
        except Exception as exc:
            print("disconnect_failed", type(exc).__name__, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
