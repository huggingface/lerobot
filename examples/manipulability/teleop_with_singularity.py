#!/usr/bin/env python3
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

"""
SO-101 singularity indicator — two modes:

  observe [--duration N]   Torque OFF.  Move the follower arm by hand.
                           Displays σ_min / cond live.  No motor commands.

  teleop  [--duration N]   Direct joint-mirroring from leader to follower
                           with live σ_min / cond overlay.
                           Emergency-stops if any joint jumps > threshold.

Example usage:

    python examples/manipulability/teleop_with_singularity.py observe \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_follower \
        --urdf=path/to/so101.urdf

    python examples/manipulability/teleop_with_singularity.py teleop \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_follower \
        --teleop.port=/dev/ttyACM1 \
        --teleop.id=my_leader \
        --urdf=path/to/so101.urdf
"""

import argparse
import logging
import time

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.model.manipulability import (
    compute_manipulability,
    extract_translational_jacobian,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

SIGMA_MIN_WARN = 0.015
SIGMA_MIN_CRITICAL = 0.006
ESTOP_JUMP_THRESHOLD = 15.0  # normalized units

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_JOINT_NAMES = MOTOR_NAMES[:5]

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SO-101 singularity indicator (observe / teleop modes)"
    )
    parser.add_argument(
        "mode", choices=["observe", "teleop"], default="observe", nargs="?",
        help="observe = torque off, move by hand; teleop = joint mirroring with e-stop",
    )
    parser.add_argument("--robot.port", dest="robot_port", required=True, help="Follower serial port")
    parser.add_argument("--robot.id", dest="robot_id", required=True, help="Follower calibration ID")
    parser.add_argument("--teleop.port", dest="teleop_port", default=None, help="Leader serial port (teleop mode)")
    parser.add_argument("--teleop.id", dest="teleop_id", default=None, help="Leader calibration ID (teleop mode)")
    parser.add_argument("--urdf", required=True, help="Path to SO-101 URDF file")
    parser.add_argument("--duration", type=float, default=None, help="Session duration in seconds (default: unlimited)")
    parser.add_argument("--fps", type=int, default=30, help="Target loop rate (default: 30)")
    parser.add_argument("--estop-threshold", type=float, default=ESTOP_JUMP_THRESHOLD, help="E-stop jump threshold in normalized units")
    return parser.parse_args()


def main():
    args = parse_args()
    observe_only = args.mode == "observe"

    if not observe_only and (args.teleop_port is None or args.teleop_id is None):
        raise ValueError("Teleop mode requires --teleop.port and --teleop.id")

    # ---- Robot setup ---------------------------------------------------------
    follower = SO101Follower(
        SO101FollowerConfig(port=args.robot_port, id=args.robot_id, use_degrees=False)
    )

    leader = None
    if not observe_only:
        leader = SO101Leader(
            SO101LeaderConfig(port=args.teleop_port, id=args.teleop_id, use_degrees=False)
        )

    motor_names = MOTOR_NAMES

    # ---- Kinematics ----------------------------------------------------------
    follower_kin = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # ---- Connect -------------------------------------------------------------
    follower.connect()
    if leader is not None:
        leader.connect()

    if observe_only:
        follower.bus.disable_torque()
        log.info("OBSERVE mode — torque DISABLED.  Move the arm by hand.")
    else:
        log.info("TELEOP mode — direct joint mirroring with e-stop.")

    log.info(f"  WARN  threshold: σ_min < {SIGMA_MIN_WARN}")
    log.info(f"  CRIT  threshold: σ_min < {SIGMA_MIN_CRITICAL}")
    if not observe_only:
        log.info(f"  E-STOP threshold: joint jump > {args.estop_threshold}")
    log.info("Press Ctrl-C to stop.\n")

    # ---- Joint-limit lookup for degree conversion ----------------------------
    joint_limits_deg = {}
    for name in ARM_JOINT_NAMES:
        lo, hi = follower_kin.robot.get_joint_limits(name)
        joint_limits_deg[name] = (np.rad2deg(lo), np.rad2deg(hi))

    def norm_to_deg(obs_norm: dict) -> np.ndarray:
        """Convert normalized [-100,100] follower observation to degrees."""
        q = np.zeros(len(motor_names))
        for i, m in enumerate(motor_names):
            v_norm = float(obs_norm.get(f"{m}.pos", 0.0))
            if m in joint_limits_deg:
                lo, hi = joint_limits_deg[m]
                q[i] = lo + (v_norm + 100.0) / 200.0 * (hi - lo)
            else:
                q[i] = v_norm
        return q

    # ---- Main loop -----------------------------------------------------------
    loop_start = time.perf_counter()
    prev_obs_vals = None
    try:
        while True:
            if args.duration is not None and (time.perf_counter() - loop_start) >= args.duration:
                print(f"\n\nDuration {args.duration:.0f}s reached. Stopping.")
                break
            t0 = time.perf_counter()

            # -- Teleop: read leader, send to follower -------------------------
            if not observe_only and leader is not None:
                leader_action = leader.get_action()
                _ = follower.send_action(leader_action)

            # -- Read follower observation -------------------------------------
            robot_obs = follower.get_observation()

            # -- E-stop: detect erratic jumps (teleop mode only) ---------------
            if not observe_only:
                cur_vals = np.array(
                    [float(robot_obs.get(f"{m}.pos", 0)) for m in motor_names],
                    dtype=float,
                )
                if prev_obs_vals is not None:
                    jumps = np.abs(cur_vals - prev_obs_vals)
                    worst_joint = motor_names[int(np.argmax(jumps))]
                    worst_jump = jumps.max()
                    if worst_jump > args.estop_threshold:
                        print(
                            f"\n\n\033[91m⚠⚠ E-STOP ⚠⚠  {worst_joint} jumped "
                            f"{worst_jump:.1f} units in one frame!\033[0m"
                        )
                        print("Disabling torque and stopping.")
                        follower.bus.disable_torque()
                        break
                prev_obs_vals = cur_vals

            # -- Singularity indicator (read-only) -----------------------------
            q_deg = norm_to_deg(robot_obs)

            j_arm = follower_kin.compute_frame_jacobian(
                q_deg, joint_names=ARM_JOINT_NAMES
            )
            jv = extract_translational_jacobian(j_arm)
            manip = compute_manipulability(jv)

            sigma = manip.sigma_min
            cond = manip.condition_number
            if sigma < SIGMA_MIN_CRITICAL:
                tag = "\033[91m▓▓ CRITICAL \033[0m"
            elif sigma < SIGMA_MIN_WARN:
                tag = "\033[93m▒▒ WARNING  \033[0m"
            else:
                tag = "\033[92m░░ OK       \033[0m"

            dt_ms = (time.perf_counter() - t0) * 1e3
            print(
                f"\r{tag}  σ_min={sigma:.4f}  cond={cond:7.1f}  "
                f"loop={dt_ms:5.1f}ms ({1e3/max(dt_ms,1e-3):4.0f}Hz)",
                end="",
                flush=True,
            )

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        follower.disconnect()
        if leader is not None:
            leader.disconnect()
        log.info("Disconnected.")


if __name__ == "__main__":
    main()
