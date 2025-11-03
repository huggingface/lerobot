#!/usr/bin/env python

import argparse
import time
from typing import List

from lerobot.utils.import_utils import register_third_party_devices
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot_robot_piper import Piper, PiperConfig


FPS = 20
LOG_EVERY = 10

# SO101 leader joint names (5-DOF arm + gripper channel)
LEADER_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Approx leader ranges in degrees for normalization (adjust to your hardware)
LEADER_RANGES_DEG = {
    "shoulder_pan": (-150.0, 150.0),
    "shoulder_lift": (-112.0, 0.0),
    "elbow_flex": (0.0, 97.0),
    "wrist_flex": (-70.0, 70.0),
    "wrist_roll": (-180.0, 180.0),
}


def deg_to_percent(value_deg: float, rng: tuple[float, float]) -> float:
    lmin, lmax = rng
    if lmax <= lmin:
        return 0.0
    return (value_deg - lmin) / (lmax - lmin) * 200.0 - 100.0


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compute_hold_percent(j_idx_1based: int, obs: dict, min_pos: List[float], max_pos: List[float], signs: List[int]) -> float:
    # obs contains sign-applied degrees
    cur = float(obs.get(f"joint_{j_idx_1based}.pos", 0.0))
    cur_hw = cur * signs[j_idx_1based - 1]  # undo sign to map within hw limits
    jmin = float(min_pos[j_idx_1based - 1])
    jmax = float(max_pos[j_idx_1based - 1])
    if jmax <= jmin:
        return 0.0
    pct = (cur_hw - jmin) / (jmax - jmin) * 200.0 - 100.0
    return clamp(pct, -100.0, 100.0)


def run_test(source: str, targets: List[str], leader_port: str, can_interface: str, bitrate: int, gain: float) -> None:
    register_third_party_devices()

    leader = SO101Leader(SO101LeaderConfig(port=leader_port, use_degrees=True))
    robot = Piper(PiperConfig(can_interface=can_interface, bitrate=bitrate, include_gripper=True))

    leader.connect()
    robot.connect()

    try:
        if robot._iface is None:
            raise RuntimeError("Piper SDK interface not available; cannot run percent-based tester")

        print("Connected. Safety tester starting.")
        print("Controls:")
        print("- Move ONLY the selected leader joint during each subtest.")
        print("- Conservative gain is applied; increase with --gain if needed (e.g., 0.6).")
        print("- Press Ctrl+C to end a subtest and proceed to the next.")

        signs = robot.config.joint_signs

        # One-time hardware info
        mins = [round(float(v), 2) for v in robot._iface.min_pos[:6]]
        maxs = [round(float(v), 2) for v in robot._iface.max_pos[:6]]
        g_min = float(robot._iface.min_pos[6])
        g_max = float(robot._iface.max_pos[6])
        print("Piper joint limits (deg):", list(zip(mins, maxs)))
        print("Piper gripper limit (mm):", round(g_min, 2), "to", round(g_max, 2))
        print("joint_signs:", signs)

        # Resolve source channel
        if source == "gripper":
            leader_key = "gripper.pos"
        else:
            if source not in LEADER_RANGES_DEG:
                raise ValueError(f"Unknown leader source '{source}'. Options: {list(LEADER_RANGES_DEG.keys()) + ['gripper']}")
            leader_key = f"{source}.pos"

        # Resolve target sequence
        resolved_targets: List[str] = []
        for t in targets:
            if t == "all":
                resolved_targets.extend(["1", "2", "3", "4", "5", "6", "gripper"])
            elif t == "joints":
                resolved_targets.extend(["1", "2", "3", "4", "5", "6"]) 
            else:
                resolved_targets.append(t)

        for t in resolved_targets:
            if t.lower() == "gripper":
                title = f"Subtest: {source} -> Piper gripper"
            else:
                title = f"Subtest: {source} -> Piper joint_{t}"
            print("\n" + "-" * len(title))
            print(title)
            print("-" * len(title))
            input("Press ENTER to start this subtest; Ctrl+C to end it at any time...")

            loop_idx = 0
            while True:
                t0 = time.perf_counter()

                obs = robot.get_observation()
                lead_act = leader.get_action()  # keys like '<name>.pos' and 'gripper.pos'

                # Build normalized percents for joints and gripper
                pct = [0.0] * 6

                # Fill holds for all 6 joints first (safe default)
                mins_hw = robot._iface.min_pos[:6]
                maxs_hw = robot._iface.max_pos[:6]
                for j in range(1, 7):
                    pct[j - 1] = compute_hold_percent(j, obs, mins_hw, maxs_hw, signs)

                # Compute control value from leader source
                if source == "gripper":
                    # leader gripper is 0..100 percent
                    src_val_pct = float(lead_act.get(leader_key, 0.0))
                    src_val_pct = clamp(src_val_pct, 0.0, 100.0)
                else:
                    # leader source in degrees -> percent
                    src_deg = float(lead_act.get(leader_key, 0.0))
                    src_pct = deg_to_percent(src_deg, LEADER_RANGES_DEG[source])
                    src_val_pct = clamp(src_pct * gain, -100.0, 100.0)

                # Route to target
                if t.lower() == "gripper":
                    g_pct = src_val_pct if source == "gripper" else 0.0
                else:
                    # map source percent to one Piper joint
                    j_idx = int(t)
                    pct[j_idx - 1] = src_val_pct if source != "gripper" else 0.0
                    # hold gripper from current obs
                    g_min = robot._iface.min_pos[6]
                    g_max = robot._iface.max_pos[6]
                    g_mm = float(obs.get("gripper.pos", (g_min + g_max) * 0.5))
                    g_pct = (g_mm - g_min) / (g_max - g_min) * 100.0 if g_max > g_min else 50.0
                    g_pct = clamp(g_pct, 0.0, 100.0)

                # Debug
                if (loop_idx % LOG_EVERY) == 0:
                    # Leader states
                    leader_joints_deg = [round(float(lead_act.get(f"{nm}.pos", 0.0)), 1) for nm in LEADER_JOINT_NAMES]
                    leader_grip_pct = round(float(lead_act.get("gripper.pos", 0.0)), 1)

                    # Piper states
                    piper_obs_deg = [round(float(obs.get(f"joint_{i}.pos", 0.0)), 1) for i in range(1, 7)]
                    piper_grip_mm = round(float(obs.get("gripper.pos", 0.0)), 2)

                    # Source debug
                    if source == "gripper":
                        src_dbg = f"src(gripper%)={src_val_pct:.1f}"
                    else:
                        src_deg = float(lead_act.get(leader_key, 0.0))
                        src_dbg = f"src({source} deg)={src_deg:.1f} -> pct={src_val_pct:.1f}"

                    print(
                        src_dbg,
                        "| target=", t,
                        "| leader_deg=", leader_joints_deg,
                        "| leader_grip%=", leader_grip_pct,
                        "| send_pct(joints)=", [round(v, 1) for v in pct],
                        "| g_pct=", round(g_pct, 1),
                        "| piper_obs_deg=", piper_obs_deg,
                        "| piper_grip_mm=", piper_grip_mm,
                    )

                # Send normalized command (6 joints + gripper)
                robot._iface.set_joint_positions(pct + [g_pct])

                # pacing
                dt = time.perf_counter() - t0
                time.sleep(max(0.0, 1.0 / FPS - dt))
                loop_idx += 1

    except KeyboardInterrupt:
        print("\nSubtest ended by user.")
    finally:
        try:
            leader.disconnect()
        finally:
            robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Safe tele-mapping tester: control one Piper joint at a time from SO101 leader")
    parser.add_argument("--leader-port", default="/dev/ttyACM0", help="Serial port for SO101 leader (e.g., /dev/ttyACM0)")
    parser.add_argument("--can-interface", default="can0", help="CAN interface for Piper (e.g., can0)")
    parser.add_argument("--bitrate", type=int, default=1_000_000, help="CAN bitrate")
    parser.add_argument("--gain", type=float, default=0.3, help="Leader -> Piper gain (0..1) for tested joint")
    parser.add_argument(
        "--source",
        default="gripper",
        choices=LEADER_JOINT_NAMES + ["gripper"],
        help="Leader source channel to test (default gripper)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["gripper", "1", "2", "3", "4", "5", "6"],
        help="Target list: 'gripper', joint numbers '1'..'6', or aliases 'all'/'joints'",
    )

    args = parser.parse_args()
    run_test(
        source=args.source,
        targets=args.targets,
        leader_port=args.leader_port,
        can_interface=args.can_interface,
        bitrate=args.bitrate,
        gain=args.gain,
    )


if __name__ == "__main__":
    main()


