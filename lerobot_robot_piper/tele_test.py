import time
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot_robot_piper import Piper, PiperConfig

FPS = 30
LOG_EVERY = 10

# SO101 leader joint names (5 DOF) in degrees
so101_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Explicit mapping from leader joint -> Piper joint index (1-based)
# Joint_4 is intentionally left unmapped and will be held steady.
leader_to_piper = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 5,
    "wrist_roll": 6,
}

# Shrink leader command to avoid saturating Piper limits too often
LEADER_GAIN = 0.9
 # 60% of leader normalized range

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

register_third_party_devices()

# 1) Teleop (SO101 leader) – set your serial port
# Use normalized mode so joints are in [-100, 100]
leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", use_degrees=False))

# 2) Piper robot – set CAN interface/bitrate. include_gripper for gripper control
robot = Piper(PiperConfig(can_interface="can0", bitrate=1_000_000, include_gripper=True))

leader.connect()
robot.connect()

try:
    # Print mapping and limits once
    if robot._iface is not None:
        mins = robot._iface.min_pos[:6]
        maxs = robot._iface.max_pos[:6]
        print("Piper limits (deg):", list(zip([round(m,1) for m in mins], [round(M,1) for M in maxs])))
        print("joint_signs:", robot.config.joint_signs)

    loop_idx = 0
    while True:
        t0 = time.perf_counter()

        # Robot obs (we use joint_6 current value to keep it steady)
        obs = robot.get_observation()


        # Leader joint actions (normalized percents for joints, 0..100 for gripper)
        lead_act = leader.get_action()

        if robot._iface is None:
            raise RuntimeError("Piper SDK interface not available; cannot perform percent-based control")

        signs = robot.config.joint_signs

        # Prefill with holds for all 6 joints
        pct = [0.0] * 6
        mins_hw = robot._iface.min_pos[:6]
        maxs_hw = robot._iface.max_pos[:6]
        for j in range(1, 7):
            cur = float(obs.get(f"joint_{j}.pos", 0.0))  # sign-applied deg
            cur_hw = cur * signs[j - 1]  # undo sign for HW mapping
            jmin = mins_hw[j - 1]
            jmax = maxs_hw[j - 1]
            if jmax > jmin:
                hold = (cur_hw - jmin) / (jmax - jmin) * 200.0 - 100.0
            else:
                hold = 0.0
            pct[j - 1] = clamp(hold, -100.0, 100.0)

        # Apply leader -> Piper mapping (SO101 normalized output)
        for name in so101_joint_names:
            key = f"{name}.pos"
            tgt = leader_to_piper.get(name)
            if tgt is None or key not in lead_act:
                continue
            ld_pct = float(lead_act[key])
            lp = clamp(ld_pct * LEADER_GAIN * signs[tgt - 1], -100.0, 100.0)
            pct[tgt - 1] = lp

        # Gripper percent 0..100
        if robot.config.include_gripper and "gripper.pos" in lead_act:
            g_pct = clamp(float(lead_act["gripper.pos"]), 0.0, 100.0)
        else:
            g_min = robot._iface.min_pos[6]
            g_max = robot._iface.max_pos[6]
            g_mm = float(obs.get("gripper.pos", (g_min + g_max) * 0.5))
            g_pct = clamp((g_mm - g_min) / (g_max - g_min) * 100.0 if g_max > g_min else 50.0, 0.0, 100.0)

        # Debug print
        if (loop_idx % LOG_EVERY) == 0:
            leader_pct = [round(float(lead_act.get(f"{nm}.pos", 0.0)), 1) for nm in so101_joint_names]
            piper_obs = [round(float(obs.get(f"joint_{i}.pos", 0.0)), 1) for i in range(1, 7)]
            print(
                "leader_pct=", leader_pct,
                "| send_pct(j1..j6)=", [round(v, 1) for v in pct],
                "| g_pct=", round(g_pct, 1),
                "| piper_obs_deg=", piper_obs,
            )

        # Send normalized command via SDK helper
        robot._iface.set_joint_positions(pct + [g_pct])

        # Pace loop
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / FPS - dt))
        loop_idx += 1
finally:
    leader.disconnect()
    robot.disconnect()