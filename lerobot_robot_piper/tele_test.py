import time
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot_robot_piper import Piper, PiperConfig

FPS = 30
LOG_EVERY = 10

# SO101 leader joint names (5 DOF) in degrees
so101_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Shrink leader command to avoid saturating Piper limits too often
LEADER_GAIN = 0.9
 # 60% of leader normalized range

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

register_third_party_devices()

# 1) Teleop (SO101 leader) – set your serial port
# Use normalized mode so joints are in [-100, 100]
leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", use_degrees=False))

# 2) Piper robot – use normalized pipeline (use_degrees=False) and optional gripper
robot = Piper(
    PiperConfig(
        can_interface="can0",
        bitrate=1_000_000,
        include_gripper=True,
        use_degrees=False,
    )
)

leader.connect()
robot.connect()

try:
    loop_idx = 0
    while True:
        t0 = time.perf_counter()

        obs = robot.get_observation()  # already normalized because use_degrees=False
        lead_act = leader.get_action()

        action = {}
        for alias in so101_joint_names:
            alias_key = f"{alias}.pos"
            if alias_key not in lead_act:
                continue
            if alias not in robot.config.joint_aliases:
                continue
            action[alias_key] = clamp(float(lead_act[alias_key]) * LEADER_GAIN, -100.0, 100.0)

        if robot.config.include_gripper and "gripper.pos" in lead_act:
            action["gripper.pos"] = clamp(float(lead_act["gripper.pos"]), 0.0, 100.0)

        # Debug print
        if (loop_idx % LOG_EVERY) == 0:
            leader_pct = [round(float(lead_act.get(f"{nm}.pos", 0.0)), 1) for nm in so101_joint_names]
            piper_obs = [round(float(obs.get(f"joint_{i}.pos", 0.0)), 1) for i in range(1, 7)]
            print(
                "leader_pct=", leader_pct,
                "| send_pct(j1..j6)=",
                [
                    round(action.get(f"joint_{i}.pos", action.get(f"{alias}.pos", 0.0)), 1)
                    for i, alias in enumerate([
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "joint_4",
                        "wrist_flex",
                        "wrist_roll",
                    ], start=1)
                ],
                "| g_pct=", round(action.get("gripper.pos", 0.0), 1),
                "| piper_obs_pct=", piper_obs,
            )

        robot.send_action(action)

        # Pace loop
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / FPS - dt))
        loop_idx += 1
finally:
    leader.disconnect()
    robot.disconnect()