import time
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot_robot_piper import Piper, PiperConfig

FPS = 30

# Mapping from SO101 leader joints (5) to Piper joints (first 5)
so101_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

register_third_party_devices()

# 1) Teleop (SO101 leader) – set your serial port
leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", use_degrees=True))

# 2) Piper robot – set CAN interface/bitrate. include_gripper for gripper control
robot = Piper(PiperConfig(can_interface="can0", bitrate=1_000_000, include_gripper=True))

leader.connect()
robot.connect()

try:
    while True:
        t0 = time.perf_counter()

        # Robot obs (we use joint_6 current value to keep it steady)
        obs = robot.get_observation()
        action = {}

        # Leader joint actions (degrees)
        lead_act = leader.get_action()  # keys like 'shoulder_pan.pos', ..., 'gripper.pos'

        # Map first 5 joints 1:1 to Piper joints 1..5
        for i, name in enumerate(so101_joint_names, start=1):
            key = f"{name}.pos"
            if key in lead_act:
                action[f"joint_{i}.pos"] = float(lead_act[key])

        # Keep Piper joint_6 at current pose (no sixth DoF on leader)
        action["joint_6.pos"] = float(obs.get("joint_6.pos", 0.0))

        # Map gripper percent (0..100) -> mm using Piper SDK limits
        if robot.config.include_gripper and "gripper.pos" in lead_act and robot._iface is not None:
            p = float(lead_act["gripper.pos"])  # 0..100 from SO101 leader
            g_min = robot._iface.min_pos[6]
            g_max = robot._iface.max_pos[6]
            action["gripper.pos"] = g_min + (g_max - g_min) * (p / 100.0)

        # Send
        robot.send_action(action)

        # Pace loop
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, 1.0 / FPS - dt))
finally:
    leader.disconnect()
    robot.disconnect()