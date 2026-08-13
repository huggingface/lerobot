"""
Usage:
  python examples/alohamini/teleoperate_bi.py \
    --remote_ip 192.168.50.84 \
    --leader_id so101_leader_bi_t6
"""

import argparse
import time

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============ Parameter Section ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--no_robot", action="store_true", help="Do not connect robot, only print actions")
parser.add_argument(
    "--no_leader",
    action="store_true",
    help="Do not connect leader arm, only perform keyboard-controlled actions.",
)
parser.add_argument("--fps", type=int, default=30, help="Main loop frequency (frames per second)")
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="AlohaMini host IP address")
parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")

args = parser.parse_args()

NO_ROBOT = args.no_robot
NO_LEADER = args.no_leader
FPS = args.fps
# ========================================== #

if NO_ROBOT:
    print("🧪 NO_ROBOT mode enabled: robot will not connect, only print actions.")

if NO_LEADER:
    print("🧪 NO_LEADER mode enabled: leader arm will not connect, only print actions.")
# Create configs
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
bi_cfg = BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(
        port="/dev/am_arm_leader_left",
    ),
    right_arm_config=SOLeaderConfig(
        port="/dev/am_arm_leader_right",
    ),
    id=args.leader_id,
)
leader = BiSOLeader(bi_cfg)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
robot = LeKiwiClient(robot_config)

# Connection logic
if not NO_ROBOT:
    robot.connect()
else:
    print("🧪 robot.connect() skipped, only printing actions.")

if not NO_LEADER:
    leader.connect()
else:
    print("🧪 robot.connect() skipped, only printing actions.")

keyboard.connect()


init_rerun(session_name="lekiwi_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ Warning: Some devices are not connected! Still running for debug.")

# Main loop
while True:
    t0 = time.perf_counter()

    observation = robot.get_observation() if not NO_ROBOT else {}
    arm_actions = leader.get_action() if not NO_LEADER else {}
    arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

    action = {**arm_actions, **base_action, **lift_action}
    log_rerun_data(observation, action)

    if NO_ROBOT:
        print(f"[NO_ROBOT] action → {action}")
    else:
        robot.send_action(action)
        print(f"Sent action → {action}")

    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
