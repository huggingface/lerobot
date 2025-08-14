import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="192.168.31.166", id="MyKiwi")
# robot_config = LeKiwiClientConfig(remote_ip="100.96.43.20", id="MyKiwi")
teleop_arm_config = SO101LeaderConfig(port="COM6", id="my_awesome_leader_arm")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
leader_arm = SO101Leader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader_arm.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_teleop")

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    arm_action = leader_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**arm_action, **base_action})

    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
