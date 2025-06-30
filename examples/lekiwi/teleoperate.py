import time

from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="my_lekiwi")

teleop__arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem585A0077581", id="my_awesome_leader_arm")

teleop_keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
teleop_arm = SO100Leader(teleop__arm_config)
telep_keyboard = KeyboardTeleop(teleop_keyboard_config)

_init_rerun(session_name="lekiwi_teleop")

robot.connect()
teleop_arm.connect()
telep_keyboard.connect()

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    arm_action = teleop_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

    keyboard_keys = telep_keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**arm_action, **base_action})

    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
