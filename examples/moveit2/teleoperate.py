from lerobot.common.robots.moveit2 import MoveIt2, MoveIt2Config
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig

robot_config = MoveIt2Config()

teleop_keyboard_config = KeyboardTeleopConfig(
    id="my_laptop_keyboard",
)

robot = MoveIt2(robot_config)
teleop_keyboard = KeyboardTeleop(teleop_keyboard_config)
robot.connect()
teleop_keyboard.connect()

while True:
    observation = robot.get_observation()
    keyboard_keys = teleop_keyboard.get_action()
    action = robot.from_keyboard_to_action(keyboard_keys)
    robot.send_action(action)
