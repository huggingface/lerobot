#!/usr/bin/env python3
"""端到端测试：keyboard_joint → processor → NERO，5秒自动退出。"""
import logging, sys, time

logging.basicConfig(level=logging.WARNING)

from lerobot.robots.nero_follower.nero_follower import NEOFollower
from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig
from lerobot.robots.nero_follower.robot_joint_delta_processor import NEROKeyboardJointDeltasToAbsolute
from lerobot.processor.converters import robot_action_observation_to_transition
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardJointTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardJointTeleopConfig
from lerobot.types import TransitionKey

cfg = NEOFollowerRobotConfig(interface='socketcan', channel='can0', speed_percent=10, disable_torque_on_disconnect=False)
robot = NEOFollower(cfg)
robot.connect()

teleop_cfg = KeyboardJointTeleopConfig(joint_step=0.05, gripper_step=2.0, require_deadman=True)
teleop = KeyboardJointTeleop(teleop_cfg)
teleop.connect()

processor = NEROKeyboardJointDeltasToAbsolute(
    joint_names=['joint1','joint2','joint3','joint4','joint5','joint6','joint7'],
    gripper_min=0.0, gripper_max=100.0,
)

print('=== 运行5秒，请按住 Space+Q 测试（ESC退出）===')
start = time.perf_counter()
count = 0
sent = 0
while time.perf_counter() - start < 5.0:
    obs = robot.get_observation()
    try:
        raw_action = teleop.get_action()
    except Exception as e:
        print(f'  get_action error: {e}')
        break
    count += 1

    # 每30次循环打印一次 raw_action
    if count % 30 == 0:
        print(f'  [{count}] raw_action={raw_action}')

    transition = robot_action_observation_to_transition((raw_action, obs))
    result_transition = processor(transition)
    robot_action = result_transition[TransitionKey.ACTION]

    if robot_action:
        robot.send_action(robot_action)
        sent += 1
        if sent <= 5:
            print(f'  [{sent}] enabled={raw_action.get("enabled")}, j1d={raw_action.get("joint1.delta")}, action={robot_action}')

    time.sleep(0.033)

print(f'\n总计: {count} 次循环, {sent} 次有效动作')
teleop.disconnect()
robot.disconnect()
print('完成。')
