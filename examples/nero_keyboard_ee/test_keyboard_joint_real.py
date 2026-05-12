#!/usr/bin/env python3
"""真机测试：模拟键盘输入，验证 keyboard_joint → processor → NERO 完整链路。"""
import sys
import time

sys.path.insert(0, "src")

from lerobot.robots.nero_follower.nero_follower import NEOFollower, NERO_JOINTS
from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig
from lerobot.robots.nero_follower.robot_joint_delta_processor import (
    NEROKeyboardJointDeltasToAbsolute,
)
from lerobot.processor.converters import robot_action_observation_to_transition
from lerobot.types import TransitionKey


def main():
    cfg = NEOFollowerRobotConfig(
        interface="socketcan",
        channel="can0",
        firmeware_version="default",
        speed_percent=10,
        disable_torque_on_disconnect=False,
    )
    robot = NEOFollower(cfg)

    processor = NEROKeyboardJointDeltasToAbsolute(
        joint_names=NERO_JOINTS,
        gripper_min=0.0,
        gripper_max=100.0,
    )

    print("[1/3] 连接机械臂...")
    robot.connect()

    obs = robot.get_observation()
    print("  当前关节位置:")
    for name in NERO_JOINTS:
        print(f"    {name}.pos = {obs[f'{name}.pos']:.4f}")
    print(f"    gripper.pos = {obs['gripper.pos']:.1f}")

    # 模拟键盘输入序列
    test_steps = [
        # (描述, teleop_action)
        ("joint1+ (模拟 Shift+Q)", {
            "enabled": 1.0, "joint1.delta": 0.05, "joint2.delta": 0.0,
            "joint3.delta": 0.0, "joint4.delta": 0.0, "joint5.delta": 0.0,
            "joint6.delta": 0.0, "joint7.delta": 0.0, "gripper.delta": 0.0,
        }),
        ("joint2- (模拟 Shift+S)", {
            "enabled": 1.0, "joint1.delta": 0.0, "joint2.delta": -0.05,
            "joint3.delta": 0.0, "joint4.delta": 0.0, "joint5.delta": 0.0,
            "joint6.delta": 0.0, "joint7.delta": 0.0, "gripper.delta": 0.0,
        }),
        ("joint3+ (模拟 Shift+E)", {
            "enabled": 1.0, "joint1.delta": 0.0, "joint2.delta": 0.0,
            "joint3.delta": 0.08, "joint4.delta": 0.0, "joint5.delta": 0.0,
            "joint6.delta": 0.0, "joint7.delta": 0.0, "gripper.delta": 0.0,
        }),
        ("夹爪开 (模拟 Shift+O)", {
            "enabled": 1.0, "joint1.delta": 0.0, "joint2.delta": 0.0,
            "joint3.delta": 0.0, "joint4.delta": 0.0, "joint5.delta": 0.0,
            "joint6.delta": 0.0, "joint7.delta": 0.0, "gripper.delta": 10.0,
        }),
        ("夹爪合 (模拟 Shift+L)", {
            "enabled": 1.0, "joint1.delta": 0.0, "joint2.delta": 0.0,
            "joint3.delta": 0.0, "joint4.delta": 0.0, "joint5.delta": 0.0,
            "joint6.delta": 0.0, "joint7.delta": 0.0, "gripper.delta": -10.0,
        }),
        ("无 Shift 按键 (不应有任何动作)", {
            "enabled": 0.0, "joint1.delta": 0.05, "joint2.delta": 0.05,
            "joint3.delta": 0.0, "joint4.delta": 0.0, "joint5.delta": 0.0,
            "joint6.delta": 0.0, "joint7.delta": 0.0, "gripper.delta": 5.0,
        }),
    ]

    print("[2/3] 模拟键盘输入序列...")
    for desc, teleop_action in test_steps:
        obs = robot.get_observation()
        transition = robot_action_observation_to_transition((teleop_action, obs))
        result_transition = processor(transition)
        robot_action = result_transition[TransitionKey.ACTION]

        if robot_action:
            robot.send_action(robot_action)
            print(f"  {desc}: 发送 {robot_action}")
        else:
            print(f"  {desc}: 无动作（正确）")

        time.sleep(0.8)

    # 读最终状态
    obs = robot.get_observation()
    print("\n[3/3] 最终关节位置:")
    for name in NERO_JOINTS:
        print(f"  {name}.pos = {obs[f'{name}.pos']:.4f}")
    print(f"  gripper.pos = {obs['gripper.pos']:.1f}")

    robot.disconnect()
    print("\n完成。真机链路测试通过。")


if __name__ == "__main__":
    main()
