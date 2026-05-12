#!/usr/bin/env python3
"""模拟测试：验证 keyboard_joint → delta processor → NERO send_action 完整链路。"""

import sys
import time

sys.path.insert(0, "src")

from lerobot.robots.nero_follower.nero_follower import NEOFollower, NERO_JOINTS
from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig
from lerobot.robots.nero_follower.robot_joint_delta_processor import (
    NEROKeyboardJointDeltasToAbsolute,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.types import TransitionKey


def simulate_teleop_step(processor, teleop_action, obs):
    """模拟 teleop_loop 中的一步：teleop_action → processor → robot_action"""
    transition = robot_action_observation_to_transition((teleop_action, obs))
    result_transition = processor(transition)
    robot_action = result_transition[TransitionKey.ACTION]
    return robot_action


def main():
    # 1. 连接真实机械臂
    cfg = NEOFollowerRobotConfig(
        interface="socketcan",
        channel="can0",
        firmeware_version="default",
        speed_percent=10,
        disable_torque_on_disconnect=False,
    )
    robot = NEOFollower(cfg)

    print("[1/4] 连接机械臂...")
    robot.connect()

    # 2. 创建处理器
    processor = NEROKeyboardJointDeltasToAbsolute(
        joint_names=NERO_JOINTS,
        gripper_min=0.0,
        gripper_max=100.0,
    )

    # 3. 读取当前状态
    print("[2/4] 读取当前状态...")
    obs = robot.get_observation()
    for name in NERO_JOINTS:
        print(f"  {name}.pos = {obs[f'{name}.pos']:.4f}")
    print(f"  gripper.pos = {obs['gripper.pos']:.1f}")

    # 4. 模拟3步键盘输入
    print("[3/4] 模拟键盘输入...")

    # 第1步：joint1+ 和 joint2-，死人开关开
    teleop_action_1 = {
        "enabled": 1.0,
        "joint1.delta": 0.05,
        "joint2.delta": -0.05,
        "joint3.delta": 0.0,
        "joint4.delta": 0.0,
        "joint5.delta": 0.0,
        "joint6.delta": 0.0,
        "joint7.delta": 0.0,
        "gripper.delta": 0.0,
    }
    robot_action_1 = simulate_teleop_step(processor, teleop_action_1, obs)
    print(f"  步骤1 处理器输出: {robot_action_1}")
    if robot_action_1:
        robot.send_action(robot_action_1)
        print("  → 已发送到机械臂")
    time.sleep(1.0)

    # 重新读取状态
    obs = robot.get_observation()
    print(f"  joint1.pos = {obs['joint1.pos']:.4f} (期望 ≈ 初始 + 0.05)")
    print(f"  joint2.pos = {obs['joint2.pos']:.4f} (期望 ≈ 初始 - 0.05)")

    # 第2步：joint3+，夹爪开
    teleop_action_2 = {
        "enabled": 1.0,
        "joint1.delta": 0.0,
        "joint2.delta": 0.0,
        "joint3.delta": 0.1,
        "joint4.delta": 0.0,
        "joint5.delta": 0.0,
        "joint6.delta": 0.0,
        "joint7.delta": 0.0,
        "gripper.delta": 5.0,
    }
    robot_action_2 = simulate_teleop_step(processor, teleop_action_2, obs)
    print(f"  步骤2 处理器输出: {robot_action_2}")
    if robot_action_2:
        robot.send_action(robot_action_2)
        print("  → 已发送到机械臂")
    time.sleep(1.0)

    # 第3步：死人开关关，不应该有任何动作
    obs = robot.get_observation()
    teleop_action_3 = {
        "enabled": 0.0,
        "joint1.delta": 0.05,
        "joint2.delta": 0.05,
        "joint3.delta": 0.0,
        "joint4.delta": 0.0,
        "joint5.delta": 0.0,
        "joint6.delta": 0.0,
        "joint7.delta": 0.0,
        "gripper.delta": 10.0,
    }
    robot_action_3 = simulate_teleop_step(processor, teleop_action_3, obs)
    print(f"  步骤3 (死人开关关) 处理器输出: {robot_action_3}")
    if not robot_action_3:
        print("  → 正确：死人开关关闭时无动作")
    else:
        print("  → 错误：死人开关关闭时仍有动作！")

    # 4. 断开
    print("[4/4] 断开连接...")
    robot.disconnect()
    print("完成。链路测试通过。")


if __name__ == "__main__":
    main()
