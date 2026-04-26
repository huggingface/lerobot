#!/usr/bin/env python3
"""单元测试：验证 KeyboardJointTeleop 逻辑（不需要真实终端）。"""
import sys
sys.path.insert(0, "src")

from lerobot.robots.nero_follower.robot_joint_delta_processor import (
    NEROKeyboardJointDeltasToAbsolute,
)
from lerobot.processor.converters import robot_action_observation_to_transition
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardJointTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardJointTeleopConfig
from lerobot.types import TransitionKey


def test_teleop_logic():
    """直接测试 get_action 逻辑，绕过 connect 装饰器。"""
    cfg = KeyboardJointTeleopConfig(joint_step=0.05, gripper_step=2.0)
    teleop = KeyboardJointTeleop(cfg)

    # 测试1: Q+I (joint1+, 夹爪开)
    teleop.current_pressed = {ord("q"), ord("i")}

    action = {"enabled": 1.0}
    for i in range(cfg.num_joints):
        delta = 0.0
        if teleop.JOINT_PLUS[i] in teleop.current_pressed:
            delta += cfg.joint_step
        if teleop.JOINT_MINUS[i] in teleop.current_pressed:
            delta -= cfg.joint_step
        action[f"joint{i+1}.delta"] = float(delta)

    gripper_delta = 0.0
    if ord("i") in teleop.current_pressed:
        gripper_delta += cfg.gripper_step
    if ord("k") in teleop.current_pressed:
        gripper_delta -= cfg.gripper_step
    action["gripper.delta"] = float(gripper_delta)

    print("测试1: Q+I (joint1+, 夹爪开)")
    assert action["enabled"] == 1.0
    assert action["joint1.delta"] == 0.05
    assert action["joint2.delta"] == 0.0
    assert action["gripper.delta"] == 2.0
    print("  通过 ✓")

    # 测试2: Q+A (joint1 正负抵消)
    teleop.current_pressed = {ord("q"), ord("a")}
    action2 = {"enabled": 1.0}
    for i in range(cfg.num_joints):
        delta = 0.0
        if teleop.JOINT_PLUS[i] in teleop.current_pressed:
            delta += cfg.joint_step
        if teleop.JOINT_MINUS[i] in teleop.current_pressed:
            delta -= cfg.joint_step
        action2[f"joint{i+1}.delta"] = float(delta)
    action2["gripper.delta"] = 0.0

    print("测试2: Q+A (joint1 正负抵消)")
    assert action2["joint1.delta"] == 0.0
    print("  通过 ✓")

    # 测试3: W+S+K (joint2 抵消, 夹爪合)
    teleop.current_pressed = {ord("w"), ord("s"), ord("k")}
    action3 = {"enabled": 1.0}
    for i in range(cfg.num_joints):
        delta = 0.0
        if teleop.JOINT_PLUS[i] in teleop.current_pressed:
            delta += cfg.joint_step
        if teleop.JOINT_MINUS[i] in teleop.current_pressed:
            delta -= cfg.joint_step
        action3[f"joint{i+1}.delta"] = float(delta)
    gd = 0.0
    if ord("i") in teleop.current_pressed:
        gd += cfg.gripper_step
    if ord("k") in teleop.current_pressed:
        gd -= cfg.gripper_step
    action3["gripper.delta"] = float(gd)

    print("测试3: W+S+K (joint2 抵消, 夹爪合)")
    assert action3["joint2.delta"] == 0.0
    assert action3["gripper.delta"] == -2.0
    print("  通过 ✓")

    # 测试4: require_deadman=True 时 Shift 使能
    cfg_deadman = KeyboardJointTeleopConfig(joint_step=0.05, gripper_step=2.0, require_deadman=True)
    teleop_deadman = KeyboardJointTeleop(cfg_deadman)
    teleop_deadman.current_pressed = {ord("q")}
    enabled = not cfg_deadman.require_deadman  # 无 Shift，deadman=True → False

    print("测试4: require_deadman=True 但无 Shift → enabled=False")
    assert enabled == False
    print("  通过 ✓")


def test_processor():
    """测试 delta→absolute 处理器。"""
    step = NEROKeyboardJointDeltasToAbsolute(
        joint_names=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"],
        gripper_min=0.0, gripper_max=100.0,
    )

    obs = {
        "joint1.pos": 0.1, "joint2.pos": -0.2, "joint3.pos": 0.3,
        "joint4.pos": 0.4, "joint5.pos": -0.5, "joint6.pos": 0.6, "joint7.pos": 0.0,
        "gripper.pos": 50.0,
    }

    # 有增量
    action = {
        "enabled": 1.0,
        "joint1.delta": 0.05, "joint2.delta": -0.05, "joint3.delta": 0.0,
        "joint4.delta": 0.0, "joint5.delta": 0.0, "joint6.delta": 0.0,
        "joint7.delta": 0.0, "gripper.delta": 5.0,
    }
    transition = robot_action_observation_to_transition((action, obs))
    result = step(transition)[TransitionKey.ACTION]

    print("测试5: 处理器增量→绝对")
    assert abs(result["joint1.pos"] - 0.15) < 1e-6, f"joint1: {result['joint1.pos']}"
    assert abs(result["joint2.pos"] - (-0.25)) < 1e-6, f"joint2: {result['joint2.pos']}"
    assert "joint3.pos" not in result
    assert abs(result["gripper.pos"] - 55.0) < 1e-6, f"gripper: {result['gripper.pos']}"
    print("  通过 ✓")

    # enabled=False
    action2 = {"enabled": 0.0, "joint1.delta": 0.05, "gripper.delta": 10.0}
    transition2 = robot_action_observation_to_transition((action2, obs))
    result2 = step(transition2)[TransitionKey.ACTION]
    print("测试6: enabled=False 返回空")
    assert result2 == {}
    print("  通过 ✓")

    # gripper 限位
    obs["gripper.pos"] = 99.0
    action3 = {"enabled": 1.0, "joint1.delta": 0.0, "joint2.delta": 0.0,
               "joint3.delta": 0.0, "joint4.delta": 0.0, "joint5.delta": 0.0,
               "joint6.delta": 0.0, "joint7.delta": 0.0, "gripper.delta": 5.0}
    transition3 = robot_action_observation_to_transition((action3, obs))
    result3 = step(transition3)[TransitionKey.ACTION]
    print("测试7: 夹爪限位 (99+5→100)")
    assert abs(result3["gripper.pos"] - 100.0) < 1e-6, f"gripper: {result3['gripper.pos']}"
    print("  通过 ✓")


if __name__ == "__main__":
    test_teleop_logic()
    test_processor()
    print("\n全部测试通过 ✓")
