#!/usr/bin/env python3
"""冒烟测试：控制 NERO 机械臂单个关节来回运动 + 夹爪开合，验证 CAN 通信与控制是否正常。"""

import sys
import time

sys.path.insert(0, "src")

from lerobot.robots.nero_follower.nero_follower import NEOFollower, NERO_JOINTS
from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig

# ---------- 可调参数 ----------
JOINT_INDEX = 2       # 测试的关节索引（0=joint1, 1=joint2, ...）
DELTA_RAD = 0.15      # 摆动幅度（弧度）
STEPS = 3             # 来回摆动次数
PAUSE_SEC = 1.0       # 每次运动后的停顿时间（秒）
GRIPPER_OPEN = 80.0   # 夹爪打开目标值（0-100，度数映射）
GRIPPER_CLOSE = 0.0   # 夹爪闭合目标值


def main():
    # 1. 构造配置：使用 socketcan 通信，10% 低速，断开时不下电
    cfg = NEOFollowerRobotConfig(
        interface="socketcan",
        channel="can0",
        firmeware_version="default",
        speed_percent=10,
        disable_torque_on_disconnect=False,  # 不断电！下电后不自锁会掉
    )

    robot = NEOFollower(cfg)

    # 2. 连接机械臂（自动使能电机 + 初始化夹爪）
    print("[1/6] 连接机械臂...")
    robot.connect()
    print("  已连接。")

    # 3. 读取当前关节角度作为初始位置
    print("[2/6] 读取初始位置...")
    obs = robot.get_observation()
    joint_name = NERO_JOINTS[JOINT_INDEX]
    init_val = obs[f"{joint_name}.pos"]
    init_gripper = obs["gripper.pos"]
    print(f"  {joint_name}.pos = {init_val:.4f} rad")
    print(f"  gripper.pos = {init_gripper:.1f}")

    # 4. 来回摆动指定关节
    print(f"[3/6] 摆动 {joint_name} ±{DELTA_RAD} rad，共 {STEPS} 个来回...")
    for i in range(STEPS):
        # 正向偏移
        target_pos = init_val + DELTA_RAD
        print(f"  第 {i+1}/{STEPS} 次: 运动到 {target_pos:.4f}")
        robot.send_action({f"{joint_name}.pos": target_pos})
        time.sleep(PAUSE_SEC)

        # 反向偏移
        target_pos = init_val - DELTA_RAD
        print(f"  第 {i+1}/{STEPS} 次: 运动到 {target_pos:.4f}")
        robot.send_action({f"{joint_name}.pos": target_pos})
        time.sleep(PAUSE_SEC)

    # 5. 回到初始位置
    print("[4/6] 关节回到初始位置...")
    robot.send_action({f"{joint_name}.pos": init_val})
    time.sleep(PAUSE_SEC)

    obs = robot.get_observation()
    final_val = obs[f"{joint_name}.pos"]
    print(f"  {joint_name}.pos = {final_val:.4f} rad（初始值 {init_val:.4f}）")

    # 6. 夹爪开合测试
    print(f"[5/6] 夹爪测试：打开到 {GRIPPER_OPEN}° ...")
    robot.send_action({"gripper.pos": GRIPPER_OPEN})
    time.sleep(PAUSE_SEC)
    obs = robot.get_observation()
    print(f"  gripper.pos = {obs['gripper.pos']:.1f}")

    print(f"  夹爪闭合到 {GRIPPER_CLOSE}° ...")
    robot.send_action({"gripper.pos": GRIPPER_CLOSE})
    time.sleep(PAUSE_SEC)
    obs = robot.get_observation()
    print(f"  gripper.pos = {obs['gripper.pos']:.1f}")

    print(f"  夹爪恢复初始值 {init_gripper:.1f}° ...")
    robot.send_action({"gripper.pos": init_gripper})
    time.sleep(PAUSE_SEC)

    # 7. 断开连接（不下电）
    print("[6/6] 断开连接...")
    robot.disconnect()
    print("完成。单关节 + 夹爪测试通过。")


if __name__ == "__main__":
    main()
