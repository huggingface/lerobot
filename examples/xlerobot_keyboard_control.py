#!/usr/bin/env python3
"""
XLeRobot键盘控制程序
基于双臂SO100控制，添加了头部电机和底盘控制
支持双臂、头部和底盘的完整控制
"""

import logging
import math
import time
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 关节校准系数 - 手动编辑
# 格式: [关节名称, 零位置偏移(度), 缩放系数]
JOINT_CALIBRATION = [
    ["shoulder_pan", 6.0, 1.0],  # 关节1: 零位置偏移, 缩放系数
    ["shoulder_lift", 2.0, 0.97],  # 关节2: 零位置偏移, 缩放系数
    ["elbow_flex", 0.0, 1.05],  # 关节3: 零位置偏移, 缩放系数
    ["wrist_flex", 0.0, 0.94],  # 关节4: 零位置偏移, 缩放系数
    ["wrist_roll", 0.0, 0.5],  # 关节5: 零位置偏移, 缩放系数
    ["gripper", 0.0, 1.0],  # 关节6: 零位置偏移, 缩放系数
]


def detect_motor_count(port):
    """
    检测指定端口的电机数量

    Args:
        port: 端口路径，如 "/dev/ttyACM0"

    Returns:
        motor_count: 检测到的电机数量，如果检测失败返回None
    """
    try:
        from lerobot.motors.feetech import FeetechMotorsBus

        # 创建一个空的MotorsBus来扫描端口
        bus = FeetechMotorsBus(port, {})
        bus._connect(handshake=False)

        # 优先尝试1000000波特率
        motor_count = 0
        preferred_baudrates = [1000000] + bus.available_baudrates

        for baudrate in preferred_baudrates:
            if baudrate in bus.available_baudrates:  # 确保波特率在支持列表中
                bus.set_baudrate(baudrate)
                ids_models = bus.broadcast_ping()
                if ids_models:
                    motor_count = len(ids_models)
                    print(f"端口 {port} 在波特率 {baudrate} 下检测到 {motor_count} 个电机: {ids_models}")
                    break

        bus.port_handler.closePort()
        return motor_count

    except Exception as e:
        print(f"检测端口 {port} 的电机数量时出错: {e}")
        return None


def auto_detect_ports():
    """
    自动检测并分配端口
    左手和头部共用一个端口（8个电机）
    右手和底盘共用一个端口（9个电机）

    Returns:
        tuple: (left_arm_port, right_arm_port, head_port, base_port)
    """
    print("正在自动检测端口...")

    # 检测两个可能的端口
    ports_to_check = ["/dev/ttyACM0", "/dev/ttyACM1"]
    port_motor_counts = {}

    for port in ports_to_check:
        print(f"正在检测端口 {port}...")
        motor_count = detect_motor_count(port)
        if motor_count is not None:
            port_motor_counts[port] = motor_count
            print(f"端口 {port} 检测到 {motor_count} 个电机")
        else:
            print(f"端口 {port} 检测失败")

    # 根据电机数量分配端口
    left_arm_port = None
    right_arm_port = None
    head_port = None
    base_port = None

    for port, motor_count in port_motor_counts.items():
        if motor_count == 8:
            # 8个电机：左手 + 头部
            left_arm_port = port
            head_port = port
            print(f"端口 {port} 分配为左手和头部 (8个电机)")
        elif motor_count == 9:
            # 9个电机：右手 + 底盘
            right_arm_port = port
            base_port = port
            print(f"端口 {port} 分配为右手和底盘 (9个电机)")
        else:
            print(f"警告: 端口 {port} 有 {motor_count} 个电机，不符合预期配置")

    # 检查是否成功检测到所有端口
    if left_arm_port is None or right_arm_port is None:
        print("错误: 无法检测到正确的端口配置")
        print("期望配置:")
        print("- 一个端口有8个电机 (左手 + 头部)")
        print("- 一个端口有9个电机 (右手 + 底盘)")
        return None, None, None, None

    print("端口自动检测完成:")
    print(f"- 左手和头部端口: {left_arm_port}")
    print(f"- 右手和底盘端口: {right_arm_port}")

    return left_arm_port, right_arm_port, head_port, base_port


def manual_port_config():
    """
    手动配置端口
    当自动检测失败时使用

    Returns:
        tuple: (left_arm_port, right_arm_port, head_port, base_port)
    """
    print("自动检测失败，请手动配置端口")
    print("根据您的描述:")
    print("- 左手和头部共用一个端口（8个电机）")
    print("- 右手和底盘共用一个端口（9个电机）")

    # 获取用户输入
    while True:
        try:
            left_port = input("请输入左手和头部端口 (例如 /dev/ttyACM0): ").strip()
            right_port = input("请输入右手和底盘端口 (例如 /dev/ttyACM1): ").strip()

            if left_port and right_port:
                print("手动配置完成:")
                print(f"- 左手和头部端口: {left_port}")
                print(f"- 右手和底盘端口: {right_port}")
                return left_port, right_port, left_port, right_port
            else:
                print("端口不能为空，请重新输入")
        except KeyboardInterrupt:
            print("\n用户取消配置")
            return None, None, None, None


def apply_joint_calibration(joint_name, raw_position):
    """
    应用关节校准系数

    Args:
        joint_name: 关节名称
        raw_position: 原始位置值

    Returns:
        calibrated_position: 校准后的位置值
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # 零位置偏移
            scale = joint_cal[2]  # 缩放系数
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # 如果没找到校准系数，返回原始值


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets

    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0

    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # Maximum reachable distance

    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)

    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    # Convert from radians to degrees
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

    return joint2_deg, joint3_deg


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    使用P控制缓慢移动XLeRobot到零位置

    Args:
        robot: XLeRobot实例
        duration: 移动到零位置所需时间(秒)
        kp: 比例增益
    """
    print("正在使用P控制缓慢移动XLeRobot到零位置...")

    # 获取当前机器人状态
    current_obs = robot.get_observation()

    # 提取当前关节位置
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith(".pos"):
            motor_name = key.removesuffix(".pos")
            current_positions[motor_name] = value

    # 零位置目标
    zero_positions = {
        # 左臂
        "left_shoulder_pan": 0.0,
        "left_shoulder_lift": 0.0,
        "left_elbow_flex": 0.0,
        "left_wrist_flex": 0.0,
        "left_wrist_roll": 0.0,
        "left_gripper": 0.0,
        # 右臂
        "right_shoulder_pan": 0.0,
        "right_shoulder_lift": 0.0,
        "right_elbow_flex": 0.0,
        "right_wrist_flex": 0.0,
        "right_wrist_roll": 0.0,
        "right_gripper": 0.0,
        # 头部
        "head_pan": 0.0,
        "head_tilt": 0.0,
    }

    # 计算控制步数
    control_freq = 50  # 50Hz控制频率
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(f"将在 {duration} 秒内使用P控制移动到零位置，控制频率: {control_freq}Hz，比例增益: {kp}")

    for step in range(total_steps):
        # 获取当前机器人状态
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                # 应用校准系数（仅对手臂关节）
                if motor_name.startswith("left_") or motor_name.startswith("right_"):
                    base_motor_name = motor_name.split("_", 1)[1]  # 移除left_或right_前缀
                    calibrated_value = apply_joint_calibration(base_motor_name, value)
                    current_positions[motor_name] = calibrated_value
                else:
                    current_positions[motor_name] = value

        # P控制计算
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos

                # P控制: 输出 = Kp * 误差
                control_output = kp * error

                # 将控制输出转换为位置命令
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # 发送动作到机器人
        if robot_action:
            robot.send_action(robot_action)

        # 显示进度
        if step % (control_freq // 2) == 0:  # 每0.5秒显示一次进度
            progress = (step / total_steps) * 100
            print(f"移动到零位置进度: {progress:.1f}%")

        time.sleep(step_time)

    print("XLeRobot已移动到零位置")


def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    使用P控制返回到起始位置

    Args:
        robot: XLeRobot实例
        start_positions: 起始关节位置字典
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    print("正在返回到起始位置...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # 最多5秒

    for step in range(max_steps):
        # 获取当前机器人状态
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                current_positions[motor_name] = value  # 不应用校准系数

        # P控制计算
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)

                # P控制: 输出 = Kp * 误差
                control_output = kp * error

                # 将控制输出转换为位置命令
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # 发送动作到机器人
        if robot_action:
            robot.send_action(robot_action)

        # 检查是否到达起始位置
        if total_error < 6.0:  # 如果总误差小于6度，认为已到达
            print("已返回到起始位置")
            break

        time.sleep(control_period)

    print("返回起始位置完成")


def p_control_loop(
    robot, keyboard, target_positions, start_positions, current_positions, kp=0.5, control_freq=50
):
    """
    P控制循环

    Args:
        robot: XLeRobot实例
        keyboard: 键盘实例
        target_positions: 目标关节位置字典
        start_positions: 起始关节位置字典
        current_positions: 当前关节位置字典
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    control_period = 1.0 / control_freq

    # 初始化双臂pitch控制变量
    pitch = {"left": 0.0, "right": 0.0}  # 初始pitch调整
    pitch_step = 1  # pitch调整步长

    # 初始化底盘速度
    base_velocities = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    base_speed_step = 0.1  # 底盘速度步长

    print(f"开始P控制循环，控制频率: {control_freq}Hz，比例增益: {kp}")

    while True:
        try:
            # 获取键盘输入
            keyboard_action = keyboard.get_action()

            if keyboard_action:
                # 处理键盘输入，更新目标位置
                for key, value in keyboard_action.items():
                    if key == "x":
                        # 退出程序，先回到起始位置
                        print("检测到退出命令，正在回到起始位置...")
                        return_to_start_position(robot, start_positions, 0.2, control_freq)
                        return

                    # 第一臂控制映射: 7y8u9i0o-p=[
                    arm1_joint_controls = {
                        "7": ("left_shoulder_pan", -1),  # 关节1减少
                        "y": ("left_shoulder_pan", 1),  # 关节1增加
                        "0": ("left_wrist_roll", -1),  # 关节5减少
                        "o": ("left_wrist_roll", 1),  # 关节5增加
                        "-": ("left_gripper", -1),  # 关节6减少
                        "p": ("left_gripper", 1),  # 关节6增加
                    }

                    # 第一臂x,y坐标控制
                    arm1_xy_controls = {
                        "8": ("x", -0.004),  # x减少
                        "u": ("x", 0.004),  # x增加
                        "9": ("y", -0.004),  # y减少
                        "i": ("y", 0.004),  # y增加
                    }

                    # 第二臂控制映射: hbjnkml,;.'/
                    arm2_joint_controls = {
                        "h": ("right_shoulder_pan", -1),  # 关节1减少
                        "b": ("right_shoulder_pan", 1),  # 关节1增加
                        ";": ("right_wrist_roll", -1),  # 关节5减少
                        "l": ("right_wrist_roll", 1),  # 关节5增加
                        "'": ("right_gripper", -1),  # 关节6减少
                        "/": ("right_gripper", 1),  # 关节6增加
                    }

                    # 第二臂x,y坐标控制
                    arm2_xy_controls = {
                        "j": ("x", -0.004),  # x减少
                        "n": ("x", 0.004),  # x增加
                        "k": ("y", -0.004),  # y减少
                        "m": ("y", 0.004),  # y增加
                    }

                    # 头部控制: rtfg
                    head_controls = {
                        "r": ("head_pan", -1),  # 头部pan减少
                        "t": ("head_pan", 1),  # 头部pan增加
                        "f": ("head_tilt", -1),  # 头部tilt减少
                        "g": ("head_tilt", 1),  # 头部tilt增加
                    }

                    # 底盘控制: wasdqe
                    base_controls = {
                        "w": ("x.vel", base_speed_step),  # 前进
                        "s": ("x.vel", -base_speed_step),  # 后退
                        "a": ("y.vel", -base_speed_step),  # 左移
                        "d": ("y.vel", base_speed_step),  # 右移
                        "q": ("theta.vel", -base_speed_step),  # 左转
                        "e": ("theta.vel", base_speed_step),  # 右转
                    }

                    # 第一臂pitch控制
                    if key == "=":
                        pitch["left"] += pitch_step
                        print(f"第一臂增加pitch调整: {pitch['left']:.3f}")
                    elif key == "[":
                        pitch["left"] -= pitch_step
                        print(f"第一臂减少pitch调整: {pitch['left']:.3f}")

                    # 第二臂pitch控制
                    elif key == ",":
                        pitch["right"] += pitch_step
                        print(f"第二臂增加pitch调整: {pitch['right']:.3f}")
                    elif key == ".":
                        pitch["right"] -= pitch_step
                        print(f"第二臂减少pitch调整: {pitch['right']:.3f}")

                    # 第一臂关节控制
                    if key in arm1_joint_controls:
                        joint_name, delta = arm1_joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"第一臂更新目标位置 {joint_name}: {current_target} -> {new_target}")

                    # 第二臂关节控制
                    elif key in arm2_joint_controls:
                        joint_name, delta = arm2_joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"第二臂更新目标位置 {joint_name}: {current_target} -> {new_target}")

                    # 头部控制
                    elif key in head_controls:
                        joint_name, delta = head_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"头部更新目标位置 {joint_name}: {current_target} -> {new_target}")

                    # 底盘控制
                    elif key in base_controls:
                        vel_name, delta = base_controls[key]
                        current_vel = base_velocities[vel_name]
                        new_vel = current_vel + delta
                        # 限制底盘速度范围
                        new_vel = max(-1.0, min(1.0, new_vel))
                        base_velocities[vel_name] = new_vel
                        print(f"底盘更新速度 {vel_name}: {current_vel:.3f} -> {new_vel:.3f}")

                    # 第一臂xy控制
                    elif key in arm1_xy_controls:
                        coord, delta = arm1_xy_controls[key]
                        if coord == "x":
                            current_positions["left"]["x"] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(
                                current_positions["left"]["x"], current_positions["left"]["y"]
                            )
                            target_positions["left_shoulder_lift"] = joint2_target
                            target_positions["left_elbow_flex"] = joint3_target
                            print(
                                f"第一臂更新x坐标: {current_positions['left']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )
                        elif coord == "y":
                            current_positions["left"]["y"] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(
                                current_positions["left"]["x"], current_positions["left"]["y"]
                            )
                            target_positions["left_shoulder_lift"] = joint2_target
                            target_positions["left_elbow_flex"] = joint3_target
                            print(
                                f"第一臂更新y坐标: {current_positions['left']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )

                    # 第二臂xy控制
                    elif key in arm2_xy_controls:
                        coord, delta = arm2_xy_controls[key]
                        if coord == "x":
                            current_positions["right"]["x"] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(
                                current_positions["right"]["x"], current_positions["right"]["y"]
                            )
                            target_positions["right_shoulder_lift"] = joint2_target
                            target_positions["right_elbow_flex"] = joint3_target
                            print(
                                f"第二臂更新x坐标: {current_positions['right']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )
                        elif coord == "y":
                            current_positions["right"]["y"] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(
                                current_positions["right"]["x"], current_positions["right"]["y"]
                            )
                            target_positions["right_shoulder_lift"] = joint2_target
                            target_positions["right_elbow_flex"] = joint3_target
                            print(
                                f"第二臂更新y坐标: {current_positions['right']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )

            # 对每个机械臂应用pitch调整到wrist_flex
            if "left_shoulder_lift" in target_positions and "left_elbow_flex" in target_positions:
                target_positions["left_wrist_flex"] = (
                    -target_positions["left_shoulder_lift"]
                    - target_positions["left_elbow_flex"]
                    + pitch["left"]
                )

            if "right_shoulder_lift" in target_positions and "right_elbow_flex" in target_positions:
                target_positions["right_wrist_flex"] = (
                    -target_positions["right_shoulder_lift"]
                    - target_positions["right_elbow_flex"]
                    + pitch["right"]
                )

            # 显示当前pitch值（每100步显示一次，避免刷屏）
            if hasattr(p_control_loop, "step_counter"):
                p_control_loop.step_counter += 1
            else:
                p_control_loop.step_counter = 0

            if p_control_loop.step_counter % 100 == 0:
                print(f"当前pitch调整: left={pitch['left']:.3f}, right={pitch['right']:.3f}")
                print(
                    f"当前底盘速度: x={base_velocities['x.vel']:.3f}, y={base_velocities['y.vel']:.3f}, theta={base_velocities['theta.vel']:.3f}"
                )

            # 获取当前机器人状态
            current_obs = robot.get_observation()

            # 提取当前关节位置
            current_joint_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    # 应用校准系数（仅对手臂关节）
                    if motor_name.startswith("left_") or motor_name.startswith("right_"):
                        base_motor_name = motor_name.split("_", 1)[1]  # 移除left_或right_前缀
                        calibrated_value = apply_joint_calibration(base_motor_name, value)
                        current_joint_positions[motor_name] = calibrated_value
                    else:
                        current_joint_positions[motor_name] = value

            # P控制计算
            robot_action = {}

            # 处理关节位置控制
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_joint_positions:
                    current_pos = current_joint_positions[joint_name]
                    error = target_pos - current_pos

                    # P控制: 输出 = Kp * 误差
                    control_output = kp * error

                    # 将控制输出转换为位置命令
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position

            # 处理底盘速度控制
            for vel_name, target_vel in base_velocities.items():
                if target_vel != 0.0:  # 只有当速度不为0时才发送命令
                    robot_action[vel_name] = target_vel

            # 发送动作到机器人
            if robot_action:
                robot.send_action(robot_action)

            time.sleep(control_period)

        except KeyboardInterrupt:
            print("用户中断程序")
            break
        except Exception as e:
            print(f"P控制循环出错: {e}")
            traceback.print_exc()
            break


def main():
    """主函数"""
    print("XLeRobot 键盘控制示例 (P控制)")
    print("=" * 50)

    try:
        # 导入必要的模块
        from lerobot.robots.xlerobot import XLeRobot, XLeRobotConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        # 自动检测端口
        left_arm_port, right_arm_port, head_port, base_port = auto_detect_ports()
        if left_arm_port is None or right_arm_port is None:
            print("自动检测失败，尝试手动配置...")
            left_arm_port, right_arm_port, head_port, base_port = manual_port_config()
            if left_arm_port is None or right_arm_port is None:
                print("无法配置端口，程序退出。")
                return

        print("端口配置:")
        print(f"- 左手和头部端口: {left_arm_port}")
        print(f"- 右手和底盘端口: {right_arm_port}")

        # 配置XLeRobot
        robot_config = XLeRobotConfig(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            head_port=head_port,
            base_port=base_port,
        )

        robot = XLeRobot(robot_config)

        # 配置键盘
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)

        # 连接设备
        print("正在连接XLeRobot...")
        robot.connect()
        print("正在连接键盘...")
        keyboard.connect()

        print("所有设备连接成功！")

        # 询问是否重新校准
        while True:
            print("\n校准说明:")
            print("- 左臂校准: 校准左手设备 (头部电机使用默认设置)")
            print("- 右臂校准: 校准右手设备")
            print("- 底盘校准: 参考LeKiwi，使用简单校准设置")
            print("- 注意: 头部电机不需要校准，使用默认设置")
            calibrate_choice = input("是否重新校准机器人? (y/n): ").strip().lower()
            if calibrate_choice in ["y", "yes", "是"]:
                print("开始重新校准XLeRobot...")
                try:
                    robot.calibrate()
                    print("XLeRobot校准完成！")
                except Exception as e:
                    print(f"校准过程中出现错误: {e}")
                    print("请检查设备连接，或尝试重新连接设备")
                    continue_choice = input("是否继续程序? (y/n): ").strip().lower()
                    if continue_choice not in ["y", "yes", "是"]:
                        print("程序退出")
                        return
                break
            elif calibrate_choice in ["n", "no", "否"]:
                print("使用之前的校准文件")
                break
            else:
                print("请输入 y 或 n")

        # 读取起始关节角度
        print("读取XLeRobot起始关节角度...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                start_positions[motor_name] = int(value)  # 不应用校准系数

        print("XLeRobot起始关节角度:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")

        # 移动到零位置
        move_to_zero_position(robot, duration=3.0)

        # 初始化目标位置为当前位置（整数）
        target_positions = {
            # 左臂
            "left_shoulder_pan": 0.0,
            "left_shoulder_lift": 0.0,
            "left_elbow_flex": 0.0,
            "left_wrist_flex": 0.0,
            "left_wrist_roll": 0.0,
            "left_gripper": 0.0,
            # 右臂
            "right_shoulder_pan": 0.0,
            "right_shoulder_lift": 0.0,
            "right_elbow_flex": 0.0,
            "right_wrist_flex": 0.0,
            "right_wrist_roll": 0.0,
            "right_gripper": 0.0,
            # 头部
            "head_pan": 0.0,
            "head_tilt": 0.0,
        }

        # 初始化双臂x,y坐标控制
        x0, y0 = 0.1629, 0.1131
        current_positions = {"left": {"x": x0, "y": y0}, "right": {"x": x0, "y": y0}}
        print(f"初始化双臂末端执行器位置: left=({x0:.4f}, {y0:.4f}), right=({x0:.4f}, {y0:.4f})")

        print("XLeRobot键盘控制说明:")
        print("第一臂控制 (7y8u9i0o-p=[):")
        print("- 7/y: 关节1 (shoulder_pan) 减少/增加")
        print("- 8/u: 控制末端执行器x坐标 (joint2+3)")
        print("- 9/i: 控制末端执行器y坐标 (joint2+3)")
        print("- =/[: pitch调整 增加/减少 (影响wrist_flex)")
        print("- 0/o: 关节5 (wrist_roll) 减少/增加")
        print("- -/p: 关节6 (gripper) 减少/增加")
        print("")
        print("第二臂控制 (hbjnkml,;.'/):")
        print("- h/b: 关节1 (shoulder_pan) 减少/增加")
        print("- j/n: 控制末端执行器x坐标 (joint2+3)")
        print("- k/m: 控制末端执行器y坐标 (joint2+3)")
        print("- ,/.: pitch调整 增加/减少 (影响wrist_flex)")
        print("- ;/l: 关节5 (wrist_roll) 减少/增加")
        print("- '/: 关节6 (gripper) 减少/增加")
        print("")
        print("头部控制 (rtfg):")
        print("- r/t: 头部pan 减少/增加")
        print("- f/g: 头部tilt 减少/增加")
        print("")
        print("底盘控制 (wasdqe):")
        print("- w/s: 前进/后退")
        print("- a/d: 左移/右移")
        print("- q/e: 左转/右转")
        print("")
        print("- X: 退出程序（先回到起始位置）")
        print("- ESC: 退出程序")
        print("=" * 50)
        print("注意: XLeRobot会持续移动到目标位置")

        # 开始P控制循环
        p_control_loop(
            robot, keyboard, target_positions, start_positions, current_positions, kp=0.5, control_freq=50
        )

        # 断开连接
        print("正在断开XLeRobot...")
        robot.disconnect()
        keyboard.disconnect()
        print("程序结束")

    except Exception as e:
        print(f"程序执行失败: {e}")
        traceback.print_exc()
        print("请检查:")
        print("1. 机器人是否正确连接")
        print("2. USB端口是否正确")
        print("3. 是否有足够的权限访问USB设备")
        print("4. 机器人是否已正确配置")


if __name__ == "__main__":
    main()
