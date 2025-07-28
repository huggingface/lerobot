#!/usr/bin/env python3
"""
双臂键盘控制SO100/SO101机器人
修复了动作格式转换问题
使用P控制，键盘只改变目标关节角度
支持两个机械臂同时控制: /dev/ttyACM0 和 /dev/ttyACM1
键盘映射: 第一臂(7y8u9i0o-p=[), 第二臂(hbjnkml,;.'/)
"""

import time
import logging
import traceback
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 关节校准系数 - 手动编辑
# 格式: [关节名称, 零位置偏移(度), 缩放系数]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # 关节1: 零位置偏移, 缩放系数
    ['shoulder_lift', 2.0, 0.97],     # 关节2: 零位置偏移, 缩放系数
    ['elbow_flex', 0.0, 1.05],        # 关节3: 零位置偏移, 缩放系数
    ['wrist_flex', 0.0, 0.94],        # 关节4: 零位置偏移, 缩放系数
    ['wrist_roll', 0.0, 0.5],        # 关节5: 零位置偏移, 缩放系数
    ['gripper', 0.0, 1.0],           # 关节6: 零位置偏移, 缩放系数
]

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
            scale = joint_cal[2]   # 缩放系数
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

    joint2_deg = 90-joint2_deg
    joint3_deg = joint3_deg-90
    
    return joint2_deg, joint3_deg

def move_to_zero_position(robots, duration=3.0, kp=0.5):
    """
    使用P控制缓慢移动所有机器人到零位置
    
    Args:
        robots: 机器人实例字典 {'arm1': robot1, 'arm2': robot2}
        duration: 移动到零位置所需时间(秒)
        kp: 比例增益
    """
    print("正在使用P控制缓慢移动所有机器人到零位置...")
    
    # 获取当前所有机器人状态
    current_obs = {}
    for arm_name, robot in robots.items():
        current_obs[arm_name] = robot.get_observation()
    
    # 提取当前关节位置
    current_positions = {}
    for arm_name, obs in current_obs.items():
        current_positions[arm_name] = {}
        for key, value in obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[arm_name][motor_name] = value
    
    # 零位置目标
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    # 计算控制步数
    control_freq = 50  # 50Hz控制频率
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"将在 {duration} 秒内使用P控制移动到零位置，控制频率: {control_freq}Hz，比例增益: {kp}")
    
    for step in range(total_steps):
        # 获取当前所有机器人状态
        current_obs = {}
        current_positions = {}
        for arm_name, robot in robots.items():
            current_obs[arm_name] = robot.get_observation()
            current_positions[arm_name] = {}
            for key, value in current_obs[arm_name].items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # 应用校准系数
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[arm_name][motor_name] = calibrated_value
        
        # 对每个机械臂进行P控制计算
        for arm_name, robot in robots.items():
            robot_action = {}
            for joint_name, target_pos in zero_positions.items():
                if joint_name in current_positions[arm_name]:
                    current_pos = current_positions[arm_name][joint_name]
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
    
    print("所有机器人已移动到零位置")

def return_to_start_position(robots, start_positions, kp=0.5, control_freq=50):
    """
    使用P控制返回到起始位置
    
    Args:
        robots: 机器人实例字典 {'arm1': robot1, 'arm2': robot2}
        start_positions: 起始关节位置字典 {'arm1': {}, 'arm2': {}}
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    print("正在返回到起始位置...")
    
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # 最多5秒
    
    for step in range(max_steps):
        # 获取当前所有机器人状态
        current_obs = {}
        current_positions = {}
        for arm_name, robot in robots.items():
            current_obs[arm_name] = robot.get_observation()
            current_positions[arm_name] = {}
            for key, value in current_obs[arm_name].items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    current_positions[arm_name][motor_name] = value  # 不应用校准系数
        
        # 对每个机械臂进行P控制计算
        total_error = 0
        for arm_name, robot in robots.items():
            robot_action = {}
            for joint_name, target_pos in start_positions[arm_name].items():
                if joint_name in current_positions[arm_name]:
                    current_pos = current_positions[arm_name][joint_name]
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
        if total_error < 4.0:  # 如果总误差小于4度（双臂），认为已到达
            print("已返回到起始位置")
            break
        
        time.sleep(control_period)
    
    print("返回起始位置完成")

def p_control_loop(robots, keyboard, target_positions, start_positions, current_positions, kp=0.5, control_freq=50):
    """
    P控制循环
    
    Args:
        robots: 机器人实例字典 {'arm1': robot1, 'arm2': robot2}
        keyboard: 键盘实例
        target_positions: 目标关节位置字典
        start_positions: 起始关节位置字典
        current_positions: 当前关节位置字典
        kp: 比例增益
        control_freq: 控制频率(Hz)
    """
    control_period = 1.0 / control_freq
    
    # 初始化双臂pitch控制变量
    pitch = {'arm1': 0.0, 'arm2': 0.0}  # 初始pitch调整
    pitch_step = 1  # pitch调整步长
    
    print(f"开始P控制循环，控制频率: {control_freq}Hz，比例增益: {kp}")
    
    while True:
        try:
            # 获取键盘输入
            keyboard_action = keyboard.get_action()
            
            if keyboard_action:
                # 处理键盘输入，更新目标位置
                for key, value in keyboard_action.items():
                    if key == 'x':
                        # 退出程序，先回到起始位置
                        print("检测到退出命令，正在回到起始位置...")
                        return_to_start_position(robots, start_positions, 0.2, control_freq)
                        return
                    
                    # 第一臂控制映射: 7y8u9i0o-p=[
                    arm1_joint_controls = {
                        '7': ('shoulder_pan', -1),    # 关节1减少
                        'y': ('shoulder_pan', 1),     # 关节1增加
                        '0': ('wrist_roll', -1),      # 关节5减少
                        'o': ('wrist_roll', 1),       # 关节5增加
                        '-': ('gripper', -1),         # 关节6减少
                        'p': ('gripper', 1),          # 关节6增加
                    }
                    
                    # 第一臂x,y坐标控制
                    arm1_xy_controls = {
                        '8': ('x', -0.004),  # x减少
                        'u': ('x', 0.004),   # x增加
                        '9': ('y', -0.004),  # y减少
                        'i': ('y', 0.004),   # y增加
                    }
                    
                    # 第二臂控制映射: hbjnkml,;.'/
                    arm2_joint_controls = {
                        'h': ('shoulder_pan', -1),    # 关节1减少
                        'b': ('shoulder_pan', 1),     # 关节1增加
                        ';': ('wrist_roll', -1),      # 关节5减少
                        'l': ('wrist_roll', 1),       # 关节5增加
                        "'": ('gripper', -1),         # 关节6减少
                        '/': ('gripper', 1),          # 关节6增加
                    }
                    
                    # 第二臂x,y坐标控制
                    arm2_xy_controls = {
                        'j': ('x', -0.004),  # x减少
                        'n': ('x', 0.004),   # x增加
                        'k': ('y', -0.004),  # y减少
                        'm': ('y', 0.004),   # y增加
                    }
                    
                    # 第一臂pitch控制
                    if key == '=':
                        pitch['arm1'] += pitch_step
                        print(f"第一臂增加pitch调整: {pitch['arm1']:.3f}")
                    elif key == '[':
                        pitch['arm1'] -= pitch_step
                        print(f"第一臂减少pitch调整: {pitch['arm1']:.3f}")
                    
                    # 第二臂pitch控制
                    elif key == ',':
                        pitch['arm2'] += pitch_step
                        print(f"第二臂增加pitch调整: {pitch['arm2']:.3f}")
                    elif key == '.':
                        pitch['arm2'] -= pitch_step
                        print(f"第二臂减少pitch调整: {pitch['arm2']:.3f}")
                    
                    # 第一臂关节控制
                    if key in arm1_joint_controls:
                        joint_name, delta = arm1_joint_controls[key]
                        if joint_name in target_positions['arm1']:
                            current_target = target_positions['arm1'][joint_name]
                            new_target = int(current_target + delta)
                            target_positions['arm1'][joint_name] = new_target
                            print(f"第一臂更新目标位置 {joint_name}: {current_target} -> {new_target}")
                    
                    # 第二臂关节控制
                    elif key in arm2_joint_controls:
                        joint_name, delta = arm2_joint_controls[key]
                        if joint_name in target_positions['arm2']:
                            current_target = target_positions['arm2'][joint_name]
                            new_target = int(current_target + delta)
                            target_positions['arm2'][joint_name] = new_target
                            print(f"第二臂更新目标位置 {joint_name}: {current_target} -> {new_target}")
                    
                    # 第一臂xy控制
                    elif key in arm1_xy_controls:
                        coord, delta = arm1_xy_controls[key]
                        if coord == 'x':
                            current_positions['arm1']['x'] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm1']['x'], current_positions['arm1']['y'])
                            target_positions['arm1']['shoulder_lift'] = joint2_target
                            target_positions['arm1']['elbow_flex'] = joint3_target
                            print(f"第一臂更新x坐标: {current_positions['arm1']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_positions['arm1']['y'] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm1']['x'], current_positions['arm1']['y'])
                            target_positions['arm1']['shoulder_lift'] = joint2_target
                            target_positions['arm1']['elbow_flex'] = joint3_target
                            print(f"第一臂更新y坐标: {current_positions['arm1']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                    
                    # 第二臂xy控制
                    elif key in arm2_xy_controls:
                        coord, delta = arm2_xy_controls[key]
                        if coord == 'x':
                            current_positions['arm2']['x'] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm2']['x'], current_positions['arm2']['y'])
                            target_positions['arm2']['shoulder_lift'] = joint2_target
                            target_positions['arm2']['elbow_flex'] = joint3_target
                            print(f"第二臂更新x坐标: {current_positions['arm2']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_positions['arm2']['y'] += delta
                            # 计算joint2和joint3的目标角度
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm2']['x'], current_positions['arm2']['y'])
                            target_positions['arm2']['shoulder_lift'] = joint2_target
                            target_positions['arm2']['elbow_flex'] = joint3_target
                            print(f"第二臂更新y坐标: {current_positions['arm2']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
            
            # 对每个机械臂应用pitch调整到wrist_flex
            for arm_name in ['arm1', 'arm2']:
                if 'shoulder_lift' in target_positions[arm_name] and 'elbow_flex' in target_positions[arm_name]:
                    target_positions[arm_name]['wrist_flex'] = - target_positions[arm_name]['shoulder_lift'] - target_positions[arm_name]['elbow_flex'] + pitch[arm_name]
            
            # 显示当前pitch值（每100步显示一次，避免刷屏）
            if hasattr(p_control_loop, 'step_counter'):
                p_control_loop.step_counter += 1
            else:
                p_control_loop.step_counter = 0
            
            if p_control_loop.step_counter % 100 == 0:
                print(f"当前pitch调整: arm1={pitch['arm1']:.3f}, arm2={pitch['arm2']:.3f}")
            
            # 获取当前所有机器人状态
            current_obs = {}
            current_joint_positions = {}
            for arm_name, robot in robots.items():
                current_obs[arm_name] = robot.get_observation()
                current_joint_positions[arm_name] = {}
                for key, value in current_obs[arm_name].items():
                    if key.endswith('.pos'):
                        motor_name = key.removesuffix('.pos')
                        # 应用校准系数
                        calibrated_value = apply_joint_calibration(motor_name, value)
                        current_joint_positions[arm_name][motor_name] = calibrated_value
            
            # 对每个机械臂进行P控制计算
            for arm_name, robot in robots.items():
                robot_action = {}
                for joint_name, target_pos in target_positions[arm_name].items():
                    if joint_name in current_joint_positions[arm_name]:
                        current_pos = current_joint_positions[arm_name][joint_name]
                        error = target_pos - current_pos
                        
                        # P控制: 输出 = Kp * 误差
                        control_output = kp * error
                        
                        # 将控制输出转换为位置命令
                        new_position = current_pos + control_output
                        robot_action[f"{joint_name}.pos"] = new_position
                
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
    print("LeRobot 双臂键盘控制示例 (P控制)")
    print("="*50)
    
    try:
        # 导入必要的模块
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
        
        # 配置双臂机器人
        arm1_port = "/dev/ttyACM0"
        arm2_port = "/dev/ttyACM1"
        
        print(f"配置第一臂: {arm1_port}")  
        print(f"配置第二臂: {arm2_port}")
        
        # 创建双臂机器人实例
        arm1_config = SO100FollowerConfig(port=arm1_port)
        arm2_config = SO100FollowerConfig(port=arm2_port)
        
        arm1_robot = SO100Follower(arm1_config)
        arm2_robot = SO100Follower(arm2_config)
        
        robots = {
            'arm1': arm1_robot,
            'arm2': arm2_robot
        }
        
        # 配置键盘
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)
        
        # 连接设备
        print("正在连接第一臂...")
        arm1_robot.connect()
        print("正在连接第二臂...")
        arm2_robot.connect()
        print("正在连接键盘...")
        keyboard.connect()
        
        print("所有设备连接成功！")
        
        # 询问是否重新校准
        while True:
            calibrate_choice = input("是否重新校准机器人? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes', '是']:
                print("开始重新校准双臂...")
                for arm_name, robot in robots.items():
                    print(f"正在校准{arm_name}...")
                    robot.calibrate()
                print("双臂校准完成！")
                break
            elif calibrate_choice in ['n', 'no', '否']:
                print("使用之前的校准文件")
                break
            else:
                print("请输入 y 或 n")
        
        # 读取起始关节角度
        print("读取双臂起始关节角度...")
        start_positions = {}
        for arm_name, robot in robots.items():
            start_obs = robot.get_observation()
            start_positions[arm_name] = {}
            for key, value in start_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    start_positions[arm_name][motor_name] = int(value)  # 不应用校准系数
        
        print("双臂起始关节角度:")
        for arm_name, positions in start_positions.items():
            print(f"{arm_name}:")
            for joint_name, position in positions.items():
                print(f"  {joint_name}: {position}°")
        
        # 移动到零位置
        move_to_zero_position(robots, duration=3.0)
        
        # 初始化双臂目标位置为当前位置（整数）
        target_positions = {
            'arm1': {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            },
            'arm2': {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            }
        }
        
        # 初始化双臂x,y坐标控制
        x0, y0 = 0.1629, 0.1131
        current_positions = {
            'arm1': {'x': x0, 'y': y0},
            'arm2': {'x': x0, 'y': y0}
        }
        print(f"初始化双臂末端执行器位置: arm1=({x0:.4f}, {y0:.4f}), arm2=({x0:.4f}, {y0:.4f})")
        
        
        print("双臂键盘控制说明:")
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
        print("- X: 退出程序（先回到起始位置）")
        print("- ESC: 退出程序")
        print("="*50)
        print("注意: 双臂机器人会持续移动到目标位置")
        
        # 开始P控制循环
        p_control_loop(robots, keyboard, target_positions, start_positions, current_positions, kp=0.5, control_freq=50)
        
        # 断开连接
        for arm_name, robot in robots.items():
            print(f"正在断开{arm_name}...")
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