import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="192.168.101.33", id="my_lekiwi")
teleop_arm_config = SO100LeaderConfig(port="/dev/ttyACM0", id="my_awesome_leader_arm")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
leader_arm = SO100Leader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

print("=== 连接状态检查 ===")
print(f"机器人配置: {robot_config}")
print(f"键盘配置: {keyboard_config}")

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
print("正在连接机器人...")
robot.connect()
print("正在连接机械臂...")
leader_arm.connect()
print("正在连接键盘...")
keyboard.connect()

print("=== 连接后状态检查 ===")
print(f"机器人已连接: {robot.is_connected}")
print(f"机械臂已连接: {leader_arm.is_connected}")
print(f"键盘已连接: {keyboard.is_connected}")

_init_rerun(session_name="lekiwi_teleop")

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

print("=== 开始遥操作循环 ===")
print("键盘控制说明:")
print("W - 前进, S - 后退, A - 左转, D - 右转")
print("Z - 左旋转, X - 右旋转")
print("R - 加速, F - 减速")
print("按 Ctrl+C 退出")

loop_count = 0
while True:
    t0 = time.perf_counter()
    loop_count += 1

    observation = robot.get_observation()

    arm_action = leader_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

    keyboard_keys = keyboard.get_action()
    
    # 添加键盘调试信息
    if loop_count % 30 == 0:  # 每30帧打印一次
        print(f"键盘按键: {keyboard_keys}")
    
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    
    # 添加动作调试信息
    if base_action and any(v != 0 for v in base_action.values()):
        print(f"检测到键盘动作: {base_action}")

    log_rerun_data(observation, {**arm_action, **base_action})

    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0)) 