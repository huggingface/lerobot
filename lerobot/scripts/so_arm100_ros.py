# ##############################################################################################
# #### 读取每个舵机角度 Demo1

from lerobot.common.robot_devices.motors.feetech import *
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
import numpy as np

so_arm100_configs=So100RobotConfig()

config = FeetechMotorsBusConfig(
    port=so_arm100_configs.follower_arms["main"].port,
    motors=so_arm100_configs.follower_arms["main"].motors
)

robot = ManipulatorRobot(so_arm100_configs)

robot.connect()

follower_pos = robot.follower_arms["main"].read("Present_Position")
print("Follower Position: ", np.array2string(follower_pos, precision=3, suppress_small=True))
robot.disconnect()

# ############################################################################################
# ##### 开启力矩自锁 Demo2

# from lerobot.common.robot_devices.motors.feetech import *
# from lerobot.common.robot_devices.robots.configs import So100RobotConfig
# from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
# import numpy as np

# so_arm100_configs=So100RobotConfig()

# config = FeetechMotorsBusConfig(
#     port=so_arm100_configs.follower_arms["main"].port,
#     motors=so_arm100_configs.follower_arms["main"].motors
# )

# robot = ManipulatorRobot(so_arm100_configs)

# robot.connect()
# robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)  #关闭
# robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)  # 开启
# robot.disconnect()

# ########################################################################################
# ######让机械臂每个关节达到指定角度 Demo3


# from lerobot.common.robot_devices.motors.feetech import *
# from lerobot.common.robot_devices.robots.configs import So100RobotConfig
# from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
# import numpy as np

# so_arm100_configs=So100RobotConfig()

# config = FeetechMotorsBusConfig(
#     port=so_arm100_configs.follower_arms["main"].port,
#     motors=so_arm100_configs.follower_arms["main"].motors
# )

# robot = ManipulatorRobot(so_arm100_configs)

# robot.connect()
# follower_pos -=[5.6,147.30469,157.06055,21.269531,185.90625,6.626506]
# robot.follower_arms["main"].write("Goal_Position", follower_pos)

# robot.disconnect()

############################################################################################
######手动控制机械臂N次记录当前位置，并依次执行 Demo4

# from lerobot.common.robot_devices.motors.feetech import *
# from lerobot.common.robot_devices.robots.configs import So100RobotConfig
# from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
# import numpy as np
# import time

# # 初始化机器人配置
# so_arm100_configs = So100RobotConfig()

# config = FeetechMotorsBusConfig(
#     port=so_arm100_configs.follower_arms["main"].port,
#     motors=so_arm100_configs.follower_arms["main"].motors
# )

# # 创建机器人对象
# robot = ManipulatorRobot(so_arm100_configs)

# # 连接到机器人
# robot.connect()

# # 记录角度的列表
# positions = []

# # 示教次数
# pose_index= 3

# # 取消自锁
# robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value) 

# # 手动操作三次
# for i in range(pose_index):
#     input(f"第 {i+1} 次手动操作，请按回车记录当前位置...")
    
#     # 读取当前位置
#     follower_pos = robot.follower_arms["main"].read("Present_Position")
    
#     # 打印当前位置
#     print(f"当前机械臂角度：{np.array2string(follower_pos, precision=3, suppress_small=True)}")
    
#     # 将当前位置添加到列表中
#     positions.append(follower_pos)
    
# # 开启自锁
# robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value) 

# # 显示记录的所有位置，并执行
# print("\n所有记录的角度：")
# for i, pos in enumerate(positions, 1):
#     time.sleep(3)
#     print(f"第 {i} 次记录的位置：{np.array2string(pos, precision=3, suppress_small=True)}")
#     print(f"\n机械臂将自动移动到目标位置：{np.array2string(pos, precision=3, suppress_small=True)}")
#     # 让机械臂移动到目标位置
#     robot.follower_arms["main"].write("Goal_Position", pos)
    
# # 断开连接
# robot.disconnect()
