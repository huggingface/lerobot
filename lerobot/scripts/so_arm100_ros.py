# ##############################################################################################
# #### 读取每个舵机角度 Demo1

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

# follower_pos = robot.follower_arms["main"].read("Present_Position")
# print("Follower Position: ", np.array2string(follower_pos, precision=3, suppress_small=True))
# robot.disconnect()

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
# #robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)  #关闭
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
#     time.sleep(1)
#     print(f"第 {i} 次记录的位置：{np.array2string(pos, precision=3, suppress_small=True)}")
#     print(f"机械臂将自动移动到目标位置：{np.array2string(pos, precision=3, suppress_small=True)}")
#     # 让机械臂移动到目标位置
#     robot.follower_arms["main"].write("Goal_Position", pos)

# # 断开连接
# robot.disconnect()


###########################################################################################
#### 真实世界机器人控制rviz机械臂
### 读取每个舵机角度 Demo1
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from lerobot.common.robot_devices.motors.feetech import *
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


class JointStateSubscriber:
    def __init__(self):
        # 初始化机械臂连接
        self.robot = ManipulatorRobot(So100RobotConfig())
        self.robot.connect()

        # 定义机械臂初始位置（根据URDF文件中的位置）
        self.urdf_initial_position = np.array([0.0, 90.0, -40.0, -2, -90, 0.0])  # 根据URDF文件设置
        self.initial_position = self.robot.follower_arms["main"].read("Present_Position")  # 初始化零点

        # 启用舵机控制
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)

        # 订阅 joint_states 话题
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)

    def joint_state_callback(self, msg):
        # 获取从话题中收到的关节角度（弧度）
        joint_angles = np.array(msg.position)

        # 计算目标位置的差值（需要考虑URDF初始角度的偏移）
        joint_angles_deg = (np.rad2deg(joint_angles) - self.urdf_initial_position) * [
            -1,
            1,
            -1,
            1,
            1,
            1,
        ] + self.initial_position
        print(joint_angles_deg)
        # 向机械臂发送目标位置
        self.robot.follower_arms["main"].write("Goal_Position", joint_angles_deg)

    def stop_robot(self):
        self.robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.disconnect()


def main():
    rospy.init_node("joint_state_subscriber_node")  # 初始化 ROS 节点
    joint_state_subscriber = JointStateSubscriber()

    # 设定 ROS 节点的频率（这里的频率是接收话题的频率）
    rate = rospy.Rate(10)  # 10 Hz
    try:
        while not rospy.is_shutdown():
            # 在这里，你的代码只需要继续运行并处理订阅的消息
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        joint_state_subscriber.stop_robot()
        rospy.loginfo("Shutting down node.")
        rospy.shutdown()


if __name__ == "__main__":
    main()
