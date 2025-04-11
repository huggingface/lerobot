import time
from hopejr import HopeJuniorRobot


def main():
    # Instantiate and connect to the robot
    robot = HopeJuniorRobot()
    robot.connect()

    # Example read of the current position
    print("Present Position:", robot.arm_bus.read("Present_Position"))

    # Enable torque and set acceleration
    robot.arm_bus.write("Torque_Enable", 1)
    robot.arm_bus.write("Acceleration", 20)
    print("Acceleration Read:", robot.arm_bus.read("Acceleration"))

    # Move elbow_flex and wrist_yaw a few times
    robot.arm_bus.write("Goal_Position", [1000, 1000], ["elbow_flex", "wrist_yaw"])
    time.sleep(2)
    robot.arm_bus.write("Goal_Position", [1500, 1500], ["elbow_flex", "wrist_yaw"])
    time.sleep(2)
    robot.arm_bus.write("Goal_Position", [1000, 1000], ["elbow_flex", "wrist_yaw"])


if __name__ == "__main__":
    main()
