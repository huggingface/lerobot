from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig
import time

REACHY2_MOTORS = {
    "neck_yaw.pos": "head.neck.yaw",
    "neck_pitch.pos": "head.neck.pitch",
    "neck_roll.pos": "head.neck.roll",
    "r_shoulder_pitch.pos": "r_arm.shoulder.pitch",
    "r_shoulder_roll.pos": "r_arm.shoulder.roll",
    "r_elbow_yaw.pos": "r_arm.elbow.yaw",
    "r_elbow_pitch.pos": "r_arm.elbow.pitch",
    "r_wrist_roll.pos": "r_arm.wrist.roll",
    "r_wrist_pitch.pos": "r_arm.wrist.pitch",
    "r_wrist_yaw.pos": "r_arm.wrist.yaw",
    "r_gripper.pos": "r_arm.gripper",
    "l_shoulder_pitch.pos": "l_arm.shoulder.pitch",
    "l_shoulder_roll.pos": "l_arm.shoulder.roll",
    "l_elbow_yaw.pos": "l_arm.elbow.yaw",
    "l_elbow_pitch.pos": "l_arm.elbow.pitch",
    "l_wrist_roll.pos": "l_arm.wrist.roll",
    "l_wrist_pitch.pos": "l_arm.wrist.pitch",
    "l_wrist_yaw.pos": "l_arm.wrist.yaw",
    "l_gripper.pos": "l_arm.gripper",
    "l_antenna.pos": "head.l_antenna",
    "r_antenna.pos": "head.r_antenna",
    # "mobile_base.vx": "mobile_base.vx",
    # "mobile_base.vy": "mobile_base.vy",
    # "mobile_base.vtheta": "mobile_base.vtheta",
}


robot_config = Reachy2RobotConfig(ip_address="localhost")
robot = Reachy2Robot(robot_config)

robot.connect()

print(f"is_connected(): {robot.is_connected}\n")

print(f"_get_state(): {robot._get_state()}\n")

obs = robot.get_observation()
print(f"get_observation(): {obs}\n")
print(f"observation_features: {robot.observation_features}\n")
print(f"action_features: {robot.action_features}\n")


def get_action(robot):
    my_keys = REACHY2_MOTORS.keys()
    my_values = [robot.reachy.joints[motor].present_position for motor in REACHY2_MOTORS.values()]
    action = dict(zip(my_keys, my_values))
    return action


action = get_action(robot)
time.sleep(5)
robot.send_action(action)
