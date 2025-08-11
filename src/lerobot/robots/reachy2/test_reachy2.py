import time

from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig

# {lerobot_keys: reachy2_sdk_keys}
REACHY2_JOINTS = {
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
}

REACHY2_VEL = {
    "mobile_base.vx": "vx",
    "mobile_base.vy": "vy",
    "mobile_base.vtheta": "vtheta",
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
    pos_dict = {k: robot.reachy.joints[v].present_position for k, v in REACHY2_JOINTS.items()}
    vel_dict = {k: robot.reachy.mobile_base.odometry[v] for k, v in REACHY2_VEL.items()}
    return {**pos_dict, **vel_dict}


action = get_action(robot)
time.sleep(5)
robot.send_action(action)
