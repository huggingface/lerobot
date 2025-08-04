import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.pipeline import RobotProcessor
from lerobot.processor.utils import to_output_robot_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

episode_idx = 0

robot_config = SO100FollowerConfig(port="/dev/tty.usbmodem58760434471", id="my_awesome_follower_arm")

robot = SO100Follower(robot_config)
robot.connect()

dataset = LeRobotDataset("<hf_username>/<dataset_repo_id>", episodes=[episode_idx])
actions = dataset.hf_dataset.select_columns("action")

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf", target_frame_name="gripper_frame_link"
)


def to_transition_from_action(action: dict):
    """Build an EnvTransition for the pipeline"""
    return (None, action, None, None, None, {}, {})


# Build pipeline to convert ee pose action to joint action
robot_ee_to_joints = RobotProcessor(
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
        ),
        GripperVelocityToJoint(
            motor_names=list(robot.bus.motors.keys()),
            speed_factor=20.0,
        ),
    ],
    to_transition=to_transition_from_action,
    to_output=to_output_robot_action,
)

log_say(f"Replaying episode {episode_idx}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }

    joints_transition = robot_ee_to_joints(action)
    robot_action_to_send = robot_ee_to_joints.to_output(joints_transition)
    robot.send_action(robot_action_to_send)

    busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

robot.disconnect()
