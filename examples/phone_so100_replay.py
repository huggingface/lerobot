import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.pipeline import RobotProcessor
from lerobot.processor.utils import to_output_robot_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    AddRobotObservationAsComplimentaryData,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

episode_idx = 2

robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471", id="my_phone_teleop_follower_arm", use_degrees=True
)
robot = SO100Follower(robot_config)
robot.connect()

dataset = LeRobotDataset("pepijn223/phone_pipeline_pickup6", episodes=[episode_idx])
actions = dataset.hf_dataset.select_columns("action")

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)


def to_transition_from_action(action: dict):
    act = {}

    # EE pose
    for k in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"):
        if k in action:
            act[f"action.{k}"] = float(action[k])

    # Gripper: your dataset has absolute position
    if "gripper.pos" in action:
        act["action.gripper.pos"] = float(action["gripper.pos"])

    return {
        "observation": None,
        "action": act,
        "reward": None,
        "done": False,
        "truncated": False,
        "info": {},
        "complementary_data": {},
    }


# Build pipeline to convert ee pose action to joint action
robot_ee_to_joints = RobotProcessor(
    steps=[
        AddRobotObservationAsComplimentaryData(robot=robot),
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=False,  # Because replay is open loop
        ),
    ],
    to_transition=to_transition_from_action,
    to_output=to_output_robot_action,
)

robot_ee_to_joints.reset()

log_say(f"Replaying episode {episode_idx}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    ee_action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }

    joint_action = robot_ee_to_joints(ee_action)
    action_sent = robot.send_action(joint_action)

    busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

robot.disconnect()
