#!/bin/bash

# Evaluate SmolVLA model on real robot
uv run python -c "
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.factory import get_policy_class
from lerobot.processor.factory import make_default_robot_observation_processor
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.utils import log_say
import logging

logging.basicConfig(level=logging.INFO)

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60

# Create the robot configuration & robot
camera_config = {
    'gripper': OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
    'front': OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS)
}
robot_config = SO100FollowerConfig(
    port='/dev/tty.usbmodem58FA0834591',
    id='so100_follower',
    cameras=camera_config,
    use_degrees=True,
)

robot = SO100Follower(robot_config)

# Load policy from HuggingFace
policy_class = get_policy_class('smolvla')
policy = policy_class.from_pretrained('helper2424/smolval_move_green_object_to_purple_plate')
policy = policy.to('mps')
policy.eval()

# Create robot observation processor
robot_observation_processor = make_default_robot_observation_processor()

log_say('Starting evaluation')

# Run evaluation
record_loop(
    robot=robot,
    policy=policy,
    robot_observation_processor=robot_observation_processor,
    num_episodes=NUM_EPISODES,
    fps=FPS,
    episode_time_s=EPISODE_TIME_SEC,
    task_description='Move green small object into the purple platform',
    save_videos=True,
    display_cameras=['gripper', 'front'],
)

robot.disconnect()
log_say('Evaluation complete')
"
