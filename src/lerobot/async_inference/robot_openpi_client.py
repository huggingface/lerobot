# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/scripts/server/robot_openpi_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --debug_visualize_queue_size=True
```
"""

from lerobot.async_inference.configs import RobotOpenpiClientConfig
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, log_rerun_action_chunk
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_koch_follower,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.async_inference.constants import SUPPORTED_ROBOTS
from lerobot.async_inference.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    raw_observation_to_observation,
)
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.processor.processor_factory import make_robot_action_processor, make_fk_processor

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.async_inference import koch_utils
from lerobot.utils.control_utils import is_headless


def init_pause_keyboard_listener():
    """
    Initializes a keyboard listener for pause/resume functionality.

    Similar to control_utils.init_keyboard_listener but specifically for pause control.

    Returns:
        A tuple containing:
        - The keyboard.Listener instance, or None if in a headless environment.
        - A dictionary with the 'paused' event flag.
    """
    events = {"paused": False}

    if is_headless():
        logging.warning(
            "Headless environment detected. Keyboard pause functionality will not be available."
        )
        return None, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                events["paused"] = not events["paused"]
                if events["paused"]:
                    print("🔴 PAUSED - Press SPACEBAR to resume")
                else:
                    print("🟢 RESUMED - Press SPACEBAR to pause")
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


class RobotOpenpiClient:
    prefix = "robot_openpi_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotOpenpiClientConfig):
        """Initialize RobotOpenpiClient with unified configuration.

        Args:
            config: RobotOpenpiClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        init_rerun(session_name="openpi_client")

        # Detect if using single or dual arm based on robot type
        self.is_bimanual = config.robot.type == "bi_koch_follower"
        self.uses_fk_ik = config.robot.type in ["koch_follower", "bi_koch_follower"]

        # Select appropriate initial EE pose based on robot type
        self.initial_ee_pose = (
            koch_utils.INITIAL_EE_POSE_BIMANUAL if self.is_bimanual else koch_utils.INITIAL_EE_POSE_SINGLE
        )

        self.lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        # TODO: this needs to be consistent with the policy config
        self.policy_image_features = {
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        if self.is_bimanual:
            self.policy_image_features["observation.images.right_wrist"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            )

        self.server_address = config.server_address
        self.host = self.server_address.split(":")[0]
        self.port = self.server_address.split(":")[1]

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.client = websocket_client_policy.WebsocketClientPolicy(host=self.host, port=self.port)

        # Use factory functions to create processors based on robot type
        self.robot_action_processor = make_robot_action_processor(config.robot, self.robot, True)

        # Create FK processor for computing current EE from joint angles
        self.fk_processor = make_fk_processor(config.robot, self.robot, display_data=False)

        # Get action features using the utility module
        self.action_features = koch_utils.get_action_features(self.robot, self.fk_processor)


    @property
    def running(self):
        return True

    def stop(self):
        """Stop the robot client"""
        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

    def control_loop(self, task: str, events: dict | None = None, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations

        Args:
            task: Task description string
            events: Dictionary with event flags (e.g., 'paused')
            verbose: Whether to log verbose output
        """
        # Wait at barrier for synchronized start
        self.logger.info("Control loop thread starting")

        if events is None:
            events = {}

        _performed_action = None
        _captured_observation = None

        timestep_count = 0

        # Get initial EE pose based on robot type
        initial_EE = self.initial_ee_pose

        # Set indices to exclude based on robot type
        if self.is_bimanual:
            # Bimanual: exclude both left and right grippers
            gripper_exclude = [6, 13]
            rotation_exclude = [3, 4, 5, 10, 11, 12]
        else:
            # Single arm: exclude single gripper
            gripper_exclude = [6]
            rotation_exclude = [3, 4, 5]

        # create a mask (True = add, False = keep original)
        non_gripper_mask = torch.ones_like(initial_EE, dtype=torch.bool)
        ee_pose_mask = torch.ones_like(initial_EE, dtype=torch.bool)
        ee_pose_mask[gripper_exclude] = False
        ee_pose_mask[rotation_exclude] = False
        non_gripper_mask[gripper_exclude] = False
        task_completed = False

        while self.running:
            # Check if paused
            if events.get("paused", False):
                time.sleep(0.1)  # Sleep briefly to avoid busy waiting
                continue

            control_loop_start = time.perf_counter()
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            observation: Observation = raw_observation_to_observation(
                raw_observation,
                self.lerobot_features,
                self.policy_image_features,
                self.config.device,
            )
            # Convert to numpy. This prompt hack is annoying.
            observation = {
                k: v.numpy().squeeze(0) for k, v in observation.items()
            }  # Remove the batch dimension for openpi policies
            observation["prompt"] = task

            timed_observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=observation,
                timestep=timestep_count,
            )
            timestep_count += 1

            obs_capture_time = time.perf_counter()
            action_chunk = self.client.infer(timed_observation.get_observation())["actions"]
            self.logger.info(f"Inference time (ms): {(time.perf_counter() - obs_capture_time) * 1000:.2f}")

            if self.config.actions_per_chunk > action_chunk.shape[0]:
                self.logger.warning(
                    f"Actions per chunk is greater than the number of actions in the chunk: {self.config.actions_per_chunk} > {action_chunk.shape[0]}"
                )

            assert action_chunk.ndim == 2, (
                "Unexpected action chunk shape, should be (num_timesteps, action_dim)"
            )
            base_action_world_tensor = koch_utils.compute_current_ee(
                raw_observation, self.fk_processor, self.action_features
            )
            action_chunk_tensor = torch.tensor(
                action_chunk[: self.config.actions_per_chunk, : base_action_world_tensor.shape[-1]]
            )  # Trim to the same number of actions as the base action
            action_chunk_world = action_chunk_tensor + non_gripper_mask * base_action_world_tensor

            # Check if we need to override the action chunk to initial EE. If so, we will generate a linearly interpolated trajectory from current EE pose to initial EE.
            if task_completed:
                task_completed = False
                # Generate linearly interpolated trajectory from current pose to initial EE
                num_steps = action_chunk_tensor.shape[0]
                action_chunk_world = koch_utils.generate_linear_trajectory(
                    start=base_action_world_tensor, target=self.initial_ee_pose, num_steps=num_steps
                )

            log_rerun_action_chunk(action_chunk_world)
            count = 0
            action_chunk_velocities =  torch.zeros_like(action_chunk_world)
            action_chunk_velocities[1:] = action_chunk_world[1:] - action_chunk_world[:-1]

            for action_idx in range(action_chunk_world.shape[0]):
                action_start_time = time.perf_counter()

                action = action_chunk_world[action_idx]
                log_rerun_action_chunk(action.unsqueeze(0))
                action_tensor_world = koch_utils.action_tensor_to_dict(action, self.action_features)

                # Bimanual-specific task completion check
                if self.is_bimanual and action_tensor_world.get("left_ee.y", 0) < -0.09:
                    count += 1

                processed_action = self.robot_action_processor((action_tensor_world, raw_observation))
                _performed_action = self.robot.send_action(processed_action)
                log_rerun_data(raw_observation, _performed_action)

                largest_left_arm_delta = torch.abs(action_chunk_velocities[action_idx][:3]).max()
                if largest_left_arm_delta > 0.05:
                    breakpoint()
                    self.logger.info(f"Large action of {largest_left_arm_delta}, sending twice to reach it")
                    _performed_action = self.robot.send_action(processed_action)

                # Dynamically adjust sleep time to maintain desired control frequency
                action_elapsed_time = time.perf_counter() - action_start_time
                time.sleep(max(0, self.config.environment_dt / self.config.speed_multiplier - action_elapsed_time))
                # The code below slows things down a lot. We need a better way to compute this
                # raw_observation = self.robot.get_observation()
                # current_ee = self._compute_current_ee(raw_observation)
                # thresh = 0.02
                # if torch.any(torch.abs(current_ee - action)[ee_pose_mask] > thresh):
                #     print("Sending action again reach it before trying to get to next action chunk")
                #     _performed_action = self.robot.send_action(processed_action)
                #     time.sleep(self.config.environment_dt / 10)
                # raw_observation = self.robot.get_observation()
                # current_ee = self._compute_current_ee(raw_observation)
                # thresh = 0.05
                # if torch.any(torch.abs(current_ee - action)[ee_pose_mask] > thresh):
                #     print("Sending action again reach it before trying to get to next action chunk")
                #     _performed_action = self.robot.send_action(processed_action)
                #     time.sleep(self.config.environment_dt / 10)


            if count == self.config.actions_per_chunk:
                task_completed = True
                self.logger.info("Task completed, going to initial EE")

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(timed_observation.get_timestamp())

                self.logger.info(
                    f"Obs #{timed_observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.info(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {(obs_capture_time - start_time):.6f}s"
                )

            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt / self.config.speed_multiplier - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def synchronous_client(cfg: RobotOpenpiClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    # Initialize keyboard listener for pause functionality
    listener, events = init_pause_keyboard_listener()
    if listener is not None:
        logging.info("Press SPACEBAR to pause/resume the robot")

    client = RobotOpenpiClient(cfg)
    try:
        client.control_loop(task=cfg.task, events=events)
    finally:
        client.stop()
        if listener is not None:
            listener.stop()
        client.logger.info("Client stopped")


if __name__ == "__main__":
    synchronous_client()  # run the client
