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

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
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
from lerobot.robots.bi_koch_follower.config_bi_koch_follower import make_bimanual_koch_robot_processors
from lerobot.teleoperators.bi_koch_leader.config_bi_koch_leader import make_bimanual_koch_teleop_processors

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features


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

        self.lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        # TODO: this needs to be consistent with the policy config
        self.policy_image_features = {
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }

        self.server_address = config.server_address
        self.host = self.server_address.split(":")[0]
        self.port = self.server_address.split(":")[1]

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.client = websocket_client_policy.WebsocketClientPolicy(host=self.host, port=self.port)
        self.robot_action_processor = make_bimanual_koch_robot_processors(self.robot, True)
        self.teleop_action_processor = make_bimanual_koch_teleop_processors(self.robot, True)

        self.action_features = aggregate_pipeline_dataset_features(
            pipeline=self.teleop_action_processor,
            initial_features=create_initial_features(
                action=self.robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=True,
        )["action"]["names"]

    @property
    def running(self):
        return True

    def stop(self):
        """Stop the robot client"""
        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.action_features)}
        return action

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        timestep_count = 0
        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""

            """Control loop: (2) Streaming observations to the remote policy server"""
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
            action_chunk = action_chunk[: self.config.actions_per_chunk]

            base_action_xyz = self.teleop_action_processor(
                (raw_observation, raw_observation)
            )  # we need to run FK and use this as our zero.

            for action in action_chunk:
                action_tensor = self._action_tensor_to_action_dict(action)
                action_tensor_world = action_tensor.copy()
                for key in action_tensor:
                    if "gripper" not in key:
                        assert key in base_action_xyz, f"Key {key} not in base_action_xyz"
                        action_tensor_world[key] = action_tensor[key] + base_action_xyz[key]
                processed_action = self.robot_action_processor((action_tensor_world, raw_observation))
                _performed_action = self.robot.send_action(processed_action)
                log_rerun_data(raw_observation, _performed_action)
                time.sleep(self.config.environment_dt)

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
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def synchronous_client(cfg: RobotOpenpiClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotOpenpiClient(cfg)
    try:
        client.control_loop(task=cfg.task)
    finally:
        client.stop()
        client.logger.info("Client stopped")


if __name__ == "__main__":
    synchronous_client()  # run the client
