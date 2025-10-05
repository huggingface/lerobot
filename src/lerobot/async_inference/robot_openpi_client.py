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
        self.teleop_action_processor = make_bimanual_koch_teleop_processors(self.robot, False)

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

    def _action_dict_to_action_tensor(self, action_dict: dict[str, float]) -> torch.Tensor:
        action_tensor = torch.tensor([action_dict[key] for key in self.action_features])
        return action_tensor

    def _compute_current_ee(self, raw_observation: RawObservation) -> dict[str, float]:
            base_action_world_dict = self.teleop_action_processor(
                (raw_observation, raw_observation)
            )  # we need to run FK and use this as our zero.
            return self._action_dict_to_action_tensor(base_action_world_dict)

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        timestep_count = 0
        # hardcode this to some value
        initial_EE = torch.tensor([
            -1.0168e-01,
            1.1525e-03,
            9.4441e-02,
            -7.4215e-01,
            -1.2467e00,
            -5.7231e-01,
            4.9067e01,
            -8.8268e-02,
            4.3833e-03,
            9.6670e-02,
            -6.4383e-01,
            -1.2725e00,
            -5.1562e-01,
            5.0397e01,
        ])

        # indices to exclude for gripper
        gripper_exclude = [6, 13]
        rotation_exclude = [3, 4, 5, 10, 11, 12]

        # create a mask (True = add, False = keep original)
        non_gripper_mask = torch.ones_like(initial_EE, dtype=torch.bool)
        ee_pose_mask = torch.ones_like(initial_EE, dtype=torch.bool)
        ee_pose_mask[gripper_exclude] = False
        ee_pose_mask[rotation_exclude] = False
        non_gripper_mask[gripper_exclude] = False
        task_completed = False

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

            assert action_chunk.ndim == 2, (
                "Unexpected action chunk shape, should be (num_timesteps, action_dim)"
            )
            base_action_world_tensor = self._compute_current_ee(raw_observation)
            action_chunk_tensor = torch.tensor(
                action_chunk[: self.config.actions_per_chunk, : base_action_world_tensor.shape[-1]]
            )  # Trim to the same number of actions as the base action
            action_chunk_world = action_chunk_tensor + non_gripper_mask * base_action_world_tensor

            # Check if we need to override the action chunk to initial EE
            if task_completed:
                task_completed = False
                # Linearly interpolate non-gripper DOFs from base -> initial across the chunk
                num_steps = action_chunk_tensor.shape[0]
                num_dims = base_action_world_tensor.shape[0]
                t_vals = (
                    torch.linspace(
                        0.0,
                        1.0,
                        steps=num_steps,
                        dtype=base_action_world_tensor.dtype,
                        device=base_action_world_tensor.device,
                    )
                    .unsqueeze(1)
                )  # (num_steps, 1)

                base = base_action_world_tensor.unsqueeze(0).expand(num_steps, num_dims)  # (num_steps, num_dims)
                target = (
                    initial_EE.to(base_action_world_tensor.dtype)
                    .to(base_action_world_tensor.device)
                    .unsqueeze(0)
                    .expand(num_steps, num_dims)
                )  # (num_steps, num_dims)

                interpolated = (1.0 - t_vals) * base + t_vals * target  # (num_steps, num_dims)
                # Keep grippers fixed at base; interpolate others
                # mask_expand = mask.unsqueeze(0).expand(num_steps, num_dims)
                # action_chunk_world = torch.where(mask_expand, interpolated, base)
                action_chunk_world = interpolated

            # log_rerun_action_chunk(action_chunk_world)
            count = 0
            action_chunk_velocities =  torch.zeros_like(action_chunk_world)
            action_chunk_velocities[1:] = action_chunk_world[1:] - action_chunk_world[:-1]

            for action_idx in range(action_chunk_world.shape[0]):
                action = action_chunk_world[action_idx]
                log_rerun_action_chunk(action.unsqueeze(0))
                action_tensor_world = self._action_tensor_to_action_dict(action)
                if action_tensor_world["left_ee.y"] < -0.09:
                    count += 1

                processed_action = self.robot_action_processor((action_tensor_world, raw_observation))
                _performed_action = self.robot.send_action(processed_action)
                log_rerun_data(raw_observation, _performed_action)
                time.sleep(self.config.environment_dt)
                largest_left_arm_delta = torch.abs(action_chunk_velocities[action_idx][:3]).max()
                if largest_left_arm_delta > 0.05:
                    breakpoint()
                    self.logger.info(f"Large action of {largest_left_arm_delta}, sending twice to reach it")
                    _performed_action = self.robot.send_action(processed_action)
                    time.sleep(self.config.environment_dt)
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
