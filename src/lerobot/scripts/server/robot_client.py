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
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any, Callable, Optional

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.constants import SUPPORTED_ROBOTS
from lerobot.scripts.server.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
    aggregate_actions,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils import queue
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.queue import get_last_item_from_queue


class RobotClient:
    prefix = "robot_client"

    def __init__(self, config: RobotClientConfig, shutdown_event: threading.Event):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        self.set_up_logger()

        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        if config.verify_robot_cameras:
            # Load policy config for validation
            policy_config = PreTrainedConfig.from_pretrained(config.pretrained_name_or_path)
            policy_image_features = policy_config.image_features

            # The cameras specified for inference must match the one supported by the policy chosen
            validate_robot_cameras_for_policy(lerobot_features, policy_image_features)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = shutdown_event

        # Lock to prevent reading from and writing to the robot at the same time
        self.robot_lock = threading.Lock()

        # Initialize client side variables
        self.action_queue_lock = threading.Lock()
        self.latest_action_timestep = -1
        self.action_chunk_size = config.actions_per_chunk

        self._chunk_size_threshold = config.chunk_size_threshold

        self.actions_bytes_queue: Queue[bytes] = Queue()
        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_size = []

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        # self.observation_bytes_queue: Queue[bytes] = Queue()

    def set_up_logger(self):
        self.logger = get_logger(self.prefix)
        self.logger.setLevel(logging.DEBUG)

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def _extract_actions_from_bytes_queue(self):
        result: list[TimedAction] = []

        while True:
            try:
                bytes = self.actions_bytes_queue.get_nowait()
                actions: list[TimedAction] = pickle.loads(bytes)
                result.extend(actions)
            except queue.Empty:
                break

        return result

    def start_policy_client_thread(self):
        self.logger.info("Starting action receiver thread")
        self.action_receiver_thread = threading.Thread(target=self.get_actions_loop, daemon=True)
        self.action_receiver_thread.start()
        self.logger.info("Action receiver thread started")

    def join_policy_client_thread(self):
        self.logger.info("Joining action receiver thread")
        self.action_receiver_thread.join()
        self.logger.info("Action receiver thread joined")

    def get_actions_loop(self):
        # This method is handle tranpsortation only, it doesn't know anything about the logic
        # it sends and receives bytes from the policy server

        self.logger.info("Starting GetActions loop")

        while self.running:
            get_observation_start = time.perf_counter()
            self.action_queue_lock.acquire()
            # If there are sufficient actions in the queue, skip the loop
            if self.action_queue.qsize() / self.action_chunk_size > self._chunk_size_threshold:
                # TODO: use cv
                self.action_queue_lock.release()
                time.sleep(self.config.environment_dt)
                continue

            obs_timestep = self.latest_action_timestep + 1
            with self.robot_lock:
                raw_observation: RawObservation = self.robot.get_observation()

            self.action_queue_lock.release()

            raw_observation["task"] = self.config.task

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=obs_timestep,
            )

            observation_bytes = pickle.dumps(observation)
            get_observation_end = time.perf_counter()

            if not self.running:
                break

            self.logger.info(f"Sending observation to policy server | get_observation={get_observation_end - get_observation_start:.4f}s")

            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )

            start_time = time.perf_counter()
            actions_bytes = self.stub.GetActions(observation_iterator)
            end_time = time.perf_counter()

            self.logger.info(f"GetActions took {end_time - start_time:.6f}s")

            actions: list[TimedAction] = pickle.loads(actions_bytes.data)

            if len(actions) > 0:
                with self.action_queue_lock:
                    # TODO: use cv
                    self.action_queue = aggregate_actions(self.action_queue, self.latest_action_timestep, actions, self.config.aggregate_fn)
            
            time.sleep(self.config.environment_dt)

        self.logger.info("GetActions loop stopped")

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.logger.info("Control loop thread starting")

        # self.push_observation_to_queue(task, verbose)
        while self.running:
            control_loop_start = time.perf_counter()

            get_action_start = time.perf_counter()
            try:
                with self.action_queue_lock:
                    timed_action = self.action_queue.get_nowait()
                    self.latest_action_timestep = timed_action.get_timestep()
                    action_queue_size = self.action_queue.qsize()
            except queue.Empty:
                self.logger.warning("Action queue is empty, skipping control loop iteration")
                time.sleep(self.config.environment_dt)
                continue

            action = self._action_tensor_to_action_dict(timed_action.get_action())
            get_action_end = time.perf_counter()

            perform_action_start = get_action_end
            with self.robot_lock:
                self.robot.send_action(action)
            perform_action_end = time.perf_counter()

            self.action_queue_size.append(action_queue_size)
            control_loop_end = time.perf_counter()

            time_to_sleep = min(self.config.environment_dt, max(0, self.config.environment_dt - (control_loop_end - control_loop_start)))

            self.logger.info(
                f"Ts={timed_action.get_timestamp()} | step=#{timed_action.get_timestep()} | control_loop={control_loop_end - control_loop_start:.4f}s | sleep={time_to_sleep:.4f}s | get_action={get_action_end - get_action_start:.4f}s | perform_action={perform_action_end - perform_action_start:.4f}s | action_queue_size={action_queue_size}"
            )

            time.sleep(time_to_sleep)

@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    shutdown_event = ProcessSignalHandler(use_threads=True).shutdown_event

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg, shutdown_event)

    if not client.start():
        return

    client.logger.info("Starting policy client thread...")

    client.start_policy_client_thread()
    client.logger.info("Policy client thread started")

    client.control_loop(task=cfg.task)

    client.stop()
    client.join_policy_client_thread()
    client.logger.info("Policy client thread joined")

    if cfg.debug_visualize_queue_size:
        visualize_action_queue_size(client.action_queue_size)

    client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
