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
        self.setup_logger()

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

        # Initialize client side variables
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.actions_bytes_queue = Queue()
        self.action_queue = Queue()
        self.action_queue_size = []

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.observation_bytes_queue = Queue()

    def setup_logger(self):
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

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        self.action_queue = future_action_queue

    def _extract_actions_from_bytes_queue(self):
        result = []

        while True:
            try:
                bytes = self.actions_bytes_queue.get_nowait()
                actions = pickle.loads(bytes)
                result.extend(actions)
            except queue.Empty:
                break

        return result

    def start_communication_thread(self):
        self.logger.info("Starting action receiver thread")
        self.action_receiver_thread = threading.Thread(target=self.get_actions_loop, daemon=True)
        self.action_receiver_thread.start()
        self.logger.info("Action receiver thread started")

    def join_communication_thread(self):
        self.logger.info("Joining action receiver thread")
        self.action_receiver_thread.join()
        self.logger.info("Action receiver thread joined")

    def get_actions_loop(self):
        # This method is handle tranpsortation only, it doesn't know anything about the logic
        # it sends and receives bytes from the policy server

        self.logger.info("Starting GetActions loop")

        while self.running:
            observation_bytes = get_last_item_from_queue(
                self.observation_bytes_queue, block=True, timeout=self.config.environment_dt
            )

            if not self.running:
                break

            if observation_bytes is None:
                continue

            self.logger.info("Received observation bytes, sending to policy server")

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

            self.actions_bytes_queue.put(actions_bytes.data)

        self.logger.info("GetActions loop stopped")

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def _update_action_queue(self):
        actions_from_policy_server = self._extract_actions_from_bytes_queue()

        if len(actions_from_policy_server) > 0:
            self._aggregate_action_queues(actions_from_policy_server, self.config.aggregate_fn)

    def track_action_queue_size(self):
        self.action_queue_size.append(self.action_queue.qsize())
        self.logger.debug(f"Action queue size: {self.action_queue_size[-1]}")

    def apply_action(self) -> dict[str, Any]:
        """Reading and performing actions in local queue"""
        try:
            timed_action = self.action_queue.get_nowait()
        except queue.Empty:
            return

        performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )

        self.latest_action = timed_action.get_timestep()

        self.logger.debug(
            f"Ts={timed_action.get_timestamp()} | Action #{timed_action.get_timestep()} performed"
        )

        return performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def push_observation_to_queue(self, task: str, verbose: bool = False) -> RawObservation | None:
        if not self._ready_to_send_observation():
            return

        # Get serialized observation bytes from the function
        raw_observation: RawObservation = self.robot.get_observation()
        raw_observation["task"] = task

        observation = TimedObservation(
            timestamp=time.time(),  # need time.time() to compare timestamps across client and server
            observation=raw_observation,
            timestep=0,
        )

        observation_bytes = pickle.dumps(observation)

        self.observation_bytes_queue.put(observation_bytes)

        return raw_observation

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.logger.info("Control loop thread starting")

        while self.running:
            control_loop_start = time.perf_counter()

            self._update_action_queue()
            self.track_action_queue_size()
            self.apply_action()
            self.push_observation_to_queue(task, verbose)

            self._sleep(control_loop_start, self.config.environment_dt)

    def _sleep(self, control_loop_start: float, max_sleep_time: float):
        time_to_sleep = min(max_sleep_time, max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        self.logger.info(
            f"Control loop took {time.perf_counter() - control_loop_start:.6f}s, will sleep for {time_to_sleep:.6f}s (max sleep time: {max_sleep_time:.6f}s)"
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

    client.logger.info("Starting action receiver thread...")

    client.start_communication_thread()

    client.control_loop(task=cfg.task)

    client.stop()
    client.join_communication_thread()
    client.logger.info("Communication thread joined")

    if cfg.debug_visualize_queue_size:
        visualize_action_queue_size(client.action_queue_size)

    client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
