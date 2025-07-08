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
from queue import Empty, Queue
from typing import Callable, Optional

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


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)  # TODO(fracapuano): Reduce logging verbosity

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)

        # Load policy config for validation
        policy_config = PreTrainedConfig.from_pretrained(config.pretrained_name_or_path)
        policy_image_features = policy_config.image_features

        # The cameras specified for inference must match the one supported by the policy chosen
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        validate_robot_cameras_for_policy(lerobot_features, policy_image_features)

        self.robot.connect()

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

        self._running_event = threading.Event()

        # Initialize client side variables
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # TODO(fracapuano): Either find a logic local to the control_loop or replace this with an event. This such that we can remove
        # the must_go in the receive_actions() thread, otherwise we need a lock.
        self.must_go = True  # does the observation qualify for direct processing on the policy server?

    @property
    def running(self):
        return self._running_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.info(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.info(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self._running_event.set()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    # TODO(fracapuano): Reduce logging verbosity
    def _inspect_action_queue(self):
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
        # TODO(fracapuano): move outside of the function and make aggregate_fn an always required argument
        if not aggregate_fn:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        current_action_queue = {
            action.get_timestep(): action.get_action() for action in self.action_queue.queue
        }

        for new_action in incoming_actions:
            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= self.latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO(fracapuano): There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        # TODO(fracapuano): Add a lock
        self.action_queue = future_action_queue

    def actions_available(self):
        """Check if there are actions available in the queue"""
        return not self.action_queue.empty()

    def _clear_action_queue(self):
        """Clear the existing queue"""
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except Empty:
                break

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self) -> Optional[Action]:
        """Reading and performing actions in local queue"""
        self.action_queue_size.append(self.action_queue.qsize())

        # Get action from queue
        get_start = time.perf_counter()
        timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        self.latest_action = timed_action.get_timestep()

        self.logger.debug(
            f"Ts={timed_action.get_timestamp()} | "
            f"Action #{timed_action.get_timestep()} performed | "
            f"Queue size: {self.action_queue.qsize()}"
        )

        self.logger.debug(
            f"Popping action from queue to perform took {get_end:.6f}s | "
            f"Queue size: {self.action_queue.qsize()}"
        )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def get_actions_unary(self, observation: TimedObservation):
        """Get actions from the policy server"""
        observation_bytes = pickle.dumps(observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        actions_chunk = self.stub.GetActionsUnary(observation_iterator)
        actions = pickle.loads(actions_chunk.data)  # nosec
        return actions

    def control_loop(self, task: str):
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                self.control_loop_action()

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                received_actions = self.get_actions_unary()
                self._aggregate_action_queues(received_actions, self.config.aggregate_fn)

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)

            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
