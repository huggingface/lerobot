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
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.cameras="{ laptop: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, phone: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=follower_so100 \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=fracapuano/act_so100_test \
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
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    send_bytes_in_chunks,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)
from lerobot.transport import (
    async_inference_pb2,  # type: ignore
    async_inference_pb2_grpc,  # type: ignore
)


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
        self.robot.connect()

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            config.lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = async_inference_pb2_grpc.AsyncInferenceStub(self.channel)
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
            self.stub.Ready(async_inference_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.info(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = async_inference_pb2.PolicySetup(data=policy_config_bytes)

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

    def stop(self):
        """Stop the robot client"""
        self._running_event.clear()

        self.robot.disconnect()
        self.logger.info("Robot disconnected")

        self.channel.close()
        self.logger.info("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                async_inference_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.info(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
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

    def receive_actions(self):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(async_inference_pb2.Empty())
                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                start_time = time.perf_counter()

                self.logger.info(f"Current latest action: {self.latest_action}")

                # Get queue state before changes
                old_size, old_timesteps = self._inspect_action_queue()
                if not old_timesteps:
                    old_timesteps = [self.latest_action]  # queue was empty

                # Log incoming actions
                incoming_timesteps = [a.get_timestep() for a in timed_actions]

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0:
                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{self.latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go = True  # after receiving actions, next empty queue triggers must-go processing!

                # Get queue state after changes
                new_size, new_timesteps = self._inspect_action_queue()

                self.logger.info(
                    f"Queue update complete ({queue_update_time:.6f}s) | "
                    f"Before: {old_size} items | "
                    f"After: {new_size} items | "
                )
                self.logger.info(
                    f"Latest action: {self.latest_action} | "
                    f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                    f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

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

    def control_loop_observation(self, task: str) -> Observation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(self.latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            observation.must_go = self.must_go and self.action_queue.empty()
            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {self.action_queue.qsize()} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go flag will be set again after receiving actions
                self.must_go = False

            # Calculate comprehensive FPS metrics
            fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

            self.logger.info(
                f"Obs #{observation.get_timestep()} | "
                f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                f"Target: {fps_metrics['target_fps']:.2f}"
            )

            self.logger.debug(
                f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
            )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action()

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task)

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    # Load policy config for validation
    policy_config = PreTrainedConfig.from_pretrained(cfg.pretrained_name_or_path)
    policy_image_features = policy_config.image_features

    # The cameras specified for inference must match the one supported by the policy chosen
    validate_robot_cameras_for_policy(cfg.lerobot_features, policy_image_features)

    # If you want to use a different aggregate function, you can do:
    # my_aggregate_fn = lambda old, new: 0.3 * old + 0.7 * new
    # cfg.aggregate_fn = my_aggregate_fn

    # Create client with unified config (robot created at init)
    logging.info(pformat(asdict(cfg)))
    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)

            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
