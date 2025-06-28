import argparse
import logging
import pickle  # nosec
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

import grpc
import torch

from lerobot.common.transport import (
    async_inference_pb2,  # type: ignore
    async_inference_pb2_grpc,  # type: ignore
)
from lerobot.common.transport.utils import send_bytes_in_chunks
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    TimedAction,
    TimedObservation,
    TinyPolicyConfig,
    get_logger,
    make_robot,
    map_robot_keys_to_lerobot_features,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)
    logger.setLevel(logging.WARNING)

    def __init__(self, config: RobotClientConfig):
        # Store configuration
        self.config = config

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = TinyPolicyConfig(
            config.policy_type, config.pretrained_name_or_path, config.lerobot_features, config.policy_device
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
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.robot = self.config.robot
        self.robot.connect()

        self.logger.info("Robot connected and ready")

        self.must_go = True  # does the observation qualify for direct processing on the policy server?

    @property
    def running(self):
        return self._running_event.is_set()

    def timestamps(self):
        """Get the timestamps of the actions in the queue"""
        return sorted([action.get_timestep() for action in self.action_queue.queue])

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
            self.action_queue_size = []

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
            self.logger.warning("Client not running")
            return False

        assert isinstance(obs, TimedObservation), "Input observation needs to be a TimedObservation!"

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

    def _validate_action(self, action: TimedAction):
        """Received actions are keps only when they have been produced for now or later, never before"""
        return not action.get_timestep() <= self.latest_action

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

        action_intersections: list[torch.Tensor] = []
        current_action_queue = {
            action.get_timestep(): action.get_action() for action in self.action_queue.queue
        }

        for new_action in incoming_actions:
            if not self._validate_action(new_action):
                continue

            elif new_action.get_timestep() not in current_action_queue:
                action_intersections.append(new_action)
                continue

            # TODO(fracapuano): There is probably a way to do this with broadcasting of the two action tensors
            action_intersections.append(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        new_queue = Queue()
        for action in action_intersections:
            # if self._validate_action(action):
            new_queue.put(action)

        self.action_queue = new_queue

    def receive_actions(self):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                for actions_chunk in self.stub.StreamActions(async_inference_pb2.Empty()):
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
                            f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                            f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                        )

                    # Update action queue
                    start_time = time.perf_counter()
                    self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                    queue_update_time = time.perf_counter() - start_time

                    self.must_go = (
                        True  # after receiving actions, next empty queue triggers must-go processing!
                    )

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

    def _actions_available(self):
        """Check if there are actions available in the queue"""
        return not self.action_queue.empty()

    def _get_next_action(self) -> Optional[TimedAction]:
        """Get the next action from the queue"""
        try:
            action = self.action_queue.get_nowait()
            return action

        except Empty:
            return None

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

    def _perform_action(self, timed_action: TimedAction) -> Action:
        self.robot.send_action(self._action_tensor_to_action_dict(timed_action.get_action()))
        self.latest_action = timed_action.get_timestep()

        self.logger.debug(
            f"Ts={timed_action.get_timestamp()} | "
            f"Action #{timed_action.get_timestep()} performed | "
            f"Queue size: {self.action_queue.qsize()}"
        )

        return timed_action.get_action()

    def control_loop_action(self) -> Optional[Action]:
        """Reading and performing actions in local queue"""
        self.action_queue_size.append(self.action_queue.qsize())

        # Get action from queue
        get_start = time.perf_counter()
        timed_action = self._get_next_action()
        get_end = time.perf_counter() - get_start

        _performed_action = self._perform_action(timed_action)

        self.logger.debug(
            f"Popping action from queue to perform took {get_end:.6f}s | "
            f"Queue size: {self.action_queue.qsize()}"
        )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, get_observation_fn) -> Observation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()
            observation = get_observation_fn()
            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            observation.must_go = self.must_go and self.action_queue.empty()
            self.send_observation(observation)

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

            # The actual observation is wrapped in a TimedObservation, get_observation() returns the tensor
            _captured_observation = observation.get_observation()
            return _captured_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, get_observation_fn: Callable[[], Observation]) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        control_loops = 0
        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self._actions_available():
                _performed_action = self.control_loop_action()

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(get_observation_fn)

            self.logger.warning(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

            control_loops += 1

        return _captured_observation, _performed_action


def parse_args():
    parser = argparse.ArgumentParser(description="Robot client for executing tasks via policy server")
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the red cube and put in the box",
        help="Task instruction for the robot to execute (e.g., 'fold my tshirt')",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level (default: 0)")
    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost:8080",
        help="Server address (default: localhost:8080)",
    )
    parser.add_argument("--policy-type", type=str, default="act", help="Policy type (default: smolvla)")
    parser.add_argument(
        "--pretrained-name-or-path",
        type=str,
        default="fracapuano/act_so100_test",
        help="Pretrained model name or path (default: lerobot/smolvla_base)",
    )
    parser.add_argument(
        "--policy-device", type=str, default="mps", help="Device for policy inference (default: cuda)"
    )
    parser.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=0.5,
        help="Chunk size threshold (`g` in the paper, default: 0.5)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="so100",
        help="Robot name, as per the `make_robot` function (default: so100)",
    )

    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/tty.usbmodem585A0076841",
        help="Port on which to read/write robot joint status (e.g., '/dev/tty.usbmodem575E0031751'). Find your port with lerobot/find_port.py",
    )

    parser.add_argument(
        "--robot-id",
        type=str,
        default="follower_so100",
        help="ID of the robot to connect to (default: follower_so100)",
    )

    parser.add_argument(
        "--robot-cameras",
        type=str,
        default='{"laptop": {"index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "phone": {"index_or_path": 1, "width": 640, "height": 480, "fps": 30}}',
        help="Cameras of the robot to connect to (default: {'laptop': {'index_or_path': 0, 'width': 1920, 'height': 1080, 'fps': 30}, 'phone': {'index_or_path': 1, 'width': 1920, 'height': 1080, 'fps': 30}})",
    )

    parser.add_argument(
        "--debug-visualize-queue-size",
        action="store_true",
        help="Trigger visualization of action queue size upon stopping the client, to tweak client hyperparameters (default: False)",
    )

    return parser.parse_args()


def async_client(args: argparse.Namespace):
    robot = make_robot(args)
    policy_config = PreTrainedConfig.from_pretrained(args.pretrained_name_or_path)
    policy_image_features = policy_config.image_features

    lerobot_features = map_robot_keys_to_lerobot_features(robot)

    # The cameras specified for inference must match the one supported by the policy chosen
    validate_robot_cameras_for_policy(lerobot_features, policy_image_features)

    # Create config from parsed arguments
    config = RobotClientConfig(
        robot=robot,
        policy_type=args.policy_type,
        pretrained_name_or_path=args.pretrained_name_or_path,
        lerobot_features=lerobot_features,
        server_address=args.server_address,
        policy_device=args.policy_device,
        chunk_size_threshold=args.chunk_size_threshold,
        aggregate_fn=lambda old, new: 0.3 * old + 0.7 * new,
    )

    # Create client with config
    client = RobotClient(config)

    if client.start():

        def make_observation() -> TimedObservation:
            # Function to make observations starting from the robot's get_observation() method
            observation: RawObservation = client.robot.get_observation()
            observation["task"] = args.task

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=observation,
                timestep=max(client.latest_action, 0),
            )

            return observation

        client.logger.info("Starting all threads...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        control_loop_thread = threading.Thread(
            target=client.control_loop, args=(make_observation,), daemon=True
        )

        # Start all threads
        action_receiver_thread.start()
        control_loop_thread.start()

        try:
            while client.running:
                time.sleep(0.1)  # tiny sleep to avoid tight loop in main thread

        except KeyboardInterrupt:
            client.stop()
            action_receiver_thread.join()
            control_loop_thread.join()

        finally:
            if args.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)

            client.logger.info("Client stopped")


if __name__ == "__main__":
    args = parse_args()
    async_client(args)  # run the client
