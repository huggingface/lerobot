import os
import pickle  # nosec
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore
import grpc
import torch

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.scripts.server.constants import environment_dt, idle_wait
from lerobot.scripts.server.helpers import TimedAction, TimedObservation, TinyPolicyConfig, setup_logging


class RobotClient:
    prefix = "robot_client"
    info_bracket = "CLIENT"
    logger = setup_logging(prefix, info_bracket)

    def __init__(
        self,
        server_address: Optional[str] = None,
        policy_type: str = "act",
        pretrained_name_or_path: str = "fracapuano/act_so100_test",
        policy_device: str = "mps",
        chunk_size_threshold: float = 0.5,
        robot: str = "so100",
        mock: bool = True,
    ):
        # Use environment variable if server_address is not provided
        if server_address is None:
            server_address = os.getenv("SERVER_ADDRESS", "localhost:8080")
            self.logger.info(f"No server address provided, using default address: {server_address}")

        self.policy_config = TinyPolicyConfig(policy_type, pretrained_name_or_path, policy_device)
        self.channel = grpc.insecure_channel(server_address)
        self.stub = async_inference_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {server_address}")

        self.running = False

        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = chunk_size_threshold

        self.action_queue = Queue()
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        start_time = time.time()
        self.robot = make_robot(robot, mock=mock)
        self.robot.connect()

        connect_time = time.time()
        self.logger.info(f"Robot connection time: {connect_time - start_time:.4f}s")

        time.sleep(idle_wait)  # sleep waiting for cameras to activate
        self.logger.info("Robot connected and ready")

    def timestamps(self):
        """Get the timestamps of the actions in the queue"""
        return sorted([action.get_timestep() for action in self.action_queue.queue])

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.time()
            self.stub.Ready(async_inference_pb2.Empty())
            end_time = time.time()
            self.logger.info(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = async_inference_pb2.PolicySetup(
                transfer_state=async_inference_pb2.TRANSFER_BEGIN, data=policy_config_bytes
            )

            self.logger.info("Sending policy instructions to policy server")
            self.logger.info(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.running = True
            self.available_actions_size = []
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.running = False

        self.robot.disconnect()
        self.logger.info("Robot disconnected")

        self.channel.close()
        self.logger.info("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
        transfer_state: async_inference_pb2.TransferState = async_inference_pb2.TRANSFER_MIDDLE,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            self.logger.warning("Client not running")
            return False

        assert isinstance(obs, TimedObservation), "Input observation needs to be a TimedObservation!"

        start_time = time.time()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.time()
        self.logger.debug(f"Observation serialization time: {serialize_time - start_time:.6f}s")

        observation = async_inference_pb2.Observation(transfer_state=transfer_state, data=observation_bytes)

        try:
            send_start = time.time()
            _ = self.stub.SendObservations(iter([observation]))
            send_end = time.time()

            obs_timestep = obs.get_timestep()

            self.logger.info(
                f"Sent observation #{obs_timestep} | "
                f"Serialize time: {serialize_time - start_time:.6f}s | "
                f"Network time: {send_end - send_start:.6f}s | "
                f"Total time: {send_end - start_time:.6f}s"
            )

            self.last_obs_sent_time = send_end
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

    def _update_action_queue(self, actions: list[TimedAction]):
        """Update the action queue with new actions, without ever emptying the queue"""
        # TODO(fracapuano): Test action aggregation self._aggregate_action_queues(actions)

        new_queue = Queue()
        for action in actions:
            if self._validate_action(action):
                new_queue.put(action)

        self.action_queue = new_queue

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
            if new_action.get_timestep() in current_action_queue:
                # TODO(fracapuano): There is probably a way to do this with broadcasting of the two action tensors
                action_intersections.append(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        action=aggregate_fn(
                            current_action_queue[new_action.get_timestep()], new_action.get_action()
                        ),
                        timestep=new_action.get_timestep(),
                    )
                )
            else:
                action_intersections.append(new_action)

        new_queue = Queue()
        for action in action_intersections:
            if self._validate_action(action):
                new_queue.put(action)

        self.action_queue = new_queue

    def _clear_action_queue(self):
        """Clear the existing queue"""
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except Empty:
                break

    def _fill_action_queue(self, actions: list[TimedAction]):
        """Fill the action queue with incoming valid actions"""
        start_time = time.time()
        valid_count = 0

        for action in actions:
            if self._validate_action(action):
                self.action_queue.put(action)
                valid_count += 1

        end_time = time.time()
        self.logger.debug(
            f"Queue filled: {valid_count}/{len(actions)} valid actions added in {end_time - start_time:.6f}s"
        )

    def _clear_and_fill_action_queue(self, actions: list[TimedAction]):
        self._clear_action_queue()
        self._fill_action_queue(actions)

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
                    deserialize_start = time.time()
                    timed_actions = pickle.loads(actions_chunk.data)  # nosec
                    deserialize_end = time.time()

                    self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                    start_time = time.time()

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
                        server_to_client_latency = receive_time - self.last_obs_sent_time

                        self.logger.info(
                            f"Received action chunk for step #{first_action_timestep} | "
                            f"Latest action: #{self.latest_action} | "
                            f"Network latency (server->client): {server_to_client_latency:.6f}s | "
                            f"Deserialization time: {deserialize_end - deserialize_start:.6f}s"
                        )

                    # Update action queue
                    start_time = time.time()
                    self._update_action_queue(timed_actions)
                    queue_update_time = time.time() - start_time

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
                # Avoid tight loop on action receiver error
                time.sleep(idle_wait)

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

    def _perform_action(self, timed_action: TimedAction):
        self.robot.send_action(timed_action.get_action())
        self.latest_action = timed_action.get_timestep()

        self.logger.debug(
            f"Ts={timed_action.get_timestamp()} | "
            f"Action #{timed_action.get_timestep()} performed | "
            f"Queue size: {self.action_queue.qsize()}"
        )

    def execute_actions(self):
        """Continuously execute actions from the queue"""
        import warnings

        warnings.warn("This method is deprecated! Will be removed soon!", stacklevel=2)
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        time.sleep(idle_wait)  # wait for observation capture to start

        self.logger.info("Action execution thread starting")

        while self.running:
            # constantly monitor the size of the action queue
            self.available_actions_size.append(self.action_queue.qsize())

            if self._actions_available():
                timed_action = self._get_next_action()
                self._perform_action(timed_action)

                time.sleep(environment_dt)

            else:
                self.logger.debug("No action available | Sleeping")
                time.sleep(idle_wait)

    def stream_observations(self, get_observation_fn):
        """Continuously stream observations to the server"""
        import warnings

        warnings.warn("This method is deprecated! Will be removed soon!", stacklevel=2)

        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Observation streaming thread starting")

        while self.running:
            try:
                # Get serialized observation bytes from the function
                start_time = time.time()
                observation = get_observation_fn()
                obs_capture_time = time.time() - start_time

                self.logger.debug(f"Capturing observation took {obs_capture_time:.6f}s")

                if not hasattr(self, "last_obs_timestamp"):
                    self.last_obs_timestamp = observation.get_timestamp()

                obs_timestep, obs_timestamp = observation.get_timestep(), observation.get_timestamp()
                self.logger.info(
                    f"Ts={obs_timestamp} | "
                    f"Captured observation #{obs_timestep} | "
                    f"1/DeltaTs (~frequency)={1 / (1e-6 + obs_timestamp - self.last_obs_timestamp):.6f}"
                )

                self.last_obs_timestamp = obs_timestamp

                # Set appropriate transfer state
                if obs_timestep == 0:
                    state = async_inference_pb2.TRANSFER_BEGIN
                else:
                    state = async_inference_pb2.TRANSFER_MIDDLE

                time.sleep(environment_dt)
                self.send_observation(observation, state)

            except Exception as e:
                self.logger.error(f"Error in observation sender: {e}")
                time.sleep(idle_wait)

    def control_loop_action(self):
        """Reading and performing actions in local queue"""
        self.available_actions_size.append(self.action_queue.qsize())
        if self._actions_available():
            # Get action from queue
            get_start = time.time()
            timed_action = self._get_next_action()
            get_end = time.time() - get_start

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | "
                f"Queue size: {self.action_queue.qsize()}"
            )

            self._perform_action(timed_action)

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, get_observation_fn):
        try:
            # Get serialized observation bytes from the function
            start_time = time.time()
            observation = get_observation_fn()
            obs_capture_time = time.time() - start_time

            if not hasattr(self, "last_obs_timestamp"):
                self.last_obs_timestamp = observation.get_timestamp()

            obs_timestep, obs_timestamp = observation.get_timestep(), observation.get_timestamp()
            self.last_obs_timestamp = obs_timestamp

            self.logger.info(
                f"Ts={obs_timestamp} | "
                f"Captured observation #{obs_timestep} | "
                f"1/DeltaTs (~frequency)={1 / (1e-6 + obs_timestamp - self.last_obs_timestamp):.6f}"
            )

            self.logger.debug(f"Capturing observation took {obs_capture_time:.6f}s")

            # Set appropriate transfer state
            if obs_timestep == 0:
                state = async_inference_pb2.TRANSFER_BEGIN
            else:
                state = async_inference_pb2.TRANSFER_MIDDLE

            self.send_observation(observation, state)

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, get_observation_fn):
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        control_loops = 0
        while self.running:
            control_loop_start = time.time()
            self.control_loop_action()

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation() or control_loops == 0:
                self.control_loop_observation(get_observation_fn)

            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, environment_dt - (time.time() - control_loop_start)))
            control_loops += 1


def async_client(verbose=0):
    client = RobotClient()

    if client.start():
        # Function to get observations from the robot
        def get_observation():
            observation_content = None
            observation_content = client.robot.capture_observation()

            observation = TimedObservation(
                timestamp=time.time(), observation=observation_content, timestep=max(client.latest_action, 0)
            )

            return observation

        client.logger.info("Starting all threads...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions)
        action_receiver_thread.daemon = True

        control_loop_thread = threading.Thread(target=client.control_loop, args=(get_observation,))
        control_loop_thread.daemon = True

        # Start all threads
        action_receiver_thread.start()
        control_loop_thread.start()

        try:
            while client.running:
                time.sleep(idle_wait)

        except KeyboardInterrupt:
            pass

        finally:
            client.stop()
            client.logger.info("Client stopped")

            with open("action_size.pkl", "wb") as f:
                pickle.dump(client.available_actions_size, f)


if __name__ == "__main__":
    async_client(verbose=0)
