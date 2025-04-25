import logging
import logging.handlers
import os
import pickle  # nosec
import threading
import time
from queue import Empty, Queue
from typing import Any, Optional

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore
import grpc
import torch

from lerobot.common.robot_devices.robots.utils import make_robot

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging with both console and file output
logger = logging.getLogger("robot_client")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [CLIENT] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(console_handler)

# File handler - creates a new log file for each run
file_handler = logging.handlers.RotatingFileHandler(
    f"logs/robot_client_{int(time.time())}.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [CLIENT] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(file_handler)

environment_dt = 1 / 30
idle_wait = 0.1


class TimedData:
    def __init__(self, timestamp: float, data: Any, timestep: int):
        """Initialize a TimedData object.

        Args:
            timestamp: Unix timestamp relative to data's creation.
            data: The actual data to wrap a timestamp around.
        """
        self.timestamp = timestamp
        self.data = data
        self.timestep = timestep

    def get_data(self):
        return self.data

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


class TimedAction(TimedData):
    def __init__(self, timestamp: float, action: torch.Tensor, timestep: int):
        super().__init__(timestamp=timestamp, data=action, timestep=timestep)

    def get_action(self):
        return self.get_data()


class TimedObservation(TimedData):
    def __init__(
        self, timestamp: float, observation: dict[str, torch.Tensor], timestep: int, transfer_state: int = 0
    ):
        super().__init__(timestamp=timestamp, data=observation, timestep=timestep)
        self.transfer_state = transfer_state

    def get_observation(self):
        return self.get_data()


class TinyPolicyConfig:
    def __init__(
        self,
        policy_type: str = "act",
        pretrained_name_or_path: str = "fracapuano/act_so100_test",
        device: str = "cpu",
    ):
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.device = device


class RobotClient:
    def __init__(
        self,
        server_address="localhost:50051",
        policy_type: str = "act",  # "pi0"
        pretrained_name_or_path: str = "fracapuano/act_so100_test",  # "lerobot/pi0"
        policy_device: str = "mps",
    ):
        self.policy_config = TinyPolicyConfig(policy_type, pretrained_name_or_path, policy_device)
        self.channel = grpc.insecure_channel(server_address)
        self.stub = async_inference_pb2_grpc.AsyncInferenceStub(self.channel)
        logger.info(f"Initializing client to connect to server at {server_address}")

        self.running = False
        self.first_observation_sent = False
        self.latest_action = 0
        self.action_chunk_size = 20

        self.action_queue = Queue()
        self.start_barrier = threading.Barrier(
            3
        )  # 3 threads: observation sender, action receiver, action executor

        # Create a lock for robot access
        self.robot_lock = threading.Lock()

        # Stats for logging
        self.obs_sent_count = 0
        self.actions_received_count = 0
        self.actions_executed_count = 0
        self.last_obs_sent_time = 0
        self.last_action_received_time = 0

        start_time = time.time()
        self.robot = make_robot("so100", mock=True)
        self.robot.connect()
        connect_time = time.time()
        logger.info(f"Robot connection time: {connect_time - start_time:.4f}s")

        time.sleep(idle_wait)  # sleep waiting for cameras to activate
        logger.info("Robot connected and ready")

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
            logger.info(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = async_inference_pb2.PolicySetup(
                transfer_state=async_inference_pb2.TRANSFER_BEGIN, data=policy_config_bytes
            )

            logger.info("Sending policy instructions to policy server")
            logger.info(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.running = True

            return True

        except grpc.RpcError as e:
            logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.running = False

        self.robot.disconnect()
        logger.info("Robot disconnected")

        self.channel.close()
        logger.info("Client stopped, channel closed")

        # Log final stats
        logger.info(
            f"Session stats - Observations sent: {self.obs_sent_count}, "
            f"Action chunks received: {self.actions_received_count}, "
            f"Actions executed: {self.actions_executed_count}"
        )

    def send_observation(
        self,
        obs: TimedObservation,
        transfer_state: async_inference_pb2.TransferState = async_inference_pb2.TRANSFER_MIDDLE,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            logger.warning("Client not running")
            return False

        assert isinstance(obs, TimedObservation), "Input observation needs to be a TimedObservation!"

        start_time = time.time()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.time()
        logger.debug(f"Observation serialization time: {serialize_time - start_time:.6f}s")

        observation = async_inference_pb2.Observation(transfer_state=transfer_state, data=observation_bytes)

        try:
            send_start = time.time()
            _ = self.stub.SendObservations(iter([observation]))
            send_end = time.time()

            self.obs_sent_count += 1
            obs_timestep = obs.get_timestep()

            logger.info(
                f"Sent observation #{obs_timestep} | "
                f"Serialize time: {serialize_time - start_time:.6f}s | "
                f"Network time: {send_end - send_start:.6f}s | "
                f"Total time: {send_end - start_time:.6f}s"
            )

            if transfer_state == async_inference_pb2.TRANSFER_BEGIN:
                self.first_observation_sent = True

            self.last_obs_sent_time = send_end
            return True

        except grpc.RpcError as e:
            logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _validate_action(self, action: TimedAction):
        """Received actions are keps only when they have been produced for now or later, never before"""
        return not action.get_timestamp() < self.latest_action

    def _validate_action_chunk(self, actions: list[TimedAction]):
        assert len(actions) == self.action_chunk_size, (
            f"Action batch size must match action chunk!size: {len(actions)} != {self.action_chunk_size}"
        )
        assert all(self._validate_action(action) for action in actions), "Invalid action in chunk"

        return True

    def _inspect_action_queue(self):
        queue_size = self.action_queue.qsize()
        timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _clear_queue(self):
        """Clear the existing queue"""
        start_time = time.time()
        old_size = self.action_queue.qsize()

        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except Empty:
                break

        end_time = time.time()
        logger.debug(f"Queue cleared: {old_size} items removed in {end_time - start_time:.6f}s")

    def _fill_action_queue(self, actions: list[TimedAction]):
        """Fill the action queue with incoming valid actions"""
        start_time = time.time()
        valid_count = 0

        for action in actions:
            if self._validate_action(action):
                self.action_queue.put(action)
                valid_count += 1

        end_time = time.time()
        logger.debug(
            f"Queue filled: {valid_count}/{len(actions)} valid actions added in {end_time - start_time:.6f}s"
        )

    def _clear_and_fill_action_queue(self, actions: list[TimedAction]):
        """Clear the existing queue and fill it with new actions.
        This is a higher-level function that combines clearing and filling operations.

        Args:
            actions: List of TimedAction instances to queue
        """
        start_time = time.time()
        logger.info(f"Current latest action: {self.latest_action}")

        # Get queue state before changes
        old_size, old_timesteps = self._inspect_action_queue()

        # Log incoming actions
        incoming_timesteps = [a.get_timestep() for a in actions]
        logger.info(f"Incoming actions: {len(actions)} items with timesteps {incoming_timesteps}")

        # Clear and fill
        clear_start = time.time()
        self._clear_queue()
        clear_end = time.time()

        fill_start = time.time()
        self._fill_action_queue(actions)
        fill_end = time.time()

        # Get queue state after changes
        new_size, new_timesteps = self._inspect_action_queue()

        end_time = time.time()
        logger.info(
            f"Queue update complete | "
            f"Before: {old_size} items | "
            f"After: {new_size} items | "
            f"Previous content: {old_timesteps} | "
            f"Incoming content: {incoming_timesteps} | "
            f"Current contents: {new_timesteps}"
        )

        logger.info(
            f"Clear time: {clear_end - clear_start:.6f}s | "
            f"Fill time: {fill_end - fill_start:.6f}s | "
            f"Total time: {end_time - start_time:.6f}s"
        )

    def receive_actions(self):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                for actions_chunk in self.stub.StreamActions(async_inference_pb2.Empty()):
                    receive_time = time.time()

                    # Deserialize bytes back into list[TimedAction]
                    deserialize_start = time.time()
                    timed_actions = pickle.loads(actions_chunk.data)  # nosec
                    deserialize_end = time.time()

                    # Calculate network latency if we have matching observations
                    if len(timed_actions) > 0:
                        first_action_timestep = timed_actions[0].get_timestep()
                        server_to_client_latency = receive_time - self.last_obs_sent_time

                        logger.info(
                            f"Received action chunk for step #{first_action_timestep} | "
                            f"Network latency (server->client): {server_to_client_latency:.6f}s | "
                            f"Deserialization time: {deserialize_end - deserialize_start:.6f}s"
                        )

                    # Update action queue
                    _ = time.time()
                    self._clear_and_fill_action_queue(timed_actions)
                    update_end = time.time()

                    self.actions_received_count += 1
                    self.last_action_received_time = receive_time

                    logger.info(
                        f"Action chunk processed | "
                        f"Total processing time: {update_end - receive_time:.6f}s | "
                        f"Round-trip time since observation sent: {receive_time - self.last_obs_sent_time:.6f}s"
                    )

            except grpc.RpcError as e:
                logger.error(f"Error receiving actions: {e}")
                time.sleep(idle_wait)  # Avoid tight loop on error

    def _get_next_action(self) -> Optional[TimedAction]:
        """Get the next action from the queue"""
        try:
            action = self.action_queue.get_nowait()
            return action

        except Empty:
            return None

    def execute_actions(self):
        """Continuously execute actions from the queue"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        logger.info("Action execution thread starting")

        while self.running:
            # Get the next action from the queue
            cycle_start = time.time()
            time.sleep(environment_dt)

            get_start = time.time()
            timed_action = self._get_next_action()
            get_end = time.time()

            if timed_action is not None:
                # self.latest_action = timed_action.get_timestep()
                _ = self.latest_action
                self.latest_action = timed_action.get_timestamp()

                action_timestep = timed_action.get_timestep()

                # Convert action to tensor and send to robot - Acquire lock before accessing the robot
                lock_start = time.time()
                if self.robot_lock.acquire(timeout=1.0):  # Wait up to 1 second to acquire the lock
                    lock_acquired = time.time()
                    try:
                        send_start = time.time()
                        self.robot.send_action(timed_action.get_action())
                        send_end = time.time()

                        self.actions_executed_count += 1
                        logger.info(
                            f"Executed action #{action_timestep} | "
                            f"Queue get time: {get_end - get_start:.6f}s | "
                            f"Lock wait time: {lock_acquired - lock_start:.6f}s | "
                            f"Action send time: {send_end - send_start:.6f}s | "
                            f"Total execution time: {send_end - cycle_start:.6f}s | "
                            f"Action latency: {send_end - timed_action.get_timestamp():.6f}s | "
                            f"Queue size: {self.action_queue.qsize()}"
                        )
                    finally:
                        # Always release the lock in a finally block to ensure it's released
                        self.robot_lock.release()
                else:
                    logger.warning("Could not acquire robot lock for action execution, retrying next cycle")
            else:
                if get_end - get_start > 0.001:  # Only log if there was a measurable delay
                    logger.debug(f"No action available, get time: {get_end - get_start:.6f}s")
                time.sleep(idle_wait)

    def stream_observations(self, get_observation_fn):
        """Continuously stream observations to the server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        logger.info("Observation streaming thread starting")

        first_observation = True
        while self.running:
            try:
                # Get serialized observation bytes from the function
                cycle_start = time.time()
                time.sleep(environment_dt)

                get_start = time.time()
                observation = get_observation_fn()
                get_end = time.time()

                # Skip if observation is None (couldn't acquire lock)
                if observation is None:
                    logger.warning("Failed to get observation, skipping cycle")
                    continue

                # Set appropriate transfer state
                if first_observation:
                    state = async_inference_pb2.TRANSFER_BEGIN
                    first_observation = False
                else:
                    state = async_inference_pb2.TRANSFER_MIDDLE

                obs_timestep = observation.get_timestep()
                logger.debug(f"Got observation #{obs_timestep} in {get_end - get_start:.6f}s, sending...")

                send_start = time.time()
                self.send_observation(observation, state)
                send_end = time.time()

                logger.info(
                    f"Observation #{obs_timestep} cycle complete | "
                    f"Get time: {get_end - get_start:.6f}s | "
                    f"Send time: {send_end - send_start:.6f}s | "
                    f"Total cycle time: {send_end - cycle_start:.6f}s"
                )

            except Exception as e:
                logger.error(f"Error in observation sender: {e}")
                time.sleep(idle_wait)


def async_client():
    # Example of how to use the RobotClient
    client = RobotClient()

    if client.start():
        # Function to generate mock observations
        def get_observation():
            # Create a counter attribute if it doesn't exist
            if not hasattr(get_observation, "counter"):
                get_observation.counter = 0

            # Acquire lock before accessing the robot
            start_time = time.time()
            observation_content = None
            if client.robot_lock.acquire(timeout=1.0):  # Wait up to 1 second to acquire the lock
                lock_time = time.time()
                try:
                    capture_start = time.time()
                    observation_content = client.robot.capture_observation()
                    capture_end = time.time()
                    logger.debug(
                        f"Observation capture | "
                        f"Lock acquisition: {lock_time - start_time:.6f}s | "
                        f"Capture time: {capture_end - capture_start:.6f}s"
                    )
                finally:
                    # Always release the lock in a finally block to ensure it's released
                    client.robot_lock.release()
            else:
                logger.warning("Could not acquire robot lock for observation capture, skipping this cycle")
                return None  # Return None to indicate no observation was captured

            current_time = time.time()
            observation = TimedObservation(
                timestamp=current_time, observation=observation_content, timestep=get_observation.counter
            )

            # Increment counter for next call
            get_observation.counter += 1

            end_time = time.time()
            logger.debug(
                f"Observation #{observation.get_timestep()} prepared | "
                f"Total time: {end_time - start_time:.6f}s"
            )

            return observation

        logger.info("Starting all threads...")

        # Create and start observation sender thread
        obs_thread = threading.Thread(target=client.stream_observations, args=(get_observation,))
        obs_thread.daemon = True

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions)
        action_receiver_thread.daemon = True

        # Create action execution thread
        action_execution_thread = threading.Thread(target=client.execute_actions)
        action_execution_thread.daemon = True

        # Start all threads
        obs_thread.start()
        action_receiver_thread.start()
        action_execution_thread.start()

        try:
            # Main thread just keeps everything alive
            while client.running:
                time.sleep(idle_wait)

        except KeyboardInterrupt:
            pass

        finally:
            client.stop()
            logger.info("Client stopped")


if __name__ == "__main__":
    async_client()
