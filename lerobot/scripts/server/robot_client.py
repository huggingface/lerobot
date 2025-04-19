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


class RobotClient:
    def __init__(
        self,
        # cfg: RobotConfig,
        server_address="localhost:50051",
        use_robot=True,
    ):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = async_inference_pb2_grpc.AsyncInferenceStub(self.channel)

        self.running = False
        self.first_observation_sent = False
        self.latest_action = 0
        self.action_chunk_size = 20

        self.action_queue = Queue()
        self.start_barrier = threading.Barrier(3)

        # Create a lock for robot access
        self.robot_lock = threading.Lock()

        self.use_robot = use_robot
        if self.use_robot:
            self.robot = make_robot("so100")
            self.robot.connect()

            time.sleep(idle_wait)  # sleep waiting for cameras to activate
            print("Robot connected")

        self.robot_reading = True

    def timestamps(self):
        """Get the timestamps of the actions in the queue"""
        return sorted([action.get_timestep() for action in self.action_queue.queue])

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            self.stub.Ready(async_inference_pb2.Empty())
            print("Connected to policy server")

            self.running = True
            return True

        except grpc.RpcError as e:
            print(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.running = False
        if self.use_robot and hasattr(self, "robot"):
            self.robot.disconnect()
        self.channel.close()

    def send_observation(
        self,
        obs: TimedObservation,
        transfer_state: async_inference_pb2.TransferState = async_inference_pb2.TRANSFER_MIDDLE,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            print("Client not running")
            return False

        assert isinstance(obs, TimedObservation), "Input observation needs to be a TimedObservation!"

        observation_bytes = pickle.dumps(obs)
        observation = async_inference_pb2.Observation(transfer_state=transfer_state, data=observation_bytes)

        try:
            _ = self.stub.SendObservations(iter([observation]))
            if transfer_state == async_inference_pb2.TRANSFER_BEGIN:
                self.first_observation_sent = True
            return True

        except grpc.RpcError as e:
            print(f"Error sending observation: {e}")
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
        print("Queue size: ", self.action_queue.qsize())
        print("Queue contents: ", sorted([action.get_timestep() for action in self.action_queue.queue]))

    def _clear_queue(self):
        """Clear the existing queue"""
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except Empty:
                break

    def _fill_action_queue(self, actions: list[TimedAction]):
        """Fill the action queue with incoming valid actions"""
        for action in actions:
            if self._validate_action(action):
                self.action_queue.put(action)

    def _update_action_queue(self, actions: list[TimedAction]):
        """Aggregate incoming actions into the action queue.
        Raises NotImplementedError as this is not implemented yet.

        Args:
            actions: List of TimedAction instances to queue
        """
        # TODO: Implement this
        raise NotImplementedError("Not implemented")

    def _clear_and_fill_action_queue(self, actions: list[TimedAction]):
        """Clear the existing queue and fill it with new actions.
        This is a higher-level function that combines clearing and filling operations.

        Args:
            actions: List of TimedAction instances to queue
        """
        print("*** Current latest action: ", self.latest_action, "***")
        print("\t**** Current queue content ****: ")
        self._inspect_action_queue()

        print("\t*** Incoming actions ****: ")
        print([a.get_timestep() for a in actions])

        self._clear_queue()
        self._fill_action_queue(actions)

        print("\t*** Queue after clearing and filling ****: ")
        self._inspect_action_queue()

    def receive_actions(self):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        print("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                for actions_chunk in self.stub.StreamActions(async_inference_pb2.Empty()):
                    # Deserialize bytes back into list[TimedAction]
                    timed_actions = pickle.loads(actions_chunk.data)  # nosec

                    # strategy for queue composition is specified in the method
                    self._clear_and_fill_action_queue(timed_actions)

            except grpc.RpcError as e:
                print(f"Error receiving actions: {e}")
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
        print("Action execution thread starting")

        while self.running:
            # Get the next action from the queue
            time.sleep(environment_dt)
            timed_action = self._get_next_action()

            if timed_action is not None:
                # self.latest_action = timed_action.get_timestep()
                self.latest_action = timed_action.get_timestamp()

                # Convert action to tensor and send to robot
                if self.use_robot:
                    # Acquire lock before accessing the robot
                    if self.robot_lock.acquire(timeout=1.0):  # Wait up to 1 second to acquire the lock
                        try:
                            self.robot.send_action(timed_action.get_action())
                        finally:
                            # Always release the lock in a finally block to ensure it's released
                            self.robot_lock.release()
                    else:
                        print("Could not acquire robot lock for action execution, retrying next cycle")

            else:
                # No action available, wait and retry fetching from queue
                time.sleep(idle_wait)

    def stream_observations(self, get_observation_fn):
        """Continuously stream observations to the server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        print("Observation streaming thread starting")

        first_observation = True
        while self.running:
            try:
                # Get serialized observation bytes from the function
                time.sleep(environment_dt)
                observation = get_observation_fn()

                # Skip if observation is None (couldn't acquire lock)
                if observation is None:
                    continue

                # Set appropriate transfer state
                if first_observation:
                    state = async_inference_pb2.TRANSFER_BEGIN
                    first_observation = False
                else:
                    state = async_inference_pb2.TRANSFER_MIDDLE

                self.send_observation(observation, state)

            except Exception as e:
                print(f"Error in observation sender: {e}")
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
            observation_content = None
            if client.robot_lock.acquire(timeout=1.0):  # Wait up to 1 second to acquire the lock
                try:
                    observation_content = client.robot.capture_observation()
                finally:
                    # Always release the lock in a finally block to ensure it's released
                    client.robot_lock.release()
            else:
                print("Could not acquire robot lock for observation capture, skipping this cycle")
                return None  # Return None to indicate no observation was captured

            observation = TimedObservation(
                timestamp=time.time(), observation=observation_content, timestep=get_observation.counter
            )

            # Increment counter for next call
            get_observation.counter += 1

            return observation

        print("Starting all threads...")

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
            print("Client stopped")


if __name__ == "__main__":
    async_client()
