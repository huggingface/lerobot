import threading
import time
from queue import Empty, Queue
from typing import Optional

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore
import grpc
import numpy as np
import torch

from lerobot.common.robot_devices.robots.utils import make_robot

environment_dt = 1 / 30
idle_wait = 0.1


class TimedData:
    def __init__(self, timestep: int, data: np.ndarray):
        self.timestep = timestep
        self.data = data

    def get_data(self):
        return self.data

    def get_timestep(self):
        return self.timestep


class TimedAction(TimedData):
    def __init__(self, timestep: int, action: np.ndarray):
        super().__init__(timestep, action)

    def get_action(self):
        return self.get_data()


class TimedObservation(TimedData):
    def __init__(self, timestep: int, observation: np.ndarray, transfer_state: int = 0):
        super().__init__(timestep, observation)
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
        self.start_barrier = threading.Barrier(3)  # Barrier for 3 threads

        self.observation_timestep = 0

        self.use_robot = use_robot
        if self.use_robot:
            self.robot = make_robot("so100")
            self.robot.connect()

            time.sleep(idle_wait)  # sleep waiting for cameras to activate
            print("Robot connected")

    def timesteps(self):
        """Get the timesteps of the actions in the queue"""
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
        self.channel.close()

    def send_observation(
        self,
        observation_data: np.ndarray,
        transfer_state: async_inference_pb2.TransferState = async_inference_pb2.TRANSFER_MIDDLE,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            print("Client not running")
            return False

        # Convert observation data to bytes
        observation_data = observation_data.tobytes()

        observation = async_inference_pb2.Observation(transfer_state=transfer_state, data=observation_data)

        try:
            _ = self.stub.SendObservations(iter([observation]))
            if transfer_state == async_inference_pb2.TRANSFER_BEGIN:
                self.first_observation_sent = True
            return True

        except grpc.RpcError as e:
            print(f"Error sending observation: {e}")
            return False

    def _validate_action(self, action: np.ndarray):
        """Validate the action"""
        assert action.shape == (7,), f"Action shape must be (7,) (including timestep), got {action.shape}"

        return True

    def _validate_action_chunk(self, actions: list[np.ndarray]):
        """Validate the action chunk"""
        assert len(actions) == self.action_chunk_size, (
            f"Action batch size must match action chunk!size: {len(actions)} != {self.action_chunk_size}"
        )

        assert all(self._validate_action(action) for action in actions), "Invalid action in chunk"

        return True

    def _inspect_action_queue(self):
        """Inspect the action queue"""
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
        """Fill the action queue with incoming actions"""
        for action in actions:
            # Only keep the actions that are newer than the latest action
            if action.get_timestep() <= self.latest_action:
                continue

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
        print("\t**** Current queue content ****: ")
        self._inspect_action_queue()

        print("\t*** Incoming actions ****: ")
        print([a.get_timestep() for a in actions])

        self._clear_queue()
        self._fill_action_queue(actions)

        print("\t*** Queue after clearing and filling ****: ")
        self._inspect_action_queue()

    def _create_timed_actions(self, action_data: np.ndarray) -> list[TimedAction]:
        """Create TimedAction instances from raw action data.

        Args:
            action_data: Numpy array of shape (chunk_size, 7) where first column
                        is timestep and remaining columns are action values.

        Returns:
            List of TimedAction instances
        """
        timed_actions = []
        for action in action_data:
            timestep = int(action[0])  # First element is the timestep
            action_values = action[1:]  # Remaining elements are the action
            timed_actions.append(TimedAction(timestep, action_values))

        return timed_actions

    def receive_actions(self):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        print("Action receiving thread starting")

        print(self.timesteps())

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                action_chunks_counter = 0
                for action in self.stub.StreamActions(async_inference_pb2.Empty()):
                    # Read the action data which includes timesteps
                    # Shape is (chunk_size, 7) where first column is timestep
                    action_data = np.frombuffer(action.data, dtype=np.float32).reshape(
                        self.action_chunk_size, 7
                    )

                    print("*** Receiving actions ****: ")
                    # Convert raw action data to TimedAction instances
                    timed_actions = self._create_timed_actions(action_data)

                    # strategy for queue composition is specified in the method
                    self._clear_and_fill_action_queue(timed_actions)

                    action_chunks_counter += 1

                if action_chunks_counter > 2:
                    raise ValueError("Too many action chunks received")

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
            timed_action = self._get_next_action()

            if timed_action is not None:
                self.latest_action = timed_action.get_timestep()

                # Convert action to tensor and send to robot
                if self.use_robot:
                    self.robot.send_action(torch.tensor(timed_action.get_action()))

                time.sleep(environment_dt)

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
                observation = get_observation_fn()

                # Set appropriate transfer state
                if first_observation:
                    state = async_inference_pb2.TRANSFER_BEGIN
                    first_observation = False
                else:
                    state = async_inference_pb2.TRANSFER_MIDDLE

                # Build timestep element in observation
                # observation = np.hstack(
                #     (np.array([self.latest_action]), observation)
                # ).astype(np.float32)

                time.sleep(environment_dt)  # Control the observation sending rate
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

            # Create observation with incrementing first element
            observation = np.array([get_observation.counter, 0, 0], dtype=np.float32)

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
