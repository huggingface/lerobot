import itertools
import time
from concurrent import futures
from queue import Queue
from typing import Generator, List

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore
import grpc
import numpy as np
import torch
from datasets import load_dataset

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.scripts.server.robot_client import TimedObservation

inference_latency = 1 / 3
idle_wait = 0.1


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, policy: PreTrainedPolicy = None):
        # TODO: Add code for loading and using policy for inference
        self.policy = policy

        # TODO: Add device specification for policy inference at init
        # Initialize dataset action generator
        self.action_generator = itertools.cycle(self._stream_action_chunks_from_dataset())

        self._setup_server()

    def _setup_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.observation_queue = Queue(maxsize=1)

    def Ready(self, request, context):  # noqa: N802
        self._setup_server()
        print("Client connected and ready")

        return async_inference_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        print(f"Receiving observations from {client_id}")
        # print("Number of observations in queue: ", self.observation_queue.qsize())

        for observation in request_iterator:
            # Increment observation timestep counter for each new observation
            observation_data = np.frombuffer(observation.data, dtype=np.float32)
            observation_timestep = observation_data[0]
            observation_content = observation_data[1:]

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(
                TimedObservation(
                    timestep=int(observation_timestep),
                    observation=observation_content,
                    transfer_state=observation.transfer_state,
                )
            )
            print("Received observation no: ", observation_timestep)

        return async_inference_pb2.Empty()

    def StreamActions(self, request, context):  # noqa: N802
        """Stream actions to the robot client"""
        # client_id = context.peer()
        # print(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        obs = self.observation_queue.get()
        print("Running inference for timestep: ", obs.get_timestep())

        if obs:
            yield self._predict_action_chunk(obs)

        else:
            print("No observation in queue yet!")
            time.sleep(idle_wait)

        return async_inference_pb2.Empty()

    def _predict_and_queue_action(self, observation):
        """Predict an action based on the observation"""
        # TODO: Implement the logic to predict an action based on the observation
        """
        Ideally, action-prediction should be general and not specific to the policy used.
        That is, this interface should be the same for ACT/VLA/RL-based etc.
        """
        # TODO: Queue the action to be sent to the robot client
        raise NotImplementedError("Not implemented")

    def _stream_action_chunks_from_dataset(self) -> Generator[List[torch.Tensor], None, None]:
        """Stream chunks of actions from a prerecorded dataset.

        Returns:
            Generator that yields chunks of actions from the dataset
        """
        dataset = load_dataset("fracapuano/so100_test", split="train").with_format("torch")

        # 1. Select the action column only, where you will find tensors with 6 elements
        actions = dataset["action"]
        action_indices = torch.arange(len(actions))

        actions_per_chunk = 20
        actions_overlap = 10

        # 2. Chunk the iterable of tensors into chunks with 10 elements each
        # sending only first element for debugging
        indices_chunks = action_indices.unfold(0, actions_per_chunk, actions_per_chunk - actions_overlap)

        for idx_chunk in indices_chunks:
            yield actions[idx_chunk[0] : idx_chunk[-1] + 1, :]

        # Non overlapping action chunks
        # actions_chunks = torch.split(actions, 20)
        # for action_chunk in actions_chunks:
        #     yield action_chunk

    def _predict_action_chunk(self, observation: TimedObservation):
        """Dummy function for predicting action chunk given observation.

        Instead of computing actions on-the-fly, this method streams
        actions from a prerecorded dataset.
        """
        transfer_state = 0 if not observation else observation.transfer_state

        # Get chunk of actions from the generator
        actions_chunk = next(self.action_generator)

        # Convert the chunk of actions to a single contiguous numpy array
        # For the so100 dataset, each action in the chunk is a tensor with 6 elements
        actions_array = actions_chunk.numpy()

        # Create timesteps starting from the observation timestep
        # Each action in the chunk gets a timestep starting from observation_timestep
        # This indicates that the first action corresponds to the current observation,
        # and subsequent actions are for future timesteps (and predicted observations!)

        timesteps = (
            np.arange(observation.timestep, observation.timestep + len(actions_array))
            .reshape(-1, 1)
            .astype(np.float32)
        )

        # Create a combined array with timesteps and actions
        # First column is the timestep, remaining columns are the action values
        combined_array = np.hstack((timesteps, actions_array))

        # Convert the numpy array to bytes for transmission
        action_data = combined_array.astype(np.float32).tobytes()

        # Create and return the Action message
        action = async_inference_pb2.Action(transfer_state=transfer_state, data=action_data)

        time.sleep(inference_latency)  # slow action generation, emulates inference time

        return action


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(PolicyServer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("PolicyServer started on port 50051")

    try:
        while True:
            time.sleep(86400)  # Sleep for a day, or until interrupted
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped")


if __name__ == "__main__":
    serve()
