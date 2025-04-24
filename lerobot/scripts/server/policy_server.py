import itertools
import pickle  # nosec
import time
from concurrent import futures
from queue import Queue
from typing import Generator, List, Optional

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore
import grpc
import torch
from datasets import load_dataset

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.server.robot_client import TimedAction, TimedObservation, environment_dt

inference_latency = 1 / 3
idle_wait = 0.1


class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    def __init__(self):
        # TODO: Add device specification for policy inference at init
        self.device = "cpu"
        start = time.time()
        self.policy = ACTPolicy.from_pretrained("fracapuano/act_so100_test")
        self.policy.to(self.device)
        end = time.time()
        print(f"Time taken to put policy on {self.device}: {end - start} seconds")

        # Initialize dataset action generator
        self.action_generator = itertools.cycle(self._stream_action_chunks_from_dataset())

        self._setup_server()

        self.actions_per_chunk = 20
        self.actions_overlap = 10

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
        # client_id = context.peer()
        # print(f"Receiving observations from {client_id}")

        for observation in request_iterator:
            timed_observation = pickle.loads(observation.data)  # nosec

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(timed_observation)
            print("Received observation no: ", timed_observation.get_timestep())

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

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(t_0 + i * environment_dt, action, i_0 + i) for i, action in enumerate(action_chunk)
        ]

    @torch.no_grad()
    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy"""
        start_time = time.time()

        # prepare observation for policy forward pass
        batch = self.policy.normalize_inputs(observation)
        if self.policy.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.policy.config.image_features]

        # forward pass outputs up to policy.config.n_action_steps != actions_per_chunk
        actions = self.policy.model(batch)[0][:, : self.actions_per_chunk]
        actions = self.policy.unnormalize_outputs({"action": actions})["action"]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Action chunk generation time: {elapsed_time:.6f} seconds")

        return actions

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action based on the observation"""
        observation = {}
        for k, v in observation_t.get_observation().items():
            if "image" in k:
                observation[k] = v.permute(2, 0, 1).unsqueeze(0).to(self.device)
            else:
                observation[k] = v.unsqueeze(0).to(self.device)

        # normalize observation
        observation = self.policy.normalize_inputs(observation)

        # Remove batch dimension
        action_tensor = self._get_action_chunk(observation)
        action_tensor = action_tensor.squeeze(0)

        if action_tensor.dim() == 1:
            # No chunk dimension, so repeat action to create a (dummy) chunk of actions
            action_tensor = action_tensor.cpu().repeat(self.actions_per_chunk, 1)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )

        action_bytes = pickle.dumps(action_chunk)  # nosec
        # Create and return the Action message
        action = async_inference_pb2.Action(transfer_state=observation_t.transfer_state, data=action_bytes)

        # time.sleep(inference_latency)  # slow action generation, emulates inference time (ACT is very fast)

        return action

    def _stream_action_chunks_from_dataset(self) -> Generator[List[torch.Tensor], None, None]:
        """Stream chunks of actions from a prerecorded dataset.

        Returns:
            Generator that yields chunks of actions from the dataset
        """
        dataset = load_dataset("fracapuano/so100_test", split="train").with_format("torch")

        # 1. Select the action column only, where you will find tensors with 6 elements
        actions = dataset["action"]
        action_indices = torch.arange(len(actions))

        # 2. Chunk the iterable of tensors into chunks with 10 elements each
        # sending only first element for debugging
        indices_chunks = action_indices.unfold(
            0, self.actions_per_chunk, self.actions_per_chunk - self.actions_overlap
        )

        for idx_chunk in indices_chunks:
            yield actions[idx_chunk[0] : idx_chunk[-1] + 1, :]

    def _read_action_chunk(self, observation: Optional[TimedObservation] = None):
        """Dummy function for predicting action chunk given observation.

        Instead of computing actions on-the-fly, this method streams
        actions from a prerecorded dataset.
        """
        import warnings

        warnings.warn(
            "This method is deprecated and will be removed in the future.", DeprecationWarning, stacklevel=2
        )

        if not observation:
            observation = TimedObservation(timestamp=time.time(), observation={}, timestep=0)
            transfer_state = 0
        else:
            transfer_state = observation.transfer_state

        # Get chunk of actions from the generator
        actions_chunk = next(self.action_generator)

        # Return a list of TimedActions, with timestamps starting from the observation timestamp
        action_data = self._time_action_chunk(
            observation.get_timestamp(), actions_chunk, observation.get_timestep()
        )
        action_bytes = pickle.dumps(action_data)  # nosec

        # Create and return the Action message
        action = async_inference_pb2.Action(transfer_state=transfer_state, data=action_bytes)

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
