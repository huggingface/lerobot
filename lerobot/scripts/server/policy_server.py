import itertools
import logging
import logging.handlers
import os
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

from lerobot.common.policies.factory import get_policy_class
from lerobot.scripts.server.robot_client import (
    TimedAction,
    TimedObservation,
    TinyPolicyConfig,
    environment_dt,
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging with both console and file output
logger = logging.getLogger("policy_server")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [SERVER] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(console_handler)

# File handler - creates a new log file for each run
file_handler = logging.handlers.RotatingFileHandler(
    f"logs/policy_server_{int(time.time())}.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [SERVER] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(file_handler)

inference_latency = 1 / 3
idle_wait = 0.1

supported_policies = ["act"]


class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    def __init__(self):
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
        client_id = context.peer()
        logger.info(f"Client {client_id} connected and ready")
        self._setup_server()

        return async_inference_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""
        client_id = context.peer()
        logger.debug(f"Receiving policy instructions from {client_id}")

        policy_specs = pickle.loads(request.data)  # nosec
        assert isinstance(policy_specs, TinyPolicyConfig), (
            f"Policy specs must be a TinyPolicyConfig. Got {type(policy_specs)}"
        )

        logger.info(
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Device: {policy_specs.device}"
        )

        assert policy_specs.policy_type in supported_policies, (
            f"Policy type {policy_specs.policy_type} not supported. Supported policies: {supported_policies}"
        )

        self.device = policy_specs.device
        policy_class = get_policy_class(policy_specs.policy_type)

        start = time.time()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        end = time.time()

        logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return async_inference_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        logger.debug(f"Receiving observations from {client_id}")

        for observation in request_iterator:
            receive_time = time.time()
            timed_observation = pickle.loads(observation.data)  # nosec
            deserialize_time = time.time()

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(timed_observation)
            queue_time = time.time()

            obs_timestep = timed_observation.get_timestep()
            obs_timestamp = timed_observation.get_timestamp()

            logger.info(
                f"Received observation #{obs_timestep} | "
                f"Client timestamp: {obs_timestamp:.6f} | "
                f"Server timestamp: {receive_time:.6f} | "
                f"Network latency: {receive_time - obs_timestamp:.6f}s | "
                f"Deserialization time: {deserialize_time - receive_time:.6f}s | "
                f"Queue time: {queue_time - deserialize_time:.6f}s"
            )

        return async_inference_pb2.Empty()

    def StreamActions(self, request, context):  # noqa: N802
        """Stream actions to the robot client"""
        client_id = context.peer()
        logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        start_time = time.time()
        try:
            obs = self.observation_queue.get()
            get_time = time.time()
            logger.info(
                f"Running inference for observation #{obs.get_timestep()} | Queue get time: {get_time - start_time:.6f}s"
            )

            if obs:
                action = self._predict_action_chunk(obs)
                inference_end_time = time.time()
                logger.info(
                    f"Action chunk #{obs.get_timestep()} generated | "
                    f"Total inference time: {inference_end_time - get_time:.6f}s"
                )
                yield action
                yield_time = time.time()
                logger.info(
                    f"Action chunk #{obs.get_timestep()} sent | Send time: {yield_time - inference_end_time:.6f}s"
                )
            else:
                logger.warning("No observation in queue yet!")
                time.sleep(idle_wait)
        except Exception as e:
            logger.error(f"Error in StreamActions: {e}")

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
        # NOTE: This temporary function only works for ACT policies (Pi0-like models are *not* supported just yet)
        """Get an action chunk from the policy"""
        start_time = time.time()

        # prepare observation for policy forward pass
        batch = self.policy.normalize_inputs(observation)
        normalize_time = time.time()
        logger.debug(f"Observation normalization time: {normalize_time - start_time:.6f}s")

        if self.policy.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.policy.config.image_features]
            prep_time = time.time()
            logger.debug(f"Observation image preparation time: {prep_time - normalize_time:.6f}s")

        # forward pass outputs up to policy.config.n_action_steps != actions_per_chunk
        forward_start = time.time()
        actions = self.policy.model(batch)[0][:, : self.actions_per_chunk]
        forward_end = time.time()
        logger.debug(f"Policy forward pass time: {forward_end - forward_start:.6f}s")

        actions = self.policy.unnormalize_outputs({"action": actions})["action"]
        unnormalize_end = time.time()
        logger.debug(f"Action unnormalization time: {unnormalize_end - forward_end:.6f}s")

        end_time = time.time()
        logger.info(f"Action chunk generation total time: {end_time - start_time:.6f}s")

        return actions

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action based on the observation"""
        start_time = time.time()
        observation = {}
        for k, v in observation_t.get_observation().items():
            if "image" in k:
                observation[k] = v.permute(2, 0, 1).unsqueeze(0).to(self.device)
            else:
                observation[k] = v.unsqueeze(0).to(self.device)

        prep_time = time.time()
        logger.debug(f"Observation preparation time: {prep_time - start_time:.6f}s")

        # normalize observation
        observation = self.policy.normalize_inputs(observation)

        # Remove batch dimension
        action_tensor = self._get_action_chunk(observation)
        action_tensor = action_tensor.squeeze(0)

        post_inference_time = time.time()
        logger.debug(f"Post-inference processing start: {post_inference_time - prep_time:.6f}s")

        if action_tensor.dim() == 1:
            # No chunk dimension, so repeat action to create a (dummy) chunk of actions
            action_tensor = action_tensor.cpu().repeat(self.actions_per_chunk, 1)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )

        chunk_time = time.time()
        logger.debug(f"Action chunk creation time: {chunk_time - post_inference_time:.6f}s")

        action_bytes = pickle.dumps(action_chunk)  # nosec
        serialize_time = time.time()
        logger.debug(f"Action serialization time: {serialize_time - chunk_time:.6f}s")

        # Create and return the Action message
        action = async_inference_pb2.Action(transfer_state=observation_t.transfer_state, data=action_bytes)

        end_time = time.time()
        logger.info(
            f"Total action prediction time: {end_time - start_time:.6f}s | "
            f"Observation #{observation_t.get_timestep()} | "
            f"Action chunk size: {len(action_chunk)}"
        )

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
    logger.info("PolicyServer started on port 50051")

    try:
        while True:
            time.sleep(86400)  # Sleep for a day, or until interrupted
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Server stopped")


if __name__ == "__main__":
    serve()
