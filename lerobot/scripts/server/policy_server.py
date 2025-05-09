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

from lerobot.common.policies.factory import get_policy_class
from lerobot.scripts.server.constants import environment_dt, idle_wait, inference_latency, supported_policies
from lerobot.scripts.server.helpers import TimedAction, TimedObservation, TinyPolicyConfig, setup_logging


class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    info_bracket = "SERVER"
    logger = setup_logging(prefix, info_bracket)

    def __init__(self):
        # Initialize dataset action generator
        self.action_generator = itertools.cycle(self._stream_action_chunks_from_dataset())

        self._setup_server()

        self.actions_per_chunk = 20
        self.actions_overlap = 10

        self.running = True

    def _setup_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.observation_queue = Queue(maxsize=1)
        self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._setup_server()

        return async_inference_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving policy instructions from {client_id}")

        policy_specs = pickle.loads(request.data)  # nosec
        assert isinstance(policy_specs, TinyPolicyConfig), (
            f"Policy specs must be a TinyPolicyConfig. Got {type(policy_specs)}"
        )

        self.logger.info(
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Device: {policy_specs.device}"
        )

        assert policy_specs.policy_type in supported_policies, (
            f"Policy type {policy_specs.policy_type} not supported. Supported policies: {supported_policies}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.

        policy_class = get_policy_class(self.policy_type)

        start = time.time()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        end = time.time()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return async_inference_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        for observation in request_iterator:
            receive_time = time.time()
            timed_observation = pickle.loads(observation.data)  # nosec
            deserialize_time = time.time()

            if timed_observation.get_timestep() in self._predicted_timesteps:
                self.logger.debug(
                    f"Skipping observation #{timed_observation.get_timestep()} - Already predicted!"
                )
                continue

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(timed_observation)
            queue_time = time.time()

            obs_timestep = timed_observation.get_timestep()
            obs_timestamp = timed_observation.get_timestamp()

            if not hasattr(self, "previous_obs_timestamp"):
                self.previous_obs_timestamp = obs_timestamp

            self.logger.info(
                f"Received observation #{obs_timestep} | "
                f"Client timestamp: {obs_timestamp:.6f} | "
                f"Server timestamp: {receive_time:.6f} | "
            )

            self.logger.debug(
                f"1/DeltaObsT (~frequency): {1 / (1e-6 + obs_timestamp - self.previous_obs_timestamp):.6f} Hz| "
                f"Network latency: {receive_time - obs_timestamp:.6f}s | "
                f"Deserialization time: {deserialize_time - receive_time:.6f}s | "
                f"Queue time: {queue_time - deserialize_time:.6f}s | "
            )

            self.previous_obs_timestamp = obs_timestamp

        return async_inference_pb2.Empty()

    def StreamActions(self, request, context):  # noqa: N802
        """Stream actions to the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            obs = self.observation_queue.get()
            self.logger.info(f"Running inference for observation #{obs.get_timestep()}")

            if obs:
                self._predicted_timesteps.add(obs.get_timestep())
                start_time = time.time()
                action_chunk = self._predict_action_chunk(obs)
                # action_chunk = self._read_action_chunk(obs)
                inference_time = time.time() - start_time

                start_time = time.time()
                action_bytes = pickle.dumps(action_chunk)  # nosec
                serialize_time = time.time() - start_time

                # Create and return the Action
                action = async_inference_pb2.Action(transfer_state=obs.transfer_state, data=action_bytes)

                self.logger.info(
                    f"Action chunk #{obs.get_timestep()} generated | Inference time: {inference_time:.6f}s |"
                )

                self.logger.debug(
                    f"Action chunk #{obs.get_timestep()} generated | "
                    f"Inference time: {inference_time:.6f}s |"
                    f"Serialize time: {serialize_time:.6f}s |"
                    f"Total time: {inference_time + serialize_time:.6f}s"
                )

                yield action
            else:
                self.logger.warning("No observation in queue yet!")
                time.sleep(idle_wait)
        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

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
    def _run_act_policy(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run ACT-like policies"""
        start_time = time.time()

        # prepare observation for policy forward pass
        batch = self.policy.normalize_inputs(observation)
        normalize_time = time.time()
        self.logger.debug(f"Observation normalization time: {normalize_time - start_time:.6f}s")

        if self.policy.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.policy.config.image_features]
            prep_time = time.time()
            self.logger.debug(f"Observation image preparation time: {prep_time - normalize_time:.6f}s")

        # forward pass outputs up to policy.config.n_action_steps != actions_per_chunk
        actions = self.policy.model(batch)[0][:, : self.actions_per_chunk]

        actions = self.policy.unnormalize_outputs({"action": actions})["action"]

        end_time = time.time()
        self.logger.info(f"[ACT] Action chunk generation total time: {end_time - start_time:.6f}s")

        return actions

    @torch.no_grad()
    def _run_pi0_policy(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run PI0-like policies"""
        raise NotImplementedError("PI0 policy not implemented yet")

    def _get_action_chunk(
        self, observation: dict[str, torch.Tensor], policy_type: str = "act"
    ) -> torch.Tensor:
        """Get an action chunk from the policy"""
        if policy_type == "act":
            return self._run_act_policy(observation)
        else:
            raise ValueError(f"Policy class {policy_type} not supported")

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action based on the observation"""
        """1. Prepare observation"""
        start_time = time.time()

        observation = {}
        for k, v in observation_t.get_observation().items():
            if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions
                if "image" in k:
                    # Add batch dimension first, then reorder to NCHW format, then normalize to [0, 1]
                    observation[k] = (
                        v.unsqueeze(0).permute(0, 3, 1, 2).to(self.device, non_blocking=True) / 255
                    )
                else:
                    observation[k] = v.unsqueeze(0).to(self.device, non_blocking=True)
            else:
                observation[k] = v  # textual instructions are passed as a list of strings

        prep_time = time.time()
        self.logger.debug(f"Observation preparation time: {prep_time - start_time:.6f}s")

        """2. Get action chunk"""
        action_tensor = self._get_action_chunk(observation, self.policy_type)
        action_tensor = action_tensor.squeeze(0)

        # Move to CPU before serializing
        action_tensor = action_tensor.cpu()

        post_inference_time = time.time()
        self.logger.debug(f"Post-inference processing start: {post_inference_time - prep_time:.6f}s")

        if action_tensor.dim() == 1:
            # No chunk dimension, so repeat action to create a (dummy) chunk of actions
            action_tensor = action_tensor.repeat(self.actions_per_chunk, 1)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )

        chunk_time = time.time()
        self.logger.debug(f"Action chunk creation time: {chunk_time - post_inference_time:.6f}s")
        time.sleep(
            max(0, inference_latency - max(0, chunk_time - start_time))
        )  # sleep to control inference latency

        return action_chunk

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

    def _read_action_chunk(self, observation: Optional[TimedObservation] = None) -> list[TimedAction]:
        """Dummy function for predicting action chunk given observation.

        Instead of computing actions on-the-fly, this method streams
        actions from a prerecorded dataset.
        """
        import warnings

        warnings.warn(
            "This method is deprecated and will be removed in the future.", DeprecationWarning, stacklevel=2
        )

        start_time = time.time()
        if not observation:
            observation = TimedObservation(timestamp=time.time(), observation={}, timestep=0)

        # Get chunk of actions from the generator
        actions_chunk = next(self.action_generator)

        # Return a list of TimedActions, with timestamps starting from the observation timestamp
        actions_chunk = self._time_action_chunk(
            observation.get_timestamp(), actions_chunk, observation.get_timestep()
        )

        chunk_time = time.time()
        self.logger.debug(f"Action chunk creation time: {chunk_time - start_time:.6f}s")

        # slow action generation, emulates inference time
        time.sleep(max(0, inference_latency - max(0, chunk_time - start_time)))

        return actions_chunk

    def stop(self):
        """Stop the server"""
        self.running = False
        self.logger.info("Server stopping...")


def serve():
    port = 8080
    # Create the server instance first
    policy_server = PolicyServer()

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    policy_server.logger.info(f"PolicyServer started on port {port}")

    try:
        # Use the running attribute to control server lifetime
        while policy_server.running:
            time.sleep(1)  # Check every second instead of sleeping indefinitely
    except KeyboardInterrupt:
        policy_server.stop()
        policy_server.logger.info("Keyboard interrupt received")


if __name__ == "__main__":
    serve()
