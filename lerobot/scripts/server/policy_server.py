import pickle  # nosec
import threading
import time
from concurrent import futures
from queue import Empty, Queue
from typing import Optional

import grpc
import torch

from lerobot.common.policies.factory import get_policy_class
from lerobot.scripts.server import (
    async_inference_pb2,  # type: ignore
    async_inference_pb2_grpc,  # type: ignore
)
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.constants import supported_policies
from lerobot.scripts.server.helpers import (
    TimedAction,
    TimedObservation,
    TinyPolicyConfig,
    observations_similar,
    setup_logging,
)


class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    info_bracket = "SERVER"
    logger = setup_logging(prefix, info_bracket)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self._running_event = threading.Event()

        self._setup_server()

    @property
    def running(self):
        return self._running_event.is_set()

    def _setup_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.observation_queue = Queue(maxsize=1)
        self._predicted_timesteps = set()
        self._predicted_observations = Queue(maxsize=1)

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._setup_server()  # new client's handshake clears server state

        self._running_event.set()

        return async_inference_pb2.Empty()

    def _validate_policy_specs(self, policy_specs: TinyPolicyConfig) -> None:
        assert isinstance(policy_specs, TinyPolicyConfig), (
            f"Policy specs must be a TinyPolicyConfig. Got {type(policy_specs)}"
        )
        assert policy_specs.policy_type in supported_policies, (
            f"Policy type {policy_specs.policy_type} not supported. Supported policies: {supported_policies}"
        )

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving policy instructions from {client_id}")

        policy_specs = pickle.loads(request.data)  # nosec
        self._validate_policy_specs(policy_specs)

        self.logger.info(
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return async_inference_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        for observation in request_iterator:
            receive_time = time.perf_counter()
            timed_observation = pickle.loads(observation.data)  # nosec
            deserialize_time = time.perf_counter()

            self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

            if not self._maybe_enqueue_observation(timed_observation):
                continue

            queue_time = time.perf_counter()

            obs_timestep = timed_observation.get_timestep()
            obs_timestamp = timed_observation.get_timestamp()

            self.logger.info(
                f"Received observation #{obs_timestep} | "
                f"Client timestamp: {obs_timestamp:.6f} | "
                f"Server timestamp: {receive_time:.6f} | "
            )

            if not hasattr(self, "previous_obs_timestamp"):
                self.previous_obs_timestamp = obs_timestamp

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
            obs = self.observation_queue.get(timeout=2)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            if obs:
                self.last_predicted_obs = obs
                self._predicted_timesteps.add(obs.get_timestep())
                start_time = time.perf_counter()
                action_chunk = self._predict_action_chunk(obs)
                inference_time = time.perf_counter() - start_time

                start_time = time.perf_counter()
                action_bytes = pickle.dumps(action_chunk)  # nosec
                serialize_time = time.perf_counter() - start_time

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
                time.sleep(self.config.idle_wait)

        except Empty:
            self.logger.warning("No observation in queue!")

            return async_inference_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return async_inference_pb2.Empty()

    def _enqueue_and_go(self, obs: TimedObservation):
        # If queue is full, get the old observation to make room
        if self.observation_queue.full():
            # pops from queue
            _ = self.observation_queue.get_nowait()
            self.logger.debug("Observation queue was full, removed oldest observation")

        # Now put the new observation (never blocks as queue is non-full here)
        self.observation_queue.put(obs)
        return True

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        if obs.get_timestep() in self._predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, atol=1):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _maybe_enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if obs.must_go or not hasattr(self, "last_predicted_obs"):
            self.logger.info(f"[MUST GO] Enqueued observation #{obs.get_timestep()} for direct processing!")
            return self._enqueue_and_go(obs)

        else:
            if self._obs_sanity_checks(obs, self.last_predicted_obs):
                return self._enqueue_and_go(obs)
            else:
                return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(t_0 + i * self.config.environment_dt, action, i_0 + i)
            for i, action in enumerate(action_chunk)
        ]

    @torch.no_grad()
    def _run_act_policy(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run ACT-like policies"""
        start_time = time.perf_counter()

        # prepare observation for policy forward pass
        batch = self.policy.normalize_inputs(observation)
        normalize_time = time.perf_counter()
        self.logger.debug(f"Observation normalization time: {normalize_time - start_time:.6f}s")

        if self.policy.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.policy.config.image_features]
            prep_time = time.perf_counter()
            self.logger.debug(f"Observation image preparation time: {prep_time - normalize_time:.6f}s")

        # forward pass outputs up to policy.config.n_action_steps != actions_per_chunk
        actions = self.policy.model(batch)[0][:, : self.config.actions_per_chunk]

        actions = self.policy.unnormalize_outputs({"action": actions})["action"]

        end_time = time.perf_counter()
        self.logger.info(f"[ACT] Action chunk generation total time: {end_time - start_time:.6f}s")

        return actions

    @torch.no_grad()
    def _run_pi0_policy(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run PI0-like policies"""
        raise NotImplementedError("PI0 policy not implemented yet")

    @torch.no_grad()
    def _run_smolvla_policy(
        self, observation: dict[str, torch.Tensor], noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run smolvla-like policies"""
        observation = self.policy.normalize_inputs(observation)

        images, img_masks = self.policy.prepare_images(observation)
        state = self.policy.prepare_state(observation)
        lang_tokens, lang_masks = self.policy.prepare_language(observation)

        actions = self.policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.policy.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.policy.unnormalize_outputs(
            {"action": actions, "robot_type": [self.policy.config.robot_type]}
        )["action"]

        return actions

    def _get_action_chunk(
        self, observation: dict[str, torch.Tensor], policy_type: str = "act"
    ) -> torch.Tensor:
        """Get an action chunk from the policy"""
        if policy_type == "act":
            return self._run_act_policy(observation)
        elif policy_type == "smolvla":
            return self._run_smolvla_policy(observation)
        else:
            raise ValueError(f"Policy class {policy_type} not supported")

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action based on the observation"""
        """1. Prepare observation"""
        start_time = time.perf_counter()

        observation = {
            "robot_type": [self.policy.config.robot_type],
        }
        for k, v in observation_t.get_observation().items():
            if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions
                if "image" in k:
                    # Add batch dimension first, then reorder to NCHW format, then normalize to [0, 1]
                    observation[k] = (
                        v.unsqueeze(0).permute(0, 3, 1, 2).to(self.device, non_blocking=True) / 255.0
                    )
                else:
                    observation[k] = v.unsqueeze(0).to(self.device, non_blocking=True)
            else:
                observation[k] = v  # textual instructions are passed as a list of strings

        prep_time = time.perf_counter()
        self.logger.debug(f"Observation preparation time: {prep_time - start_time:.6f}s")

        """2. Get action chunk"""
        action_tensor = self._get_action_chunk(observation, self.policy_type)

        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)

        post_inference_time = time.perf_counter()
        self.logger.debug(f"Post-inference processing start: {post_inference_time - prep_time:.6f}s")

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )

        chunk_time = time.perf_counter()
        self.logger.debug(f"Action chunk creation time: {chunk_time - post_inference_time:.6f}s")

        time.sleep(
            max(0, self.config.inference_latency - max(0, chunk_time - start_time))
        )  # sleep to control inference latency

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._running_event.clear()
        self.logger.info("Server stopping...")


def serve(config: Optional[PolicyServerConfig] = None, host: str = "localhost", port: int = 8080):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    if config is None:
        config = PolicyServerConfig(host=host, port=port)

    # Create the server instance first
    policy_server = PolicyServer(config)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{config.host}:{config.port}")

    policy_server.logger.info(f"PolicyServer started on {config.host}:{config.port}")
    server.start()

    try:
        # Use the running event to control server lifetime
        while policy_server.running:
            time.sleep(1.0)  # Check every second

    except KeyboardInterrupt:
        policy_server.logger.info("Keyboard interrupt received")

        policy_server.stop()
        server.stop(grace=None)


if __name__ == "__main__":
    serve()  # use default network configuration
