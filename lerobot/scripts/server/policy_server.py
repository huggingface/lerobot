import pickle  # nosec
import threading
import time
from concurrent import futures
from queue import Empty, Queue
from typing import Optional

import grpc
import torch

from lerobot.common.policies.factory import get_policy_class
from lerobot.common.transport import (
    async_inference_pb2,  # type: ignore
    async_inference_pb2_grpc,  # type: ignore
)
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.constants import supported_policies
from lerobot.scripts.server.helpers import (
    FPSTracker,
    Observation,
    TimedAction,
    TimedObservation,
    TinyPolicyConfig,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
    receive_bytes_in_chunks,
)


class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self._running_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self._setup_server()

    @property
    def running(self):
        return self._running_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

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
        self.lerobot_features = policy_specs.lerobot_features

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

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, self._running_event
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.info(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._maybe_enqueue_observation(
            timed_observation
        ):  # TODO(fracapuano): check does not work on raw observsation need to transform it first
            self.logger.info(f"Observation #{obs_timestep} has been filtered out")

        return async_inference_pb2.Empty()

    def StreamActions(self, request, context):  # noqa: N802
        """Stream actions to the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            self.last_processed_obs = obs
            self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            action_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the Action
            action = async_inference_pb2.Action(data=action_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time * 1000:.2f}ms | "
                f"Serialize time: {serialize_time * 1000:.2f}ms | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            yield action

        except Empty:  # no observation added to queue in obs_queue_timeout
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

        if obs.must_go or not hasattr(self, "last_processed_obs"):
            self.logger.info(f"[MUST GO] Enqueued observation #{obs.get_timestep()} for direct processing!")
            return self._enqueue_and_go(obs)

        else:
            if self._obs_sanity_checks(obs, self.last_processed_obs):
                return self._enqueue_and_go(obs)
            else:
                return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """
        Prepare observation, ready for policy inference.
        E.g.: To keep observation sampling rate high (and network packet tiny) we send int8 [0,255] images from the
        client and then convert them to float32 [0,1] images here, before running inference.
        """
        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy.config.image_features,
            self.device,
        )
        # processed Observation - right keys, right dtype, right image shape

        return observation

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy"""
        return self.policy.predict_action_chunk(observation)

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation"""
        inference_starts = time.perf_counter()

        """1. Prepare observation"""
        start_time = time.perf_counter()
        observation = self._prepare_observation(observation_t)
        preprocessing_time = time.perf_counter() - start_time

        """2. Get action chunk"""
        start_time = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_time

        """3. Post-inference processing"""
        start_time = time.perf_counter()
        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocessing_time = time.perf_counter() - start_time
        inference_stops = time.perf_counter()

        self.logger.info(
            f"Observation {observation_t.get_timestep()} |"
            f"Inference time: {1000 * (inference_stops - inference_starts):.2f}ms"
        )

        # full-process latency breakdown for debugging purposes
        self.logger.debug(
            f"Observation {observation_t.get_timestep()} |"
            f"Preprocessing time: {1000 * (preprocessing_time - inference_starts):.2f}ms |"
            f"Inference time: {1000 * (inference_time - preprocessing_time):.2f}ms |"
            f"Postprocessing time: {1000 * (postprocessing_time - inference_time):.2f}ms |"
            f"Total time: {1000 * (postprocessing_time - inference_starts):.2f}ms"
        )

        time.sleep(
            max(0, self.config.inference_latency - max(0, time.perf_counter() - inference_starts))
        )  # sleep controls inference latency

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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{config.host}:{config.port}")

    policy_server.logger.info(f"PolicyServer started on {config.host}:{config.port}")
    server.start()

    try:
        server.wait_for_termination()

    except KeyboardInterrupt:
        policy_server.logger.info("Keyboard interrupt received")

        policy_server.stop()


if __name__ == "__main__":
    serve()  # pass a
