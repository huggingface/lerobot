# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Examples:

1. gRPC over TCP (default):
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --transport=grpc \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```

2. ZeroMQ over TCP:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --transport=zmq \
     --fps=30
```

3. ZeroMQ over IPC (Unix Domain Socket - optimal for local hardware running both robot & server):
```shell
python -m lerobot.async_inference.policy_server \
     --host=ipc:///tmp/policy_server.sock \
     --transport=zmq \
     --fps=30
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.lerobot_types import PolicyAction
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def _handle_ready(self, client_id: str) -> None:
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self._handle_ready(client_id)
        return services_pb2.Empty()

    def _handle_policy_instructions(self, policy_specs: RemotePolicyConfig, client_id: str) -> None:
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""
        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec
        self._handle_policy_instructions(policy_specs, client_id)
        return services_pb2.Empty()

    def _handle_send_observation(self, timed_observation: TimedObservation, client_id: str) -> None:
        receive_time = time.time()  # comparing timestamps so need time.time()
        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        if not self._enqueue_observation(timed_observation):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        self._handle_send_observation(timed_observation, client_id)

        return services_pb2.Empty()

    def _handle_get_actions(self, client_id: str, block: bool = True) -> bytes:
        try:
            getactions_starts = time.perf_counter()
            if block:
                obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            else:
                obs = self.observation_queue.get_nowait()
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions_bytes

        except Empty:  # no observation added to queue in obs_queue_timeout
            return b""

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")
            return b""

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")
        actions_bytes = self._handle_get_actions(client_id)
        if not actions_bytes:
            return services_pb2.Empty()
        return services_pb2.Actions(data=actions_bytes)

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

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

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


def serve_zmq(cfg: PolicyServerConfig, policy_server: PolicyServer):
    """Start the PolicyServer using ZeroMQ transport."""
    import zmq
    from lerobot.utils.import_utils import require_package

    require_package("pyzmq", extra="async", import_name="zmq")

    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)

    if cfg.host.startswith("tcp://") or cfg.host.startswith("ipc://") or cfg.host.startswith("inproc://"):
        bind_address = cfg.host
    else:
        bind_address = f"tcp://{cfg.host}:{cfg.port}"

    socket.bind(bind_address)
    policy_server.logger.info(f"PolicyServer (ZeroMQ) bound to {bind_address}")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    try:
        while policy_server.running:
            events = dict(poller.poll(timeout=200))
            if socket in events:
                frames = socket.recv_multipart()
                if not frames:
                    continue

                client_id_bytes = frames[0]
                client_id = client_id_bytes.hex()

                idx = 1
                if idx < len(frames) and frames[idx] == b"":
                    idx += 1

                if idx >= len(frames):
                    continue

                cmd = frames[idx]
                payload = frames[idx + 1] if len(frames) > idx + 1 else b""

                if cmd == b"ready":
                    policy_server._handle_ready(client_id)
                    socket.send_multipart([client_id_bytes, b"", b"ok"])

                elif cmd == b"send_policy_instructions":
                    policy_specs = pickle.loads(payload)  # nosec
                    policy_server._handle_policy_instructions(policy_specs, client_id)
                    socket.send_multipart([client_id_bytes, b"", b"ok"])

                elif cmd == b"send_observation":
                    timed_obs = pickle.loads(payload)  # nosec
                    policy_server._handle_send_observation(timed_obs, client_id)
                    socket.send_multipart([client_id_bytes, b"", b"ok"])

                elif cmd == b"get_actions":
                    actions_bytes = policy_server._handle_get_actions(client_id, block=False)
                    socket.send_multipart([client_id_bytes, b"", actions_bytes])

                else:
                    policy_server.logger.warning(f"Unknown ZMQ command: {cmd}")
                    socket.send_multipart([client_id_bytes, b"", b"error: unknown command"])

    finally:
        try:
            poller.unregister(socket)
        except Exception:
            pass
        socket.close(linger=0)
        policy_server.logger.info("ZeroMQ PolicyServer socket closed")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    if cfg.transport == "zmq":
        serve_zmq(cfg, policy_server)
    else:
        # Setup and start gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
        server.add_insecure_port(f"{cfg.host}:{cfg.port}")

        policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
        server.start()

        server.wait_for_termination()

        policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
