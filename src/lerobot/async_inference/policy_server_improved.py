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
Latency-Adaptive Asynchronous Inference Policy Server

This implementation follows the latency-adaptive async inference algorithm with:
- 2-thread architecture (observation receiver + main inference loop)
- SPSC one-slot mailbox for observation queue

Threading model (2 threads):
- Main thread: inference loop, runs policy, sends actions
- Observation receiver thread: receives observations from clients via gRPC

Example:
```shell
python -m lerobot.async_inference.policy_server_improved \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --obs_queue_timeout=2
```
"""

# ruff: noqa: E402, I001

import os as _os
import sys as _sys
import time as _time

_IMPORT_TIMING_ENABLED = _os.getenv("LEROBOT_IMPORT_TIMING", "0") == "1"
_IMPORT_T0 = _time.perf_counter() if _IMPORT_TIMING_ENABLED else 0.0

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import cv2  # type: ignore
import numpy as np
import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .constants import DEFAULT_FPS, DEFAULT_OBS_QUEUE_TIMEOUT, SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    raw_observation_to_observation,
)

if _IMPORT_TIMING_ENABLED:
    _sys.stderr.write(
        f"[import-timing] {__name__} imports: {(_time.perf_counter() - _IMPORT_T0) * 1000.0:.2f}ms\n"
    )


# =============================================================================
# Policy Server Configuration
# =============================================================================


@dataclass
class PolicyServerImprovedConfig:
    """Configuration for the improved PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    following the 2-thread model from the latency-adaptive async inference paper.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second (control frequency)"})

    # Observation queue timeout
    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT,
        metadata={"help": "Timeout for observation queue in seconds"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step in seconds."""
        return 1.0 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")


# =============================================================================
# Improved Policy Server
# =============================================================================


class PolicyServerImproved(services_pb2_grpc.AsyncInferenceServicer):
    """Latency-adaptive asynchronous inference policy server.

    This implementation follows the 2-thread model from the paper:
    - Main thread: runs the inference loop
    - Observation receiver thread: receives observations from clients via gRPC

    Thread communication uses a SPSC one-slot mailbox (Queue with maxsize=1).
    """

    prefix = "policy_server_improved"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerImprovedConfig):
        """Initialize the policy server.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS tracking for debugging
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        # SPSC one-slot mailbox for observations
        # The receiver thread is the producer, main inference thread is the consumer
        self._observation_mailbox: Queue[TimedObservation] = Queue(maxsize=1)

        # Policy components (set by SendPolicyInstructions)
        self.device: str | None = None
        self.policy_type: str | None = None
        self.lerobot_features: dict[str, Any] | None = None
        self.actions_per_chunk: int | None = None
        self.policy: Any = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

        self.logger.info("PolicyServerImproved initialized")

    @staticmethod
    def _ms(seconds: float) -> float:
        return seconds * 1000.0

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Reset server state when a new client connects."""
        self.shutdown_event.set()
        # Clear the observation mailbox
        self._observation_mailbox = Queue(maxsize=1)
        self.fps_tracker.reset()

    # -------------------------------------------------------------------------
    # gRPC Service Methods (called by receiver thread)
    # -------------------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        """Handle client ready signal. Resets server state for new session."""
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive and load policy from client instructions."""
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        t_total_start = time.perf_counter()

        # Deserialize policy configuration
        t_deser_start = time.perf_counter()
        policy_specs = pickle.loads(request.data)  # nosec
        t_deser_done = time.perf_counter()

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
            f"Pretrained: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        # Load policy
        policy_class = get_policy_class(self.policy_type)

        t_load_start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        t_load_done = time.perf_counter()

        t_to_start = time.perf_counter()
        self.policy.to(self.device)
        t_to_done = time.perf_counter()

        # Load preprocessor and postprocessor
        device_override = {"device": self.device}
        t_pp_start = time.perf_counter()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        t_pp_done = time.perf_counter()

        self.logger.info(
            "Policy loaded | from_pretrained: %.2fms | to(%s): %.2fms | processors: %.2fms | total: %.2fms",
            self._ms(t_load_done - t_load_start),
            self.device,
            self._ms(t_to_done - t_to_start),
            self._ms(t_pp_done - t_pp_start),
            self._ms(time.perf_counter() - t_total_start),
        )

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from client and enqueue for inference.

        This method is called by the gRPC receiver thread.
        """
        client_id = context.peer()
        self.logger.debug(f"Receiving observation from {client_id}")

        t_total_start = time.perf_counter()
        receive_time = time.time()

        # Receive observation bytes
        t_recv_start = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )
        t_recv_done = time.perf_counter()

        # Deserialize
        t_deser_start = time.perf_counter()
        timed_observation = pickle.loads(received_bytes)  # nosec
        t_deser_done = time.perf_counter()

        # Decode images
        t_decode_start = time.perf_counter()
        decoded_observation, decode_stats = _decode_images_from_transport(timed_observation.get_observation())
        timed_observation.observation = decoded_observation
        t_decode_done = time.perf_counter()

        if decode_stats["images_decoded"] > 0:
            self.logger.debug(
                "Decoded %s images in %.2fms | encoded=%s -> raw=%s",
                decode_stats["images_decoded"],
                self._ms(t_decode_done - t_decode_start),
                decode_stats["encoded_bytes_total"],
                decode_stats["raw_bytes_total"],
            )

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # FPS tracking
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            "Received observation #%s | Avg FPS: %.2f | One-way latency: %.2fms",
            obs_timestep,
            fps_metrics["avg_fps"],
            self._ms(receive_time - obs_timestamp),
        )

        # Enqueue observation (one-slot mailbox: overwrite if full)
        if self._observation_mailbox.full():
            try:
                _ = self._observation_mailbox.get_nowait()
                self.logger.debug("Observation mailbox was full, removed old observation")
            except Empty:
                pass

        try:
            self._observation_mailbox.put_nowait(timed_observation)
            self.logger.debug(
                "Observation #%s enqueued | recv: %.2fms | deser: %.2fms | decode: %.2fms | total: %.2fms",
                obs_timestep,
                self._ms(t_recv_done - t_recv_start),
                self._ms(t_deser_done - t_deser_start),
                self._ms(t_decode_done - t_decode_start),
                self._ms(time.perf_counter() - t_total_start),
            )
        except Exception:
            self.logger.debug(f"Failed to enqueue observation #{obs_timestep}")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Get actions from the inference queue.

        This method blocks until an observation is available, runs inference,
        and returns the action chunk.
        """
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} requesting actions")

        try:
            t_total_start = time.perf_counter()

            # Wait for observation from mailbox
            t_wait_start = time.perf_counter()
            obs = self._observation_mailbox.get(timeout=self.config.obs_queue_timeout)
            t_wait_done = time.perf_counter()

            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )
            self.logger.debug(
                "Waited %.2fms for observation",
                self._ms(t_wait_done - t_wait_start),
            )

            # Run inference
            t_infer_start = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            t_infer_done = time.perf_counter()

            # Serialize action chunk
            t_ser_start = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)
            t_ser_done = time.perf_counter()

            self.logger.info(
                "Action chunk #%s generated | inference: %.2fms | serialize: %.2fms | total: %.2fms",
                obs.get_timestep(),
                self._ms(t_infer_done - t_infer_start),
                self._ms(t_ser_done - t_ser_start),
                self._ms(time.perf_counter() - t_total_start),
            )

            return services_pb2.Actions(data=actions_bytes)

        except Empty:
            # No observation available within timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in GetActions: {e}")
            return services_pb2.Empty()

    # -------------------------------------------------------------------------
    # Inference Pipeline
    # -------------------------------------------------------------------------

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Run inference on an observation and return timestamped action chunk.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        # 1. Prepare observation
        t_prepare_start = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        t_prepare_done = time.perf_counter()

        # 2. Apply preprocessor
        t_preprocess_start = time.perf_counter()
        observation = self.preprocessor(observation)
        t_preprocess_done = time.perf_counter()

        # 3. Get action chunk from policy
        t_infer_start = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        t_infer_done = time.perf_counter()

        self.logger.debug(
            "Model timings | prepare: %.2fms | preprocess: %.2fms | inference: %.2fms | shape: %s",
            self._ms(t_prepare_done - t_prepare_start),
            self._ms(t_preprocess_done - t_preprocess_start),
            self._ms(t_infer_done - t_infer_start),
            tuple(action_tensor.shape),
        )

        # 4. Apply postprocessor to each action
        t_postprocess_start = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        processed_actions = []
        for i in range(chunk_size):
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        t_postprocess_done = time.perf_counter()

        # 5. Convert to TimedAction list
        t_time_start = time.perf_counter()
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
        )
        t_time_done = time.perf_counter()

        self.logger.debug(
            "Observation #%s pipeline | prepare: %.2fms | preprocess: %.2fms | "
            "inference: %.2fms | postprocess: %.2fms | timing: %.2fms | total: %.2fms",
            observation_t.get_timestep(),
            self._ms(t_prepare_done - t_prepare_start),
            self._ms(t_preprocess_done - t_preprocess_start),
            self._ms(t_infer_done - t_infer_start),
            self._ms(t_postprocess_done - t_postprocess_start),
            self._ms(t_time_done - t_time_start),
            self._ms(t_time_done - t_prepare_start),
        )

        return action_chunk

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get action chunk from the policy."""
        t0 = time.perf_counter()
        chunk = self.policy.predict_action_chunk(observation)
        t1 = time.perf_counter()
        self.logger.debug("Policy predict_action_chunk: %.2fms", self._ms(t1 - t0))

        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # Add batch dimension: (chunk_size, action_dim) -> (1, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _time_action_chunk(
        self,
        t_0: float,
        action_chunk: list[torch.Tensor],
        i_0: int,
    ) -> list[TimedAction]:
        """Convert action chunk to list of TimedAction with timestamps.

        Args:
            t_0: Timestamp of the source observation.
            action_chunk: List of action tensors.
            i_0: Action step of the source observation.

        Returns:
            List of TimedAction, each with timestamp and action step.
        """
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                # Convert to numpy for transport (client doesn't need torch)
                action=action.detach().cpu().numpy(),
            )
            for i, action in enumerate(action_chunk)
        ]

    def stop(self) -> None:
        """Stop the server."""
        self._reset_server()
        self.logger.info("Server stopping...")


# =============================================================================
# Image Decoding from Transport
# =============================================================================


def _decode_images_from_transport(observation: Any) -> tuple[Any, dict[str, int]]:
    """Recursively decode JPEG-marked images back into uint8 HWC3 RGB numpy arrays."""
    stats = {"images_decoded": 0, "raw_bytes_total": 0, "encoded_bytes_total": 0}

    def _maybe_decode_payload(x: Any) -> Any:
        if isinstance(x, dict) and x.get("__lerobot_image_encoding__") == "jpeg":
            data = x.get("data")
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("JPEG payload missing bytes 'data'")

            buf = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("OpenCV failed to decode JPEG payload")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stats["images_decoded"] += 1
            stats["encoded_bytes_total"] += len(data)
            stats["raw_bytes_total"] += int(rgb.nbytes)
            return rgb

        if isinstance(x, dict):
            return {k: _maybe_decode_payload(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_maybe_decode_payload(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_maybe_decode_payload(v) for v in x)
        return x

    return _maybe_decode_payload(observation), stats


# =============================================================================
# Entry Point
# =============================================================================


@draccus.wrap()
def serve_improved(cfg: PolicyServerImprovedConfig) -> None:
    """Start the improved PolicyServer."""
    logging.info(pformat(cfg.__dict__))

    # Create server instance
    policy_server = PolicyServerImproved(cfg)

    # Setup gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServerImproved started on {cfg.host}:{cfg.port}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.stop()
        server.stop(grace=5)

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve_improved()

