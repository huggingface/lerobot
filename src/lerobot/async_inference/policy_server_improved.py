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
from contextlib import suppress
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

    # Low-jitter actions delivery
    actions_dense_enabled: bool = field(
        default=True,
        metadata={"help": "Produce and serve dense action chunks (lower jitter than pickled TimedAction list)"},
    )
    actions_stream_enabled: bool = field(
        default=True,
        metadata={"help": "Enable server-streaming of actions (lower jitter than client polling)"},
    )
    actions_wait_timeout_s: float = field(
        default=1.0,
        metadata={"help": "Max time GetActionsDense/GetActions waits for the first available actions"},
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
        if self.actions_wait_timeout_s <= 0:
            raise ValueError(
                f"actions_wait_timeout_s must be positive, got {self.actions_wait_timeout_s}"
            )


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

        # Latest produced actions (one-slot 'mailbox' semantics, but kept as latest+seq for streaming)
        self._actions_lock = threading.Lock()
        self._actions_cv = threading.Condition(self._actions_lock)
        self._actions_seq: int = 0
        self._latest_actions_dense: services_pb2.ActionsDense | None = None

        self._policy_ready = threading.Event()
        self._producer_thread: threading.Thread | None = None

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
        self._policy_ready.clear()
        with self._actions_cv:
            self._actions_seq = 0
            self._latest_actions_dense = None

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
        policy_specs = pickle.loads(request.data)  # nosec

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

        self._policy_ready.set()
        # Start producer thread (if needed) to generate actions outside the RPC path (lower jitter).
        if self._producer_thread is None or not self._producer_thread.is_alive():
            self._producer_thread = threading.Thread(
                target=self._inference_producer_loop,
                name="policy_server_improved_inference_producer",
                daemon=True,
            )
            self._producer_thread.start()

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
            with suppress(Empty):
                _ = self._observation_mailbox.get_nowait()
                self.logger.debug("Observation mailbox was full, removed old observation")

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
        """Back-compat action RPC.

        For lower jitter, inference runs in a dedicated producer loop, and this RPC returns the
        latest available actions (converted to the legacy pickled TimedAction list format).
        """
        if not self._policy_ready.is_set():
            return services_pb2.Empty()

        dense = self._wait_for_latest_dense(timeout_s=self.config.actions_wait_timeout_s)
        if dense is None:
            return services_pb2.Empty()

        try:
            timed_actions = self._dense_to_timed_actions(dense)
            return services_pb2.Actions(data=pickle.dumps(timed_actions))  # nosec
        except Exception as e:
            self.logger.error(f"Error in GetActions (compat): {e}")
            return services_pb2.Empty()

    def GetActionsDense(self, request, context):  # noqa: N802
        """Unary dense actions RPC (lower jitter than pickled TimedAction list)."""
        if not self._policy_ready.is_set() or not self.config.actions_dense_enabled:
            return services_pb2.ActionsDense()

        dense = self._wait_for_latest_dense(timeout_s=self.config.actions_wait_timeout_s)
        return dense if dense is not None else services_pb2.ActionsDense()

    def StreamActionsDense(self, request, context):  # noqa: N802
        """Server-streaming dense actions RPC (lowest jitter path)."""
        if not self._policy_ready.is_set() or not self.config.actions_dense_enabled or not self.config.actions_stream_enabled:
            return

        last_seq = -1
        while self.running and context.is_active():
            with self._actions_cv:
                while (
                    self.running
                    and context.is_active()
                    and (self._latest_actions_dense is None or int(self._actions_seq) == last_seq)
                ):
                    self._actions_cv.wait(timeout=1.0)
                if not self.running or not context.is_active():
                    break
                dense = self._latest_actions_dense
                seq = int(self._actions_seq)

            if dense is None or seq == last_seq:
                continue
            last_seq = seq
            yield dense

    # -------------------------------------------------------------------------
    # Inference Pipeline
    # -------------------------------------------------------------------------

    def _wait_for_latest_dense(self, timeout_s: float) -> services_pb2.ActionsDense | None:
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        with self._actions_cv:
            while self._latest_actions_dense is None and self.running:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._actions_cv.wait(timeout=remaining)
            return self._latest_actions_dense

    def _publish_dense(self, dense: services_pb2.ActionsDense) -> None:
        with self._actions_cv:
            self._actions_seq += 1
            dense.seq = int(self._actions_seq)
            self._latest_actions_dense = dense
            self._actions_cv.notify_all()

    def _dense_to_timed_actions(self, dense: services_pb2.ActionsDense) -> list[TimedAction]:
        t = int(dense.t)
        a = int(dense.a)
        if t <= 0 or a <= 0:
            return []
        buf = dense.actions_f32
        actions = np.frombuffer(buf, dtype=np.float32)
        if actions.size != t * a:
            raise ValueError(f"ActionsDense buffer size mismatch: {actions.size} != {t*a}")
        actions = actions.reshape(t, a)
        t0 = float(dense.t0)
        i0 = int(dense.i0)
        dt = float(dense.dt)
        return [
            TimedAction(timestamp=t0 + i * dt, timestep=i0 + i, action=actions[i])
            for i in range(t)
        ]

    def _inference_producer_loop(self) -> None:
        """Continuously produce the latest action chunk from the latest observation (low jitter)."""
        self.logger.info("Inference producer thread starting")

        while self.running:
            if not self._policy_ready.is_set():
                time.sleep(0.01)
                continue

            try:
                obs = self._observation_mailbox.get(timeout=0.1)
            except Empty:
                continue

            try:
                t_total_start = time.perf_counter()
                t_infer_start = time.perf_counter()
                dense = self._predict_action_chunk_dense(obs)
                t_infer_done = time.perf_counter()

                self._publish_dense(dense)

                self.logger.info(
                    "Dense action chunk #%s produced | inference_total: %.2fms",
                    obs.get_timestep(),
                    self._ms(t_infer_done - t_infer_start),
                )
                self.logger.debug("Producer loop total: %.2fms", self._ms(time.perf_counter() - t_total_start))
            except Exception as e:
                self.logger.error(f"Error in inference producer loop: {e}")

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Legacy path: return `list[TimedAction]`.\n+\n+        For jitter reduction, this uses the dense pipeline (vectorized postprocess + single CPU copy)\n+        and then converts to TimedAction objects.\n+        """
        dense = self._predict_action_chunk_dense(observation_t)
        return self._dense_to_timed_actions(dense)

    def _predict_action_chunk_dense(self, observation_t: TimedObservation) -> services_pb2.ActionsDense:
        """Run inference on an observation and return dense packed actions (lower jitter)."""
        if self.actions_per_chunk is None:
            raise RuntimeError("actions_per_chunk is not set; did SendPolicyInstructions run?")
        if self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("pre/post processors not initialized; did SendPolicyInstructions run?")

        # 1. Prepare observation
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )

        # 2. Preprocess
        observation = self.preprocessor(observation)

        # 3. Inference (avoid autograd / reduce variance)
        with torch.inference_mode():
            action_tensor = self._get_action_chunk(observation)

        # Ensure (B, T, A)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, : self.actions_per_chunk, :]

        b, t, a = action_tensor.shape

        # 4. Vectorized postprocess: (B, T, A) -> (B*T, A) -> (B, T, A)
        flat = action_tensor.reshape(b * t, a)
        flat = self.postprocessor(flat)
        if not isinstance(flat, torch.Tensor):
            raise TypeError(f"postprocessor must return torch.Tensor, got {type(flat)}")
        action_tensor = flat.reshape(b, t, a)

        # Drop batch dim and move to CPU once
        actions_cpu = action_tensor.squeeze(0).detach().to("cpu")
        actions_np = actions_cpu.to(torch.float32).numpy()

        payload = np.asarray(actions_np, dtype=np.float32, order="C")

        dense = services_pb2.ActionsDense(
            t0=float(observation_t.get_timestamp()),
            i0=int(observation_t.get_timestep()),
            dt=float(self.config.environment_dt),
            t=int(payload.shape[0]),
            a=int(payload.shape[1]),
            actions_f32=payload.tobytes(order="C"),
            seq=0,
        )
        return dense

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

