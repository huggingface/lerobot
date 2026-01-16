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
from collections import OrderedDict
from contextlib import suppress
from concurrent import futures
from dataclasses import dataclass
from pprint import pformat
from queue import Empty, Full, Queue
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

from .configs_improved import PolicyServerImprovedConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedObservation,
    get_logger,
    raw_observation_to_observation,
)
from .rtc_guidance import AsyncRTCConfig, AsyncRTCProcessor
from .utils.simulation import SpikeDelayConfig, SpikeDelaySimulator
from .utils.trajectory_viz import TrajectoryVizServer
from .utils.diagnostics import EvActionChunk

if _IMPORT_TIMING_ENABLED:
    _sys.stderr.write(
        f"[import-timing] {__name__} imports: {(_time.perf_counter() - _IMPORT_T0) * 1000.0:.2f}ms\n"
    )


# =============================================================================
# Action Chunk Cache (for RTC with raw model actions)
# =============================================================================


class ActionChunkCache:
    """LRU cache for raw action chunks, keyed by source step.

    Used for RTC inpainting: the server caches raw (pre-postprocess) action chunks
    so the client can reference them by source step + index range instead of
    sending post-processed actions (which have different dimensions).
    """

    def __init__(self, max_size: int = 10):
        """Initialize the cache.

        Args:
            max_size: Maximum number of chunks to cache (oldest evicted first).
        """
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._max_size = max_size

    def put(self, src_step: int, raw_actions: torch.Tensor) -> None:
        """Store a raw action chunk keyed by source step.

        Args:
            src_step: The source step (observation timestep) for this chunk.
            raw_actions: Raw action tensor of shape (B, T, A) or (T, A).
        """
        # If already exists, remove it first so it goes to the end (most recent)
        if src_step in self._cache:
            del self._cache[src_step]

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        # Store a detached clone to avoid holding onto computation graph
        self._cache[src_step] = raw_actions.detach().clone()

    def get(self, src_step: int) -> torch.Tensor | None:
        """Retrieve a cached chunk by source step.

        Args:
            src_step: The source step to look up.

        Returns:
            The cached tensor or None if not found.
        """
        return self._cache.get(src_step)

    def clear(self) -> None:
        """Clear all cached chunks."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# Track which (schedule, d, overlap_end) combos have been logged to avoid spam
_prefix_weights_logged: set[tuple[str, int, int]] = set()


def compute_prefix_weights_for_viz(d: int, overlap_end: int, H: int, schedule: str = "linear") -> list[float]:
    """Compute prefix weights for RTC visualization.

    Args:
        d: Inference delay (frozen region ends at d).
        overlap_end: Where soft masking ends (H - d with s=d).
        H: Total chunk size.
        schedule: Weight schedule ("linear" or "exp").

    Returns:
        List of H floats, each in [0, 1]:
        - [0, d): weight = 1.0 (frozen)
        - [d, overlap_end): weight decays 1->0 (soft mask)
        - [overlap_end, H): weight = 0.0 (fresh)
    """
    import math

    weights = []
    for i in range(H):
        if i < d:
            # Frozen region
            weights.append(1.0)
        elif i < overlap_end:
            # Soft masking region - linear decay from 1 to 0
            if overlap_end > d:
                t = (i - d) / (overlap_end - d)  # t goes from 0 to 1
                w = 1.0 - t  # Linear decay
                if schedule.lower() == "exp":
                    # Exponential decay (steeper at start)
                    w = w * (math.expm1(w) / (math.e - 1)) if w > 0 else 0.0
                weights.append(w)
            else:
                weights.append(0.0)
        else:
            # Fresh region
            weights.append(0.0)

    # Log weight samples once per unique (schedule, d, overlap_end) to verify formula
    _log_key = (schedule.lower(), d, overlap_end)
    if _log_key not in _prefix_weights_logged and H > 0:
        _prefix_weights_logged.add(_log_key)
        logger = logging.getLogger("policy_server_improved")
        sample_indices = [d, (d + overlap_end) // 2, overlap_end - 1]
        samples = [(i, weights[i]) for i in sample_indices if 0 <= i < len(weights)]
        logger.info(
            "RTC prefix weights (%s): d=%d, overlap_end=%d, H=%d, samples=%s",
            schedule, d, overlap_end, H,
            [(f"w[{i}]", f"{w:.3f}") for i, w in samples],
        )

    return weights


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

        # SPSC one-slot mailbox for actions (single client / single stream only)
        # The producer thread is the producer, StreamActionsDense is the consumer
        self._actions_mailbox: Queue[services_pb2.ActionsDense] = Queue(maxsize=1)
        self._actions_seq: int = 0

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

        # Client-driven RTC (optional)
        self._rtc_cfg: AsyncRTCConfig | None = None

        # Action chunk cache for RTC (stores raw actions before postprocessing)
        self._action_cache = ActionChunkCache(max_size=config.rtc_chunk_cache_size)

        # Spike delay simulator for experiments
        self._delay_simulator = SpikeDelaySimulator(config=config.mock_spike_config)

        # Trajectory visualization server (HTTP + WebSocket)
        self._trajectory_viz_server: TrajectoryVizServer | None = None
        self._trajectory_viz_thread: threading.Thread | None = None
        if config.trajectory_viz_enabled:
            self._trajectory_viz_server = TrajectoryVizServer(
                ws_port=config.trajectory_viz_ws_port,
                http_port=config.trajectory_viz_http_port,
            )
            self._trajectory_viz_thread = threading.Thread(
                target=self._trajectory_viz_server.start,
                name="trajectory_viz_server",
                daemon=True,
            )
            self._trajectory_viz_thread.start()
            self.logger.info(
                f"Trajectory visualization server started on "
                f"http://0.0.0.0:{config.trajectory_viz_http_port} "
                f"(WebSocket: ws://0.0.0.0:{config.trajectory_viz_ws_port})"
            )

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
        self._actions_seq = 0
        self._actions_mailbox = Queue(maxsize=1)
        self._action_cache.clear()

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

    def SendTrajectoryChunk(self, request, context):  # noqa: N802
        """Receive trajectory chunk from robot client for visualization."""
        if self._trajectory_viz_server is None:
            return services_pb2.Empty()

        # Decode the packed float32 actions
        num_actions = request.num_actions
        action_dim = request.action_dim
        if num_actions > 0 and action_dim > 0:
            actions_flat = np.frombuffer(request.actions_f32, dtype=np.float32)
            actions = actions_flat.reshape(num_actions, action_dim).tolist()
        else:
            actions = []

        # Create EvActionChunk event and forward to viz server
        event = EvActionChunk(
            source_step=request.source_step,
            actions=actions,
            frozen_len=request.frozen_len,
            timestamp=request.timestamp,
        )
        self._trajectory_viz_server.on_chunk(event)

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

        # Skip loading real policy in mock mode
        if self.config.mock_policy:
            self.logger.info("Mock policy mode: skipping real model loading")
            self._policy_ready.set()
            return services_pb2.Empty()

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

        # Apply num_flow_matching_steps override if provided by client
        # (Alex Soare optimization: Beta should scale with n)
        num_flow_steps = getattr(policy_specs, "num_flow_matching_steps", None)
        if num_flow_steps is not None:
            cfg_obj = getattr(self.policy, "config", None)
            if cfg_obj is not None:
                # PI0/PI05 use num_inference_steps, SmolVLA uses num_steps
                if hasattr(cfg_obj, "num_inference_steps"):
                    old_val = cfg_obj.num_inference_steps
                    cfg_obj.num_inference_steps = num_flow_steps
                    self.logger.info(
                        "Overriding num_inference_steps: %d -> %d",
                        old_val,
                        num_flow_steps,
                    )
                elif hasattr(cfg_obj, "num_steps"):
                    old_val = cfg_obj.num_steps
                    cfg_obj.num_steps = num_flow_steps
                    self.logger.info(
                        "Overriding num_steps: %d -> %d",
                        old_val,
                        num_flow_steps,
                    )
                else:
                    self.logger.warning(
                        "Could not find num_inference_steps or num_steps on policy config; "
                        "num_flow_matching_steps override ignored"
                    )

        self._policy_ready.set()

        # Optional: enable RTC via client instructions (server-side inpainting)
        if getattr(policy_specs, "rtc_enabled", False):
            # Handle optional max_guidance_weight (None = use num_flow_matching_steps, Alex Soare opt)
            max_gw_raw = getattr(policy_specs, "rtc_max_guidance_weight", None)
            max_gw = float(max_gw_raw) if max_gw_raw is not None else None

            self._rtc_cfg = AsyncRTCConfig(
                enabled=True,
                prefix_attention_schedule=str(getattr(policy_specs, "rtc_prefix_attention_schedule", "linear")),
                max_guidance_weight=max_gw,
                sigma_d=float(getattr(policy_specs, "rtc_sigma_d", 1.0)),
                full_trajectory_alignment=bool(getattr(policy_specs, "rtc_full_trajectory_alignment", False)),
            )
            # NOTE: We do NOT pass self.postprocessor to RTC guidance because:
            # - RTC operates INSIDE the model's denoising loop in raw action space (e.g. 32 dims)
            # - The postprocessor (NormalizeProcessor) expects executable action space (e.g. 6 dims)
            # - These dimensions are incompatible; the model's action head converts at the end
            # - For now, RTC guidance compares in raw model space (prev must match model dims)
            rtc = AsyncRTCProcessor(self._rtc_cfg, postprocess=None)

            # Flow policies expect `policy.rtc_processor` and `policy.model.rtc_processor`.
            setattr(self.policy, "rtc_processor", rtc)
            model_value = getattr(self.policy, "model", None)
            if model_value is not None:
                setattr(model_value, "rtc_processor", rtc)

            # Satisfy policy-side `_rtc_enabled()` checks without importing RTCConfig.
            cfg_obj = getattr(self.policy, "config", None)
            if cfg_obj is not None:
                with suppress(Exception):
                    setattr(cfg_obj, "rtc_config", type("RTCConfigShim", (), {"enabled": True})())

        # Apply spike configuration from client (for experiments)
        spikes = getattr(policy_specs, "spikes", [])
        if spikes:
            self._delay_simulator = SpikeDelaySimulator.from_dicts(spikes)
            self.logger.info(
                "Spike injection configured from client: %d spike events",
                len(spikes),
            )
            for i, spike in enumerate(spikes):
                self.logger.info(
                    "  Spike %d: start=%.1fs, delay=%.0fms",
                    i + 1,
                    spike.get("start_s", 0),
                    spike.get("delay_ms", 0),
                )

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

    def StreamActionsDense(self, request, context):  # noqa: N802
        """Server-streaming dense actions RPC (streaming-only action transport)."""
        if not self._policy_ready.is_set():
            return

        while self.running and context.is_active():
            try:
                dense = self._actions_mailbox.get(timeout=0.25)
            except Empty:
                continue
            yield dense

    # -------------------------------------------------------------------------
    # Inference Pipeline
    # -------------------------------------------------------------------------

    def _publish_dense(self, dense: services_pb2.ActionsDense) -> None:
        self._actions_seq += 1
        dense.seq = int(self._actions_seq)

        # One-slot mailbox semantics: overwrite if full
        if self._actions_mailbox.full():
            with suppress(Empty):
                _ = self._actions_mailbox.get_nowait()
        with suppress(Full):
            self._actions_mailbox.put_nowait(dense)

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

                # Apply simulated delay (for experiments)
                self._delay_simulator.apply_delay()

                t_infer_start = time.perf_counter()

                # Use mock policy or real policy
                if self.config.mock_policy:
                    dense = self._mock_predict_action_chunk_dense(obs)
                else:
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

    def _mock_predict_action_chunk_dense(self, observation_t: TimedObservation) -> services_pb2.ActionsDense:
        """Generate mock actions for simulation experiments (no real model)."""
        action_dim = self.config.mock_action_dim
        actions_per_chunk = self.actions_per_chunk or 50

        # Generate random actions
        actions_np = np.random.randn(actions_per_chunk, action_dim).astype(np.float32) * 0.1
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

    def _predict_action_chunk_dense(self, observation_t: TimedObservation) -> services_pb2.ActionsDense:
        """Run inference on an observation and return dense packed actions (lower jitter)."""
        if self.actions_per_chunk is None:
            raise RuntimeError("actions_per_chunk is not set; did SendPolicyInstructions run?")
        if self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("pre/post processors not initialized; did SendPolicyInstructions run?")

        # Optional RTC metadata (client-provided frozen prefix + estimated delay).
        rtc_meta = None
        raw_obs_any = observation_t.get_observation()
        if isinstance(raw_obs_any, dict):
            rtc_meta = raw_obs_any.get("__rtc__")

        # Remove RTC metadata before policy preprocessing (avoid surprising processors).
        if rtc_meta is not None and isinstance(raw_obs_any, dict):
            raw_obs = dict(raw_obs_any)
            raw_obs.pop("__rtc__", None)
        else:
            raw_obs = raw_obs_any

        # 1. Prepare observation
        observation: Observation = raw_observation_to_observation(
            raw_obs,
            self.lerobot_features,
            self.policy_image_features,
        )

        # 2. Preprocess
        observation = self.preprocessor(observation)

        # 3. Inference (avoid autograd / reduce variance)
        # NOTE: Do NOT use `torch.inference_mode()` here: RTC guidance needs to temporarily
        # enable gradients for the inpainting correction term, and inference_mode cannot be
        # overridden. `torch.no_grad()` keeps the normal path efficient while still allowing
        # nested `torch.enable_grad()` for RTC.
        src_step = int(observation_t.get_timestep())

        with torch.no_grad():
            rtc_kwargs: dict[str, Any] = {}
            if rtc_meta is not None and self._rtc_cfg is not None and self._rtc_cfg.enabled:
                try:
                    d = int(rtc_meta.get("latency_steps", 0))
                    # Accept prefix_chunks (new) or frozen_chunks (backward compat)
                    prefix_chunks = rtc_meta.get("prefix_chunks") or rtc_meta.get("frozen_chunks")

                    # Get execution_horizon from client (d + epsilon)
                    # Falls back to H - d for backward compatibility
                    H = self.actions_per_chunk
                    execution_horizon = int(rtc_meta.get("execution_horizon", H - d))

                    # Log what we received
                    self.logger.info(
                        "RTC: src_step=%s, H=%s, d=%s, execution_horizon=%s, schedule=%s, prefix_chunks=%s, cache_size=%d",
                        src_step,
                        H,
                        d,
                        execution_horizon,
                        self._rtc_cfg.prefix_attention_schedule if self._rtc_cfg else "N/A",
                        prefix_chunks,
                        len(self._action_cache),
                    )

                    # Reconstruct prefix tensor from multiple cached chunks
                    if prefix_chunks:
                        slices: list[torch.Tensor] = []
                        for chunk_src_step, start_idx, end_idx in prefix_chunks:
                            cached_chunk = self._action_cache.get(int(chunk_src_step))
                            self.logger.debug(
                                "RTC: cache lookup src_step=%s, found=%s",
                                chunk_src_step,
                                cached_chunk is not None,
                            )
                            if cached_chunk is not None:
                                # Extract slice from cached chunk (B, T, A) or (T, A)
                                if cached_chunk.ndim == 2:
                                    slices.append(cached_chunk[start_idx:end_idx, :])
                                else:
                                    # Squeeze batch dim for concatenation
                                    slices.append(cached_chunk[0, start_idx:end_idx, :])

                        if slices:
                            # Concatenate all slices along time dimension -> (T_total, A)
                            prefix_tensor = torch.cat(slices, dim=0)
                            prefix_tensor = prefix_tensor.unsqueeze(0)  # (1, T_total, A)
                            T_prefix = prefix_tensor.shape[1]

                            # Clamp execution_horizon to what we actually have in the prefix
                            # This allows graceful degradation when cache is incomplete
                            effective_horizon = min(execution_horizon, T_prefix)

                            # Zero-pad to max_action_dim if model uses padded action space
                            max_action_dim = getattr(self.policy.config, "max_action_dim", None)
                            if max_action_dim is not None and prefix_tensor.shape[-1] < max_action_dim:
                                b, t, a = prefix_tensor.shape
                                padded = torch.zeros(
                                    b, t, max_action_dim,
                                    device=prefix_tensor.device,
                                    dtype=prefix_tensor.dtype,
                                )
                                padded[:, :, :a] = prefix_tensor
                                prefix_tensor = padded

                            rtc_kwargs = {
                                "inference_delay": d,
                                "prev_chunk_left_over": prefix_tensor.to(device=self.device),
                                "execution_horizon": effective_horizon,
                            }
                            self.logger.debug(
                                "RTC: APPLYING with shape=%s, d=%s, execution_horizon=%s, T_prefix=%s",
                                prefix_tensor.shape,
                                d,
                                effective_horizon,
                                T_prefix,
                            )
                        else:
                            self.logger.debug("RTC: NOT applying (no slices found from cache)")
                    else:
                        self.logger.debug("RTC: NOT applying (prefix_chunks is empty/None)")
                except Exception as e:
                    self.logger.warning("RTC metadata lookup failed: %s", e)
                    rtc_kwargs = {}

            action_tensor = self._get_action_chunk(observation, **rtc_kwargs)

        # Ensure (B, T, A)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, : self.actions_per_chunk, :]

        b, t, a = action_tensor.shape

        # Cache raw action chunk BEFORE postprocessing (for future RTC inpainting)
        # Skip caching during priming phase (negative timesteps)
        if src_step >= 0:
            self._action_cache.put(src_step, action_tensor)

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

        # Emit action chunk to trajectory visualization (if enabled)
        if self._trajectory_viz_server is not None:
            # Build RTC params dict for visualization
            rtc_params_viz: dict[str, Any] | None = None
            prefix_weights_viz: list[float] | None = None

            if self._rtc_cfg is not None and self._rtc_cfg.enabled and rtc_kwargs:
                d_viz = rtc_kwargs.get("inference_delay", 0)
                exec_h = rtc_kwargs.get("execution_horizon", self.actions_per_chunk)
                H_viz = self.actions_per_chunk
                overlap_end_viz = exec_h  # execution_horizon is passed as overlap_end

                rtc_params_viz = {
                    "d": d_viz,
                    "H": H_viz,
                    "overlap_end": overlap_end_viz,
                    "sigma_d": self._rtc_cfg.sigma_d,
                    "schedule": self._rtc_cfg.prefix_attention_schedule,
                    "max_guidance_weight": self._rtc_cfg.max_guidance_weight,
                    "full_trajectory_alignment": self._rtc_cfg.full_trajectory_alignment,
                }
                prefix_weights_viz = compute_prefix_weights_for_viz(
                    d_viz, overlap_end_viz, H_viz, self._rtc_cfg.prefix_attention_schedule
                )

            # Create and emit the event
            actions_list = actions_np.tolist()
            event = EvActionChunk(
                source_step=src_step,
                actions=actions_list,
                frozen_len=rtc_kwargs.get("inference_delay", 0) if rtc_kwargs else 0,
                timestamp=time.time(),
                rtc_params=rtc_params_viz,
                prefix_weights=prefix_weights_viz,
            )
            self._trajectory_viz_server.on_chunk(event)

        dense_kwargs: dict[str, Any] = dict(
            t0=float(observation_t.get_timestamp()),
            i0=int(observation_t.get_timestep()),
            dt=float(self.config.environment_dt),
            t=int(payload.shape[0]),
            a=int(payload.shape[1]),
            actions_f32=payload.tobytes(order="C"),
            seq=0,
        )
        dense = services_pb2.ActionsDense(**dense_kwargs)
        return dense

    def _get_action_chunk(self, observation: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """Get action chunk from the policy."""
        t0 = time.perf_counter()
        chunk = self.policy.predict_action_chunk(observation, **kwargs)
        t1 = time.perf_counter()
        self.logger.debug("Policy predict_action_chunk: %.2fms", self._ms(t1 - t0))

        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # Add batch dimension: (chunk_size, action_dim) -> (1, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

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

