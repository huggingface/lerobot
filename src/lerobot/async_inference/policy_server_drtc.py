"""
DRTC Policy Server

This implementation follows the DRTC algorithm with:
- 2-thread architecture (observation receiver + main inference loop)
- SPSC last-write-wins registers for observation/actions handoff

Threading model (2 threads):
- Main thread: inference loop, runs policy, sends actions
- Observation receiver thread: receives observations from clients via gRPC

Example:
```shell
python -m lerobot.async_inference.policy_server_drtc \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --obs_queue_timeout=2
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from concurrent import futures
from typing import Any

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

from .configs_drtc import PolicyServerDrtcConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    Observation,
    RemotePolicyConfig,
    TimedObservation,
    get_logger,
    raw_observation_to_observation,
)
from .rtc_guidance import AsyncRTCConfig, AsyncRTCProcessor
from .utils.simulation import SpikeDelaySimulator
from .utils.trajectory_viz import TrajectoryVizServer
from .utils.metrics import DiagnosticMetrics, EvActionChunk, Metrics
from .utils.viz_utils import compute_prefix_weights_for_viz
from .utils.compression import decode_images_from_transport
from .lww_register import LWWRegister

_INITIAL_K = -(2**63)

class ActionChunkCache:
    """LRU cache for raw action chunks, keyed by source control step (t).

    Used for RTC inpainting: the server caches raw (pre-postprocess) action chunks
    so the client can reference them by source control step + index range instead
    of sending post-processed actions (which have different dimensions).
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

class PolicyServerDrtc(services_pb2_grpc.AsyncInferenceServicer):
    """DRTC policy server.

    This implementation follows the 2-thread model from the paper:
    - Main thread: runs the inference loop
    - Observation receiver thread: receives observations from clients via gRPC

    Thread communication uses SPSC last-write-wins registers (keyed by timesteps).
    """

    prefix = "policy_server_drtc"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerDrtcConfig):
        """Initialize the policy server.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.shutdown_event = threading.Event()

        # Diagnostic metrics (console only; avg/max timings).
        diag = DiagnosticMetrics(
            fps=config.fps,
            window_s=config.metrics_diagnostic_window_s,
            interval_s=config.metrics_diagnostic_interval_s,
            enabled=config.metrics_diagnostic_enabled,
            verbose=config.metrics_diagnostic_verbose,
            prefix="DIAG_SERVER",
        )
        diag.start()
        self._metrics = Metrics(experiment=None, diagnostic=diag)

        # SPSC LWW registers
        # - Receiver thread -> inference producer: latest observation (by control_step)
        # - Inference producer -> StreamActionsDense: latest dense actions (by control_step)
        self._obs_reg: LWWRegister[TimedObservation | None] = LWWRegister(
            initial_control_step=_INITIAL_K, initial_value=None
        )
        self._action_reg: LWWRegister[services_pb2.ActionsDense | None] = LWWRegister(
            initial_control_step=_INITIAL_K, initial_value=None
        )

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

        # Action chunk cache for RTC (stores raw actions before postprocessing).
        # Placeholder; resized to match actions_per_chunk in SendPolicyInstructions.
        self._action_cache = ActionChunkCache(max_size=10)

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
            print(
                "Trajectory visualization server started on "
                f"http://0.0.0.0:{config.trajectory_viz_http_port} "
                f"(WebSocket: ws://0.0.0.0:{config.trajectory_viz_ws_port})"
            )

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Reset server state when a new client connects."""
        self.shutdown_event.set()
        # Reset registers (avoid leaking prior session values)
        self._obs_reg = LWWRegister(initial_control_step=_INITIAL_K, initial_value=None)
        self._policy_ready.clear()
        self._action_reg = LWWRegister(initial_control_step=_INITIAL_K, initial_value=None)
        self._action_cache.clear()

    # -------------------------------------------------------------------------
    # gRPC Service Methods (called by receiver thread)
    # -------------------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        """Handle client ready signal. Resets server state for new session."""
        self._metrics.diagnostic.counter("client_ready", 1)
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
            src_control_step=request.source_step,  # proto field is source_step
            actions=actions,
            frozen_len=request.frozen_len,
            timestamp=request.timestamp,
        )
        self._trajectory_viz_server.on_chunk(event)

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive and load policy from client instructions."""
        if not self.running:
            return services_pb2.Empty()

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

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        # Resize RTC chunk cache to match the client's chunk size so we always
        # keep enough history for the full action horizon.
        self._action_cache = ActionChunkCache(max_size=self.actions_per_chunk)

        # Skip loading real policy in mock mode
        if self.config.mock_policy:
            self._metrics.diagnostic.counter("mock_policy_mode", 1)
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
        self._metrics.diagnostic.timing_s("policy_load_ms", t_load_done - t_load_start)
        self._metrics.diagnostic.timing_s("policy_to_ms", t_to_done - t_to_start)
        self._metrics.diagnostic.timing_s("policy_processors_ms", t_pp_done - t_pp_start)
        self._metrics.diagnostic.timing_s("policy_total_ms", time.perf_counter() - t_total_start)

        # Apply num_flow_matching_steps override if provided by client
        # (Alex Soare optimization: Beta should scale with n)
        num_flow_steps = getattr(policy_specs, "num_flow_matching_steps", None)
        if num_flow_steps is not None:
            cfg_obj = getattr(self.policy, "config", None)
            if cfg_obj is not None:
                # PI0/PI05 use num_inference_steps, SmolVLA uses num_steps
                if hasattr(cfg_obj, "num_inference_steps"):
                    cfg_obj.num_inference_steps = num_flow_steps
                elif hasattr(cfg_obj, "num_steps"):
                    cfg_obj.num_steps = num_flow_steps
                else:
                    self._metrics.diagnostic.counter("num_flow_steps_override_ignored", 1)

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
            self._metrics.diagnostic.counter("spike_events_configured", len(spikes))

        # Warmup: run dummy inference passes to trigger CUDA kernel compilation
        # and memory allocation so the first real measurement isn't inflated.
        if self.config.warmup_passes > 0:
            self._warmup_model(num_passes=self.config.warmup_passes)

        self._policy_ready.set()

        # Start producer thread (if needed) to generate actions outside the RPC path (lower jitter).
        if self._producer_thread is None or not self._producer_thread.is_alive():
            self._producer_thread = threading.Thread(
                target=self._inference_producer_loop,
                name="policy_server_drtc_inference_producer",
                daemon=True,
            )
            self._producer_thread.start()

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from client and enqueue for inference.

        This method is called by the gRPC receiver thread.
        """
        t_total_start = time.perf_counter()

        # Receive observation bytes (stamp receive_time AFTER full payload
        # arrives so that client-to-server latency captures the actual
        # network transfer of the chunked image payload, not just the
        # gRPC handler dispatch time).
        t_recv_start = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )
        t_recv_done = time.perf_counter()
        receive_time = time.time()

        # Deserialize
        t_deser_start = time.perf_counter()
        timed_observation = pickle.loads(received_bytes)  # nosec
        t_deser_done = time.perf_counter()

        # Decode images
        t_decode_start = time.perf_counter()
        decoded_observation, _ = decode_images_from_transport(timed_observation.get_observation())
        timed_observation.observation = decoded_observation
        t_decode_done = time.perf_counter()

        # Stamp the server receive time for granular latency decomposition
        timed_observation.server_received_ts = receive_time

        obs_control_step = timed_observation.get_control_step()
        obs_timestamp = timed_observation.get_timestamp()

        # Diagnostics
        # Provide a stable `step` field for compact diagnostics.
        self._metrics.diagnostic.set_context(step=obs_control_step, last_obs_step=obs_control_step)
        self._metrics.diagnostic.timing_s("obs_recv_ms", t_recv_done - t_recv_start)
        self._metrics.diagnostic.timing_s("deser_ms", t_deser_done - t_deser_start)
        self._metrics.diagnostic.timing_s("obs_decode_ms", t_decode_done - t_decode_start)
        self._metrics.diagnostic.timing_s("obs_one_way_latency_ms", receive_time - obs_timestamp)
        self._metrics.diagnostic.timing_s("obs_total_ms", time.perf_counter() - t_total_start)

        # Publish newest observation (monotone w.r.t. control_step)
        self._obs_reg.update_if_newer(obs_control_step, timed_observation)

        return services_pb2.Empty()

    def StreamActionsDense(self, request, context):  # noqa: N802
        """Server-streaming dense actions RPC (streaming-only action transport)."""
        if not self._policy_ready.is_set():
            return
        reader = self._action_reg.reader(initial_watermark=_INITIAL_K)
        while self.running and context.is_active():
            state, _, is_new = reader.read_if_newer()
            dense = state.value
            if not is_new or dense is None:
                time.sleep(0.01)
                continue
            yield dense

    # -------------------------------------------------------------------------
    # Inference Pipeline
    # -------------------------------------------------------------------------

    def _publish_dense(self, dense: services_pb2.ActionsDense) -> None:
        control_step = int(dense.source_control_step)
        self._action_reg.update_if_newer(control_step, dense)

    def _warmup_model(self, num_passes: int = 2) -> None:
        """Run dummy inference passes to warm up CUDA kernels and memory allocations.

        The first forward pass through a PyTorch model on GPU triggers JIT compilation
        of CUDA kernels and cuDNN workspace allocation, adding hundreds of milliseconds
        to inference time. Running a few dummy passes here ensures this overhead is paid
        during startup, not during the first real measurement.

        Args:
            num_passes: Number of dummy inference passes to run.
        """
        if self.preprocessor is None or self.postprocessor is None:
            self.logger.warning("Cannot warmup: pre/post processors not initialized")
            return
        if self.policy is None:
            self.logger.warning("Cannot warmup: policy not loaded")
            return

        self.logger.info(f"Warming up model with {num_passes} dummy inference pass(es)...")
        t_warmup_start = time.perf_counter()

        try:
            # Build a dummy observation matching the format produced by
            # raw_observation_to_observation(): {OBS_STATE: (1, state_dim), image_keys: (1, C, H, W), task: str}
            dummy_obs: dict[str, Any] = {}

            # State: derive dimensionality from lerobot_features
            if self.lerobot_features:
                state_features = self.lerobot_features.get("observation.state", [])
                state_dim = len(state_features) if isinstance(state_features, (list, tuple)) else 6
            else:
                state_dim = 6
            dummy_obs["observation.state"] = torch.zeros(1, state_dim)

            # Images: use policy's image_features to get (C, H, W) shapes
            for key, feat in self.policy_image_features.items():
                c, h, w = feat.shape
                # After prepare_image + unsqueeze: float32 in [0, 1], shape (1, C, H, W)
                dummy_obs[key] = torch.zeros(1, c, h, w, dtype=torch.float32)

            # Task string (VLA models require this)
            dummy_obs["task"] = "warmup"

            for i in range(num_passes):
                t_pass_start = time.perf_counter()

                # Preprocess
                obs = self.preprocessor(dummy_obs)

                # Inference -- call policy directly (not _get_action_chunk)
                # to avoid recording warmup timings in diagnostic metrics.
                with torch.no_grad():
                    action_tensor = self.policy.predict_action_chunk(obs)

                # Postprocess (same path as real inference)
                if action_tensor.ndim != 3:
                    action_tensor = action_tensor.unsqueeze(0)
                action_tensor = action_tensor[:, : self.actions_per_chunk, :]
                b, t_dim, a = action_tensor.shape
                flat = action_tensor.reshape(b * t_dim, a)
                flat = self.postprocessor(flat)

                t_pass_done = time.perf_counter()
                self.logger.info(
                    f"  Warmup pass {i + 1}/{num_passes}: {(t_pass_done - t_pass_start) * 1000:.1f}ms"
                )

            t_warmup_done = time.perf_counter()
            warmup_total_ms = (t_warmup_done - t_warmup_start) * 1000
            self.logger.info(f"Model warmup complete ({warmup_total_ms:.0f}ms total)")
            self._metrics.diagnostic.timing_ms("warmup_total_ms", warmup_total_ms)

        except Exception as e:
            self.logger.error(f"Warmup failed (non-fatal, first inference may be slow): {e}")
            self._metrics.diagnostic.counter("warmup_failed", 1)

    def _inference_producer_loop(self) -> None:
        """Continuously produce the latest action chunk from the latest observation (low jitter)."""
        reader = self._obs_reg.reader(initial_watermark=_INITIAL_K)

        while self.running:
            if not self._policy_ready.is_set():
                time.sleep(0.01)
                continue

            state, _, is_new = reader.read_if_newer()
            obs = state.value
            if not is_new or obs is None:
                time.sleep(0.01)
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

                # Stamp server-side timestamps for granular latency decomposition
                dense.server_obs_received_ts = float(getattr(obs, "server_received_ts", 0.0))
                dense.server_action_sent_ts = time.time()

                self._publish_dense(dense)
                # Provide a stable `step` field for compact diagnostics.
                self._metrics.diagnostic.set_context(
                    step=int(obs.get_control_step()),
                    last_infer_src_step=int(obs.get_control_step()),
                )
                self._metrics.diagnostic.timing_s("infer_total_ms", t_infer_done - t_infer_start)
                self._metrics.diagnostic.timing_s("producer_loop_total_ms", time.perf_counter() - t_total_start)
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
            timestamp=float(observation_t.get_timestamp()),
            source_control_step=int(observation_t.get_control_step()),
            chunk_start_step=int(observation_t.chunk_start_step),
            dt=float(self.config.environment_dt),
            num_actions=int(payload.shape[0]),
            action_dim=int(payload.shape[1]),
            actions_f32=payload.tobytes(order="C"),
        )
        return dense

    def _predict_action_chunk_dense(self, observation_t: TimedObservation) -> services_pb2.ActionsDense:
        """Run inference on an observation and return dense packed actions (lower jitter)."""
        if self.actions_per_chunk is None:
            raise RuntimeError("actions_per_chunk is not set; did SendPolicyInstructions run?")
        if self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("pre/post processors not initialized; did SendPolicyInstructions run?")

        # Optional RTC metadata (client-provided hard-mask prefix + estimated delay).
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
        src_control_step = int(observation_t.get_control_step())
        chunk_start_step = int(observation_t.chunk_start_step)

        with torch.no_grad():
            rtc_kwargs: dict[str, Any] = {}
            if rtc_meta is not None and self._rtc_cfg is not None and self._rtc_cfg.enabled:
                try:
                    d = int(rtc_meta.get("latency_steps", 0))
                    # Accept prefix_chunks (new) or frozen_chunks (backward compat; hard-mask prefix)
                    # TODO - we can remove this, don't need backwards compat
                    prefix_chunks = rtc_meta.get("prefix_chunks") or rtc_meta.get("frozen_chunks")

                    # Get overlap_end from client: where fresh region starts (H - max(s_min, d))
                    # Falls back to execution_horizon (old name) or H - d for backward compatibility
                    H = self.actions_per_chunk
                    overlap_end = int(
                        rtc_meta.get("overlap_end") or rtc_meta.get("execution_horizon") or (H - d)
                    )
                    self._metrics.diagnostic.counter("rtc_meta_seen", 1)

                    # Reconstruct prefix tensor from multiple cached chunks
                    if prefix_chunks:
                        slices: list[torch.Tensor] = []
                        for chunk_src_step, start_idx, end_idx in prefix_chunks:
                            cached_chunk = self._action_cache.get(int(chunk_src_step))
                            if cached_chunk is None:
                                self._metrics.diagnostic.counter("rtc_cache_miss", 1)
                            else:
                                self._metrics.diagnostic.counter("rtc_cache_hit", 1)
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

                            # Clamp overlap_end to what we actually have in the prefix
                            # This allows graceful degradation when cache is incomplete
                            effective_overlap_end = min(overlap_end, T_prefix)

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
                                "overlap_end": effective_overlap_end,  # Clamped for RTC guidance
                                "overlap_end_intended": overlap_end,  # Original for visualization
                            }
                            self._metrics.diagnostic.counter("rtc_applied", 1)
                        else:
                            self._metrics.diagnostic.counter("rtc_not_applied_no_slices", 1)
                    else:
                        self._metrics.diagnostic.counter("rtc_not_applied_empty_prefix", 1)
                except Exception as e:
                    self._metrics.diagnostic.counter("rtc_meta_error", 1)
                    rtc_kwargs = {}

            action_tensor = self._get_action_chunk(observation, **rtc_kwargs)

        # Ensure (B, T, A)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, : self.actions_per_chunk, :]

        b, t, a = action_tensor.shape

        # Cache raw action chunk BEFORE postprocessing (for future RTC inpainting)
        # Key by control_step so RTC prefix_chunks spans can look up the right chunk.
        if src_control_step >= 0:
            self._action_cache.put(src_control_step, action_tensor)

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
                # Use intended overlap_end for visualization (not clamped to prefix length)
                overlap_end_viz = rtc_kwargs.get("overlap_end_intended", rtc_kwargs.get("overlap_end", self.actions_per_chunk))
                H_viz = self.actions_per_chunk

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
                src_control_step=src_control_step,
                actions=actions_list,
                frozen_len=rtc_kwargs.get("inference_delay", 0) if rtc_kwargs else 0,
                timestamp=time.time(),
                rtc_params=rtc_params_viz,
                prefix_weights=prefix_weights_viz,
            )
            self._trajectory_viz_server.on_chunk(event)

        dense_kwargs: dict[str, Any] = dict(
            timestamp=float(observation_t.get_timestamp()),
            source_control_step=int(observation_t.get_control_step()),
            chunk_start_step=int(observation_t.chunk_start_step),
            dt=float(self.config.environment_dt),
            num_actions=int(payload.shape[0]),
            action_dim=int(payload.shape[1]),
            actions_f32=payload.tobytes(order="C"),
        )
        dense = services_pb2.ActionsDense(**dense_kwargs)
        return dense

    def _get_action_chunk(self, observation: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """Get action chunk from the policy."""
        t0 = time.perf_counter()
        chunk = self.policy.predict_action_chunk(observation, **kwargs)
        t1 = time.perf_counter()
        self._metrics.diagnostic.timing_s("policy_predict_ms", t1 - t0)

        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # Add batch dimension: (chunk_size, action_dim) -> (1, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def stop(self) -> None:
        """Stop the server."""
        self._reset_server()
        self._metrics.diagnostic.stop()


@draccus.wrap()
def serve_drtc(cfg: PolicyServerDrtcConfig) -> None:
    """Start the DRTC PolicyServer."""
    # Create server instance
    policy_server = PolicyServerDrtc(cfg)

    # Setup gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    bound_port = server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    if bound_port == 0:
        raise RuntimeError(
            f"Failed to bind gRPC server to {cfg.host}:{cfg.port}. "
            "Is the port already in use, or are you binding to an unavailable interface?"
        )

    print(f"PolicyServerDrtc started on {cfg.host}:{cfg.port}")
    server_started = False
    try:
        server.start()
        server_started = True
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received; shutting down")
    except Exception:
        policy_server.logger.error("Policy server crashed", exc_info=True)
        raise
    finally:
        # Best-effort cleanup to avoid dangling threads on failures.
        try:
            policy_server.stop()
        except Exception:
            policy_server.logger.error("Error while stopping policy server", exc_info=True)
        if server_started:
            server.stop(grace=5)
    print("Server terminated")


if __name__ == "__main__":
    serve_drtc()

