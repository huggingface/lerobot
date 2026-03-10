#!/usr/bin/env python

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
RTC Policy Server - Remote inference server with Real-Time Chunking support.

This server runs diffusion-based policies (SmolVLA, Pi0, Pi0.5) with RTC on a powerful
remote machine, allowing lightweight robot computers to control robots smoothly.

Usage:
    python examples/remote_rtc/rtc_policy_server.py \
        --host=0.0.0.0 \
        --port=8080
"""

import contextlib
import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass, field
from pprint import pformat
from queue import Empty, Queue

import draccus
import grpc
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.remote import (
    RTCActionData,
    RTCObservationData,
    RTCRemotePolicyConfig,
    RTCTimingData,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


SUPPORTED_POLICIES = ["smolvla", "pi0", "pi05"]


@dataclass
class RTCPolicyServerConfig:
    """Configuration for RTC Policy Server."""

    host: str = field(default="0.0.0.0", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})
    obs_queue_timeout: float = field(
        default=1.0, metadata={"help": "Timeout for observation queue in seconds"}
    )

    verbose_request_logging: bool = field(
        default=False,
        metadata={"help": "Enable detailed per-request timing logs"},
    )
    client_unavailable_timeout_s: float = field(
        default=2.0,
        metadata={
            "help": (
                "Reset/unload the server (freeing VRAM) if no client RPCs arrive for this many seconds. "
                "Set <= 0 to disable."
            )
        },
    )

    def __post_init__(self):
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")


class RTCPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    """gRPC server for RTC policy inference."""

    def __init__(self, config: RTCPolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self.observation_queue: Queue[RTCObservationData] = Queue(maxsize=1)
        self._rpc_state_lock = threading.Lock()
        self._active_rpcs = 0
        self._client_unavailable_timer: threading.Timer | None = None
        self._has_received_observation = False

        # Policy components (initialized by SendPolicyInstructions)
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline | None = None
        self.postprocessor: PolicyProcessorPipeline | None = None

        logger.info(f"RTCPolicyServer initialized with config: {config}")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _rpc_enter(self) -> None:
        with self._rpc_state_lock:
            self._active_rpcs += 1
            if self._client_unavailable_timer is not None:
                self._client_unavailable_timer.cancel()
                self._client_unavailable_timer = None

    def _rpc_exit(self) -> None:
        with self._rpc_state_lock:
            self._active_rpcs = max(0, self._active_rpcs - 1)
            if self._active_rpcs == 0:
                if not self._has_received_observation:
                    return
                timeout_s = self.config.client_unavailable_timeout_s
                if timeout_s <= 0:
                    return
                if self._client_unavailable_timer is not None:
                    self._client_unavailable_timer.cancel()
                self._client_unavailable_timer = threading.Timer(timeout_s, self._on_client_unavailable)
                self._client_unavailable_timer.daemon = True
                self._client_unavailable_timer.start()

    def _unload_policy(self, reason: str) -> None:
        with self._rpc_state_lock:
            self._has_received_observation = False

        if self.policy is None:
            return

        logger.warning("Unloading policy to free VRAM (reason=%s)", reason)

        policy = self.policy
        preprocessor = self.preprocessor
        postprocessor = self.postprocessor
        device = self.device

        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.observation_queue = Queue(maxsize=1)

        del policy, preprocessor, postprocessor

        try:
            import gc

            gc.collect()
        except Exception as e:
            logger.debug("gc.collect failed: %s", e)

        if device is not None and torch.cuda.is_available() and "cuda" in str(device):
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug("Failed to clear CUDA cache: %s", e)

    def _on_client_unavailable(self) -> None:
        with self._rpc_state_lock:
            if self._active_rpcs != 0:
                return
            self._client_unavailable_timer = None

        self._reset_server()
        self._unload_policy(reason=f"client_unavailable_{self.config.client_unavailable_timeout_s}s")

    def _reset_server(self) -> None:
        """Reset server state when new client connects."""
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

    def Ready(self, request, context):  # noqa: N802
        """Handle client ready signal."""
        self._rpc_enter()
        context.add_callback(self._rpc_exit)
        client_id = context.peer()
        logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        with self._rpc_state_lock:
            no_other_rpcs = self._active_rpcs == 1
        if no_other_rpcs:
            self._unload_policy(reason="new_client_ready")
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy configuration from client and initialize policy."""
        self._rpc_enter()
        context.add_callback(self._rpc_exit)
        if not self.running:
            logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RTCRemotePolicyConfig):
            raise TypeError(f"Expected RTCRemotePolicyConfig, got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained: {policy_specs.pretrained_name_or_path} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features

        # Load policy
        self._unload_policy(reason="replacing_existing_policy")
        policy_class = get_policy_class(self.policy_type)
        start = time.perf_counter()

        use_compile = getattr(policy_specs, "use_torch_compile", False)
        compile_mode = getattr(policy_specs, "torch_compile_mode", "reduce-overhead")

        # Load policy config, applying client overrides
        policy_cfg = PreTrainedConfig.from_pretrained(policy_specs.pretrained_name_or_path)
        policy_cfg.device = policy_specs.device

        chunk_size = getattr(policy_specs, "chunk_size", None)
        n_action_steps = getattr(policy_specs, "n_action_steps", None)
        if chunk_size is not None:
            policy_cfg.chunk_size = chunk_size
            logger.info(f"Overriding chunk_size={chunk_size}")
        if n_action_steps is not None:
            policy_cfg.n_action_steps = n_action_steps
            logger.info(f"Overriding n_action_steps={n_action_steps}")

        if use_compile and self.policy_type in ["pi05", "pi0"]:
            torch._inductor.config.fx_graph_cache = True
            torch._inductor.config.fx_graph_remote_cache = False
            logger.info("Enabled persistent FX graph cache for torch.compile")
            policy_cfg.compile_model = True
            if compile_mode == "max-autotune":
                compile_mode = "max-autotune-no-cudagraphs"
            policy_cfg.compile_mode = compile_mode

        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path, config=policy_cfg)
        self.policy.to(self.device)
        self.policy.eval()

        # Configure RTC from client config
        rtc_config = getattr(policy_specs, "rtc_config", None)
        if rtc_config is not None:
            self.policy.config.rtc_config = rtc_config
        self.policy.init_rtc_processor()

        # Apply torch.compile for non-pi0 policies (pi0/pi05 handle it internally)
        if use_compile and self.policy_type not in ("pi05", "pi0"):
            try:
                logger.info("Applying torch.compile to predict_action_chunk...")
                self.policy.predict_action_chunk = torch.compile(
                    self.policy.predict_action_chunk,
                    backend="inductor",
                    mode=compile_mode,
                )
                logger.info("Successfully compiled predict_action_chunk")
            except Exception as e:
                logger.error(f"Failed to apply torch.compile: {e}")

        # Load preprocessor and postprocessor
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
        logger.info(f"Policy loaded on {self.device} in {end - start:.4f} seconds")
        logger.info(f"RTC config: {self.policy.config.rtc_config}")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations with RTC parameters from client."""
        self._rpc_enter()
        context.add_callback(self._rpc_exit)
        logger.debug("SendObservations called, receiving data...")
        t_start = time.perf_counter()

        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, logger)
        if received_bytes is None:
            return services_pb2.Empty()

        with self._rpc_state_lock:
            self._has_received_observation = True

        t_receive = time.perf_counter()
        receive_ms = (t_receive - t_start) * 1000

        rtc_obs_data: RTCObservationData = pickle.loads(received_bytes)  # nosec
        t_unpickle = time.perf_counter()
        unpickle_ms = (t_unpickle - t_receive) * 1000

        if self.config.verbose_request_logging:
            prev_shape = (
                tuple(rtc_obs_data.prev_chunk_left_over.shape)
                if rtc_obs_data.prev_chunk_left_over is not None
                else None
            )
            logger.info(
                f"Observation received | "
                f"bytes: {len(received_bytes)} | "
                f"receive: {receive_ms:.1f}ms | "
                f"unpickle: {unpickle_ms:.1f}ms | "
                f"inference_delay: {rtc_obs_data.inference_delay} | "
                f"execution_horizon: {rtc_obs_data.execution_horizon} | "
                f"prev_chunk_left_over: {prev_shape}"
            )

        # Enqueue observation (replacing old one if queue is full)
        if self.observation_queue.full():
            with contextlib.suppress(Empty):
                self.observation_queue.get_nowait()

        rtc_obs_data._server_receive_time = t_start  # Store for end-to-end timing
        self.observation_queue.put(rtc_obs_data)
        logger.debug("Observation queued")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Run RTC inference and return actions to client."""
        self._rpc_enter()
        context.add_callback(self._rpc_exit)
        try:
            if self.policy is None or self.preprocessor is None or self.postprocessor is None:
                return services_pb2.Actions(data=b"")

            logger.debug("GetActions called, waiting for observation...")
            wait_start = time.perf_counter()
            rtc_obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            wait_end = time.perf_counter()

            logger.debug(
                f"Running inference | delay={rtc_obs.inference_delay} | horizon={rtc_obs.execution_horizon}"
            )

            t_start = time.perf_counter()

            # Preprocess observation
            logger.debug("Preprocessing observation...")
            observation = rtc_obs.observation
            preprocessed_obs = self.preprocessor(observation)
            t_preprocess = time.perf_counter()

            # Run policy with RTC parameters
            logger.debug("Running predict_action_chunk...")
            with torch.no_grad():
                actions = self.policy.predict_action_chunk(
                    preprocessed_obs,
                    inference_delay=rtc_obs.inference_delay,
                    prev_chunk_left_over=rtc_obs.prev_chunk_left_over,
                    execution_horizon=rtc_obs.execution_horizon,
                )
            t_inference = time.perf_counter()
            logger.debug("predict_action_chunk completed")

            logger.debug("Postprocessing actions...")
            # Store original actions for RTC tracking
            original_actions = actions.squeeze(0).clone()

            # Postprocess actions
            postprocessed_actions = self.postprocessor(actions)
            postprocessed_actions = postprocessed_actions.squeeze(0)
            t_postprocess = time.perf_counter()

            # Calculate detailed timing
            queue_wait_ms = (wait_end - wait_start) * 1000
            preprocess_ms = (t_preprocess - t_start) * 1000
            inference_ms = (t_inference - t_preprocess) * 1000
            postprocess_ms = (t_postprocess - t_inference) * 1000
            server_compute_total_ms = queue_wait_ms + preprocess_ms + inference_ms + postprocess_ms

            # Create response
            rtc_action_data = RTCActionData(
                actions=postprocessed_actions.cpu(),
                original_actions=original_actions.cpu(),
                timestamp=time.time(),
                timestep=rtc_obs.timestep,
                timing=RTCTimingData(
                    queue_wait_ms=queue_wait_ms,
                    preprocess_ms=preprocess_ms,
                    inference_ms=inference_ms,
                    postprocess_ms=postprocess_ms,
                    total_ms=server_compute_total_ms,
                ),
            )

            actions_bytes = pickle.dumps(rtc_action_data)
            t_pickle = time.perf_counter()
            pickle_ms = (t_pickle - t_postprocess) * 1000
            total_ms = (t_pickle - t_start) * 1000

            # Calculate server-side total if we have receive time
            server_total_ms = None
            if hasattr(rtc_obs, "_server_receive_time"):
                server_total_ms = (t_pickle - rtc_obs._server_receive_time) * 1000

            log_message = (
                f"Actions ready | "
                f"queue_wait: {queue_wait_ms:.1f}ms | "
                f"preprocess: {preprocess_ms:.1f}ms | "
                f"inference: {inference_ms:.1f}ms | "
                f"postprocess: {postprocess_ms:.1f}ms | "
                f"pickle: {pickle_ms:.1f}ms | "
                f"total: {total_ms:.1f}ms"
                + (f" | server_total: {server_total_ms:.1f}ms" if server_total_ms else "")
                + f" | shape: {postprocessed_actions.shape}"
            )
            if self.config.verbose_request_logging:
                logger.info(log_message)
            else:
                logger.debug(log_message)

            return services_pb2.Actions(data=actions_bytes)

        except Empty:
            logger.debug("GetActions timeout - no observation in queue")
            return services_pb2.Actions(data=b"")

        except Exception as e:
            logger.error(f"Error in GetActions: {e}")
            import traceback

            traceback.print_exc()
            return services_pb2.Actions(data=b"")

    def stop(self):
        """Stop the server."""
        with self._rpc_state_lock:
            if self._client_unavailable_timer is not None:
                self._client_unavailable_timer.cancel()
                self._client_unavailable_timer = None
        self._reset_server()
        self._unload_policy(reason="server_stop")
        logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: RTCPolicyServerConfig):
    """Start the RTC Policy Server."""
    init_logging()
    logger.info("Configuration:\n%s", pformat(asdict(cfg)))

    logger.info("Creating RTCPolicyServer...")
    policy_server = RTCPolicyServer(cfg)

    logger.info("Creating gRPC server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    server.start()

    logger.info("=" * 60)
    logger.info(f"RTC Policy Server running on {cfg.host}:{cfg.port}")
    logger.info("Waiting for client connections...")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        policy_server.stop()
        server.stop(grace=5)

    logger.info("Server terminated")


if __name__ == "__main__":
    serve()
