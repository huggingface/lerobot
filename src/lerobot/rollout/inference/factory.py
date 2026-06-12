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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|remote``.  Adding a
new backend requires registering its config subclass and dispatching it
in :func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event

import draccus

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


class FallbackMode(StrEnum):
    """What ``get_action`` returns when the remote queue runs dry (STALLED)."""

    HOLD = "hold"  # return None: the robot holds its last commanded position
    REPEAT_LAST = "repeat_last"  # re-send the last executed action
    ZERO = "zero"  # explicit zero command (required for velocity-controlled robots)


@InferenceEngineConfig.register_subclass("remote")
@dataclass
class RemoteInferenceConfig(InferenceEngineConfig):
    """Network inference against a ``lerobot-policy-server`` over Zenoh.

    The edge stays weightless: ``--policy.path`` resolves to a
    config-only ``PreTrainedConfig`` (no weight download) used for
    pre-flight validation and action ordering.  Requires the ``async``
    extra (``pip install 'lerobot[async]'``).
    """

    # Transport: robots dial out to a zenoh router (NAT-friendly).
    connect_endpoint: str = "tcp/localhost:7447"
    # "client" via a zenohd router (production) | "peer" direct (LAN/tests).
    zenoh_mode: str = "client"
    tls_ca: str | None = None
    tls_cert: str | None = None
    tls_key: str | None = None

    # Service addressing: which (model, revision, task) key tree to dial.
    # service_model_id defaults to --policy.path; service_task to the
    # rollout task.  These must match the server manifest's namespace.
    service_model_id: str = ""
    service_revision: str = "main"
    service_task: str = ""

    # Identity: "" → a fresh uuid4 per run.  Set a stable ID per robot for
    # fleet-wide log correlation and per-robot router ACLs.
    client_uuid: str = ""

    # Observation encoding: JPEG quality (0 = raw, LAN/debug only).
    jpeg_quality: int = 90

    # Self-clocking: request the next chunk when the local queue holds
    # less than this many seconds of playback.
    buffer_time_s: float = 0.5

    # Safety: never execute an action whose source observation is older
    # than this (bounds open-loop execution after a network stall).
    max_action_age_s: float = 3.0
    # Fallback when the queue runs dry (see FallbackMode).
    fallback: FallbackMode = FallbackMode.HOLD

    # Watchdogs & reconnection.
    degraded_after_s: float = 1.0
    request_timeout_s: float = 5.0
    handshake_timeout_s: float = 2.0
    reconnect_initial_backoff_s: float = 0.5
    reconnect_max_backoff_s: float = 10.0
    max_offline_s: float = 60.0

    # RTC settings (enabled → replace-merge with prefix conditioning when
    # the server supports it; otherwise downgraded to chunk-append).
    rtc: RTCConfig = field(default_factory=RTCConfig)

    # Free-form labels forwarded in the session handshake (telemetry only).
    tags: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy | None,
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
    policy_config: PreTrainedConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object.

    ``policy``/``preprocessor``/``postprocessor`` are required for the
    local backends (``sync``, ``rtc``) and must be ``None``-free there;
    the ``remote`` backend is weightless and needs only ``policy_config``.
    """
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, SyncInferenceConfig):
        if policy is None or preprocessor is None or postprocessor is None:
            raise ValueError("sync inference requires a loaded policy and processors")
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        if policy is None or preprocessor is None or postprocessor is None:
            raise ValueError("rtc inference requires a loaded policy and processors")
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    if isinstance(config, RemoteInferenceConfig):
        if policy_config is None:
            raise ValueError("remote inference requires policy_config (from config-only --policy.path)")
        if use_torch_compile:
            logger.warning("--use_torch_compile is ignored with remote inference (server-side concern)")
        if device not in (None, "cpu"):
            logger.warning("--device=%s is ignored with remote inference (server-side concern)", device)
        # Lazy import: eclipse-zenoh/msgpack live behind the 'async' extra.
        from .remote import RemoteInferenceEngine

        return RemoteInferenceEngine(
            config=config,
            policy_config=policy_config,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            fps=fps,
            robot_type=robot_wrapper.robot_type,
            rename_map=rename_map,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
