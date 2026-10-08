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

Selection is explicit via ``--inference.type=sync|rtc|remote``. This factory builds
local backends with an already loaded policy. Remote inference is dispatched by
``build_rollout_context`` to ``build_remote_rollout_context``, which establishes
the remote session instead of loading a local policy.
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from threading import Event
from typing import Literal

import draccus

from lerobot.policies import PreTrainedPolicy
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine, InferenceRobot
from .chunk_settings import chunk_settings
from .contracts import ExecutionMode
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

    # INFO summarizes operation; DEBUG includes request-level diagnostics.
    log_level: Literal["INFO", "DEBUG"] = "INFO"

    def __post_init__(self) -> None:
        """Validate the console verbosity shared by every backend."""
        if self.log_level not in {"INFO", "DEBUG"}:
            raise ValueError("Inference log_level must be INFO or DEBUG")

    @property
    def type(self) -> str:
        """Return the CLI registry name for this backend."""
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
    max_observation_age_s: float = 5.0
    action_timeout_s: float = 10.0
    startup_timeout_s: float = 120.0
    language_timeout_s: float = 120.0
    action_starvation_grace_s: float = 1.0

    def __post_init__(self) -> None:
        """Validate the bounded local waiting budget."""
        super().__post_init__()
        if not math.isfinite(self.action_starvation_grace_s) or self.action_starvation_grace_s < 0:
            raise ValueError("Action starvation grace must be finite and nonnegative")


@InferenceEngineConfig.register_subclass("remote")
@dataclass
class RemoteInferenceConfig(InferenceEngineConfig):
    """Exclusive remote deployment with explicit robot semantics.

    Timing defaults are conservative starting values for lab validation; operators must
    measure their deployment's turnaround and choose a usable playback/age budget.
    """

    endpoint: str = "tcp/127.0.0.1:7447"
    deployment: str = ""
    instance: str | None = None
    expected_artifact: str | None = None
    mode: str = "chunk"
    semantics: str = ""
    # Request when remaining playback reaches this threshold (append, aligned and RTC).
    # Effective floor: recent maximum turnaround + one policy action interval.
    # Smaller values allow more aligned follow-through; aligned task changes bypass this gate.
    refill_seconds: float = 0.5
    # Auto selects aligned plain chunks, leaving RTC continuation unchanged.
    chunk_merge: Literal["auto", "append", "aligned"] = "auto"
    blend_steps: int = 0
    blend_weight: float = 0.5
    blend_components: list[str] = field(default_factory=list)
    max_observation_age_s: float = 5.0
    handshake_timeout_s: float = 10.0
    action_timeout_s: float = 5.0
    language_timeout_s: float = 60.0
    startup_timeout_s: float = 10.0
    action_starvation_grace_s: float = 1.0
    encoding: Literal["raw", "jpeg"] = "raw"
    jpeg_quality: int = 90
    zenoh_config_path: str | None = None
    zenoh_mode: Literal["peer", "client"] = "peer"

    def __post_init__(self) -> None:
        """Validate transport and scheduling settings."""
        super().__post_init__()
        if not math.isfinite(self.action_starvation_grace_s) or self.action_starvation_grace_s < 0:
            raise ValueError("Action starvation grace must be finite and nonnegative")
        if self.zenoh_mode not in {"peer", "client"}:
            raise ValueError("zenoh_mode must be peer (direct) or client (router)")
        if not self.deployment or not self.semantics:
            raise ValueError(
                "Remote inference requires --inference.deployment and explicit robot/action "
                "semantics via --inference.semantics"
            )
        try:
            ExecutionMode(self.mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported remote execution mode: {self.mode!r}") from exc
        if self.chunk_merge == "auto":
            self.chunk_merge = "aligned" if self.mode == "chunk" else "append"
        chunk_settings(self.chunk_merge, self.blend_steps, self.blend_weight, self.blend_components)
        if self.mode != "chunk" and self.chunk_merge != "append":
            raise ValueError("chunk_merge=aligned is available only with mode=chunk")
        if self.encoding not in {"raw", "jpeg"} or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("Remote encoding must be raw or jpeg, with jpeg_quality in [1, 100]")
        for name in (
            "refill_seconds",
            "max_observation_age_s",
            "handshake_timeout_s",
            "action_timeout_s",
            "language_timeout_s",
            "startup_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Remote {name} must be finite and positive")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: InferenceRobot,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
) -> InferenceEngine:
    """Build a local engine; remote sessions are built by ``build_rollout_context``."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, RemoteInferenceConfig):
        raise ValueError(
            "Remote inference requires a connected RemoteClient; use build_rollout_context "
            "(which delegates to build_remote_rollout_context) instead of create_inference_engine"
        )
    if isinstance(config, SyncInferenceConfig):
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
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            dataset_features=dataset_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            max_observation_age_s=config.max_observation_age_s,
            action_timeout_s=config.action_timeout_s,
            startup_timeout_s=config.startup_timeout_s,
            language_timeout_s=config.language_timeout_s,
            action_starvation_grace_s=config.action_starvation_grace_s,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
