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

Selection is explicit via ``--inference.type=<name>``. Built-in and third-party
backends register a config subclass and an inference-engine builder.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event

import draccus

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


# ---------------------------------------------------------------------------
# Third-party engine builders
# ---------------------------------------------------------------------------


InferenceEngineBuilder = Callable[..., InferenceEngine]
_INFERENCE_ENGINE_BUILDERS: dict[type[InferenceEngineConfig], InferenceEngineBuilder] = {}


def register_inference_engine(
    config_type: type[InferenceEngineConfig],
) -> Callable[[InferenceEngineBuilder], InferenceEngineBuilder]:
    """Register the builder for an inference config type.

    LeRobot already discovers third-party packages before parsing ``lerobot-rollout``
    arguments.  Keeping engine dispatch in this registry lets those packages add an
    ``InferenceEngineConfig`` choice without patching this factory's ``isinstance``
    chain.  Registration is deliberately exact-type and duplicate-safe so a plugin
    cannot silently replace a built-in backend.
    """
    if not issubclass(config_type, InferenceEngineConfig):
        raise TypeError("config_type must inherit InferenceEngineConfig")

    def decorator(builder: InferenceEngineBuilder) -> InferenceEngineBuilder:
        if config_type in _INFERENCE_ENGINE_BUILDERS:
            raise ValueError(f"Inference engine already registered for {config_type.__name__}")
        _INFERENCE_ENGINE_BUILDERS[config_type] = builder
        return builder

    return decorator


@register_inference_engine(SyncInferenceConfig)
def _build_sync_inference_engine(
    config: SyncInferenceConfig,
    **kwargs,
) -> InferenceEngine:
    del config
    robot_wrapper = kwargs["robot_wrapper"]
    return SyncInferenceEngine(
        policy=kwargs["policy"],
        preprocessor=kwargs["preprocessor"],
        postprocessor=kwargs["postprocessor"],
        dataset_features=kwargs["dataset_features"],
        ordered_action_keys=kwargs["ordered_action_keys"],
        task=kwargs["task"],
        device=kwargs["device"],
        robot_type=robot_wrapper.robot_type,
    )


@register_inference_engine(RTCInferenceConfig)
def _build_rtc_inference_engine(
    config: RTCInferenceConfig,
    **kwargs,
) -> InferenceEngine:
    return RTCInferenceEngine(
        policy=kwargs["policy"],
        preprocessor=kwargs["preprocessor"],
        postprocessor=kwargs["postprocessor"],
        robot_wrapper=kwargs["robot_wrapper"],
        rtc_config=config.rtc,
        hw_features=kwargs["hw_features"],
        task=kwargs["task"],
        fps=kwargs["fps"],
        device=kwargs["device"],
        use_torch_compile=kwargs["use_torch_compile"],
        compile_warmup_inferences=kwargs["compile_warmup_inferences"],
        rtc_queue_threshold=config.queue_threshold,
        shutdown_event=kwargs["shutdown_event"],
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
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
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    builder = _INFERENCE_ENGINE_BUILDERS.get(type(config))
    if builder is None:
        raise ValueError(
            f"No inference engine registered for config type: {type(config).__name__}. "
            "Third-party configs must register a builder with register_inference_engine()."
        )
    return builder(
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task,
        fps=fps,
        device=device,
        use_torch_compile=use_torch_compile,
        compile_warmup_inferences=compile_warmup_inferences,
        shutdown_event=shutdown_event,
    )
