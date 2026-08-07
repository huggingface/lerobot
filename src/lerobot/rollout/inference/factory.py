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

Selection is explicit via ``--inference.type=sync|rtc``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
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
        """The registered name of this backend (e.g. `"sync"`, `"rtc"`)."""
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread.

    Args:
        rtc (`RTCConfig`, *optional*):
            RTC-specific configuration (e.g. prefix-attention schedule, execution horizon). Eagerly
            constructed so draccus exposes nested fields directly on the CLI (e.g.
            `--inference.rtc.execution_horizon=...`).
        queue_threshold (`int`, *optional*, defaults to 30):
            Action-queue size below which the background RTC thread starts producing a new chunk.
    """

    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


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
    """Instantiate the appropriate inference engine from a config object.

    Args:
        config (`InferenceEngineConfig`):
            Backend selector (`SyncInferenceConfig` or `RTCInferenceConfig`).
        policy (`PreTrainedPolicy`):
            The loaded policy to run inference with.
        preprocessor (`PolicyProcessorPipeline`):
            Observation pre-processor pipeline.
        postprocessor (`PolicyProcessorPipeline`):
            Action post-processor pipeline.
        robot_wrapper (`ThreadSafeRobot`):
            Thread-safe robot handle, used for RTC's background thread and to resolve `robot_type`.
        hw_features (`dict`):
            Raw hardware observation feature spec, used by RTC to rebuild dataset frames.
        dataset_features (`dict`):
            Dataset feature spec, used by sync inference to reorder policy outputs.
        ordered_action_keys (`list[str]`):
            Action key ordering the returned tensor should be mapped to.
        task (`str`):
            Task string passed through to the policy.
        fps (`float`):
            Control loop frequency, used by RTC to size its time-per-chunk estimate.
        device (`str | None`):
            Torch device to run inference on.
        use_torch_compile (`bool`, *optional*, defaults to `False`):
            Whether to `torch.compile` the policy's action-prediction call.
        compile_warmup_inferences (`int`, *optional*, defaults to 2):
            Number of warmup inferences before compiled inference is considered ready.
        shutdown_event (`Event | None`, *optional*):
            Global shutdown event RTC sets on an unrecoverable background-thread error.

    Returns:
        InferenceEngine: The instantiated `SyncInferenceEngine` or `RTCInferenceEngine`.

    Raises:
        ValueError: If `config` is not a recognized `InferenceEngineConfig` subclass.
    """
    logger.info("Creating inference engine: %s", config.type)
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
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
