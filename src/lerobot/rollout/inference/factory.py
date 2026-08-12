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
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""

    # When True, postprocessed action chunks are cached locally and served one
    # per tick, so the policy (and the per-tick observation upload + normalize)
    # only runs when the cache is empty.  Currently supported only for ACT
    # policies without temporal ensembling, where it is behaviour-preserving.
    # Disabled by default; the factory raises for unsupported policies.
    chunked_action_cache: bool = False

    # When True (requires chunked_action_cache), the next chunk is computed by
    # a background thread once the cache drops to ``prefetch_watermark``
    # actions, instead of blocking the control loop when the cache empties.
    # The first ``watermark`` actions of the prefetched chunk are skipped so it
    # stays time-aligned with the actions served from the old cache in the
    # meantime.  This removes the periodic inference stall from the control
    # loop entirely.
    prefetch_chunks: bool = False

    # Cache level (in actions) at which the background prefetch is triggered.
    # Must be >= 1 and < the policy's n_action_steps.  At 30 FPS the default of
    # 20 gives ~0.66 s of runway to hide the forward pass.
    prefetch_watermark: int = 20


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_chunk_action_steps(policy_config) -> int:
    """Validate the policy and return how many chunk actions to cache per query.

    The chunked-action cache bypasses ``select_action``, so it is only enabled
    for policies where serving a cached chunk is behaviour-preserving.  Today
    that is ACT without temporal ensembling: ``select_action`` ignores the
    observation while its internal queue is non-empty, and the postprocessor is
    a stateless per-action transform.  Anything else raises a clear error.
    """
    policy_type = getattr(policy_config, "type", None)
    if policy_type != "act":
        raise ValueError(
            f"chunked_action_cache currently supports only ACT policies, got '{policy_type}'. "
            "Disable inference.chunked_action_cache for this policy."
        )
    if getattr(policy_config, "temporal_ensemble_coeff", None) is not None:
        raise ValueError(
            "chunked_action_cache is incompatible with ACT temporal ensembling: the ensembler "
            "requires a fresh forward pass every step. Disable one of the two."
        )
    n_action_steps = getattr(policy_config, "n_action_steps", None)
    if not isinstance(n_action_steps, int) or n_action_steps < 1:
        raise ValueError(
            f"chunked_action_cache requires a positive integer n_action_steps, got {n_action_steps!r}."
        )
    return n_action_steps


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
    if isinstance(config, SyncInferenceConfig):
        chunk_action_steps = None
        if config.chunked_action_cache:
            chunk_action_steps = _resolve_chunk_action_steps(policy.config)
        prefetch_watermark = None
        if config.prefetch_chunks:
            if chunk_action_steps is None:
                raise ValueError(
                    "inference.prefetch_chunks requires inference.chunked_action_cache=true "
                    "(the prefetch worker produces whole chunks for the action cache)."
                )
            if not (1 <= config.prefetch_watermark < chunk_action_steps):
                raise ValueError(
                    f"inference.prefetch_watermark must be in [1, n_action_steps), got "
                    f"{config.prefetch_watermark} with n_action_steps={chunk_action_steps}."
                )
            prefetch_watermark = config.prefetch_watermark
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
            chunk_action_steps=chunk_action_steps,
            prefetch_watermark=prefetch_watermark,
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
