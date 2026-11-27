# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Shared post-preprocessing chunk prediction for local and remote policy owners."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch

from lerobot.lerobot_types import PolicyPrediction
from lerobot.policies import PreTrainedPolicy, reanchor_relative_rtc_prefix
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    PolicyProcessorStep,
)
from lerobot.utils.constants import (
    ACTION,
    ACTION_ANCHOR,
    PREDICTION,
    RTC_INFERENCE_DELAY,
    RTC_PREV_CHUNK_LEFT_OVER,
)

from .contracts import ChunkPolicySpec, ExecutionMode


@contextmanager
def chunk_inference_context(mode: ExecutionMode) -> Iterator[None]:
    """Disable gradients while allowing guided RTC to enable its correction graph.

    Policy owners also use this context for preprocessing, so guided RTC receives
    ordinary tensors rather than tensors created under inference mode.
    """
    with torch.inference_mode(mode is not ExecutionMode.RTC_GUIDED), torch.no_grad():
        yield


@dataclass(frozen=True)
class ChunkPrediction:
    """Validated ``[B, T, A]`` actions on the policy device, before transport packing.

    Model coordinates are detached and cloned before canonical postprocessing can
    mutate them. ``predicted_at`` separates policy and postprocessing diagnostics.
    """

    model_actions: torch.Tensor
    canonical_actions: torch.Tensor
    predicted_at: float
    # What the policy returned besides the actions (see `PreTrainedPolicy.predict_action_chunk`), if anything.
    prediction: PolicyPrediction | None = None


def _prepare_continuation(
    model: torch.Tensor | None,
    canonical: torch.Tensor | None,
    *,
    model_action_dim: int,
    horizon: int,
    device: str | torch.device,
    anchor: torch.Tensor | None,
    normalizer_step: NormalizerProcessorStep | None,
) -> torch.Tensor | None:
    if model is None or model.numel() == 0:
        return None
    if (
        model.ndim != 2
        or model.shape[1] != model_action_dim
        or not model.is_floating_point()
        or not torch.isfinite(model).all()
    ):
        raise ValueError("Invalid model-space continuation.")
    if anchor is not None:
        if canonical is None or canonical.shape != model.shape or not torch.isfinite(canonical).all():
            raise ValueError("Relative RTC requires matching finite canonical continuation.")
        model = reanchor_relative_rtc_prefix(canonical, anchor, normalizer_step, device)
    model = model.to(device)
    if len(model) < horizon:
        # Zero in normalized coordinates decodes to the dataset mean, not hold.
        model = torch.cat([model, model[-1:].expand(horizon - len(model), -1)])
    return model[:horizon]


def predict_chunk(
    policy: PreTrainedPolicy,
    postprocessor: PolicyProcessorPipeline,
    prepared: dict[str, Any],
    *,
    spec: ChunkPolicySpec,
    mode: ExecutionMode,
    canonical_action_dim: int,
    model_action_dim: int | None,
    device: str | torch.device,
    rtc_horizon: int = 0,
    inference_delay: int = 0,
    model_continuation: torch.Tensor | None = None,
    canonical_continuation: torch.Tensor | None = None,
    relative: bool = False,
    normalizer_step: NormalizerProcessorStep | None = None,
) -> ChunkPrediction:
    """Predict and validate under exclusive policy/processor ownership.

    ``prepared`` is the preprocessor's output. The policy runs as a
    :class:`~lerobot.processor.PolicyProcessorStep` and the postprocessor on what it
    returned, so the model chunk (``prediction["action"]``) and the canonical command
    (``action``) come out of one pass. Retain the first model width across calls; canonical
    postprocessing may crop padding; relative RTC requires matching widths and re-anchors
    the continuation on the batch's ``action_anchor``. A complete-execution policy may
    return fewer than ``prediction_steps`` actions. Plain chunks may be batched (batched
    environments); RTC continues a single robot's chunk. This function owns no scheduling.
    """
    batch = dict(prepared)
    with chunk_inference_context(mode):
        if mode is not ExecutionMode.CHUNK:
            prefix = _prepare_continuation(
                model_continuation,
                canonical_continuation,
                model_action_dim=model_action_dim or canonical_action_dim,
                horizon=rtc_horizon,
                device=device,
                anchor=batch.get(ACTION_ANCHOR) if relative else None,
                normalizer_step=normalizer_step,
            )
            if inference_delay < 0:
                raise ValueError("RTC delay must be nonnegative.")
            if prefix is None and inference_delay:
                raise ValueError("RTC delay requires a real continuation prefix.")
            if mode is ExecutionMode.RTC_TRAINED:
                available = 0 if model_continuation is None else len(model_continuation)
                if inference_delay > min(spec.training_max_delay, available, rtc_horizon):
                    raise ValueError("Trained RTC delay exceeds checkpoint or available continuation limits.")
            batch[RTC_INFERENCE_DELAY] = inference_delay
            batch[RTC_PREV_CHUNK_LEFT_OVER] = prefix
        output = PolicyProcessorStep(policy).predict(batch)
        predicted_at = time.perf_counter()
        prediction: dict[str, Any] = dict(output[PREDICTION])
        predicted = prediction.pop("action")
        if (
            predicted.ndim != 3
            or (predicted.shape[0] != 1 and mode is not ExecutionMode.CHUNK)
            or not spec.accepts(predicted.shape[1])
            or predicted.shape[2] <= 0
            or not predicted.is_floating_point()
            or not torch.isfinite(predicted).all()
        ):
            raise ValueError(
                f"{type(policy).__name__} must return [B, {spec.prediction_steps}, A] finite floating "
                f"actions (B = 1 with RTC), got {tuple(predicted.shape)}."
            )
        if model_action_dim is not None and predicted.shape[2] != model_action_dim:
            raise ValueError("Policy model action width changed after initial prediction.")
        # Older local policies can omit action feature metadata. Keep their
        # existing model-width fallback; remote admission requires explicit width.
        canonical_width = canonical_action_dim or predicted.shape[2]
        if relative and rtc_horizon and predicted.shape[2] != canonical_width:
            raise ValueError("Relative RTC with different model/canonical widths requires a custom adapter.")
        canonical = postprocessor(output)[ACTION]
        prediction_steps = predicted.shape[1]
        expected = (predicted.shape[0], prediction_steps, canonical_width)
        if not isinstance(canonical, torch.Tensor) or tuple(canonical.shape) != expected:
            raise ValueError(f"Policy/processor must return action shape {expected}.")
        if not canonical.is_floating_point() or not torch.isfinite(canonical).all():
            raise ValueError("Policy/processor returned non-finite or non-floating actions.")
        steps = spec.steps_to_execute(prediction_steps) if mode is ExecutionMode.CHUNK else prediction_steps
        return ChunkPrediction(
            predicted.detach()[:, :steps],
            canonical[:, :steps].detach(),
            predicted_at,
            cast(PolicyPrediction, prediction) if prediction else None,
        )
