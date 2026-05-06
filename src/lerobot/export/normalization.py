# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Normalization statistics extraction and optional ONNX fold.

Reads stats via the public ``step.stats`` attribute on
``NormalizerProcessorStep`` / ``UnnormalizerProcessorStep``. The processor
pipeline must be loaded explicitly via ``make_pre_post_processors`` — calling
``policy_cls.from_pretrained()`` does NOT load preprocessors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lerobot.processor.pipeline import PolicyProcessorPipeline

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Convert tensors / ndarrays to nested Python lists for JSON serialization."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):  # numpy arrays
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _iter_normalizer_steps(pipeline: PolicyProcessorPipeline | None):
    """Yield NormalizerProcessorStep / UnnormalizerProcessorStep instances from a pipeline."""
    if pipeline is None:
        return
    # Local import to avoid circular import at package init.
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )

    for step in getattr(pipeline, "steps", []):
        if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
            yield step


def save_normalization_stats(
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    output_dir: Path | str,
) -> Path:
    """Extract per-feature normalization stats and save them as JSON.

    Walks the preprocessor (and postprocessor) pipelines, collecting stats from
    every ``NormalizerProcessorStep`` / ``UnnormalizerProcessorStep``. The result
    is written to ``normalization_stats.json`` in ``output_dir``.

    The canonical ``policy_preprocessor.json`` + safetensors artifacts should be
    saved separately by the caller via ``preprocessor.save_pretrained(output_dir)``.

    Args:
        preprocessor:  Preprocessor pipeline (input normalization).
        postprocessor: Postprocessor pipeline (action denormalization).
        output_dir:    Directory where ``normalization_stats.json`` will be written.

    Returns:
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, Any]] = {}
    for step in (*_iter_normalizer_steps(preprocessor), *_iter_normalizer_steps(postprocessor)):
        for key, sub in (step.stats or {}).items():
            stats.setdefault(key, {})
            for stat_name, value in sub.items():
                stats[key][stat_name] = _to_jsonable(value)

    out_path = output_dir / "normalization_stats.json"
    out_path.write_text(json.dumps(stats, indent=2))
    if not stats:
        logger.warning(
            "No normalization stats found in preprocessor/postprocessor — wrote empty JSON. "
            "If you expect stats, ensure the pipeline was loaded via make_pre_post_processors() "
            "with the correct pretrained_path."
        )
    else:
        logger.info(f"Normalization stats saved to {out_path} ({len(stats)} features)")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Optional: fold normalization into the ONNX graph
# ──────────────────────────────────────────────────────────────────────────────


class NormalizedWrapper(nn.Module):
    """Wraps a core export wrapper with pre- and post-normalization layers.

    When ``--fold-normalization`` is set, this wrapper is applied before ONNX
    export so that clients do not need to normalize inputs or denormalize outputs.

    Pre-processing per non-image input feature: ``(x - mean) / std``
    Post-processing on the action output:        ``y * std + mean``
    """

    def __init__(
        self,
        core_wrapper: nn.Module,
        input_stats: dict[str, dict[str, Tensor]],
        action_stats: dict[str, Tensor],
    ) -> None:
        super().__init__()
        self.core = core_wrapper
        # Register as buffers so they appear as constants in the ONNX graph
        for feat_key, s in input_stats.items():
            safe_key = feat_key.replace(".", "_")
            if "mean" in s:
                self.register_buffer(f"_in_mean_{safe_key}", s["mean"].float())
            if "std" in s:
                self.register_buffer(f"_in_std_{safe_key}", s["std"].float())

        if "mean" in action_stats:
            self.register_buffer("_out_mean", action_stats["mean"].float())
        if "std" in action_stats:
            self.register_buffer("_out_std", action_stats["std"].float())

        self._input_stats = input_stats
        self._has_action_stats = "mean" in action_stats and "std" in action_stats

    def forward(self, *args: Tensor) -> Tensor:
        # Apply input normalization — works for state/env_state (not image tensors)
        normalized: list[Tensor] = list(args)
        for i, (feat_key, s) in enumerate(self._input_stats.items()):
            if i >= len(normalized):
                break
            if "mean" in s and "std" in s:
                safe_key = feat_key.replace(".", "_")
                mean = getattr(self, f"_in_mean_{safe_key}")
                std = getattr(self, f"_in_std_{safe_key}")
                normalized[i] = (normalized[i] - mean) / (std + 1e-8)

        result = self.core(*normalized)

        # Apply action denormalization
        if self._has_action_stats:
            result = result * self._out_std + self._out_mean

        return result


def _to_tensor_stats(raw_stats: dict[str, Any]) -> dict[str, Tensor]:
    """Convert raw stats dict (lists/ndarrays/tensors) to a dict of float tensors."""
    out: dict[str, Tensor] = {}
    for stat_name, value in raw_stats.items():
        if isinstance(value, Tensor):
            out[stat_name] = value.detach().float()
        else:
            out[stat_name] = torch.as_tensor(value, dtype=torch.float32)
    return out


def build_normalized_wrapper(
    core_wrapper: nn.Module,
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> NormalizedWrapper | nn.Module:
    """Wrap ``core_wrapper`` with pre/post normalization layers.

    Stats are read from the public ``step.stats`` attribute on
    ``NormalizerProcessorStep`` / ``UnnormalizerProcessorStep``. Returns the
    original ``core_wrapper`` unchanged if no stats are found.
    """
    try:
        input_stats: dict[str, dict[str, Tensor]] = {}
        action_stats: dict[str, Tensor] = {}

        for step in (*_iter_normalizer_steps(preprocessor), *_iter_normalizer_steps(postprocessor)):
            for key, sub in (step.stats or {}).items():
                if "action" in key.lower():
                    action_stats = _to_tensor_stats(sub)
                else:
                    input_stats[key] = _to_tensor_stats(sub)

        if not input_stats and not action_stats:
            logger.warning("No normalization stats found — skipping normalization fold.")
            return core_wrapper

        return NormalizedWrapper(core_wrapper, input_stats, action_stats)
    except Exception as exc:
        logger.warning(f"Failed to build NormalizedWrapper ({exc}) — skipping normalization fold.")
        return core_wrapper
