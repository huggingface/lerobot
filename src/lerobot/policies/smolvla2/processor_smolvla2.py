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
"""SmolVLA2 processor pipelines.

When ``config.recipe_path`` is set, the pre-processor pipeline becomes:

    rename observations
    add batch dim
    RenderMessagesStep(recipe)              # PR 1: language_*  → messages
    SmolVLA2ChatTokenizerStep(...)          # chat template + label mask + predict_actions
    DeviceProcessorStep
    NormalizerProcessorStep

When ``config.recipe_path`` is ``None``, we delegate to SmolVLA's
plain task-string pipeline so unannotated datasets still work.

Post-processor is unchanged from SmolVLA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lerobot.configs.recipe import TrainingRecipe
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    RenderMessagesStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from ..smolvla.processor_smolvla import make_smolvla_pre_post_processors
from .chat_processor_smolvla2 import SmolVLA2ChatTokenizerStep
from .configuration_smolvla2 import SmolVLA2Config


def make_smolvla2_pre_post_processors(
    config: SmolVLA2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build SmolVLA2's pre/post-processor pipelines.

    With ``recipe_path`` set, inserts the recipe-rendering step and the
    chat-template tokenizer that emits ``text_labels`` and
    ``predict_actions`` for the dual-loss path. Without it, falls back
    to SmolVLA's plain task-string pipeline so unannotated datasets
    keep working unchanged.
    """
    if not config.recipe_path:
        return make_smolvla_pre_post_processors(config, dataset_stats=dataset_stats)

    recipe = _load_recipe(config.recipe_path)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        RenderMessagesStep(recipe=recipe),
        SmolVLA2ChatTokenizerStep(
            tokenizer_name=config.vlm_model_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            plan_dropout_prob=getattr(config, "plan_dropout_prob", 0.0),
            memory_dropout_prob=getattr(config, "memory_dropout_prob", 0.0),
            subtask_dropout_prob=getattr(config, "subtask_dropout_prob", 0.0),
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _load_recipe(path_str: str) -> TrainingRecipe:
    """Resolve ``path_str`` to a ``TrainingRecipe``.

    Accepts an absolute path or a path relative to
    ``src/lerobot/configs/`` so recipe authors can write
    ``--policy.recipe_path=recipes/hirobot.yaml``.
    """
    p = Path(path_str)
    if not p.is_absolute() and not p.exists():
        from lerobot.configs import recipe as _recipe_module  # noqa: PLC0415

        configs_dir = Path(_recipe_module.__file__).resolve().parent
        candidate = configs_dir / path_str
        if candidate.exists():
            p = candidate
    return TrainingRecipe.from_yaml(p)
