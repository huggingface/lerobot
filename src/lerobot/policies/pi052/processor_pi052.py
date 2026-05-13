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

"""π0.5 v2 pre/post-processor factory.

When ``config.recipe_path`` is set, the pre-processor pipeline becomes:

    rename observations
    add batch dim
    relative-action prep      (inherited from π0.5)
    NormalizerProcessorStep
    RenderMessagesStep        — recipe → messages, target_message_indices,
                                message_streams (PR 1 of the steerable
                                stack)
    PI052TextTokenizerStep    — messages → input_ids + label mask +
                                predict_actions
    DeviceProcessorStep

When ``recipe_path`` is ``None`` we delegate to the plain π0.5 pipeline
so unannotated datasets keep working.

Post-processor is unchanged from π0.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lerobot.configs.recipe import TrainingRecipe
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    RenderMessagesStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from ..pi05.processor_pi05 import make_pi05_pre_post_processors
from .configuration_pi052 import PI052Config
from .text_processor_pi052 import PI052TextTokenizerStep


def make_pi052_pre_post_processors(
    config: PI052Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build PI0.5-v2's pre/post-processor pipelines.

    Falls through to π0.5's stock pipeline when ``recipe_path`` is unset.
    """
    if not config.recipe_path:
        return make_pi05_pre_post_processors(config, dataset_stats=dataset_stats)

    recipe = _load_recipe(config.recipe_path)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        RenderMessagesStep(recipe=recipe),
        PI052TextTokenizerStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            plan_dropout_prob=getattr(config, "plan_dropout_prob", 0.0),
            memory_dropout_prob=getattr(config, "memory_dropout_prob", 0.0),
            subtask_dropout_prob=getattr(config, "subtask_dropout_prob", 0.0),
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(
            enabled=config.use_relative_actions,
            relative_step=relative_step,
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
    ``src/lerobot/configs/`` (same lookup rules as
    ``make_smolvla2_pre_post_processors``).
    """
    p = Path(path_str)
    if not p.is_absolute() and not p.exists():
        from lerobot.configs import recipe as _recipe_module  # noqa: PLC0415

        configs_dir = Path(_recipe_module.__file__).resolve().parent
        candidate = configs_dir / path_str
        if candidate.exists():
            p = candidate
    return TrainingRecipe.from_yaml(p)
