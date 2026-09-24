#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    load_pretrained_policy_processors,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_dm05 import DM05Config
from .conversion_dm05 import (
    DM05ClipNormalizedProcessorStep,
    DM05StateBinsProcessorStep,
    DM05TokenizerProcessorStep,
)


@dataclass
@ProcessorStepRegistry.register(name="dm05_task_processor")
class DM05TaskProcessor(ComplementaryDataProcessorStep):
    """Normalize the task prompt field expected by DM05 tokenization."""

    default_task: str = "Execute the robot action."

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize missing or blank task prompts in complementary data."""
        if (task := complementary_data.get("task")) is None:
            return {**complementary_data, "task": self.default_task}

        if isinstance(task, str):
            return {**complementary_data, "task": task.strip() or self.default_task}
        if isinstance(task, list):
            return {
                **complementary_data,
                "task": [str(item).strip() or self.default_task for item in task],
            }
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        """Return the serializable processor-step configuration."""
        return {"default_task": self.default_task}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave policy feature metadata unchanged."""
        return features


def dm05_image_keys(config: DM05Config) -> list[str]:
    """Return the ordered camera set the tokenizer renders, pinned from the policy config.

    Pinned rather than read from each observation, so an extra camera at inference cannot change
    the prompt with no error. Empty for a base checkpoint, whose cameras come from the dataset.
    """
    image_keys = (
        list(config.image_keys)
        if config.image_keys
        else sorted(
            key for key, feature in config.input_features.items() if feature.type is FeatureType.VISUAL
        )
    )
    if unknown := [key for key in image_keys if key not in config.input_features]:
        raise ValueError(f"DM05 image_keys are not declared in input_features: {unknown}.")
    return image_keys


def dm05_clip_flags(config: DM05Config) -> dict[str, bool]:
    """Return which quantile-normalized fields are clipped to the range DM05 was trained on."""

    def clips_quantiles(feature_type: str) -> bool:
        return config.norm_clip and config.normalization_mapping.get(feature_type) in {
            NormalizationMode.QUANTILES,
            NormalizationMode.QUANTILE10,
        }

    return {"clip_state": clips_quantiles("STATE"), "clip_action": clips_quantiles("ACTION")}


def make_dm05_pre_post_processors(
    config: DM05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the LeRobot processor pipeline for the OpenDM adapter."""

    config.validate_features()
    # OpenDM normalizes only numeric state/action fields.
    normalizer = NormalizerProcessorStep(
        features={
            OBS_STATE: config.input_features[OBS_STATE],
            ACTION: config.output_features[ACTION],
        },
        norm_map=config.normalization_mapping,
        stats=dataset_stats,
        normalize_observation_keys={OBS_STATE},
        eps=1e-6,
    )
    unnormalizer = UnnormalizerProcessorStep(
        features=config.output_features,
        norm_map=config.normalization_mapping,
        stats=dataset_stats,
        eps=1e-6,
    )
    relative_actions = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    processor_source = config.processor_name_or_path or config.pretrained_name_or_path
    if not processor_source:
        raise ValueError("DM05 requires processor_name_or_path when creating a new processor pipeline.")

    if not (image_keys := dm05_image_keys(config)):
        raise ValueError("DM05 requires at least one visual input feature.")

    return make_policy_processor_pipelines(
        input_steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            DM05TaskProcessor(),
            relative_actions,
            normalizer,
            DM05ClipNormalizedProcessorStep(**dm05_clip_flags(config)),
            DM05StateBinsProcessorStep(),
            DeviceProcessorStep(device=config.device),
            DM05TokenizerProcessorStep(
                processor_name_or_path=processor_source,
                tokenizer_max_length=config.tokenizer_max_length,
                add_state=config.add_state,
                image_keys=image_keys,
            ),
        ],
        output_steps=[
            DeviceProcessorStep(device="cpu", float_dtype="float32"),
            unnormalizer,
            AbsoluteActionsProcessorStep(
                enabled=config.use_relative_actions,
                relative_step=relative_actions,
            ),
        ],
    )


def make_dm05_pre_post_processors_from_pretrained(
    config: DM05Config,
    pretrained_path: str,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the checkpoint's pipelines, re-pinning the settings the active config owns.

    The camera set, state prompting, prompt length and clipping belong to the policy config, so a
    fine-tuning run honours them instead of the saved values: a base checkpoint picks up the
    dataset's cameras, and `--policy.add_state=false` takes effect. Loading for eval or resume uses
    the checkpoint's own config, so this changes nothing there. Explicit overrides still win.
    """
    # Fine-tuning stats arrive as the caller's normalizer overrides (lerobot-train injects the
    # dataset's); eval and resume keep the checkpoint's own, which the model was trained against.
    del dataset_stats, dataset_meta
    config_overrides = {
        "dm05_tokenizer_processor": {
            "image_keys": dm05_image_keys(config),
            "add_state": config.add_state,
            "tokenizer_max_length": config.tokenizer_max_length,
        },
        "dm05_clip_normalized_processor": dm05_clip_flags(config),
    }
    for step, values in (preprocessor_overrides or {}).items():
        config_overrides[step] = {**config_overrides.get(step, {}), **values}
    return load_pretrained_policy_processors(
        pretrained_path,
        revision=revision,
        preprocessor_overrides=config_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
