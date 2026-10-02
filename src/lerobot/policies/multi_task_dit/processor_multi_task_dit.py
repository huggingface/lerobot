#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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
from torch import Tensor

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.pipeline import RobotObservation
from lerobot.utils.constants import OBS_LANGUAGE, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_multi_task_dit import MultiTaskDiTConfig

# Maps each sample in the batch to its row in the de-duplicated token tensors.
OBS_LANGUAGE_ROW_INDEX = OBS_LANGUAGE + ".row_index"


@dataclass
@ProcessorStepRegistry.register(name="multi_task_dit_task_tokenizer")
class DedupTaskTokenizerProcessorStep(TokenizerProcessorStep):
    """Tokenizes each distinct task string once per batch, and only once ever per string.

    A batch of hundreds of samples carries a handful of distinct task strings, and the frozen
    CLIP text tower is a pure function of the tokens. This step emits tokens for the distinct
    strings only, plus `observation.language.row_index` mapping every sample to its row, so
    the policy runs the tower on U rows instead of B and gathers. Tokenization is cached per
    string, so the Hugging Face tokenizer runs once per task over a whole training run
    rather than on every batch. Done here, on host strings, rather than with `torch.unique`
    on the device, which sorts rows and forces a device sync.

    Subtasks are not tokenized; this policy does not consume them.
    """

    def __post_init__(self):
        super().__post_init__()
        self._token_cache: dict[str, tuple[Tensor, Tensor]] = {}

    def observation(self, observation: RobotObservation) -> RobotObservation:
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        unique = list(dict.fromkeys(task))
        missing = [t for t in unique if t not in self._token_cache]
        if missing:
            tokenized = self._tokenize_text(missing)
            for i, text in enumerate(missing):
                self._token_cache[text] = (
                    tokenized["input_ids"][i],
                    tokenized["attention_mask"][i].to(dtype=torch.bool),
                )
        row_of = {text: i for i, text in enumerate(unique)}
        tokens = torch.stack([self._token_cache[t][0] for t in unique])
        mask = torch.stack([self._token_cache[t][1] for t in unique])
        row_index = torch.tensor([row_of[t] for t in task], dtype=torch.long)

        target_device = self._detect_device(self.transition)
        if target_device is not None:
            tokens, mask, row_index = (
                tokens.to(target_device),
                mask.to(target_device),
                row_index.to(target_device),
            )

        new_observation = dict(observation)
        new_observation[OBS_LANGUAGE_TOKENS] = tokens
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = mask
        new_observation[OBS_LANGUAGE_ROW_INDEX] = row_index
        return new_observation


def make_multi_task_dit_pre_post_processors(
    config: MultiTaskDiTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a Multi-Task DiT policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Adding a batch dimension.
    3. Tokenizing the distinct language task descriptions (if present), with a per-sample row index.
    4. Moving the data to the specified device.
    5. Normalizing the input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output features to their original scale.
    2. Moving the data to the CPU.

    Args:
        config: The configuration object for the Multi-Task DiT policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        DedupTaskTokenizerProcessorStep(
            tokenizer_name=config.text_encoder_name,
            padding=config.tokenizer_padding,
            padding_side=config.tokenizer_padding_side,
            max_length=config.tokenizer_max_length,
            truncation=config.tokenizer_truncation,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
