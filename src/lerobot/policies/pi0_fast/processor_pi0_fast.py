#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ActionTokenizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    load_pretrained_policy_processors,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.tokenizer_processor import ACTION_TOKENIZER_DIR
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.hub import split_hub_reference

from .configuration_pi0_fast import PI0FastConfig


@ProcessorStepRegistry.register(name="pi0_fast_prepare_state_tokenizer_processor_step")
@dataclass
class Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        state = observation.get(OBS_STATE) if observation is not None else None
        if state is None:
            raise ValueError("State is required for PI0Fast")
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is required for PI0Fast")
        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np: np.ndarray = state.cpu().numpy()
        discretized_states = np.asarray(np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1])) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
            full_prompts.append(full_prompt)

        complementary_data[self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi0_fast_pre_post_processors(
    config: PI0FastConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0Fast policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0Fast policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # Pi0Fast order: relative → normalize → tokenize → model → unnormalize → absolute
    # This matches pi0/pi0.5: RelativeActionsProcessorStep runs first on raw absolute actions,
    # caching the raw state. NormalizerProcessorStep then normalizes the raw relative actions,
    # so the normalizer (and action tokenizer) sees delta values — relative stats are required.
    # NOTE: RelativeActionsProcessorStep only modifies the action in the transition; it reads
    # state from the observation but does not change it. NormalizerProcessorStep still runs
    # before Pi0FastPrepareStateAndLanguageTokenizerProcessorStep, so the state tokenizer
    # continues to receive normalized state in [-1, 1] as expected.
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,
        steps.normalize,
        Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        ActionTokenizerProcessorStep(
            action_tokenizer_name=config.action_tokenizer_name,
            trust_remote_code=config.trust_remote_code,
            max_action_tokens=config.max_action_tokens,
            fast_skip_tokens=config.fast_skip_tokens,
            paligemma_tokenizer_name=config.text_tokenizer_name,
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def find_saved_action_tokenizer(
    pretrained_path: str | Path, action_tokenizer_name: str, *, revision: str | None = None
) -> str | None:
    """Return the FAST tokenizer folder the processor saved in a checkpoint, if it can be used.

    The policy decodes with this folder and the processor encodes with it, so both sides use the
    same tokenizer.

    Args:
        pretrained_path: Local checkpoint directory or Hub repository.
        action_tokenizer_name: Tokenizer the policy is configured with.
        revision: Hub revision of the checkpoint.

    Returns:
        The saved folder, or ``None`` when the checkpoint has none (checkpoints saved before the
        processor stored its tokenizer) or was trained with a different ``action_tokenizer_name``.
    """
    if Path(pretrained_path).is_dir():
        root = Path(pretrained_path)
    else:
        try:
            root = Path(
                snapshot_download(
                    str(pretrained_path),
                    allow_patterns=["config.json", f"{ACTION_TOKENIZER_DIR}/*"],
                    revision=revision,
                )
            )
        except OSError:
            # Missing from the Hub, or not cached while offline: use the configured tokenizer.
            return None
    folder = root / ACTION_TOKENIZER_DIR
    config_file = root / "config.json"
    if not (folder / "processor_config.json").is_file() or not config_file.is_file():
        return None
    # Fine-tuning with another tokenizer keeps the configured one.
    if json.loads(config_file.read_text()).get("action_tokenizer_name") != action_tokenizer_name:
        return None
    return str(folder)


def make_pi0_fast_pre_post_processors_from_pretrained(
    config: PI0FastConfig, pretrained_path: str, **kwargs: Any
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Apply repo@revision overrides before loading saved processor steps.

    The action tokenizer step also encodes with the tokenizer the policy decodes with: the copy saved
    in the checkpoint, or ``config.action_tokenizer_name`` when fine-tuning with another tokenizer.
    """
    kwargs.pop("dataset_stats", None)
    kwargs.pop("dataset_meta", None)
    overrides = deepcopy(kwargs.pop("preprocessor_overrides", None) or {})
    for step, key, name in (
        ("tokenizer_processor", "tokenizer_name", config.text_tokenizer_name),
        ("action_tokenizer_processor", "paligemma_tokenizer_name", config.text_tokenizer_name),
        ("action_tokenizer_processor", "action_tokenizer_name", config.action_tokenizer_name),
    ):
        if split_hub_reference(name)[1] is not None:
            overrides.setdefault(step, {}).setdefault(key, name)
    saved_tokenizer = find_saved_action_tokenizer(
        pretrained_path, config.action_tokenizer_name, revision=kwargs.get("revision")
    )
    if saved_tokenizer is None:
        overrides.setdefault("action_tokenizer_processor", {}).setdefault(
            "action_tokenizer_name", config.action_tokenizer_name
        )
    return load_pretrained_policy_processors(
        pretrained_path,
        preprocessor_overrides=overrides,
        trust_remote_code=config.trust_remote_code,
        **kwargs,
    )
