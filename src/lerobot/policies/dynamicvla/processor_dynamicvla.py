#!/usr/bin/env python

# Copyright 2026 S-Lab and The HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoTokenizer

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NewLineTaskProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE

from .configuration_dynamicvla import DynamicVLAConfig

# Number of per-observation states kept for anchoring streamed action chunks
_MAX_ANCHOR_STATES = 64


@ProcessorStepRegistry.register("dynamicvla_relative_actions_processor")
@dataclass
class DynamicVLARelativeActionsProcessorStep(RelativeActionsProcessorStep):
    """Relative (delta) actions as defined by DynamicVLA.

    Compared with `RelativeActionsProcessorStep`:
    - the gripper (the last action dimension) stays absolute, so the state may have fewer
      dimensions than the action;
    - a temporally stacked state ends at the current frame (e.g. offsets [-2, 0]);
    - with streaming inference, an action chunk is anchored to the state of the observation it was
      predicted from, looked up by the observation `index` returned by `anchor_index_fn`.
    """

    _anchor_index_fn: Callable[[], int | None] | None = field(default=None, init=False, repr=False)
    _states_by_index: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        return [True] * (action_dim - 1)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None:
            return super().__call__(transition)

        if state.ndim == 3:
            state = state[:, -1]
        index = (transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}).get("index")
        if isinstance(index, int):
            self._states_by_index[index] = state
            if len(self._states_by_index) > _MAX_ANCHOR_STATES:
                self._states_by_index.pop(min(self._states_by_index))

        # The parent caches and subtracts the current state; keep the stacked state for the model
        new_transition = super().__call__(
            {**transition, TransitionKey.OBSERVATION: {**observation, OBS_STATE: state}}
        )
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def bind_anchor_index(self, anchor_index_fn: Callable[[], int | None] | None) -> None:
        self._anchor_index_fn = anchor_index_fn

    def get_cached_state(self) -> torch.Tensor | None:
        index = self._anchor_index_fn() if self._anchor_index_fn is not None else None
        if index in self._states_by_index:
            return self._states_by_index[index]
        return super().get_cached_state()

    def reset(self) -> None:
        super().reset()
        self._states_by_index.clear()


def make_dynamicvla_pre_post_processors(
    config: DynamicVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the DynamicVLA policy.

    The pre-processing pipeline:
    1.  Renames features to match pretrained configurations.
    2.  Adds a batch dimension.
    3.  Converts actions to relative actions w.r.t. the raw state (if `use_delta_action`).
    4.  Ensures the task description ends with a newline and tokenizes it.
    5.  Moves all data to the specified device.
    6.  Normalizes input and output features based on dataset statistics.

    The post-processing pipeline:
    1.  Unnormalizes the output actions to their original scale.
    2.  Converts relative actions back to absolute actions (if `use_delta_action`).
    3.  Moves data to the CPU.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)

    tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    relative_step = DynamicVLARelativeActionsProcessorStep(enabled=config.use_delta_action)
    input_steps = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,  # Before normalization: deltas are taken in the raw space
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer=tokenizer,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_delta_action, relative_step=relative_step),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
