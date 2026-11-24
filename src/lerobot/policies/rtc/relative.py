#!/usr/bin/env python

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

"""Relative-action helpers for Real-Time Chunking (RTC)."""

from __future__ import annotations

import torch

from lerobot.processor import (
    NormalizerProcessorStep,
    TransitionKey,
    create_transition,
    shift_actions,
)


def reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    anchor: torch.Tensor,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    """Convert absolute leftover actions into model-space for relative-action RTC policies.

    When using relative actions, the RTC prefix (previous chunk's unexecuted tail) is stored
    in absolute coordinates. Before feeding it back to the policy, this helper re-expresses
    those actions relative to the new chunk's ``action_anchor`` (the offset the relative
    preprocessor step wrote for the current state) and optionally normalizes them so the
    policy receives correctly scaled inputs.
    """
    relative_actions = shift_actions(prev_actions_absolute.detach().cpu(), anchor.detach().cpu(), sign=-1)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    action = transition[TransitionKey.ACTION]
    if not isinstance(action, torch.Tensor):
        raise ValueError(f"Expected a tensor action after normalization, got {type(action).__name__}.")
    return action.to(policy_device)
