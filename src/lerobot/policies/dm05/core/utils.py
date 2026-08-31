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

import torch

IGNORE_INDEX = -100
DM05_STATE_BINS = 256


def build_action_prefix_mask(
    action_prefill_len: torch.Tensor | None,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a mask for timesteps that are fixed by action prefill."""
    if action_prefill_len is None:
        return None
    lengths = action_prefill_len.to(device=device, dtype=torch.long)
    positions = torch.arange(horizon, device=device)
    return positions[None, :] < lengths[:, None]


def validate_action_prefill_pair(
    prefill_actions: torch.Tensor | None,
    action_prefill_len: torch.Tensor | None,
) -> None:
    """Require action prefill values and lengths to be provided together."""
    if (prefill_actions is None) != (action_prefill_len is None):
        raise ValueError("prefill_actions and action_prefill_len must be provided together.")
