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

from typing import NotRequired, TypedDict

import torch

from .constants import ACTION


class Transition(TypedDict):
    """One RL environment transition.

    Args:
        state (`dict[str, torch.Tensor]`): Observation before the action.
        action (`torch.Tensor`): Action taken.
        reward (`float`): Reward received.
        next_state (`dict[str, torch.Tensor]`): Observation after the action.
        done (`bool`): Whether the episode terminated.
        truncated (`bool`): Whether the episode was truncated (time limit).
        complementary_info (`dict[str, torch.Tensor | float | int] | None`, *optional*): Extra
            per-transition metadata (e.g. for reward shaping or debugging).
    """

    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: dict[str, torch.Tensor]
    done: bool
    truncated: bool
    complementary_info: NotRequired[dict[str, torch.Tensor | float | int] | None]


def move_transition_to_device(transition: Transition, device: str = "cpu") -> Transition:
    """Move every tensor field of `transition` (in place) to `device`.

    Args:
        transition (`Transition`): Transition to move.
        device (`str`, *optional*, defaults to `"cpu"`): Target device.

    Returns:
        `Transition`: The same `transition` dict, mutated in place.

    Raises:
        ValueError: If a `complementary_info` value isn't a tensor, `int`, `float`, or `bool`.
    """
    device = torch.device(device)
    non_blocking = device.type == "cuda"

    # Move state tensors to device
    transition["state"] = {
        key: val.to(device, non_blocking=non_blocking) for key, val in transition["state"].items()
    }

    # Move action to device
    transition[ACTION] = transition[ACTION].to(device, non_blocking=non_blocking)

    # Move reward and done if they are tensors
    if isinstance(transition["reward"], torch.Tensor):
        transition["reward"] = transition["reward"].to(device, non_blocking=non_blocking)

    if isinstance(transition["done"], torch.Tensor):
        transition["done"] = transition["done"].to(device, non_blocking=non_blocking)

    if isinstance(transition["truncated"], torch.Tensor):
        transition["truncated"] = transition["truncated"].to(device, non_blocking=non_blocking)

    # Move next_state tensors to device
    transition["next_state"] = {
        key: val.to(device, non_blocking=non_blocking) for key, val in transition["next_state"].items()
    }

    # Move complementary_info tensors if present
    if transition.get("complementary_info") is not None:
        for key, val in transition["complementary_info"].items():
            if isinstance(val, torch.Tensor):
                transition["complementary_info"][key] = val.to(device, non_blocking=non_blocking)
            elif isinstance(val, (int | float | bool)):
                transition["complementary_info"][key] = torch.tensor(val, device=device)
            else:
                raise ValueError(f"Unsupported type {type(val)} for complementary_info[{key}]")
    return transition


def move_state_dict_to_device(state_dict, device="cpu"):
    """Recursively move all tensors in a (potentially) nested dict/list/tuple structure to `device`.

    Args:
        state_dict (`Any`): A tensor, or a nested dict/list/tuple structure containing tensors.
        device (`str`, *optional*, defaults to `"cpu"`): Target device.

    Returns:
        `Any`: The same structure, with every tensor moved to `device`.
    """
    if isinstance(state_dict, torch.Tensor):
        return state_dict.to(device)
    elif isinstance(state_dict, dict):
        return {k: move_state_dict_to_device(v, device=device) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [move_state_dict_to_device(v, device=device) for v in state_dict]
    elif isinstance(state_dict, tuple):
        return tuple(move_state_dict_to_device(v, device=device) for v in state_dict)
    else:
        return state_dict
