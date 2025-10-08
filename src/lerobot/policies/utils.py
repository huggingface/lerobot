#!/usr/bin/env python

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

import logging
from collections import deque

import torch
from torch import nn

from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """Log missing and unexpected keys when loading a model.

    Args:
        missing_keys (list[str]): Keys that were expected but not found.
        unexpected_keys (list[str]): Keys that were found but not expected.
    """
    if missing_keys:
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")


def build_inference_frame(
    observation: dict[str, torch.Tensor],
    ds_features: dict[str, dict],
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> dict[str, torch.Tensor]:
    """Build a inference frame from a raw observation."""

    # Extracts the inference keys from the raw observation
    observation = build_dataset_frame(ds_features, observation, prefix=OBS_STR)

    # Performs the necessary conversions to the observation
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()

        # Needs to add a batch dimension when running inference
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    return observation
