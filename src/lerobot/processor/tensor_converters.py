# !/usr/bin/env python

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

from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import torch


@singledispatch
def to_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Convert various data types to PyTorch tensors with configurable options.

    This is a unified tensor conversion function using single dispatch to handle
    different input types appropriately.

    Args:
        value: Input value to convert (tensor, array, scalar, sequence, etc.).
        dtype: Target tensor dtype. If None, preserves original dtype.
        device: Target device for the tensor.

    Returns:
        A PyTorch tensor.

    Raises:
        TypeError: If the input type is not supported.
    """
    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle conversion for existing PyTorch tensors."""
    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device=device)
    return value


@to_tensor.register(np.ndarray)
def _(
    value: np.ndarray,
    *,
    dtype=torch.float32,
    device=None,
    **kwargs,
) -> torch.Tensor:
    """Handle conversion for numpy arrays."""
    # Check for numpy scalars (0-dimensional arrays) and treat them as scalars.
    if value.ndim == 0:
        # Numpy scalars should be converted to 0-dimensional tensors.
        return torch.tensor(value.item(), dtype=dtype, device=device)

    # Create tensor from numpy array.
    tensor = torch.from_numpy(value)

    # Apply dtype and device conversion if specified.
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


@to_tensor.register(int)
@to_tensor.register(float)
@to_tensor.register(np.integer)
@to_tensor.register(np.floating)
def _(value, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle conversion for scalar values including numpy scalars."""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(list)
@to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:
    """Handle conversion for sequences (lists, tuples)."""
    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(dict)
def _(value: dict, *, device=None, **kwargs) -> dict:
    """Handle conversion for dictionaries by recursively converting their values to tensors."""
    if not value:
        return {}

    result = {}
    for key, sub_value in value.items():
        if sub_value is None:
            continue

        if isinstance(sub_value, dict):
            # Recursively process nested dictionaries.
            result[key] = to_tensor(
                sub_value,
                device=device,
                **kwargs,
            )
            continue

        # Convert individual values to tensors.
        result[key] = to_tensor(
            sub_value,
            device=device,
            **kwargs,
        )
    return result


def from_tensor_to_numpy(x: torch.Tensor | Any) -> np.ndarray | float | int | Any:
    """
    Convert a PyTorch tensor to a numpy array or scalar if applicable.

    If the input is not a tensor, it is returned unchanged.

    Args:
        x: The input, which can be a tensor or any other type.

    Returns:
        A numpy array, a scalar, or the original input.
    """
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x
