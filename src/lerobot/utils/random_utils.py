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
import random
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.utils import flatten_dict, unflatten_dict
from lerobot.utils.constants import RNG_STATE


def serialize_python_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `random` in the form of a flat dict[str, torch.Tensor] to be saved using
    `safetensors.save_file()` or `torch.save()`.
    """
    py_state = random.getstate()
    return {
        "py_rng_version": torch.tensor([py_state[0]], dtype=torch.int64),
        "py_rng_state": torch.tensor(py_state[1], dtype=torch.int64),
    }


def deserialize_python_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `random` from a dictionary produced by `serialize_python_rng_state()`.
    """
    py_state = (rng_state_dict["py_rng_version"].item(), tuple(rng_state_dict["py_rng_state"].tolist()), None)
    random.setstate(py_state)


def serialize_numpy_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `numpy` in the form of a flat dict[str, torch.Tensor] to be saved using
    `safetensors.save_file()` or `torch.save()`.
    """
    np_state = np.random.get_state()
    # Ensure no breaking changes from numpy
    assert np_state[0] == "MT19937"
    return {
        "np_rng_state_values": torch.tensor(np_state[1], dtype=torch.int64),
        "np_rng_state_index": torch.tensor([np_state[2]], dtype=torch.int64),
        "np_rng_has_gauss": torch.tensor([np_state[3]], dtype=torch.int64),
        "np_rng_cached_gaussian": torch.tensor([np_state[4]], dtype=torch.float32),
    }


def deserialize_numpy_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `numpy` from a dictionary produced by `serialize_numpy_rng_state()`.
    """
    np_state = (
        "MT19937",
        rng_state_dict["np_rng_state_values"].numpy(),
        rng_state_dict["np_rng_state_index"].item(),
        rng_state_dict["np_rng_has_gauss"].item(),
        rng_state_dict["np_rng_cached_gaussian"].item(),
    )
    np.random.set_state(np_state)


def serialize_torch_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `torch` in the form of a flat dict[str, torch.Tensor] to be saved using
    `safetensors.save_file()` or `torch.save()`.
    """
    torch_rng_state_dict = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        torch_rng_state_dict["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
    return torch_rng_state_dict


def deserialize_torch_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `torch` from a dictionary produced by `serialize_torch_rng_state()`.
    """
    torch.set_rng_state(rng_state_dict["torch_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state" in rng_state_dict:
        torch.cuda.set_rng_state(rng_state_dict["torch_cuda_rng_state"])


def serialize_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `random`, `numpy`, and `torch`, in the form of a flat
    dict[str, torch.Tensor] to be saved using `safetensors.save_file()` `torch.save()`.
    """
    py_rng_state_dict = serialize_python_rng_state()
    np_rng_state_dict = serialize_numpy_rng_state()
    torch_rng_state_dict = serialize_torch_rng_state()

    return {
        **py_rng_state_dict,
        **np_rng_state_dict,
        **torch_rng_state_dict,
    }


def deserialize_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `random`, `numpy`, and `torch` from a dictionary produced by
    `serialize_rng_state()`.
    """
    py_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("py")}
    np_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("np")}
    torch_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("torch")}

    deserialize_python_rng_state(py_rng_state_dict)
    deserialize_numpy_rng_state(np_rng_state_dict)
    deserialize_torch_rng_state(torch_rng_state_dict)


def save_rng_state(save_dir: Path) -> None:
    rng_state_dict = serialize_rng_state()
    flat_rng_state_dict = flatten_dict(rng_state_dict)
    save_file(flat_rng_state_dict, save_dir / RNG_STATE)


def load_rng_state(save_dir: Path) -> None:
    flat_rng_state_dict = load_file(save_dir / RNG_STATE)
    rng_state_dict = unflatten_dict(flat_rng_state_dict)
    deserialize_rng_state(rng_state_dict)


def get_rng_state() -> dict[str, Any]:
    """Get the random state for `random`, `numpy`, and `torch`."""
    random_state_dict = {
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        random_state_dict["torch_cuda_random_state"] = torch.cuda.random.get_rng_state()
    return random_state_dict


def set_rng_state(random_state_dict: dict[str, Any]):
    """Set the random state for `random`, `numpy`, and `torch`.

    Args:
        random_state_dict: A dictionary of the form returned by `get_rng_state`.
    """
    random.setstate(random_state_dict["random_state"])
    np.random.set_state(random_state_dict["numpy_random_state"])
    torch.random.set_rng_state(random_state_dict["torch_random_state"])
    if torch.cuda.is_available():
        torch.cuda.random.set_rng_state(random_state_dict["torch_cuda_random_state"])


def set_seed(seed, accelerator: Callable | None = None) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if accelerator:
        from accelerate.utils import set_seed as _accelerate_set_seed

        _accelerate_set_seed(seed)


@contextmanager
def seeded_context(seed: int) -> Generator[None, None, None]:
    """Set the seed when entering a context, and restore the prior random state at exit.

    Example usage:

    ```
    a = random.random()  # produces some random number
    with seeded_context(1337):
        b = random.random()  # produces some other random number
    c = random.random()  # produces yet another random number, but the same it would have if we never made `b`
    ```
    """
    random_state_dict = get_rng_state()
    set_seed(seed)
    yield None
    set_rng_state(random_state_dict)
