# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import inspect
import json
from collections.abc import Iterator
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor

# `backend` arrived on `safe_open` after `load_file`. Older builds still load the whole shard.
_SAFE_OPEN_TAKES_BACKEND = "backend" in inspect.signature(safe_open).parameters


def safetensors_shard_names(index_file: Path) -> list[str]:
    """Return the relative shard filenames listed in a checkpoint index."""
    weight_map = json.loads(index_file.read_text())["weight_map"]
    names = sorted(set(weight_map.values()))
    if not names:
        raise ValueError(f"Checkpoint index {index_file} contains no shards")
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid checkpoint shard path: {name}")
    return names


class SafetensorsCheckpoint:
    """Read one logical checkpoint from one or more files, without merging files on disk.

    Metadata inspection does not materialize tensors. `tensors` reads one key at a time;
    `load_state_dict` retains the whole checkpoint for loaders that need all weights together.
    CPU tensors use safetensors' memory mapping unless another backend is requested.
    """

    def __init__(self, files: list[Path]) -> None:
        if not files:
            raise ValueError("A checkpoint must contain at least one weights file")
        self.files = tuple(files)

    def keys(self) -> set[str]:
        """Return tensor names across every shard."""
        keys: set[str] = set()
        for file in self.files:
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                keys.update(handle.keys())
        return keys

    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Inspect tensor shapes without reading their data."""
        shapes = {}
        for file in self.files:
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118
                    shapes[key] = tuple(handle.get_slice(key).get_shape())
        return shapes

    def tensors(self, device: str = "cpu", *, backend: str | None = None) -> Iterator[tuple[str, Tensor]]:
        """Yield one tensor at a time; callers decide which tensors to retain."""
        for file in self.files:
            if backend is not None and not _SAFE_OPEN_TAKES_BACKEND:
                yield from load_file(str(file), device=device, backend=backend).items()
                continue
            kwargs = {"backend": backend} if backend is not None else {}
            with safe_open(str(file), framework="pt", device=device, **kwargs) as handle:
                for key in handle.keys():  # noqa: SIM118
                    yield key, handle.get_tensor(key)

    def load_state_dict(self, device: str = "cpu", *, backend: str | None = None) -> dict[str, Tensor]:
        """Read all weights for conversion or assignment without a temporary checkpoint file."""
        return dict(self.tensors(device=device, backend=backend))
