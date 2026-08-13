#!/usr/bin/env python

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

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .dataset_metadata import LeRobotDatasetMetadata

DEFAULT_STORAGE_FORMAT = "parquet"

# Supported non-default storage formats and the module implementing each.
# Modules are imported lazily so their optional dependencies stay optional;
# each must expose a ``STORAGE_BACKEND`` class implementing StorageBackend
# (constructed with the keyword arguments listed on the protocol below) and a
# ``localize_root(repo_id, root, revision)`` hook for object-store roots.
_STORAGE_BACKEND_MODULES = {"lance": "lerobot.datasets.lance_backend"}


class StorageBackend(Protocol):
    """Read-side data access for one storage format.

    A backend owns row fetching and video decoding for its format and returns
    fully assembled frame dicts — tabular features, delta-timestamp windows,
    padding masks, decoded video frames — identical to the default parquet/mp4
    pipeline's output. ``LeRobotDataset`` delegates ``__getitem__`` and
    ``__getitems__`` to it and keeps everything else (metadata, episode
    selection, the public API). Instances are constructed with the keyword
    arguments ``meta``, ``root``, ``episodes``, ``delta_timestamps``,
    ``image_transforms``, ``tolerance_s``, ``revision``, ``return_uint8`` and
    ``depth_output_unit``, and must be picklable so ``DataLoader`` workers can
    reopen their own connections.
    """

    def __len__(self) -> int: ...

    def get_item(self, idx: int) -> dict: ...

    def get_items(self, indices: list[int]) -> list[dict]: ...

    def set_image_transforms(self, image_transforms: Callable | None) -> None: ...

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None: ...


def is_remote_uri(root: str | Path) -> bool:
    """True for object-store style roots (``hf://…``, ``file://…``, …)."""
    return "://" in str(root)


def _backend_module(storage_format: str):
    module_name = _STORAGE_BACKEND_MODULES.get(storage_format)
    if module_name is None:
        raise ValueError(
            f"Unknown storage_format {storage_format!r}. Supported formats: "
            f"{[DEFAULT_STORAGE_FORMAT, *_STORAGE_BACKEND_MODULES]}."
        )
    return importlib.import_module(module_name)


def make_storage_backend(storage_format: str, **kwargs) -> StorageBackend:
    """Instantiate the backend class serving ``storage_format``."""
    return _backend_module(storage_format).STORAGE_BACKEND(**kwargs)


def localize_remote_root(repo_id: str | None, root: str | Path, revision: str | None = None) -> Path:
    """Materialize ``meta/`` for an object-store dataset and return the local dir holding it.

    The format cannot be read from ``meta/info.json`` before ``meta/`` exists
    locally, so each backend is asked in turn to recognize and localize the
    root. Data files are never downloaded — backends read them in place.
    """
    errors = []
    for storage_format in _STORAGE_BACKEND_MODULES:
        try:
            return _backend_module(storage_format).localize_root(repo_id, root, revision)
        except FileNotFoundError as error:
            errors.append(f"{storage_format}: {error}")
    raise FileNotFoundError(f"No storage backend found a dataset at {str(root)!r}. Tried {errors}.")


def load_dataset_metadata(
    repo_id: str,
    root: str | Path | None = None,
    revision: str | None = None,
    repo_type: str = "dataset",
) -> LeRobotDatasetMetadata:
    """Load dataset metadata wherever the dataset lives.

    Same as constructing :class:`LeRobotDatasetMetadata` directly, except that a
    remote object-store ``root`` has its ``meta/`` localized first.
    """
    from .dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415  (import cycle)

    if root is not None and is_remote_uri(root):
        root = localize_remote_root(repo_id, root, revision)
    return LeRobotDatasetMetadata(repo_id, root=root, revision=revision, repo_type=repo_type)
