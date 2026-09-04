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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset_metadata import LeRobotDatasetMetadata
    from .dataset_reader import BaseDatasetReader

DEFAULT_STORAGE_FORMAT = "lerobot"

# Supported non-default storage formats and the module implementing each.
# Modules are imported lazily so their optional dependencies stay optional;
# each must expose a ``DATASET_READER`` class implementing
# :class:`~lerobot.datasets.dataset_reader.BaseDatasetReader` (constructed with
# the keyword arguments ``meta``, ``root``, ``episodes``, ``delta_timestamps``,
# ``image_transforms``, ``tolerance_s``, ``revision``, ``return_uint8``,
# ``depth_output_unit`` and ``token``) and a ``localize_root`` hook for
# object-store roots.
_DATASET_READER_MODULES: dict[str, str] = {}


def register_dataset_reader(storage_format: str, module: str) -> None:
    """Register ``module`` (implementing the contract above) to serve ``storage_format``."""
    existing = _DATASET_READER_MODULES.get(storage_format, module)
    if storage_format == DEFAULT_STORAGE_FORMAT or existing != module:
        raise ValueError(f"storage_format {storage_format!r} is already registered.")
    _DATASET_READER_MODULES[storage_format] = module


register_dataset_reader("lance", "lerobot.datasets.lance_backend")


def is_remote_uri(root: str | Path) -> bool:
    """True for object-store style roots (``hf://…``, ``file://…``, …)."""
    return "://" in str(root)


def _reader_module(storage_format: str):
    module_name = _DATASET_READER_MODULES.get(storage_format)
    if module_name is None:
        raise ValueError(
            f"Unknown storage_format {storage_format!r}. Supported formats: "
            f"{[DEFAULT_STORAGE_FORMAT, *_DATASET_READER_MODULES]}."
        )
    return importlib.import_module(module_name)


def make_dataset_reader(storage_format: str, **kwargs) -> BaseDatasetReader:
    """Instantiate the reader class serving ``storage_format``."""
    if storage_format == DEFAULT_STORAGE_FORMAT:
        from .dataset_reader import DatasetReader  # noqa: PLC0415  (import cycle)

        return DatasetReader(**kwargs)
    return _reader_module(storage_format).DATASET_READER(**kwargs)


def localize_remote_root(
    repo_id: str | None,
    root: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> Path:
    """Materialize ``meta/`` for an object-store dataset and return the local dir holding it.

    The format cannot be read from ``meta/info.json`` before ``meta/`` exists
    locally, so each backend is asked in turn to recognize and localize the
    root. Data files are never downloaded — backends read them in place.
    """
    errors = []
    for storage_format in _DATASET_READER_MODULES:
        try:
            return _reader_module(storage_format).localize_root(
                repo_id, root, revision, token=token, force_cache_sync=force_cache_sync
            )
        except (FileNotFoundError, ImportError) as error:
            # ImportError: this format's optional dependencies are missing, which
            # must not stop the probe from reaching other registered formats.
            errors.append(f"{storage_format}: {error}")
    raise FileNotFoundError(
        f"No dataset found at {str(root)!r}. Tried {errors}. "
        f"For {DEFAULT_STORAGE_FORMAT!r} datasets on an HF Storage Bucket, use "
        "repo_type='bucket' with dataset.streaming=true."
    )


def load_dataset_metadata(
    repo_id: str,
    root: str | Path | None = None,
    revision: str | None = None,
    repo_type: str = "dataset",
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> LeRobotDatasetMetadata:
    """Load dataset metadata wherever the dataset lives.

    Same as constructing :class:`LeRobotDatasetMetadata` directly, except that a
    remote object-store ``root`` has its ``meta/`` localized first.
    """
    from .dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415  (import cycle)

    if root is not None and is_remote_uri(root):
        root = localize_remote_root(repo_id, root, revision, token=token, force_cache_sync=force_cache_sync)
        force_cache_sync = False  # the localized meta/ is already fresh
    return LeRobotDatasetMetadata(
        repo_id,
        root=root,
        revision=revision,
        repo_type=repo_type,
        token=token,
        force_cache_sync=force_cache_sync,
    )
