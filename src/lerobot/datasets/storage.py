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
import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset_metadata import LeRobotDatasetMetadata
    from .dataset_reader import BaseDatasetReader

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_FORMAT = "lerobot"

# Entry point group an installed package declares to serve a storage format
# without the application having to import it first::
#
#     [project.entry-points."lerobot.dataset_readers"]
#     my_format = "my_package.my_reader"
#
# The value names the module implementing the contract below. Entry points are
# read for their names only -- the module is still imported lazily, on first
# use -- so a plugin's optional dependencies stay optional.
DATASET_READER_ENTRY_POINT_GROUP = "lerobot.dataset_readers"

# Supported non-default storage formats and the module implementing each.
# Modules are imported lazily so their optional dependencies stay optional;
# each must expose a ``DATASET_READER`` class implementing
# :class:`~lerobot.datasets.dataset_reader.BaseDatasetReader` (constructed with
# the keyword arguments ``meta``, ``root``, ``episodes``, ``delta_timestamps``,
# ``image_transforms``, ``tolerance_s``, ``revision``, ``return_uint8``,
# ``depth_output_unit`` and ``token``) and a ``localize_root`` hook for
# object-store roots.
_DATASET_READER_MODULES: dict[str, str] = {}
_PLUGINS_DISCOVERED = False


def register_dataset_reader(storage_format: str, module: str) -> None:
    """Register ``module`` (implementing the contract above) to serve ``storage_format``."""
    existing = _DATASET_READER_MODULES.get(storage_format, module)
    if storage_format == DEFAULT_STORAGE_FORMAT or existing != module:
        raise ValueError(f"storage_format {storage_format!r} is already registered.")
    _DATASET_READER_MODULES[storage_format] = module


register_dataset_reader("lance", "lerobot.datasets.lance_backend")


def _discover_plugin_readers() -> None:
    """Register storage formats advertised by installed packages, once.

    Called before every registry lookup rather than at import, so the scan costs
    nothing until a dataset is actually opened.

    Built-in formats win: an entry point may not take a name that is already
    registered, so installing a package cannot silently change how ``lerobot``
    or ``lance`` datasets are read. A plugin that fails to register is skipped
    with a warning -- one broken package must not stop the others, nor stop
    datasets loading at all.
    """
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return
    # Set before scanning: a failing scan must not be retried on every lookup.
    _PLUGINS_DISCOVERED = True
    try:
        discovered = list(entry_points(group=DATASET_READER_ENTRY_POINT_GROUP))
    except Exception as error:  # pragma: no cover -- importlib.metadata is robust
        logger.warning("Could not read %r entry points: %s", DATASET_READER_ENTRY_POINT_GROUP, error)
        return

    for entry_point in discovered:
        try:
            # Registering through the public function is what keeps built-ins
            # safe: it rejects DEFAULT_STORAGE_FORMAT and any name already taken,
            # so an installed package cannot claim "lerobot" or "lance".
            # ``.module`` (not ``.value``) so a "pkg.mod:attr" spelling still
            # resolves to the module the contract is defined on.
            register_dataset_reader(entry_point.name, entry_point.module)
        except Exception as error:
            logger.warning(
                "Ignoring dataset reader plugin %r (from %r): %s",
                entry_point.name,
                entry_point.value,
                error,
            )


def is_remote_uri(root: str | Path) -> bool:
    """True for object-store style roots (``hf://…``, ``file://…``, …)."""
    return "://" in str(root)


def _reader_module(storage_format: str):
    _discover_plugin_readers()
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
    _discover_plugin_readers()
    errors = []
    for storage_format in list(_DATASET_READER_MODULES):
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
