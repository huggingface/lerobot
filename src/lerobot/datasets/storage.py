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
import threading
from importlib.metadata import EntryPoint, entry_points
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

# Formats two or more installed packages disagree over, mapped to a description
# of each claimant. Such a format is deliberately absent from the registry
# above: reading it raises rather than picking a winner. See
# :func:`_discover_plugin_readers`.
_AMBIGUOUS_READER_PROVIDERS: dict[str, list[str]] = {}
_PLUGINS_DISCOVERED = False
_DISCOVERY_LOCK = threading.Lock()


def register_dataset_reader(storage_format: str, module: str) -> None:
    """Register ``module`` (implementing the contract above) to serve ``storage_format``."""
    existing = _DATASET_READER_MODULES.get(storage_format, module)
    if storage_format == DEFAULT_STORAGE_FORMAT or existing != module:
        raise ValueError(f"storage_format {storage_format!r} is already registered.")
    _DATASET_READER_MODULES[storage_format] = module
    # An explicit call is how a user settles a format installed plugins disagree
    # over, so it has to clear the conflict -- whichever order the two happen in.
    _AMBIGUOUS_READER_PROVIDERS.pop(storage_format, None)


register_dataset_reader("lance", "lerobot.datasets.lance_backend")


def _plugin_provider(entry_point: EntryPoint) -> str:
    """Name the installed package behind ``entry_point``, for error messages."""
    dist = getattr(entry_point, "dist", None)
    name = getattr(dist, "name", None) or "unknown distribution"
    version = getattr(dist, "version", None)
    origin = f"{name} {version}" if version else name
    return f"{origin} -> {entry_point.value}"


def _discover_plugin_readers() -> None:
    """Register storage formats advertised by installed packages, once.

    Called before every registry lookup rather than at import, so the scan costs
    nothing until a dataset is actually opened.

    The scan is order-independent by construction: claims are collected per
    format first, and only a format claimed by exactly one module is registered.
    Two packages claiming the same format is an error the user has to resolve,
    not a race the loader settles for them -- which package ``entry_points()``
    lists first depends on ``sys.path``, so silently taking it would mean the
    same install reading datasets differently on different machines. The
    conflict is logged here and raised, naming both packages, if that format is
    ever asked for.

    Built-in formats win outright: an entry point may not take ``lerobot`` or
    ``lance``, so installing a package cannot change how they are read. An
    explicit :func:`register_dataset_reader` call wins the same way, which is
    what makes it the documented way out of a conflict.

    A plugin that fails to register is skipped with a warning -- one broken
    package must not stop the others, nor stop datasets loading at all.

    Because the trigger is first use, two threads can arrive together; the scan
    runs once and the second waits for it, rather than reading a registry that
    is still being built.
    """
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return
    with _DISCOVERY_LOCK:
        if _PLUGINS_DISCOVERED:
            return
        try:
            _scan_plugin_readers()
        finally:
            # Set after the scan, under the lock, so a thread that arrives while
            # the scan is running waits for it rather than reading a half-built
            # registry. The finally is what keeps a failing scan from being
            # retried on every subsequent lookup.
            _PLUGINS_DISCOVERED = True


def _scan_plugin_readers() -> None:
    """Read the entry point group and register what it unambiguously advertises."""
    try:
        discovered = list(entry_points(group=DATASET_READER_ENTRY_POINT_GROUP))
    except Exception as error:  # pragma: no cover -- importlib.metadata is robust
        logger.warning("Could not read %r entry points: %s", DATASET_READER_ENTRY_POINT_GROUP, error)
        return

    # Collect first, register second, so no claim depends on scan order.
    claims: dict[str, dict[str, list[str]]] = {}  # format -> module -> providers
    for entry_point in discovered:
        try:
            # ``.module`` (not ``.value``) so a "pkg.mod:attr" spelling still
            # resolves to the module the contract is defined on.
            module = entry_point.module
        except Exception as error:
            logger.warning(
                "Ignoring malformed dataset reader entry point %r (%r): %s",
                entry_point.name,
                entry_point.value,
                error,
            )
            continue
        claims.setdefault(entry_point.name, {}).setdefault(module, []).append(_plugin_provider(entry_point))

    for storage_format, by_module in claims.items():
        providers = sorted(provider for group in by_module.values() for provider in group)
        if storage_format == DEFAULT_STORAGE_FORMAT or storage_format in _DATASET_READER_MODULES:
            incumbent = _DATASET_READER_MODULES.get(storage_format, "lerobot.datasets.dataset_reader")
            logger.warning(
                "Ignoring dataset reader plugin(s) for storage_format %r [%s]: that format is "
                "already served by %r.",
                storage_format,
                "; ".join(providers),
                incumbent,
            )
            continue
        if len(by_module) > 1:
            # Recorded, not raised: a conflict over one format must not stop
            # datasets in every other format from loading. _reader_module()
            # raises when this format is the one actually asked for.
            _AMBIGUOUS_READER_PROVIDERS[storage_format] = providers
            logger.warning(
                "storage_format %r is claimed by %d installed packages [%s]; it will not be "
                "loaded until one is uninstalled or register_dataset_reader() picks one.",
                storage_format,
                len(providers),
                "; ".join(providers),
            )
            continue
        try:
            # register_dataset_reader() stays the single gate on the registry;
            # if it ever grows further validation, a rejection must still leave
            # the remaining plugins registered.
            register_dataset_reader(storage_format, next(iter(by_module)))
        except Exception as error:
            logger.warning(
                "Ignoring dataset reader plugin %r [%s]: %s", storage_format, "; ".join(providers), error
            )


def is_remote_uri(root: str | Path) -> bool:
    """True for object-store style roots (``hf://…``, ``file://…``, …)."""
    return "://" in str(root)


def _ambiguous_format_error(storage_format: str) -> ValueError:
    """Explain a format two installed packages claim, and how to settle it."""
    providers = "\n  ".join(_AMBIGUOUS_READER_PROVIDERS[storage_format])
    return ValueError(
        f"storage_format {storage_format!r} is claimed by more than one installed package, so "
        f"which reader serves it is ambiguous:\n  {providers}\n"
        f"Uninstall all but one, or choose explicitly before opening the dataset:\n"
        f"  from lerobot.datasets.storage import register_dataset_reader\n"
        f'  register_dataset_reader("{storage_format}", "<module>")'
    )


def _reader_module(storage_format: str):
    _discover_plugin_readers()
    module_name = _DATASET_READER_MODULES.get(storage_format)
    if module_name is None:
        if storage_format in _AMBIGUOUS_READER_PROVIDERS:
            raise _ambiguous_format_error(storage_format)
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
    if _AMBIGUOUS_READER_PROVIDERS:
        # A format no backend could be chosen for was never probed; say so here
        # rather than let it read as "that dataset does not exist".
        errors.append(
            f"not probed, claimed by more than one installed package: {sorted(_AMBIGUOUS_READER_PROVIDERS)}"
        )
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
