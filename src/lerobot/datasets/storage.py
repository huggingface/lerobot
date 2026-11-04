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

import contextlib
import hashlib
import importlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import fsspec
from huggingface_hub.utils import WeakFileLock

# Root parsing is shared with the streaming primitives, which cannot depend on this package.
from lerobot.streaming.location import (
    LocationKind,
    StorageLocation,
    hf_bucket_uri,
    hf_dataset_uri,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

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
# object-store roots. The hook receives ``storage_options`` (fsspec options)
# as a keyword only when the caller sets them.
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
    return StorageLocation.parse(root).is_remote


def dataset_location(
    repo_id: str, root: str | Path | None = None, repo_type: str = "dataset"
) -> StorageLocation:
    """Where a dataset's files live: an explicit ``root``, else its Hub repository or bucket."""
    if root is not None:
        return StorageLocation.parse(root)
    return StorageLocation.parse(hf_bucket_uri(repo_id) if repo_type == "bucket" else hf_dataset_uri(repo_id))


def is_bucket_root(repo_id: str, root: str | Path | None) -> bool:
    """True when ``root`` is the ``hf://buckets/OWNER/BUCKET`` URI of ``repo_id``'s bucket.

    The inverse of ``LeRobotDataset`` deriving ``hf://buckets/{repo_id}`` from
    ``repo_type="bucket"``: streaming readers treat such a root as bucket mode.
    """
    if root is None:
        return False
    location = StorageLocation.parse(root)
    if location.kind is not LocationKind.HF_BUCKET:
        return False
    if location.path_in_repo:
        raise ValueError(
            f"Expected a bucket root of the form {hf_bucket_uri('OWNER/BUCKET')}, got {str(root)!r}. "
            "Datasets stored under a bucket sub-directory are not supported yet."
        )
    if location.repo_id != repo_id:
        raise ValueError(f"Bucket root {str(root)!r} does not match repo_id {repo_id!r}.")
    return True


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


def remote_meta_cache_dir(key: str) -> Path:
    """Local directory caching ``meta/`` of the object-store dataset identified by ``key``."""
    # The hash keeps keys that sanitize alike (``a_b`` vs ``a/b``) apart and the name short.
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return HF_LEROBOT_HOME / "remote" / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', key)[:100]}-{digest}"


def remote_meta_lock(cache: Path) -> contextlib.AbstractContextManager:
    """Lock serializing reads of ``cache/meta`` with :func:`localize_lerobot_root` refreshes.

    Hold it while loading metadata from a cache this module returned: a refresh swaps
    ``meta/`` with two renames, so an unlocked reader can see it missing in between.
    Same lock file as the bucket metadata cache in ``LeRobotDatasetMetadata``.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    return WeakFileLock(cache.parent / f".{cache.name}.lock")


def localize_lerobot_root(
    root: str,
    storage_options: Mapping[str, Any] | None = None,
    force_cache_sync: bool = False,
) -> Path:
    """Copy ``meta/`` of a default-format dataset at an fsspec ``root`` to a local cache.

    Data files stay remote. Load metadata from the returned directory under
    :func:`remote_meta_lock`, since a concurrent refresh replaces ``meta/``.
    Raises ``FileNotFoundError`` when ``root`` has no ``meta/info.json`` or holds
    another storage format, and fsspec's ``ImportError`` when the protocol's
    package (e.g. ``s3fs``) is missing.
    """
    uri = root.rstrip("/")
    filesystem, path = fsspec.core.url_to_fs(
        uri, **StorageLocation.parse(uri).storage_options(None, storage_options)
    )
    try:
        info = json.loads(filesystem.cat_file(f"{path}/meta/info.json"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{uri}/meta/info.json not found") from error
    # Other formats (e.g. Lance) also ship meta/; leave them to their own backend.
    storage_format = info.get("storage_format") or DEFAULT_STORAGE_FORMAT
    if storage_format != DEFAULT_STORAGE_FORMAT:
        raise FileNotFoundError(
            f"{uri} has storage_format {storage_format!r}, not {DEFAULT_STORAGE_FORMAT!r}"
        )
    cache = remote_meta_cache_dir(uri)
    with remote_meta_lock(cache):
        meta = cache / "meta"
        if force_cache_sync or not (meta / "info.json").is_file():
            tmp = cache / "meta.tmp"
            for stale in [tmp, *cache.glob("meta.old-*")]:  # left behind by a crashed refresh
                shutil.rmtree(stale, ignore_errors=True)
            cache.mkdir(exist_ok=True)
            # tmp must not exist: fsspec copies into an existing directory as tmp/meta.
            filesystem.get(f"{path}/meta", str(tmp), recursive=True)
            # Not atomic: meta/ is missing between the two renames. Readers hold
            # remote_meta_lock, and a failed promotion puts the previous meta/ back.
            old = cache / f"meta.old-{os.getpid()}"
            if meta.exists():
                os.replace(meta, old)
            try:
                os.replace(tmp, meta)
            except BaseException:
                if old.exists():
                    os.replace(old, meta)
                raise
            shutil.rmtree(old, ignore_errors=True)
    return cache


def localize_remote_root(
    repo_id: str | None,
    root: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
    storage_options: Mapping[str, Any] | None = None,
) -> Path:
    """Materialize ``meta/`` for an object-store dataset and return the local dir holding it.

    The format cannot be read from ``meta/info.json`` before ``meta/`` exists
    locally, so each backend is asked in turn to recognize and localize the
    root. Data files are never downloaded — backends read them in place.
    """
    errors = []
    missing_protocol: ImportError | None = None
    if StorageLocation.parse(root).kind is LocationKind.FSSPEC:
        try:
            return localize_lerobot_root(str(root), storage_options, force_cache_sync)
        except FileNotFoundError as error:
            errors.append(f"{DEFAULT_STORAGE_FORMAT}: {error}")
        except ImportError as error:
            # The protocol's fsspec package (e.g. s3fs) is missing. Other formats may not need it
            # (Lance reads S3 natively), so keep probing and raise this only if none matches.
            missing_protocol = error
            errors.append(f"{DEFAULT_STORAGE_FORMAT}: {error}")
    extra = {"storage_options": storage_options} if storage_options else {}
    for storage_format in _DATASET_READER_MODULES:
        try:
            return _reader_module(storage_format).localize_root(
                repo_id, root, revision, token=token, force_cache_sync=force_cache_sync, **extra
            )
        except (FileNotFoundError, ImportError) as error:
            # ImportError: this format's optional dependencies are missing, which
            # must not stop the probe from reaching other registered formats.
            errors.append(f"{storage_format}: {error}")
    if missing_protocol is not None:
        raise missing_protocol
    raise FileNotFoundError(
        f"No dataset found at {str(root)!r}. Tried {errors}. "
        f"For {DEFAULT_STORAGE_FORMAT!r} datasets on an HF Storage Bucket, use "
        "repo_type='bucket' with dataset.streaming=true."
    )


def load_dataset_metadata(
    repo_id: str,
    root: str | Path | None = None,
    revision: str | None = None,
    repo_type: Literal["dataset", "bucket"] = "dataset",
    token: str | bool | None = None,
    force_cache_sync: bool = False,
    storage_options: Mapping[str, Any] | None = None,
) -> LeRobotDatasetMetadata:
    """Load dataset metadata wherever the dataset lives.

    Same as constructing :class:`LeRobotDatasetMetadata` directly, except that a
    remote object-store ``root`` has its ``meta/`` localized first.
    """
    from .dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415  (import cycle)

    meta_lock: contextlib.AbstractContextManager = contextlib.nullcontext()
    if root is not None and is_remote_uri(root):
        root = localize_remote_root(
            repo_id,
            root,
            revision,
            token=token,
            force_cache_sync=force_cache_sync,
            storage_options=storage_options,
        )
        force_cache_sync = False  # the localized meta/ is already fresh
        # repo_type="bucket" takes this same lock in LeRobotDatasetMetadata; it is not reentrant.
        if repo_type != "bucket":
            meta_lock = remote_meta_lock(root)
    with meta_lock:
        return LeRobotDatasetMetadata(
            repo_id,
            root=root,
            revision=revision,
            repo_type=repo_type,
            token=token,
            force_cache_sync=force_cache_sync,
        )
