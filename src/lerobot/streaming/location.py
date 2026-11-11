# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""One parser for dataset roots: local paths, Hub dataset/bucket URIs and other fsspec URLs.

Kept free of the ``datasets`` extra so the streaming primitives can use it;
``lerobot.datasets.storage`` builds its dataset-level helpers on top of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem

HF_URI_PREFIX = "hf://"
HF_BUCKET_URI_PREFIX = "hf://buckets/"
HF_DATASET_URI_PREFIX = "hf://datasets/"
# A dataset URI pinned to an immutable commit: (base URI, commit SHA, optional "/path").
_PINNED_DATASET_URI = re.compile(r"^(hf://datasets/[^/]+/[^/@]+)@([0-9a-f]{40})(/.*)?$")


class LocationKind(str, Enum):
    """Where a dataset root lives, which decides how its files are listed and read."""

    LOCAL = "local"
    HF_DATASET = "hf_dataset"
    HF_BUCKET = "hf_bucket"
    # Any other ``hf://`` URI, e.g. a model repository.
    HF_OTHER = "hf_other"
    # Any other fsspec protocol (``s3://``, ``memory://``, ``file://``...).
    FSSPEC = "fsspec"


@dataclass(frozen=True)
class StorageLocation:
    """A parsed dataset root. ``uri`` is the root as given, without a trailing slash."""

    uri: str
    kind: LocationKind
    # Hub repository or bucket identifier (``OWNER/NAME``), for Hub locations.
    repo_id: str | None = None
    # Dataset revision as written in the URI (branch, tag or commit), if any.
    revision: str | None = None
    # Sub-directory inside the repository or bucket, without surrounding slashes.
    path_in_repo: str = ""

    @classmethod
    def parse(cls, root: str | Path) -> StorageLocation:
        """Classify ``root`` without any network access."""
        uri = str(root)
        if "://" not in uri:
            return cls(uri=uri, kind=LocationKind.LOCAL)
        uri = uri.rstrip("/")
        if uri.startswith(HF_BUCKET_URI_PREFIX):
            parts = uri.removeprefix(HF_BUCKET_URI_PREFIX).split("/", 2)
            if len(parts) < 2 or not all(parts[:2]):
                raise ValueError(f"Invalid bucket root: {uri}")
            path = parts[2].strip("/") if len(parts) == 3 else ""
            return cls(uri=uri, kind=LocationKind.HF_BUCKET, repo_id="/".join(parts[:2]), path_in_repo=path)
        if uri.startswith(HF_DATASET_URI_PREFIX):
            rest = uri.removeprefix(HF_DATASET_URI_PREFIX)
            # Canonical datasets have no owner ("hf://datasets/NAME[@rev]"). Splitting is
            # best effort for other unpinned forms; pinned() resolves them through the Hub.
            parts = rest.split("/", 2) if "/" in rest.split("@", 1)[0] else [rest]
            name, _, revision = (parts[0] if len(parts) == 1 else parts[1]).partition("@")
            repo_id = name if len(parts) == 1 else f"{parts[0]}/{name}"
            path = parts[2].strip("/") if len(parts) == 3 else ""
            return cls(
                uri=uri,
                kind=LocationKind.HF_DATASET,
                repo_id=repo_id,
                revision=revision or None,
                path_in_repo=path,
            )
        if uri.startswith(HF_URI_PREFIX):
            return cls(uri=uri, kind=LocationKind.HF_OTHER)
        return cls(uri=uri, kind=LocationKind.FSSPEC)

    @property
    def is_remote(self) -> bool:
        """True for any URI root, including ``file://``."""
        return self.kind is not LocationKind.LOCAL

    @property
    def is_hf(self) -> bool:
        """True for ``hf://`` roots, which take a Hub token and support direct HTTP range reads."""
        return self.kind in (LocationKind.HF_DATASET, LocationKind.HF_BUCKET, LocationKind.HF_OTHER)

    @property
    def range_backend(self) -> str:
        """Byte-range reader: direct HTTP for Hub roots, generic fsspec for everything else."""
        return "native-http" if self.is_hf else "fsspec"

    @property
    def pinned_commit(self) -> str | None:
        """Commit SHA of a dataset URI pinned with ``@<40-hex>``, else None."""
        match = _PINNED_DATASET_URI.match(self.uri)
        return match[2] if match else None

    @property
    def local_path(self) -> Path:
        """Filesystem path of a local root, with ``~`` expanded."""
        if self.kind is not LocationKind.LOCAL:
            raise ValueError(f"{self.uri!r} is not a local path")
        return Path(self.uri).expanduser()

    def storage_options(self, token: str | bool | None) -> dict[str, str | bool]:
        """Fsspec options for reading this root: the Hub token, for Hub roots only."""
        return {"token": token} if token is not None and self.is_hf else {}

    def pinned(self, *, token: str | bool | None = None) -> StorageLocation:
        """Resolve a dataset branch or tag to its commit; other roots are returned unchanged."""
        if self.kind is not LocationKind.HF_DATASET or self.pinned_commit is not None:
            return self
        resolved = HfFileSystem(token=token).resolve_path(self.uri)
        sha = HfApi(token=token).dataset_info(resolved.repo_id, revision=resolved.revision).sha
        if sha is None:
            raise ValueError(f"Could not resolve a commit for {self.uri!r}")
        suffix = f"/{resolved.path_in_repo}" if resolved.path_in_repo else ""
        return StorageLocation.parse(f"{hf_dataset_uri(resolved.repo_id, sha)}{suffix}")


def hf_dataset_uri(repo_id: str, revision: str | None = None) -> str:
    """``hf://datasets/OWNER/NAME[@revision]``."""
    return f"{HF_DATASET_URI_PREFIX}{repo_id}" + (f"@{revision}" if revision else "")


def hf_bucket_uri(bucket_id: str, path: str = "") -> str:
    """``hf://buckets/OWNER/BUCKET[/path]``."""
    return f"{HF_BUCKET_URI_PREFIX}{bucket_id}" + (f"/{path.strip('/')}" if path.strip("/") else "")
