# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Low-level primitives for training-time dataset streaming."""

from lerobot.utils.import_utils import require_package

require_package("datasets", extra="dataset")
require_package("av", extra="dataset")

from .episode_cache import EpisodeByteCache
from .episode_parquet import EpisodeParquetReader
from .episode_pool import ExactCoveragePool
from .location import LocationKind, StorageLocation, hf_bucket_uri, hf_dataset_uri
from .manifest import EpisodeVideoManifest, video_file_groups
from .sidecar import SidecarSpec, ensure_mp4_sidecar, sidecar_cache_path

__all__ = [
    "EpisodeByteCache",
    "EpisodeParquetReader",
    "EpisodeVideoManifest",
    "ExactCoveragePool",
    "LocationKind",
    "SidecarSpec",
    "StorageLocation",
    "ensure_mp4_sidecar",
    "hf_bucket_uri",
    "hf_dataset_uri",
    "sidecar_cache_path",
    "video_file_groups",
]
