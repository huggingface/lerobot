#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
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

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.utils import flatten_dict, unflatten_dict

POLICY_DATASET_METADATA_NAME = "dataset_metadata.json"
POLICY_DATASET_METADATA_SCHEMA_VERSION = 1


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _cast_stats_to_numpy(stats: dict[str, Any] | None) -> dict[str, dict[str, np.ndarray]]:
    if not stats:
        return {}
    flattened = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(flattened)


def _normalize_feature_shapes(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized = {}
    for key, feature in features.items():
        normalized_feature = dict(feature)
        if isinstance(normalized_feature.get("shape"), list):
            normalized_feature["shape"] = tuple(normalized_feature["shape"])
        normalized[key] = normalized_feature
    return normalized


@dataclass
class PolicyDatasetMetadata:
    """Lightweight training-dataset metadata snapshot stored with a policy artifact.

    This intentionally exposes only the pieces needed for policy portability and
    provenance. It is not a full LeRobotDatasetMetadata and should not be used
    as a dataset handle.
    """

    repo_id: str | None
    revision: str | None
    info: dict[str, Any]
    stats: dict[str, dict[str, np.ndarray]]

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.info["features"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["image", "video"]]

    @property
    def action_names(self) -> list[str] | None:
        names = self.features.get(ACTION, {}).get("names")
        return list(names) if names is not None else None

    @classmethod
    def from_dataset_metadata(cls, ds_meta: Any) -> PolicyDatasetMetadata:
        info = ds_meta.info.to_dict() if hasattr(ds_meta.info, "to_dict") else dict(ds_meta.info)
        info["features"] = _normalize_feature_shapes(info["features"])
        return cls(
            repo_id=getattr(ds_meta, "repo_id", None),
            revision=getattr(ds_meta, "revision", None),
            info=info,
            stats=_cast_stats_to_numpy(getattr(ds_meta, "stats", None)),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str | Path) -> PolicyDatasetMetadata:
        if payload.get("schema_version") != POLICY_DATASET_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported {POLICY_DATASET_METADATA_NAME} schema in {source}: "
                f"{payload.get('schema_version')!r}"
            )
        try:
            dataset = payload.get("dataset", {})
            info = dict(payload["info"])
            info["features"] = _normalize_feature_shapes(info["features"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed {POLICY_DATASET_METADATA_NAME} in {source}") from exc
        if "fps" not in info:
            raise ValueError(f"Malformed {POLICY_DATASET_METADATA_NAME} in {source}: missing info.fps")
        return cls(
            repo_id=dataset.get("repo_id"),
            revision=dataset.get("revision"),
            info=info,
            stats=_cast_stats_to_numpy(payload.get("stats")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_DATASET_METADATA_SCHEMA_VERSION,
            "dataset": {
                "repo_id": self.repo_id,
                "revision": self.revision,
            },
            "info": _jsonify(self.info),
            "stats": _jsonify(self.stats),
        }


def save_policy_dataset_metadata(save_directory: str | Path, ds_meta: Any | None) -> None:
    if ds_meta is None:
        return
    metadata = PolicyDatasetMetadata.from_dataset_metadata(ds_meta)
    write_json(metadata.to_dict(), Path(save_directory) / POLICY_DATASET_METADATA_NAME)


def load_policy_dataset_metadata(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> PolicyDatasetMetadata | None:
    model_id = str(pretrained_name_or_path)
    metadata_file: str | os.PathLike[str] | None

    if Path(model_id).is_dir():
        metadata_file = Path(model_id) / POLICY_DATASET_METADATA_NAME
        if not metadata_file.exists():
            return None
    else:
        try:
            metadata_file = hf_hub_download(
                repo_id=model_id,
                filename=POLICY_DATASET_METADATA_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        except (EntryNotFoundError, HfHubHTTPError):
            return None

    try:
        payload = load_json(Path(metadata_file))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {POLICY_DATASET_METADATA_NAME} in {metadata_file}") from exc
    return PolicyDatasetMetadata.from_dict(payload, metadata_file)
