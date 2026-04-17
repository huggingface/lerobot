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

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


def utc_timestamp_slug(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%dT%H%M%SZ")


def make_hub_file_url(repo_id: str, path_in_repo: str, repo_type: str = "dataset") -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{path_in_repo}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


@dataclass(frozen=True)
class UploadTarget:
    local_path: Path
    path_in_repo: str


def upload_targets(
    repo_id: str,
    targets: list[UploadTarget],
    *,
    repo_type: str = "dataset",
    token: str | None = None,
    private: bool | None = None,
    commit_message: str | None = None,
) -> dict[str, str]:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    uploaded: dict[str, str] = {}
    for target in targets:
        api.upload_file(
            path_or_fileobj=str(target.local_path),
            path_in_repo=target.path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {target.path_in_repo}",
        )
        uploaded[target.path_in_repo] = make_hub_file_url(repo_id, target.path_in_repo, repo_type=repo_type)
    return uploaded
