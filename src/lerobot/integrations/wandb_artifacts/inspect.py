# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Lightweight structural validation and metadata extraction for local dataset directories.

Deliberately shallow: this never loads model weights or decodes a single frame. It only checks
the on-disk shape ``LeRobotDataset`` expects (``meta/info.json``, ``meta/stats.json``, ``data/``,
plus ``meta/tasks.parquet`` / ``meta/episodes/`` when the dataset's own info declares nonzero
counts for them) and reads ``meta/info.json`` for self-describing metadata.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.io_utils import load_info
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    STATS_PATH,
    DatasetInfo,
)


class DatasetDirectoryError(ValueError):
    """A local directory doesn't have the structure a LeRobot dataset requires."""


@dataclass(frozen=True, slots=True)
class DatasetDirectoryMetadata:
    """Metadata extracted from a validated local dataset directory."""

    schema_version: str
    robot_type: str | None
    fps: int
    total_episodes: int
    total_frames: int
    total_tasks: int
    camera_keys: tuple[str, ...]
    video_keys: tuple[str, ...]
    source_path: Path
    git_commit: str | None

    def to_wandb_metadata(self) -> dict[str, Any]:
        """JSON-safe dict form, suitable for a W&B Artifact's ``metadata`` argument."""
        return {
            "schema_version": self.schema_version,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": self.total_tasks,
            "camera_keys": list(self.camera_keys),
            "video_keys": list(self.video_keys),
            "source_path": str(self.source_path),
            "git_commit": self.git_commit,
        }


def validate_dataset_directory(root: Path | str) -> DatasetInfo:
    """Validate that ``root`` has the on-disk shape a LeRobot dataset requires.

    Checks the required top-level files/directories first, then — only once ``meta/info.json``
    is known to be readable — cross-checks the task/episode metadata files against the counts
    ``info.json`` itself declares. A dataset can pass the top-level check (info/stats/data all
    present) while still being missing ``meta/tasks.parquet`` or ``meta/episodes/``, which is
    exactly the case that causes ``LeRobotDatasetMetadata`` to fail deep inside metadata loading;
    catching it here gives a single, clear error instead.

    Returns:
        The loaded ``DatasetInfo``, so callers that go on to extract metadata don't re-read it.

    Raises:
        DatasetDirectoryError: ``root`` doesn't have the required structure.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetDirectoryError(f"{root} is not a directory.")

    missing = [path for path in (INFO_PATH, STATS_PATH) if not (root / path).is_file()]
    if not (root / DATA_DIR).is_dir():
        missing.append(f"{DATA_DIR}/")
    if missing:
        raise DatasetDirectoryError(
            f"{root} is missing required dataset file(s)/directory(ies): {', '.join(missing)}."
        )

    try:
        info = load_info(root)
    except Exception as e:
        raise DatasetDirectoryError(f"{root}/{INFO_PATH} could not be read as dataset info: {e}") from e

    if info.total_tasks > 0 and not (root / DEFAULT_TASKS_PATH).is_file():
        raise DatasetDirectoryError(
            f"{root} declares total_tasks={info.total_tasks} in {INFO_PATH} but "
            f"{DEFAULT_TASKS_PATH} is missing."
        )

    if info.total_episodes > 0:
        episodes_dir = root / EPISODES_DIR
        if not episodes_dir.is_dir() or not any(episodes_dir.rglob("*.parquet")):
            raise DatasetDirectoryError(
                f"{root} declares total_episodes={info.total_episodes} in {INFO_PATH} but "
                f"{EPISODES_DIR}/ has no episode metadata parquet files."
            )

    return info


def inspect_dataset_directory(root: Path | str) -> DatasetDirectoryMetadata:
    """Validate ``root`` and extract its self-describing metadata.

    Raises:
        DatasetDirectoryError: ``root`` doesn't have the required structure.
    """
    root = Path(root)
    info = validate_dataset_directory(root)

    camera_keys = tuple(sorted(key for key, ft in info.features.items() if ft["dtype"] in ("image", "video")))
    video_keys = tuple(sorted(key for key, ft in info.features.items() if ft["dtype"] == "video"))

    return DatasetDirectoryMetadata(
        schema_version=info.codebase_version,
        robot_type=info.robot_type,
        fps=info.fps,
        total_episodes=info.total_episodes,
        total_frames=info.total_frames,
        total_tasks=info.total_tasks,
        camera_keys=camera_keys,
        video_keys=video_keys,
        source_path=root.resolve(),
        git_commit=_current_git_commit(),
    )


def _current_git_commit() -> str | None:
    """Best-effort git commit of the running LeRobot checkout; ``None`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
