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

Deliberately shallow: this never loads model weights or decodes a single frame. It checks the
on-disk shape ``LeRobotDataset`` expects and verifies that every data, episode-metadata, and
video file referenced by that metadata is locally materialized.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.io_utils import load_episodes, load_info
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
    """Validate that ``root`` has the complete on-disk shape a LeRobot dataset requires.

    Checks required top-level files first, then cross-checks task, episode, data, and video payloads
    against ``meta/info.json`` and episode metadata. Parquet discovery deliberately matches
    ``load_nested_dataset()`` exactly: one chunk directory below the corresponding root. Video
    validation derives the same paths as ``LeRobotDatasetMetadata.get_video_file_path()`` without
    decoding media.

    Returns:
        The loaded ``DatasetInfo``, so callers that go on to extract metadata don't re-read it.

    Raises:
        DatasetDirectoryError: ``root`` is incomplete or its metadata cannot describe local files.
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

    if info.total_episodes > 0 and not _has_chunked_parquet(root / EPISODES_DIR):
        raise DatasetDirectoryError(
            f"{root} declares total_episodes={info.total_episodes} in {INFO_PATH} but "
            f"{EPISODES_DIR}/ has no episode metadata parquet files."
        )

    if info.total_frames > 0 and not _has_chunked_parquet(root / DATA_DIR):
        raise DatasetDirectoryError(
            f"{root} declares total_frames={info.total_frames} in {INFO_PATH} but "
            f"{DATA_DIR}/ has no data parquet files."
        )

    _validate_video_files(root, info)
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


def _has_chunked_parquet(root: Path) -> bool:
    """Return whether ``root`` contains a parquet exactly one chunk directory below it."""
    return any(root.glob("*/*.parquet"))


def _validate_video_files(root: Path, info: DatasetInfo) -> None:
    """Require every video path referenced by episode metadata to be materialized locally."""
    video_keys = tuple(key for key, feature in info.features.items() if feature["dtype"] == "video")
    if info.total_episodes == 0 or not video_keys:
        return
    if info.video_path is None:
        raise DatasetDirectoryError(
            f"{root}/{INFO_PATH} declares video features but does not define a video_path template."
        )

    try:
        episodes = load_episodes(root)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} could not be read to validate referenced video files: {e}"
        ) from e

    if len(episodes) < info.total_episodes:
        raise DatasetDirectoryError(
            f"{root} declares total_episodes={info.total_episodes} in {INFO_PATH} but episode "
            f"metadata contains only {len(episodes)} row(s)."
        )

    missing: set[Path] = set()
    try:
        for episode_index in range(info.total_episodes):
            episode = episodes[episode_index]
            for video_key in video_keys:
                relative_path = Path(
                    info.video_path.format(
                        video_key=video_key,
                        chunk_index=episode[f"videos/{video_key}/chunk_index"],
                        file_index=episode[f"videos/{video_key}/file_index"],
                    )
                )
                if not (root / relative_path).is_file():
                    missing.add(relative_path)
    except (IndexError, KeyError, TypeError, ValueError) as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} cannot resolve every declared video path from episode metadata: {e}"
        ) from e

    if missing:
        missing_paths = sorted(missing)
        preview = ", ".join(str(path) for path in missing_paths[:3])
        remainder = f", and {len(missing_paths) - 3} more" if len(missing_paths) > 3 else ""
        raise DatasetDirectoryError(
            f"{root} is missing {len(missing_paths)} video file(s) referenced by episode metadata: "
            f"{preview}{remainder}."
        )


def _current_git_commit() -> str | None:
    """Best-effort commit of the LeRobot checkout; ``None`` for wheel/site-packages installs."""
    package_dir = Path(__file__).resolve().parents[2]

    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if root_result.returncode != 0:
        return None

    repo_root = Path(root_result.stdout.strip()).resolve()
    if (repo_root / "src" / "lerobot").resolve() != package_dir:
        # An installed package may sit inside another repository's virtualenv. That repository's
        # HEAD is not LeRobot provenance and must not be reported as such.
        return None

    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if commit_result.returncode != 0:
        return None
    return commit_result.stdout.strip() or None
