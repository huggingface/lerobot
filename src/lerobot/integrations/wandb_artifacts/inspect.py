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
"""Local-only validation and metadata extraction for materialized LeRobot datasets.

Validation uses LeRobot's own metadata and parquet loaders. It does not decode frames or contact a
remote store, but it proves that required metadata parses and every episode-referenced payload is
present before an artifact is accepted.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.io_utils import (
    load_episodes,
    load_info,
    load_nested_dataset,
    load_stats,
    load_tasks,
)
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    STATS_PATH,
    DatasetInfo,
)


class DatasetDirectoryError(ValueError):
    """A local directory is not a complete, readable LeRobot dataset."""


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
    """Prove that ``root`` is a complete, locally readable LeRobot dataset.

    The check follows the real read path instead of approximating it with filename heuristics:
    required JSON/parquet metadata is parsed, episode metadata is cross-checked against
    ``info.json``, every referenced data/video shard must exist, and the real nested-data loader
    must observe exactly ``total_frames`` rows across exactly the declared episodes.

    Returns:
        The loaded ``DatasetInfo``, so metadata extraction does not re-read ``meta/info.json``.

    Raises:
        DatasetDirectoryError: ``root`` is incomplete, malformed, or internally inconsistent.
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

    info = _load_info(root)
    _validate_stats(root)
    _validate_tasks(root, info)
    episodes = _load_and_validate_episodes(root, info)
    _validate_episode_references(root, info, episodes)
    _validate_data(root, info, episodes)
    return info


def inspect_dataset_directory(root: Path | str) -> DatasetDirectoryMetadata:
    """Validate ``root`` and extract its self-describing metadata."""
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


def _load_info(root: Path) -> DatasetInfo:
    try:
        return load_info(root)
    except Exception as e:
        raise DatasetDirectoryError(f"{root}/{INFO_PATH} could not be read as dataset info: {e}") from e


def _validate_stats(root: Path) -> None:
    try:
        stats = load_stats(root)
    except Exception as e:
        raise DatasetDirectoryError(f"{root}/{STATS_PATH} could not be read as dataset stats: {e}") from e
    if stats is None:
        raise DatasetDirectoryError(f"{root}/{STATS_PATH} is missing.")


def _validate_tasks(root: Path, info: DatasetInfo) -> None:
    if info.total_tasks == 0:
        return
    try:
        tasks = load_tasks(root)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{DEFAULT_TASKS_PATH} could not be read as task metadata: {e}"
        ) from e
    if len(tasks) != info.total_tasks:
        raise DatasetDirectoryError(
            f"{root} declares total_tasks={info.total_tasks} in {INFO_PATH} but "
            f"{DEFAULT_TASKS_PATH} contains {len(tasks)} row(s)."
        )


def _load_and_validate_episodes(root: Path, info: DatasetInfo) -> Any | None:
    if info.total_episodes == 0:
        return None

    try:
        episodes = load_episodes(root)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} could not be read as episode metadata: {e}"
        ) from e

    if len(episodes) != info.total_episodes:
        raise DatasetDirectoryError(
            f"{root} declares total_episodes={info.total_episodes} in {INFO_PATH} but episode "
            f"metadata contains {len(episodes)} row(s)."
        )
    if "episode_index" not in episodes.column_names:
        raise DatasetDirectoryError(f"{root}/{EPISODES_DIR} has no episode_index column.")

    try:
        indices = [int(index) for index in episodes["episode_index"]]
    except (TypeError, ValueError) as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} contains invalid episode_index values: {e}"
        ) from e

    expected = set(range(info.total_episodes))
    if len(indices) != len(set(indices)) or set(indices) != expected:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} must contain each episode_index from 0 to "
            f"{info.total_episodes - 1} exactly once."
        )
    return episodes


def _validate_episode_references(root: Path, info: DatasetInfo, episodes: Any | None) -> None:
    if episodes is None:
        if info.total_frames > 0:
            raise DatasetDirectoryError(
                f"{root} declares total_frames={info.total_frames} but total_episodes=0."
            )
        return

    video_keys = tuple(key for key, feature in info.features.items() if feature["dtype"] == "video")
    video_path = info.video_path
    if video_keys and video_path is None:
        raise DatasetDirectoryError(
            f"{root}/{INFO_PATH} declares video features but does not define a video_path template."
        )

    referenced_data: set[Path] = set()
    referenced_videos: set[Path] = set()
    try:
        for row in episodes:
            referenced_data.add(
                Path(
                    info.data_path.format(
                        chunk_index=row["data/chunk_index"],
                        file_index=row["data/file_index"],
                    )
                )
            )
            for video_key in video_keys:
                assert video_path is not None
                referenced_videos.add(
                    Path(
                        video_path.format(
                            video_key=video_key,
                            chunk_index=row[f"videos/{video_key}/chunk_index"],
                            file_index=row[f"videos/{video_key}/file_index"],
                        )
                    )
                )
    except (KeyError, TypeError, ValueError) as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} cannot resolve all declared payload paths: {e}"
        ) from e

    _require_files(root, referenced_data, "data")
    _require_files(root, referenced_videos, "video")


def _require_files(root: Path, relative_paths: set[Path], payload: str) -> None:
    missing = sorted(path for path in relative_paths if not (root / path).is_file())
    if not missing:
        return

    preview = ", ".join(str(path) for path in missing[:3])
    remainder = f", and {len(missing) - 3} more" if len(missing) > 3 else ""
    raise DatasetDirectoryError(
        f"{root} is missing {len(missing)} {payload} file(s) referenced by episode metadata: "
        f"{preview}{remainder}."
    )


def _validate_data(root: Path, info: DatasetInfo, episodes: Any | None) -> None:
    data_files = sorted((root / DATA_DIR).glob("*/*.parquet"))
    if not data_files:
        if info.total_frames == 0:
            return
        raise DatasetDirectoryError(
            f"{root} declares total_frames={info.total_frames} in {INFO_PATH} but "
            f"{DATA_DIR}/ has no loader-visible parquet files."
        )

    try:
        data = load_nested_dataset(root / DATA_DIR)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{DATA_DIR} could not be read as frame data: {e}"
        ) from e

    if len(data) != info.total_frames:
        raise DatasetDirectoryError(
            f"{root} declares total_frames={info.total_frames} in {INFO_PATH} but "
            f"{DATA_DIR}/ contains {len(data)} row(s)."
        )

    if info.total_frames == 0:
        return
    if episodes is None:
        raise DatasetDirectoryError(f"{root} has frame data but no episode metadata.")
    if "episode_index" not in data.column_names:
        raise DatasetDirectoryError(f"{root}/{DATA_DIR} has no episode_index column.")

    try:
        data_episode_indices = {int(index) for index in data.unique("episode_index")}
    except (TypeError, ValueError) as e:
        raise DatasetDirectoryError(
            f"{root}/{DATA_DIR} contains invalid episode_index values: {e}"
        ) from e

    expected = set(range(info.total_episodes))
    if data_episode_indices != expected:
        raise DatasetDirectoryError(
            f"{root}/{DATA_DIR} covers episode_index values {sorted(data_episode_indices)}, "
            f"expected {sorted(expected)}."
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
