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

Validation follows the reader's actual contract without decoding media or contacting a remote
store: metadata must parse, frame parquet must match the declared schema, referenced payloads must
stay inside the artifact root, and frame/task/episode indices must agree.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import load_episodes, load_info, load_nested_dataset, load_stats, load_tasks
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    STATS_PATH,
    DatasetInfo,
    check_version_compatibility,
)
from lerobot.utils.constants import DEFAULT_FEATURES

_TIMESTAMP_TOLERANCE_S = 1e-4

# PEFT saves an adapter, not a full model, via `peft_model.save_pretrained(...)`. These filenames
# are hardcoded rather than imported from `peft` because `peft` is an optional `lerobot[peft]`
# extra, and importing it here would break directory inspection for every base-install user.
PEFT_ADAPTER_CONFIG_NAME = "adapter_config.json"
PEFT_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


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
    """Prove that ``root`` is locally consumable by LeRobot without remote fallback."""
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

    info = _read_info(root)
    frame_features = _frame_features(root, info)
    _read_stats(root)
    tasks = _read_tasks(root, info)
    episodes = _read_episodes(root, info)
    _validate_payloads(root, info, episodes)
    frames = _read_frames(root, info, frame_features)
    _validate_frame_metadata(root, info, tasks, episodes, frames)
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


class ModelDirectoryError(ValueError):
    """A local directory is not a loadable LeRobot policy checkpoint."""


@dataclass(frozen=True, slots=True)
class ModelDirectoryMetadata:
    """Metadata extracted from a validated local model directory."""

    has_full_weights: bool
    has_adapter_weights: bool
    policy_type: str | None
    source_path: Path
    git_commit: str | None

    def to_wandb_metadata(self) -> dict[str, Any]:
        """JSON-safe dict form, suitable for a W&B Artifact's ``metadata`` argument."""
        return {
            "has_full_weights": self.has_full_weights,
            "has_adapter_weights": self.has_adapter_weights,
            "policy_type": self.policy_type,
            "source_path": str(self.source_path),
            "git_commit": self.git_commit,
        }


def validate_model_directory(root: Path | str) -> dict[str, Any] | None:
    """Prove that ``root`` is a locally loadable LeRobot policy checkpoint.

    Checks only file *existence*, never opening or parsing a weights file: ``root`` must contain
    ``config.json`` plus either full weights (``model.safetensors``) or a complete PEFT adapter pair
    (``adapter_config.json`` and ``adapter_model.safetensors``). Optional processor files and
    ``train_config.json`` — and any other extra content — are tolerated and ignored.

    Returns the parsed contents of ``config.json`` (best-effort — ``None`` if it isn't valid JSON or
    isn't a JSON object), so a caller like :func:`inspect_model_directory` doesn't need to re-read it.
    """
    root = Path(root)
    if not root.is_dir():
        raise ModelDirectoryError(f"{root} is not a directory.")

    config_path = root / CONFIG_NAME
    if not config_path.is_file():
        raise ModelDirectoryError(f"{root} is missing required model config file: {CONFIG_NAME}.")

    has_full_weights = (root / SAFETENSORS_SINGLE_FILE).is_file()
    has_adapter_weights = (root / PEFT_ADAPTER_CONFIG_NAME).is_file() and (
        root / PEFT_ADAPTER_WEIGHTS_NAME
    ).is_file()
    if not has_full_weights and not has_adapter_weights:
        raise ModelDirectoryError(
            f"{root} has {CONFIG_NAME} but no model weights: expected either {SAFETENSORS_SINGLE_FILE} "
            f"(full weights) or both {PEFT_ADAPTER_CONFIG_NAME} and {PEFT_ADAPTER_WEIGHTS_NAME} "
            "(PEFT adapter weights)."
        )

    try:
        with config_path.open() as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return config if isinstance(config, dict) else None


def inspect_model_directory(root: Path | str) -> ModelDirectoryMetadata:
    """Validate ``root`` and extract its self-describing metadata."""
    root = Path(root)
    config = validate_model_directory(root)
    policy_type = config.get("type") if config is not None else None
    return ModelDirectoryMetadata(
        has_full_weights=(root / SAFETENSORS_SINGLE_FILE).is_file(),
        has_adapter_weights=(root / PEFT_ADAPTER_CONFIG_NAME).is_file()
        and (root / PEFT_ADAPTER_WEIGHTS_NAME).is_file(),
        policy_type=policy_type if isinstance(policy_type, str) else None,
        source_path=root.resolve(),
        git_commit=_current_git_commit(),
    )


def _read_info(root: Path) -> DatasetInfo:
    try:
        info = load_info(root)
        check_version_compatibility(str(root), info.codebase_version, CODEBASE_VERSION)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{INFO_PATH} could not be read as compatible dataset info: {e}"
        ) from e
    return info


def _frame_features(root: Path, info: DatasetInfo) -> datasets.Features:
    missing = sorted(set(DEFAULT_FEATURES) - set(info.features))
    incompatible = [
        key
        for key, expected in DEFAULT_FEATURES.items()
        if key in info.features
        and (
            info.features[key].get("dtype") != expected["dtype"]
            or tuple(info.features[key].get("shape", ())) != expected["shape"]
        )
    ]
    if missing or incompatible:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if incompatible:
            details.append(f"incompatible {sorted(incompatible)}")
        raise DatasetDirectoryError(
            f"{root}/{INFO_PATH} has invalid required frame features: {', '.join(details)}."
        )
    try:
        return get_hf_features_from_features(info.features)
    except Exception as e:
        raise DatasetDirectoryError(f"{root}/{INFO_PATH} contains an invalid frame schema: {e}") from e


def _read_stats(root: Path) -> None:
    try:
        if load_stats(root) is None:
            raise FileNotFoundError(STATS_PATH)
    except Exception as e:
        raise DatasetDirectoryError(f"{root}/{STATS_PATH} could not be read as dataset stats: {e}") from e


def _read_tasks(root: Path, info: DatasetInfo) -> pd.DataFrame | None:
    if info.total_frames > 0 and info.total_tasks == 0:
        raise DatasetDirectoryError(f"{root} declares total_frames={info.total_frames} but total_tasks=0.")
    if info.total_tasks == 0:
        return None
    try:
        tasks = load_tasks(root)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{DEFAULT_TASKS_PATH} could not be read as task metadata: {e}"
        ) from e
    if len(tasks) != info.total_tasks or not tasks.index.is_unique or "task_index" not in tasks:
        raise DatasetDirectoryError(
            f"{root}/{DEFAULT_TASKS_PATH} does not describe exactly {info.total_tasks} unique tasks."
        )
    try:
        indices = [int(value) for value in tasks["task_index"].tolist()]
    except (TypeError, ValueError) as e:
        raise DatasetDirectoryError(f"{root}/{DEFAULT_TASKS_PATH} has invalid task_index values: {e}") from e
    if indices != list(range(info.total_tasks)):
        raise DatasetDirectoryError(
            f"{root}/{DEFAULT_TASKS_PATH} task_index values must be ordered from 0 to {info.total_tasks - 1}."
        )
    return tasks


def _read_episodes(root: Path, info: DatasetInfo) -> datasets.Dataset | None:
    if info.total_episodes == 0:
        if info.total_frames > 0:
            raise DatasetDirectoryError(
                f"{root} declares total_frames={info.total_frames} but total_episodes=0."
            )
        return None
    try:
        episodes = load_episodes(root)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} could not be read as episode metadata: {e}"
        ) from e

    required = {
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    }
    for key in _video_keys(info):
        required |= {
            f"videos/{key}/chunk_index",
            f"videos/{key}/file_index",
            f"videos/{key}/from_timestamp",
            f"videos/{key}/to_timestamp",
        }
    missing = sorted(required - set(episodes.column_names))
    if missing:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} is missing required column(s): {', '.join(missing)}."
        )
    indices = _integer_values(episodes["episode_index"], f"{root}/{EPISODES_DIR} episode_index")
    if len(episodes) != info.total_episodes or indices != list(range(info.total_episodes)):
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} must contain {info.total_episodes} rows ordered by episode_index."
        )
    return episodes


def _validate_payloads(root: Path, info: DatasetInfo, episodes: datasets.Dataset | None) -> None:
    if episodes is None:
        return
    video_keys = _video_keys(info)
    if video_keys and info.video_path is None:
        raise DatasetDirectoryError(f"{root}/{INFO_PATH} declares videos without a video_path template.")

    referenced_data: set[Path] = set()
    referenced_videos: set[Path] = set()
    for episode_index, row in enumerate(episodes):
        length = _row_int(root, row, "length", episode_index)
        if length <= 0:
            raise DatasetDirectoryError(f"{root}/{EPISODES_DIR} episode {episode_index} has length={length}.")

        data_path = _safe_path(
            root,
            info.data_path,
            "data",
            chunk_index=_row_int(root, row, "data/chunk_index", episode_index),
            file_index=_row_int(root, row, "data/file_index", episode_index),
        )
        if len(data_path.parts) != 3 or data_path.parts[0] != DATA_DIR or data_path.suffix != ".parquet":
            raise DatasetDirectoryError(
                f"{root}/{INFO_PATH} data_path resolves outside the reader's {DATA_DIR}/*/*.parquet layout."
            )
        referenced_data.add(data_path)

        for key in video_keys:
            referenced_videos.add(
                _safe_path(
                    root,
                    info.video_path,
                    f"video {key!r}",
                    video_key=key,
                    chunk_index=_row_int(root, row, f"videos/{key}/chunk_index", episode_index),
                    file_index=_row_int(root, row, f"videos/{key}/file_index", episode_index),
                )
            )
            start = _row_float(root, row, f"videos/{key}/from_timestamp", episode_index)
            end = _row_float(root, row, f"videos/{key}/to_timestamp", episode_index)
            if start < 0 or end <= start or end - start + _TIMESTAMP_TOLERANCE_S < (length - 1) / info.fps:
                raise DatasetDirectoryError(
                    f"{root}/{EPISODES_DIR} episode {episode_index} has an invalid time range for {key!r}."
                )

    _require_files(root, referenced_data, "data")
    _require_files(root, referenced_videos, "video")


def _safe_path(root: Path, template: str | None, payload: str, **values: Any) -> Path:
    try:
        path = Path(template.format(**values))
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise DatasetDirectoryError(f"{root}/{INFO_PATH} cannot resolve the {payload} path: {e}") from e
    if path.is_absolute() or ".." in path.parts:
        raise DatasetDirectoryError(
            f"{root}/{INFO_PATH} resolves {payload} outside the dataset root: {path}."
        )
    try:
        (root / path).resolve(strict=False).relative_to(root.resolve())
    except ValueError as e:
        raise DatasetDirectoryError(
            f"{root}/{INFO_PATH} resolves {payload} outside the dataset root: {path}."
        ) from e
    return path


def _require_files(root: Path, paths: set[Path], payload: str) -> None:
    missing = sorted(path for path in paths if not (root / path).is_file())
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = f", and {len(missing) - 3} more" if len(missing) > 3 else ""
        raise DatasetDirectoryError(
            f"{root} is missing {len(missing)} {payload} file(s) referenced by episode metadata: "
            f"{preview}{suffix}."
        )


def _read_frames(root: Path, info: DatasetInfo, features: datasets.Features) -> datasets.Dataset | None:
    if not any((root / DATA_DIR).glob("*/*.parquet")):
        if info.total_frames == 0:
            return None
        raise DatasetDirectoryError(f"{root}/{DATA_DIR} has no loader-visible parquet files.")
    try:
        frames = load_nested_dataset(root / DATA_DIR, features=features)
    except Exception as e:
        raise DatasetDirectoryError(
            f"{root}/{DATA_DIR} does not match the frame schema declared in {INFO_PATH}: {e}"
        ) from e
    if set(frames.column_names) != set(features) or len(frames) != info.total_frames:
        raise DatasetDirectoryError(
            f"{root}/{DATA_DIR} does not contain exactly {info.total_frames} rows with the declared columns."
        )
    return frames


def _validate_frame_metadata(
    root: Path,
    info: DatasetInfo,
    tasks: pd.DataFrame | None,
    episodes: datasets.Dataset | None,
    frames: datasets.Dataset | None,
) -> None:
    if frames is None:
        if any((info.total_frames, info.total_episodes, info.total_tasks)):
            raise DatasetDirectoryError(f"{root} has nonzero metadata for an empty dataset.")
        return
    if episodes is None or tasks is None:
        raise DatasetDirectoryError(f"{root} has frame data without episode and task metadata.")

    indices = _integer_values(frames["index"], f"{root}/{DATA_DIR} index")
    episode_indices = _integer_values(frames["episode_index"], f"{root}/{DATA_DIR} episode_index")
    frame_indices = _integer_values(frames["frame_index"], f"{root}/{DATA_DIR} frame_index")
    task_indices = _integer_values(frames["task_index"], f"{root}/{DATA_DIR} task_index")
    timestamps = _float_values(frames["timestamp"], f"{root}/{DATA_DIR} timestamp")
    if indices != list(range(info.total_frames)):
        raise DatasetDirectoryError(f"{root}/{DATA_DIR} index values are not globally contiguous.")
    invalid_tasks = sorted({value for value in task_indices if not 0 <= value < info.total_tasks})
    if invalid_tasks:
        raise DatasetDirectoryError(
            f"{root}/{DATA_DIR} contains task_index values with no matching task row: {invalid_tasks}."
        )

    cursor = 0
    for episode_index, row in enumerate(episodes):
        length = _row_int(root, row, "length", episode_index)
        start = _row_int(root, row, "dataset_from_index", episode_index)
        end = _row_int(root, row, "dataset_to_index", episode_index)
        if start != cursor or end != start + length or end > info.total_frames:
            raise DatasetDirectoryError(
                f"{root}/{EPISODES_DIR} episode {episode_index} has inconsistent frame range/length."
            )
        if episode_indices[start:end] != [episode_index] * length:
            raise DatasetDirectoryError(
                f"{root}/{DATA_DIR} rows [{start}, {end}) do not match episode {episode_index}."
            )
        if frame_indices[start:end] != list(range(length)):
            raise DatasetDirectoryError(
                f"{root}/{DATA_DIR} frame_index values for episode {episode_index} are not contiguous."
            )
        for frame_index, timestamp in enumerate(timestamps[start:end]):
            if abs(timestamp - frame_index / info.fps) > _TIMESTAMP_TOLERANCE_S:
                raise DatasetDirectoryError(
                    f"{root}/{DATA_DIR} timestamp for episode {episode_index}, frame {frame_index} "
                    f"is not synchronized to fps={info.fps}."
                )
        cursor = end
    if cursor != info.total_frames:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} frame ranges cover {cursor} rows, expected {info.total_frames}."
        )


def _video_keys(info: DatasetInfo) -> tuple[str, ...]:
    return tuple(key for key, feature in info.features.items() if feature["dtype"] == "video")


def _row_int(root: Path, row: dict, key: str, episode_index: int) -> int:
    try:
        return _strict_int(row[key])
    except (KeyError, TypeError, ValueError) as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} episode {episode_index} has invalid {key!r}: {e}"
        ) from e


def _row_float(root: Path, row: dict, key: str, episode_index: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as e:
        raise DatasetDirectoryError(
            f"{root}/{EPISODES_DIR} episode {episode_index} has invalid {key!r}: {e}"
        ) from e
    if not math.isfinite(value):
        raise DatasetDirectoryError(f"{root}/{EPISODES_DIR} episode {episode_index} has non-finite {key!r}.")
    return value


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{value!r} is not an integer")
    return int(value)


def _integer_values(values: Any, label: str) -> list[int]:
    try:
        return [_strict_int(value) for value in values]
    except (TypeError, ValueError) as e:
        raise DatasetDirectoryError(f"{label} contains invalid integer values: {e}") from e


def _float_values(values: Any, label: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as e:
        raise DatasetDirectoryError(f"{label} contains invalid numeric values: {e}") from e
    if not all(math.isfinite(value) for value in result):
        raise DatasetDirectoryError(f"{label} contains non-finite values.")
    return result


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
