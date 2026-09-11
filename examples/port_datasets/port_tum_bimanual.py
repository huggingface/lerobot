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

"""Port generic bimanual TUM-style captures to LeRobot Dataset v3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_PARENTHESIZED = re.compile(r"\([^)]*\)")


class PortingError(RuntimeError):
    """An expected source-format or conversion error."""


@dataclass(frozen=True)
class HandPaths:
    root: Path
    trajectory: Path
    clamp: Path
    imu: Path
    video: Path
    video_timestamps: Path
    intrinsics: Path

    @property
    def name(self) -> str:
        return self.root.name


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    left: HandPaths
    right: HandPaths
    relative_pose: Path

    @property
    def name(self) -> str:
        return self.root.name


@dataclass(frozen=True)
class TimedArray:
    timestamps: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class HandData:
    video_path: Path
    video_timestamps: np.ndarray
    trajectory: TimedArray
    clamp: TimedArray
    imu: TimedArray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class SessionData:
    name: str
    left: HandData
    right: HandData
    relative_pose: TimedArray


@dataclass(frozen=True)
class SyncedEpisode:
    name: str
    timestamps: np.ndarray
    left_video_indices: np.ndarray
    right_video_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    imu: np.ndarray
    relative_pose: np.ndarray
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class PortOptions:
    raw_dir: Path
    repo_id: str
    root: Path | None = None
    fps: int = 60
    task: str = "bimanual hand manipulation"
    push_to_hub: bool = False
    overwrite: bool = False
    skip_invalid_session: bool = False


def _one_directory(session_root: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in session_root.glob(pattern) if path.is_dir())
    if len(matches) != 1:
        raise PortingError(
            f"{session_root.name}: expected exactly one {label} directory, found {len(matches)}"
        )
    return matches[0]


def _hand_paths(root: Path) -> HandPaths:
    return HandPaths(
        root=root,
        trajectory=root / "Merged_Trajectory" / "merged_trajectory.txt",
        clamp=root / "Clamp_Data" / "clamp_data_tum.txt",
        imu=root / "IMU" / "imu.txt",
        video=root / "RGB_Images" / "video.mp4",
        video_timestamps=root / "RGB_Images" / "timestamps.csv",
        intrinsics=root / "Calibration" / "rgb_intrinsic.json",
    )


def _build_session_paths(root: Path) -> SessionPaths:
    left = _one_directory(root, "left_hand_*", "left_hand")
    right = _one_directory(root, "right_hand_*", "right_hand")
    return SessionPaths(
        root=root,
        left=_hand_paths(left),
        right=_hand_paths(right),
        relative_pose=root / "relative_transforms_left_to_right.txt",
    )


def discover_sessions(raw_dir: Path) -> list[SessionPaths]:
    """Discover sorted, structurally complete session directories."""

    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise PortingError(f"raw directory does not exist: {raw_dir}")
    session_roots = sorted(path for path in raw_dir.glob("session_*") if path.is_dir())
    if not session_roots:
        raise PortingError(f"no session_* directories found: {raw_dir}")
    return [_build_session_paths(path) for path in session_roots]


def _validate_series(path: Path, timestamps: np.ndarray, values: np.ndarray) -> TimedArray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise PortingError(f"{path}: empty timestamp series")
    if values.ndim != 2 or values.shape[0] != timestamps.size:
        raise PortingError(f"{path}: invalid value matrix")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(values)):
        raise PortingError(f"{path}: values must be finite")
    if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0):
        raise PortingError(f"{path}: timestamps must be strictly increasing")
    return TimedArray(timestamps=timestamps, values=values)


def _numeric_rows(path: Path, value_width: int) -> TimedArray:
    timestamps: list[float] = []
    values: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != value_width + 1:
                    raise PortingError(
                        f"{path}: expected {value_width + 1} numeric columns at line {line_number}"
                    )
                timestamps.append(float(fields[0]))
                values.append([float(item) for item in fields[1:]])
    except OSError as error:
        raise PortingError(f"{path}: unable to read source file") from error
    except ValueError as error:
        raise PortingError(f"{path}: invalid numeric value") from error
    return _validate_series(path, np.asarray(timestamps), np.asarray(values))


def read_tum_pose(path: Path) -> TimedArray:
    series = _numeric_rows(path, value_width=7)
    quaternions = series.values[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise PortingError(f"{path}: quaternion has zero norm")
    values = series.values.copy()
    values[:, 3:7] = quaternions / norms
    return TimedArray(series.timestamps, values)


def read_clamp(path: Path) -> TimedArray:
    return _numeric_rows(path, value_width=1)


def read_imu(path: Path) -> TimedArray:
    timestamps: list[float] = []
    values: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = _PARENTHESIZED.sub("", line).split()
                if len(fields) != 7:
                    raise PortingError(f"{path}: expected timestamp, gyro3 and accel3 at line {line_number}")
                timestamps.append(float(fields[0]))
                values.append([float(item) for item in fields[1:]])
    except OSError as error:
        raise PortingError(f"{path}: unable to read source file") from error
    except ValueError as error:
        raise PortingError(f"{path}: invalid numeric value") from error
    return _validate_series(path, np.asarray(timestamps), np.asarray(values))


def read_video_timestamps(path: Path) -> np.ndarray:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == ["timestamp"]:
                timestamps = [float(row["timestamp"]) for row in reader]
                indices = list(range(len(timestamps)))
            elif reader.fieldnames == ["frame_index", "seq", "header_stamp"]:
                rows = list(reader)
                indices = [int(row["frame_index"]) for row in rows]
                timestamps = [float(row["header_stamp"]) for row in rows]
            else:
                raise PortingError(f"{path}: unsupported timestamp CSV header")
    except OSError as error:
        raise PortingError(f"{path}: unable to read source file") from error
    except (KeyError, TypeError, ValueError) as error:
        raise PortingError(f"{path}: invalid timestamp CSV value") from error
    if indices != list(range(len(indices))):
        raise PortingError(f"{path}: frame_index must be zero-based and contiguous")
    series = _validate_series(
        path,
        np.asarray(timestamps),
        np.asarray(indices, dtype=np.float64).reshape(-1, 1),
    )
    return series.timestamps


def read_intrinsics(path: Path) -> tuple[int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        height = payload["height"]
        width = payload["width"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PortingError(f"{path}: invalid camera intrinsics JSON") from error
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or isinstance(width, bool)
        or height <= 0
        or width <= 0
    ):
        raise PortingError(f"{path}: camera width and height must be positive integers")
    return height, width


def _read_hand(paths: HandPaths) -> HandData:
    return HandData(
        video_path=paths.video,
        video_timestamps=read_video_timestamps(paths.video_timestamps),
        trajectory=read_tum_pose(paths.trajectory),
        clamp=read_clamp(paths.clamp),
        imu=read_imu(paths.imu),
        image_size=read_intrinsics(paths.intrinsics),
    )


def read_session(paths: SessionPaths) -> SessionData:
    left = _read_hand(paths.left)
    right = _read_hand(paths.right)
    relative = read_tum_pose(paths.relative_pose)
    relative = TimedArray(
        timestamps=(left.trajectory.timestamps[0] + relative.timestamps - relative.timestamps[0]),
        values=relative.values,
    )
    return SessionData(
        name=paths.name,
        left=left,
        right=right,
        relative_pose=relative,
    )


def nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    source = np.asarray(source_timestamps, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise PortingError("source timestamp series is empty")
    if target.ndim != 1 or target.size == 0:
        raise PortingError("target timestamp series is empty")
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    return np.where(choose_right, right, left).astype(np.int64)


def _validate_target(series: TimedArray, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or target.size == 0:
        raise PortingError("resampling target is empty")
    if target[0] < series.timestamps[0] - 1e-9 or target[-1] > series.timestamps[-1] + 1e-9:
        raise PortingError("resampling would require extrapolation")
    return target


def linear_series(series: TimedArray, target: np.ndarray) -> np.ndarray:
    target = _validate_target(series, target)
    return np.column_stack(
        [
            np.interp(target, series.timestamps, series.values[:, column])
            for column in range(series.values.shape[1])
        ]
    )


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(norms <= 1e-12):
        raise PortingError("quaternion is non-finite or has zero norm")
    return values / norms


def slerp_series(series: TimedArray, target: np.ndarray) -> np.ndarray:
    target = _validate_target(series, target)
    quaternions = _normalize_quaternions(series.values)
    if quaternions.shape[1] != 4:
        raise PortingError("SLERP input must contain four quaternion values")
    if len(series.timestamps) == 1:
        return np.repeat(quaternions, len(target), axis=0)

    right_indices = np.searchsorted(series.timestamps, target, side="right")
    right_indices = np.clip(right_indices, 1, len(series.timestamps) - 1)
    left_indices = right_indices - 1
    left_times = series.timestamps[left_indices]
    right_times = series.timestamps[right_indices]
    alpha = ((target - left_times) / (right_times - left_times)).reshape(-1, 1)

    left = quaternions[left_indices]
    right = quaternions[right_indices]
    dots = np.sum(left * right, axis=1, keepdims=True)
    right = np.where(dots < 0, -right, right)
    dots = np.abs(dots)

    linear = _normalize_quaternions((1.0 - alpha) * left + alpha * right)
    theta = np.arccos(np.clip(dots, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe_denominator = np.where(np.abs(sin_theta) < 1e-12, 1.0, sin_theta)
    spherical = (
        np.sin((1.0 - alpha) * theta) / safe_denominator * left
        + np.sin(alpha * theta) / safe_denominator * right
    )
    result = np.where(dots > 0.9995, linear, spherical)
    return _normalize_quaternions(result)


def _pose_series(series: TimedArray, target: np.ndarray) -> np.ndarray:
    translation = linear_series(TimedArray(series.timestamps, series.values[:, :3]), target)
    quaternion = slerp_series(TimedArray(series.timestamps, series.values[:, 3:7]), target)
    return np.concatenate([translation, quaternion], axis=1)


def _build_actions(states: np.ndarray) -> np.ndarray:
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


def sync_session(data: SessionData, fps: int) -> SyncedEpisode:
    if fps <= 0:
        raise PortingError("fps must be positive")
    required = [
        data.left.video_timestamps,
        data.right.video_timestamps,
        data.left.trajectory.timestamps,
        data.right.trajectory.timestamps,
        data.left.clamp.timestamps,
        data.right.clamp.timestamps,
        data.left.imu.timestamps,
        data.right.imu.timestamps,
        data.relative_pose.timestamps,
    ]
    start = max(float(values[0]) for values in required)
    end = min(float(values[-1]) for values in required)
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        raise PortingError(f"{data.name}: sensor streams have no common time range")
    count = int(np.floor((end - start) * fps + 1e-9)) + 1
    if count < 2:
        raise PortingError(f"{data.name}: common time range has fewer than two frames")
    target = start + np.arange(count, dtype=np.float64) / fps

    left_pose = _pose_series(data.left.trajectory, target)
    right_pose = _pose_series(data.right.trajectory, target)
    left_gripper = linear_series(data.left.clamp, target)
    right_gripper = linear_series(data.right.clamp, target)
    left_imu = linear_series(data.left.imu, target)
    right_imu = linear_series(data.right.imu, target)
    relative_pose = _pose_series(data.relative_pose, target)

    states = np.concatenate([left_pose, left_gripper, right_pose, right_gripper], axis=1).astype(np.float32)
    imu = np.concatenate([left_imu, right_imu], axis=1).astype(np.float32)
    relative_pose = relative_pose.astype(np.float32)
    if not all(np.all(np.isfinite(values)) for values in (states, imu, relative_pose)):
        raise PortingError(f"{data.name}: synchronized values must be finite")

    left_indices = nearest_indices(data.left.video_timestamps, target)
    right_indices = nearest_indices(data.right.video_timestamps, target)
    left_errors = np.abs(data.left.video_timestamps[left_indices] - target)
    right_errors = np.abs(data.right.video_timestamps[right_indices] - target)
    metrics: dict[str, float | int] = {
        "common_start": float(start),
        "common_end": float(target[-1]),
        "output_frames": int(count),
        "left_video_max_match_error_s": float(left_errors.max()),
        "right_video_max_match_error_s": float(right_errors.max()),
    }
    return SyncedEpisode(
        name=data.name,
        timestamps=target,
        left_video_indices=left_indices,
        right_video_indices=right_indices,
        states=states,
        actions=_build_actions(states),
        imu=imu,
        relative_pose=relative_pose,
        metrics=metrics,
    )


STATE_NAMES = (
    "left_tx",
    "left_ty",
    "left_tz",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "left_gripper",
    "right_tx",
    "right_ty",
    "right_tz",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "right_gripper",
)

IMU_NAMES = (
    "left_gyro_x",
    "left_gyro_y",
    "left_gyro_z",
    "left_accel_x",
    "left_accel_y",
    "left_accel_z",
    "right_gyro_x",
    "right_gyro_y",
    "right_gyro_z",
    "right_accel_x",
    "right_accel_y",
    "right_accel_z",
)


def make_features(height: int, width: int) -> dict[str, dict]:
    if height <= 0 or width <= 0:
        raise PortingError("camera dimensions must be positive")
    image = {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        "observation.images.left_hand": dict(image),
        "observation.images.right_hand": dict(image),
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(STATE_NAMES),
        },
        "observation.imu": {
            "dtype": "float32",
            "shape": (12,),
            "names": list(IMU_NAMES),
        },
        "observation.relative_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(STATE_NAMES),
        },
        "source_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["device_time_s"],
        },
    }


def read_video_indices(path: Path, indices: np.ndarray) -> Iterator[np.ndarray]:
    requested = np.asarray(indices)
    if requested.ndim != 1 or requested.size == 0:
        raise PortingError(f"{path}: requested video indices are empty")
    if not np.issubdtype(requested.dtype, np.integer):
        raise PortingError(f"{path}: video indices must be integers")
    requested = requested.astype(np.int64)
    if np.any(requested < 0) or np.any(np.diff(requested) < 0):
        raise PortingError(f"{path}: video indices must be nonnegative and nondecreasing")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PortingError(f"{path}: unable to open video")
    decoded_index = -1
    cached_index = -1
    cached_rgb: np.ndarray | None = None
    try:
        for wanted in requested:
            if int(wanted) == cached_index and cached_rgb is not None:
                yield cached_rgb.copy()
                continue
            while decoded_index < int(wanted):
                ok, frame = capture.read()
                if not ok:
                    raise PortingError(f"{path}: unable to decode requested frame {int(wanted)}")
                decoded_index += 1
            cached_index = decoded_index
            cached_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield cached_rgb.copy()
    finally:
        capture.release()


def _build_frame(
    episode: SyncedEpisode,
    index: int,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    task: str,
) -> dict:
    if not task.strip():
        raise PortingError(f"{episode.name}: task text cannot be empty")
    if (
        left_rgb.dtype != np.uint8
        or right_rgb.dtype != np.uint8
        or left_rgb.ndim != 3
        or right_rgb.ndim != 3
        or left_rgb.shape[2] != 3
        or right_rgb.shape[2] != 3
        or left_rgb.shape != right_rgb.shape
    ):
        raise PortingError(f"{episode.name}: camera frames must be matching RGB uint8")
    return {
        "observation.images.left_hand": left_rgb,
        "observation.images.right_hand": right_rgb,
        "observation.state": episode.states[index],
        "observation.imu": episode.imu[index],
        "observation.relative_pose": episode.relative_pose[index],
        "action": episode.actions[index],
        "source_timestamp": np.asarray([episode.timestamps[index]], dtype=np.float64),
        "task": task,
    }


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PortingError(f"{path}: unable to open video")
    try:
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if count <= 0:
        raise PortingError(f"{path}: video contains no decodable frames")
    return count


def _validate_video_timestamp_counts(data: SessionData) -> None:
    for label, hand in (("left", data.left), ("right", data.right)):
        frame_count = _video_frame_count(hand.video_path)
        timestamp_count = len(hand.video_timestamps)
        if frame_count != timestamp_count:
            raise PortingError(
                f"{data.name}: {label} video frames ({frame_count}) do not match "
                f"timestamps ({timestamp_count})"
            )


def _source_ranges(data: SessionData) -> dict[str, list[float]]:
    streams = {
        "left_video": data.left.video_timestamps,
        "right_video": data.right.video_timestamps,
        "left_trajectory": data.left.trajectory.timestamps,
        "right_trajectory": data.right.trajectory.timestamps,
        "left_gripper": data.left.clamp.timestamps,
        "right_gripper": data.right.clamp.timestamps,
        "left_imu": data.left.imu.timestamps,
        "right_imu": data.right.imu.timestamps,
        "relative_pose": data.relative_pose.timestamps,
    }
    return {name: [float(timestamps[0]), float(timestamps[-1])] for name, timestamps in streams.items()}


def _configure_local_dataset_cache(output_root: Path) -> None:
    cache_root = output_root.parent / ".cache" / "lerobot"
    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    try:
        import datasets.config

        datasets.config.HF_DATASETS_CACHE = datasets_cache
    except ImportError:
        pass


def prepare_output(raw_dir: Path, root: Path | None, overwrite: bool) -> tuple[Path, Path | None]:
    raw_dir = raw_dir.expanduser().resolve()
    if root is None:
        return raw_dir, None
    root = root.expanduser().resolve()
    if raw_dir == root or raw_dir in root.parents or root in raw_dir.parents:
        raise PortingError("raw and output paths overlap")
    if root.exists():
        if not overwrite:
            raise PortingError(f"output already exists: {root}")
        if root == Path("/") or root == root.parent:
            raise PortingError(f"refusing to remove unsafe output path: {root}")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    return raw_dir, root


def _sanitized_error(error: Exception, raw_dir: Path) -> str:
    return str(error).replace(str(raw_dir), "<raw_dir>")


def convert_dataset(options: PortOptions) -> dict:
    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise PortingError("LeRobot with dataset support is required") from error

    raw_dir, output_root = prepare_output(options.raw_dir, options.root, options.overwrite)
    paths = discover_sessions(raw_dir)
    sessions: list[SessionData] = []
    report_by_name: dict[str, dict] = {}
    expected_size: tuple[int, int] | None = None
    for session_paths in paths:
        try:
            data = read_session(session_paths)
            if data.left.image_size != data.right.image_size:
                raise PortingError(f"{data.name}: left and right camera dimensions differ")
            _validate_video_timestamp_counts(data)
            if expected_size is None:
                expected_size = data.left.image_size
            elif data.left.image_size != expected_size:
                raise PortingError(f"{data.name}: camera dimensions differ from earlier sessions")
            sessions.append(data)
        except Exception as error:
            if not options.skip_invalid_session:
                raise
            report_by_name[session_paths.name] = {
                "session": session_paths.name,
                "status": "failed",
                "error": _sanitized_error(error, raw_dir),
            }
    if not sessions or expected_size is None:
        raise PortingError("no valid sessions to convert")
    height, width = expected_size
    if output_root is not None:
        _configure_local_dataset_cache(output_root)

    dataset = LeRobotDataset.create(
        repo_id=options.repo_id,
        root=output_root,
        fps=options.fps,
        robot_type="generic_bimanual_tum",
        features=make_features(height, width),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_queue_maxsize=30,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264",
            pix_fmt="yuv420p",
            g=2,
            crf=23,
            preset="veryfast",
        ),
        encoder_threads=4,
    )

    frames_total = 0
    try:
        for data in sessions:
            try:
                episode = sync_session(data, options.fps)
                left_frames = read_video_indices(data.left.video_path, episode.left_video_indices)
                right_frames = read_video_indices(data.right.video_path, episode.right_video_indices)
                for index, (left_rgb, right_rgb) in enumerate(zip(left_frames, right_frames, strict=True)):
                    dataset.add_frame(
                        _build_frame(
                            episode,
                            index,
                            left_rgb,
                            right_rgb,
                            options.task,
                        )
                    )
                dataset.save_episode(parallel_encoding=False)
                frames_total += len(episode.timestamps)
                report_by_name[data.name] = {
                    "session": data.name,
                    "status": "success",
                    "frames": len(episode.timestamps),
                    "metrics": episode.metrics,
                    "source_ranges": _source_ranges(data),
                }
            except Exception as error:
                if not options.skip_invalid_session:
                    raise
                dataset.clear_episode_buffer(delete_images=True)
                report_by_name[data.name] = {
                    "session": data.name,
                    "status": "failed",
                    "error": _sanitized_error(error, raw_dir),
                }
    finally:
        dataset.finalize()

    reports = [report_by_name[item.name] for item in paths]
    episode_count = sum(item["status"] == "success" for item in reports)
    if episode_count == 0:
        raise PortingError("no sessions were written successfully")
    output_root = Path(dataset.root)
    loaded = LeRobotDataset(repo_id=options.repo_id, root=output_root)
    if int(loaded.num_episodes) != episode_count or len(loaded) != frames_total:
        raise PortingError("output dataset episode or frame count validation failed")
    required_video_keys = {
        "observation.images.left_hand",
        "observation.images.right_hand",
    }
    if not required_video_keys.issubset(loaded.features):
        raise PortingError("output dataset is missing required video features")
    for index in sorted({0, len(loaded) - 1}):
        frame = loaded[index]
        if not required_video_keys.issubset(frame):
            raise PortingError("output dataset video decoding validation failed")
    validation = {
        "loadable": True,
        "episodes": int(loaded.num_episodes),
        "frames": len(loaded),
    }
    if options.push_to_hub:
        dataset.push_to_hub()
    report = {
        "format": "LeRobotDataset v3",
        "repo_id": options.repo_id,
        "fps": options.fps,
        "episodes": episode_count,
        "frames": frames_total,
        "sessions": reports,
        "validation": validation,
    }
    (output_root / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Port generic bimanual TUM-style captures to LeRobot Dataset v3."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--fps", type=_positive_int, default=60)
    parser.add_argument("--task", default="bimanual hand manipulation")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-invalid-session", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = convert_dataset(
            PortOptions(
                raw_dir=args.raw_dir,
                repo_id=args.repo_id,
                root=args.root,
                fps=args.fps,
                task=args.task,
                push_to_hub=args.push_to_hub,
                overwrite=args.overwrite,
                skip_invalid_session=args.skip_invalid_session,
            )
        )
    except PortingError as error:
        print(f"porting error: {error}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    print(
        f"port complete: repo_id={report['repo_id']} episodes={report['episodes']} frames={report['frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
