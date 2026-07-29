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

"""Create a modern episode video with RECAP value and advantage overlays."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

from lerobot.datasets import LeRobotDatasetMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlayTheme:
    background: tuple[int, int, int] = (22, 24, 28)
    surface: tuple[int, int, int] = (34, 37, 43)
    text: tuple[int, int, int] = (242, 244, 248)
    muted: tuple[int, int, int] = (158, 164, 176)
    positive: tuple[int, int, int] = (103, 198, 94)
    negative: tuple[int, int, int] = (91, 91, 235)
    intervention: tuple[int, int, int] = (72, 184, 238)
    value: tuple[int, int, int] = (230, 178, 76)
    target: tuple[int, int, int] = (190, 194, 204)
    raw_value: tuple[int, int, int] = (104, 111, 125)
    marker: tuple[int, int, int] = (255, 255, 255)


THEME = OverlayTheme()


def _fit_text(text: str, max_width: int, font_scale: float, thickness: int = 1) -> str:
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    while (
        text
        and cv2.getTextSize(
            text + suffix,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]
        > max_width
    ):
        text = text[:-1]
    return text + suffix


def _draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    x0, y0 = top_left
    x1, y1 = bottom_right
    radius = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
    cv2.rectangle(image, (x0 + radius, y0), (x1 - radius, y1), color, -1)
    cv2.rectangle(image, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    for center in (
        (x0 + radius, y0 + radius),
        (x1 - radius, y0 + radius),
        (x0 + radius, y1 - radius),
        (x1 - radius, y1 - radius),
    ):
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def _alpha_rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    *,
    alpha: float,
    radius: int,
) -> None:
    overlay = image.copy()
    _rounded_rectangle(overlay, top_left, bottom_right, color, radius)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def _contiguous_segments(labels: np.ndarray) -> list[tuple[int, int, str]]:
    if len(labels) == 0:
        return []
    segments: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(labels)):
        if labels[index] != labels[start]:
            segments.append((start, index, str(labels[start])))
            start = index
    segments.append((start, len(labels), str(labels[start])))
    return segments


def _draw_timeline(
    dashboard: np.ndarray,
    labels: np.ndarray,
    interventions: np.ndarray,
    current_index: int,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> None:
    _rounded_rectangle(dashboard, (x0, y0), (x1, y1), THEME.surface, 5)
    width = x1 - x0
    count = max(len(labels), 1)
    for start, end, label in _contiguous_segments(labels):
        segment_x0 = x0 + round(width * start / count)
        segment_x1 = x0 + round(width * end / count)
        color = THEME.positive if label == "positive" else THEME.negative
        cv2.rectangle(
            dashboard,
            (segment_x0, y0 + 2),
            (max(segment_x0 + 1, segment_x1), y1 - 2),
            color,
            -1,
        )
    for index in np.flatnonzero(interventions):
        x = x0 + round(width * index / count)
        cv2.line(dashboard, (x, y0), (x, y1), THEME.intervention, 2, cv2.LINE_AA)
    marker_x = x0 + round(width * current_index / max(count - 1, 1))
    cv2.line(dashboard, (marker_x, y0 - 3), (marker_x, y1 + 3), THEME.marker, 2, cv2.LINE_AA)


def _value_to_y(value: float, y0: int, y1: int) -> int:
    normalized = float(np.clip(value, -1.0, 0.0) + 1.0)
    return round(y1 - normalized * (y1 - y0))


def _draw_curve(
    image: np.ndarray,
    values: np.ndarray,
    color: tuple[int, int, int],
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    thickness: int,
) -> None:
    if len(values) < 2:
        return
    x = np.linspace(x0, x1, len(values)).astype(np.int32)
    y = np.asarray([_value_to_y(float(value), y0, y1) for value in values], dtype=np.int32)
    points = np.stack((x, y), axis=1).reshape(-1, 1, 2)
    cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)


def _draw_dashboard(
    width: int,
    height: int,
    episode: pd.DataFrame,
    current_index: int,
    fps: float,
    task: str,
) -> np.ndarray:
    dashboard = np.full((height, width, 3), THEME.background, dtype=np.uint8)
    padding = max(14, width // 45)
    value = float(episode.iloc[current_index]["predicted_value"])
    advantage = float(episode.iloc[current_index]["advantage"])
    time_s = current_index / fps
    duration_s = max((len(episode) - 1) / fps, 0)

    task_text = _fit_text(task, int(width * 0.52), 0.48)
    _draw_text(
        dashboard,
        task_text,
        (padding, 24),
        scale=0.48,
        color=THEME.text,
        thickness=1,
    )
    metric_text = f"VALUE  {value:+.3f}     ADV  {advantage:+.3f}"
    _draw_text(
        dashboard,
        metric_text,
        (padding, 49),
        scale=0.5,
        color=THEME.text,
        thickness=1,
    )
    time_text = f"{time_s:05.1f}s  /  {duration_s:05.1f}s"
    time_width = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
    _draw_text(
        dashboard,
        time_text,
        (width - padding - time_width, 24),
        scale=0.48,
        color=THEME.muted,
    )

    labels = episode["advantage_label"].astype(str).to_numpy()
    interventions = (
        episode["intervention"].astype(bool).to_numpy()
        if "intervention" in episode
        else np.zeros(len(episode), dtype=bool)
    )
    timeline_y0 = 59
    timeline_y1 = 73
    _draw_timeline(
        dashboard,
        labels,
        interventions,
        current_index,
        x0=padding,
        x1=width - padding,
        y0=timeline_y0,
        y1=timeline_y1,
    )

    curve_y0 = 82
    curve_y1 = height - 13
    cv2.line(
        dashboard,
        (padding, _value_to_y(0.0, curve_y0, curve_y1)),
        (width - padding, _value_to_y(0.0, curve_y0, curve_y1)),
        THEME.surface,
        1,
        cv2.LINE_AA,
    )
    _draw_curve(
        dashboard,
        episode["mc_return"].to_numpy(float),
        THEME.target,
        x0=padding,
        x1=width - padding,
        y0=curve_y0,
        y1=curve_y1,
        thickness=1,
    )
    if "predicted_value_raw" in episode:
        _draw_curve(
            dashboard,
            episode["predicted_value_raw"].to_numpy(float),
            THEME.raw_value,
            x0=padding,
            x1=width - padding,
            y0=curve_y0,
            y1=curve_y1,
            thickness=1,
        )
    _draw_curve(
        dashboard,
        episode["predicted_value"].to_numpy(float),
        THEME.value,
        x0=padding,
        x1=width - padding,
        y0=curve_y0,
        y1=curve_y1,
        thickness=2,
    )
    marker_x = padding + round((width - 2 * padding) * current_index / max(len(episode) - 1, 1))
    marker_y = _value_to_y(value, curve_y0, curve_y1)
    cv2.circle(dashboard, (marker_x, marker_y), 4, THEME.marker, -1, cv2.LINE_AA)
    return dashboard


def _draw_label_timeline(
    frame: np.ndarray,
    labels: np.ndarray,
    current_index: int,
) -> None:
    height, width = frame.shape[:2]
    horizontal_margin = max(18, round(width * 0.04))
    bar_height = max(26, round(height * 0.055))
    bottom_margin = max(14, round(height * 0.03))
    x0 = horizontal_margin
    x1 = width - horizontal_margin
    y1 = height - bottom_margin
    y0 = y1 - bar_height
    radius = max(6, bar_height // 3)

    _alpha_rounded_rectangle(
        frame,
        (x0, y0),
        (x1, y1),
        THEME.background,
        alpha=0.36,
        radius=radius,
    )

    inset = max(3, bar_height // 8)
    inner_x0, inner_x1 = x0 + inset, x1 - inset
    inner_y0, inner_y1 = y0 + inset, y1 - inset
    timeline_width = inner_x1 - inner_x0
    count = max(len(labels), 1)
    overlay = frame.copy()
    for start, end, label in _contiguous_segments(labels):
        segment_x0 = inner_x0 + round(timeline_width * start / count)
        segment_x1 = inner_x0 + round(timeline_width * end / count)
        color = THEME.positive if label == "positive" else THEME.negative
        cv2.rectangle(
            overlay,
            (segment_x0, inner_y0),
            (max(segment_x0 + 1, segment_x1), inner_y1),
            color,
            -1,
        )
    cv2.addWeighted(overlay, 0.56, frame, 0.44, 0, dst=frame)

    marker_x = inner_x0 + round(timeline_width * current_index / max(len(labels) - 1, 1))
    cv2.line(
        frame,
        (marker_x, y0 - 3),
        (marker_x, y1 + 3),
        THEME.marker,
        2,
        cv2.LINE_AA,
    )


def _resolve_dataset(
    repo_id: str,
    root: Path | None,
) -> tuple[Path, LeRobotDatasetMetadata]:
    if root is None:
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=["meta/**"],
            )
        )
    metadata = LeRobotDatasetMetadata(repo_id="local", root=root)
    return root, metadata


def _resolve_video_path(
    repo_id: str,
    root: Path,
    relative_path: Path,
    local_root: bool,
) -> Path:
    path = root / relative_path
    if path.is_file():
        return path
    if local_root:
        raise FileNotFoundError(f"Video not found: {path}")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=str(relative_path),
        )
    )


def _encode_h264(temp_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        temp_path.replace(output_path)
        logger.warning("ffmpeg not found; kept OpenCV mp4v output at %s", output_path)
        return
    result = subprocess.run(  # nosec B603
        [
            ffmpeg,
            "-y",
            "-i",
            str(temp_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("H.264 encoding failed; retaining mp4v output: %s", result.stderr[-500:])
        temp_path.replace(output_path)
        return
    temp_path.unlink(missing_ok=True)


def _create_decode_proxy(
    source_path: Path,
    proxy_path: Path,
    *,
    start_timestamp: float,
    end_timestamp: float,
    fps: float,
    expected_frames: int,
) -> Path:
    """Decode an episode segment with system FFmpeg into an OpenCV-safe proxy."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "System ffmpeg is required to decode this dataset's video codec. "
            "Install ffmpeg and rerun the command."
        )
    duration = end_timestamp - start_timestamp
    result = subprocess.run(  # nosec B603
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_timestamp:.9f}",
            "-i",
            str(source_path),
            "-t",
            f"{duration:.9f}",
            "-an",
            "-vf",
            f"fps={fps:.9f}",
            "-frames:v",
            str(expected_frames),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(proxy_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not proxy_path.is_file():
        raise RuntimeError(f"FFmpeg episode decode failed:\n{result.stderr[-1500:]}")
    return proxy_path


def create_advantage_video(
    *,
    repo_id: str,
    predictions_path: Path,
    episode_index: int,
    camera_key: str | None,
    output_dir: Path,
    root: Path | None = None,
) -> Path:
    predictions = pd.read_csv(predictions_path)
    required = {
        "episode_index",
        "frame_index",
        "mc_return",
        "predicted_value",
        "advantage",
        "advantage_label",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {sorted(missing)}")
    episode = (
        predictions[predictions["episode_index"] == episode_index]
        .sort_values("frame_index")
        .reset_index(drop=True)
    )
    if episode.empty:
        raise ValueError(f"Episode {episode_index} is absent from {predictions_path}")

    local_root = root is not None
    root, metadata = _resolve_dataset(repo_id, root)
    available_cameras = list(metadata.video_keys)
    if not available_cameras:
        raise ValueError("Dataset has no video features")
    if camera_key is None:
        camera_key = available_cameras[0]
    if camera_key not in available_cameras:
        raise ValueError(f"Unknown camera {camera_key!r}; available cameras: {available_cameras}")

    relative_video_path = metadata.get_video_file_path(episode_index, camera_key)
    video_path = _resolve_video_path(repo_id, root, relative_video_path, local_root)
    episode_metadata = metadata.episodes[episode_index]
    start_timestamp = float(episode_metadata[f"videos/{camera_key}/from_timestamp"])
    end_timestamp = float(episode_metadata[f"videos/{camera_key}/to_timestamp"])
    fps = float(metadata.fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    camera_name = camera_key.replace(".", "_").replace("/", "_")
    output_path = output_dir / f"episode_{episode_index:06d}_{camera_name}_advantage.mp4"
    decode_proxy_path = output_path.with_name(output_path.stem + "_decode_proxy.mp4")
    _create_decode_proxy(
        video_path,
        decode_proxy_path,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        fps=fps,
        expected_frames=len(episode),
    )

    capture = cv2.VideoCapture(str(decode_proxy_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or frame_height <= 0:
        capture.release()
        decode_proxy_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not read video dimensions from {video_path}")
    temp_path = output_path.with_name(output_path.stem + "_temp.mp4")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        decode_proxy_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not open video writer for {temp_path}")

    expected_frames = len(episode)
    logger.info(
        "Rendering episode %d, camera=%s, frames=%d, source interval=%.3f–%.3fs",
        episode_index,
        camera_key,
        expected_frames,
        start_timestamp,
        end_timestamp,
    )
    written = 0
    labels = episode["advantage_label"].astype(str).to_numpy()
    try:
        for index in range(expected_frames):
            ok, frame = capture.read()
            if not ok:
                logger.warning("Video ended after %d/%d frames", index, expected_frames)
                break
            _draw_label_timeline(
                frame,
                labels,
                index,
            )
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
        decode_proxy_path.unlink(missing_ok=True)

    if written != expected_frames:
        raise RuntimeError(
            f"Rendered {written} frames but predictions contain {expected_frames}; "
            "video/prediction alignment is incomplete"
        )
    _encode_h264(temp_path, output_path)
    logger.info("Wrote %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--camera-key")
    parser.add_argument("--all-cameras", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("advantage_videos"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    predictions = pd.read_csv(args.predictions_path, usecols=["episode_index"])
    episode = args.episode
    if episode is None:
        episode = random.Random(args.seed).choice(sorted(predictions["episode_index"].unique()))
        logger.info("Randomly selected episode %d (seed=%d)", episode, args.seed)

    if args.all_cameras:
        _, metadata = _resolve_dataset(args.repo_id, args.root)
        camera_keys: list[str | None] = list(metadata.video_keys)
    else:
        camera_keys = [args.camera_key]
    for camera_key in camera_keys:
        create_advantage_video(
            repo_id=args.repo_id,
            predictions_path=args.predictions_path,
            episode_index=episode,
            camera_key=camera_key,
            output_dir=args.output_dir,
            root=args.root,
        )


if __name__ == "__main__":
    main()
