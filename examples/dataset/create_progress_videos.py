#!/usr/bin/env python

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

"""
Create MP4 (or GIF) videos with per-frame progress overlay for specified episodes.

Downloads datasets from HuggingFace, seeks directly into the episode segment
of the source video, draws a progress line on each frame, and writes the result.
The progress data is read from a parquet file that lives alongside the dataset
(configurable via ``--progress-file``).

Usage:
    python examples/dataset/create_progress_videos.py \
        --repo-id lerobot-data-collection/level2_final_quality3 \
        --episode 1100

    python examples/dataset/create_progress_videos.py \
        --repo-id lerobot-data-collection/level2_final_quality3 \
        --episode 1100 \
        --camera-key observation.images.top \
        --output-dir ./my_videos \
        --gif

    # Plot native RynnValue remaining-time predictions from a local parquet
    python examples/dataset/create_progress_videos.py \
        --repo-id lilkm/stackblocks_recap_all_for_vf_v2 \
        --episode 0 \
        --camera-key observation.images.top \
        --progress-path outputs/rynnvalue/stackblocks_top.parquet \
        --value-column remaining_time_s \
        --value-label "Remaining time (s)" \
        --output-dir outputs/rynnvalue/value_videos
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

GRAPH_Y_TOP_FRAC = 0.01
GRAPH_Y_BOT_FRAC = 0.99
LINE_THICKNESS = 3
SHADOW_THICKNESS = 6
REF_ALPHA = 0.45
FILL_ALPHA = 0.55
SCORE_FONT_SCALE = 0.8
TASK_FONT_SCALE = 0.55


def download_episode_metadata(
    repo_id: str,
    episode: int,
    progress_file: str = "sarm_progress.parquet",
    progress_path: Path | None = None,
) -> Path:
    """Download only the metadata and per-frame progress file for a dataset.

    Args:
        repo_id: HuggingFace dataset repository ID.
        episode: Episode index (used for logging only; all meta is fetched).
        progress_file: Filename of the per-frame progress parquet inside the dataset repo.
        progress_path: Optional local parquet path. When provided, only dataset metadata is downloaded.

    Returns:
        Local cache path for the downloaded snapshot.
    """
    source = progress_path if progress_path is not None else progress_file
    logging.info("[1/4] Downloading metadata for %s (episode %d); values=%s ...", repo_id, episode, source)
    allow_patterns = ["meta/**"]
    if progress_path is None:
        allow_patterns.append(progress_file)
    local_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            ignore_patterns=["*.mp4"],
        )
    )
    return local_path


def load_episode_meta(local_path: Path, episode: int, camera_key: str | None) -> dict:
    """Read info.json and episode parquet to resolve fps, video path, and timestamps.

    Args:
        local_path: Local cache directory containing meta/.
        episode: Episode index to look up.
        camera_key: Camera observation key (e.g. "observation.images.base").
            If None, the first available video key is used.

    Returns:
        Dict with keys: fps, camera, video_rel, chunk_index, file_index,
        from_ts, to_ts, task_name.
    """
    info = json.loads((local_path / "meta" / "info.json").read_text())
    fps = info["fps"]
    features = info["features"]

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if not video_keys:
        raise RuntimeError("No video keys found in dataset features")

    if camera_key is not None:
        if camera_key not in video_keys:
            raise RuntimeError(f"camera_key='{camera_key}' not found. Available: {video_keys}")
        selected_camera = camera_key
    else:
        selected_camera = video_keys[0]
    logging.info("   fps=%d  camera='%s'  all_cams=%s", fps, selected_camera, video_keys)

    episode_rows = []
    for parquet_file in sorted((local_path / "meta" / "episodes").glob("**/*.parquet")):
        episode_rows.append(pd.read_parquet(parquet_file))
    episode_df = pd.concat(episode_rows, ignore_index=True)
    row = episode_df[episode_df["episode_index"] == episode]
    if row.empty:
        raise RuntimeError(f"Episode {episode} not found in episode metadata")
    row = row.iloc[0]

    chunk_col = f"videos/{selected_camera}/chunk_index"
    file_col = f"videos/{selected_camera}/file_index"
    ts_from_col = f"videos/{selected_camera}/from_timestamp"
    ts_to_col = f"videos/{selected_camera}/to_timestamp"

    if chunk_col not in row.index:
        chunk_col = f"{selected_camera}/chunk_index"
        file_col = f"{selected_camera}/file_index"
        ts_from_col = f"{selected_camera}/from_timestamp"
        ts_to_col = f"{selected_camera}/to_timestamp"
    if chunk_col not in row.index:
        raise RuntimeError(
            f"Cannot find video metadata columns for {selected_camera}.\nAvailable: {list(row.index)}"
        )

    chunk_index = int(row[chunk_col])
    file_index = int(row[file_col])
    from_timestamp = float(row[ts_from_col])
    to_timestamp = float(row[ts_to_col])

    video_template = info.get(
        "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    video_rel = video_template.format(
        video_key=selected_camera,
        chunk_index=chunk_index,
        file_index=file_index,
    )

    task_name = _resolve_task_name(row, local_path)

    return {
        "fps": fps,
        "camera": selected_camera,
        "video_rel": video_rel,
        "chunk_index": chunk_index,
        "file_index": file_index,
        "from_ts": from_timestamp,
        "to_ts": to_timestamp,
        "task_name": task_name,
    }


def _resolve_task_name(row: pd.Series, local_path: Path) -> str:
    """Best-effort extraction of the task name for an episode row.

    Args:
        row: Single-episode row from the episodes parquet.
        local_path: Dataset cache root.

    Returns:
        Task name string, or empty string if unavailable.
    """
    try:
        if "tasks" in row.index and row["tasks"] is not None:
            tasks_val = row["tasks"]
            if isinstance(tasks_val, (list, tuple, np.ndarray)) and len(tasks_val) > 0:
                return str(tasks_val[0])
            return str(tasks_val).strip("[]'")

        tasks_parquet = local_path / "meta" / "tasks.parquet"
        if tasks_parquet.exists():
            tasks_df = pd.read_parquet(tasks_parquet)
            task_idx = int(row.get("task_index", 0)) if "task_index" in row.index else 0
            match = tasks_df[tasks_df["task_index"] == task_idx]
            if not match.empty:
                return str(match.index[0])
    except Exception as exc:
        logging.warning("Could not load task name: %s", exc)
    return ""


def download_video_file(repo_id: str, local_path: Path, video_rel: str) -> Path:
    """Download the specific video file if not already cached.

    Args:
        repo_id: HuggingFace dataset repository ID.
        local_path: Local cache directory.
        video_rel: Relative path to the video file within the dataset.

    Returns:
        Absolute path to the downloaded video file.
    """
    video_path = local_path / video_rel
    if video_path.exists():
        logging.info("   Video already cached: %s", video_path)
        return video_path
    logging.info("[2/4] Downloading video file %s ...", video_rel)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path),
        allow_patterns=[video_rel],
    )
    if not video_path.exists():
        raise RuntimeError(f"Video not found after download: {video_path}")
    return video_path


def load_progress_data(
    local_path: Path,
    episode: int,
    progress_file: str = "sarm_progress.parquet",
    progress_path: Path | None = None,
    value_column: str | None = None,
) -> np.ndarray | None:
    """Load per-frame scalar values for an episode.

    Args:
        local_path: Dataset cache root.
        episode: Episode index.
        progress_file: Filename of the per-frame progress parquet.
        progress_path: Optional local CSV or parquet path.
        value_column: Explicit scalar column to visualize. Progress columns are auto-detected when omitted.

    Returns:
        Sorted (N, 2) array of (frame_index, value), or None if unavailable.
    """
    value_path = progress_path if progress_path is not None else local_path / progress_file
    if not value_path.exists():
        logging.warning("%s not found", value_path)
        return None
    df = pd.read_csv(value_path) if value_path.suffix.lower() == ".csv" else pd.read_parquet(value_path)
    logging.info("   %s columns: %s", value_path, list(df.columns))
    episode_df = df[df["episode_index"] == episode].copy()
    if episode_df.empty:
        logging.warning("No value rows for episode %d in %s", episode, value_path)
        return None
    episode_df = episode_df.sort_values("frame_index")

    if value_column is not None:
        if value_column not in episode_df.columns:
            raise ValueError(
                f"Value column {value_column!r} not found in {value_path}. "
                f"Available columns: {list(episode_df.columns)}"
            )
        progress_column = value_column
    elif "progress_dense" in episode_df.columns and episode_df["progress_dense"].notna().any():
        progress_column = "progress_dense"
    elif "progress_sparse" in episode_df.columns:
        progress_column = "progress_sparse"
    else:
        progress_columns = [c for c in episode_df.columns if "progress" in c.lower()]
        if not progress_columns:
            logging.warning("No progress column found in %s; pass --value-column explicitly", value_path)
            return None
        progress_column = progress_columns[0]

    logging.info("   Using value column: '%s'", progress_column)
    return episode_df[["frame_index", progress_column]].rename(columns={progress_column: "progress"}).values


def _precompute_pixel_coords(
    progress_data: np.ndarray,
    num_frames: int,
    frame_width: int,
    frame_height: int,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    """Map scalar samples to pixel coordinates for overlay drawing.

    Args:
        progress_data: (N, 2) array of (frame_index, progress).
        num_frames: Total number of video frames.
        frame_width: Video width in pixels.
        frame_height: Video height in pixels.
        value_min: Value mapped to the bottom of the graph.
        value_max: Value mapped to the top of the graph.

    Returns:
        (N, 2) array of (x, y) pixel coordinates.
    """
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max <= value_min:
        raise ValueError(f"Expected a finite value range with max > min, got [{value_min}, {value_max}]")
    frame_indices = progress_data[:, 0].astype(float)
    values = progress_data[:, 1].astype(float)
    normalized_values = np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0)

    y_top = int(frame_height * GRAPH_Y_TOP_FRAC)
    y_bot = int(frame_height * GRAPH_Y_BOT_FRAC)
    graph_height = y_bot - y_top

    x_coords = (frame_indices / max(num_frames - 1, 1) * (frame_width - 1)).astype(int)
    y_coords = (y_bot - normalized_values * graph_height).astype(int)

    return np.stack([x_coords, y_coords], axis=1)


def _progress_color(normalized_position: float) -> tuple[int, int, int]:
    """Interpolate BGR color from red to green based on position in [0, 1].

    Args:
        normalized_position: Value in [0, 1] indicating how far along the episode.

    Returns:
        BGR color tuple.
    """
    red = int(255 * (1.0 - normalized_position))
    green = int(255 * normalized_position)
    return (0, green, red)


def _prerender_fill_polygon(
    pixel_coords: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Pre-render the grey fill polygon under the progress curve as a BGRA image.

    Args:
        pixel_coords: (N, 2) array of (x, y) pixel coordinates.
        frame_width: Video width in pixels.
        frame_height: Video height in pixels.

    Returns:
        BGRA image array of shape (frame_height, frame_width, 4).
    """
    y_bot = int(frame_height * GRAPH_Y_BOT_FRAC)
    fill_image = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
    polygon = np.concatenate(
        [
            pixel_coords,
            [[pixel_coords[-1][0], y_bot], [pixel_coords[0][0], y_bot]],
        ],
        axis=0,
    ).astype(np.int32)
    cv2.fillPoly(fill_image, [polygon], color=(128, 128, 128, int(255 * FILL_ALPHA)))
    return fill_image


def _alpha_composite_region(base: np.ndarray, overlay_bgra: np.ndarray, x_limit: int) -> None:
    """Blend BGRA overlay onto BGR base in-place, up to x_limit columns.

    Args:
        base: BGR frame to draw on (modified in-place).
        overlay_bgra: BGRA overlay image.
        x_limit: Only blend columns [0, x_limit).
    """
    if x_limit <= 0:
        return
    region_base = base[:, :x_limit]
    region_overlay = overlay_bgra[:, :x_limit]
    alpha = region_overlay[:, :, 3:4].astype(np.float32) / 255.0
    region_base[:] = np.clip(
        region_overlay[:, :, :3].astype(np.float32) * alpha + region_base.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def _draw_text_outlined(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    """Draw white text with a dark outline for readability on any background.

    Args:
        frame: BGR image to draw on (modified in-place).
        text: String to render.
        position: (x, y) bottom-left corner of the text.
        font_scale: OpenCV font scale.
        thickness: Text stroke thickness.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _iter_episode_frames_pyav(
    video_path: Path,
    from_timestamp: float,
    num_frames: int,
    fps: float,
):
    """Yield an episode's frames as BGR arrays using PyAV software decoding."""
    tolerance_s = 0.5 / fps
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(
            max(int((from_timestamp - tolerance_s) * av.time_base), 0),
            backward=True,
            any_frame=False,
        )

        yielded = 0
        for decoded_frame in container.decode(stream):
            if decoded_frame.pts is None:
                continue
            timestamp = float(decoded_frame.pts * decoded_frame.time_base)
            if timestamp + tolerance_s < from_timestamp:
                continue
            yield decoded_frame.to_ndarray(format="bgr24")
            yielded += 1
            if yielded >= num_frames:
                return


def composite_progress_video(
    video_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    progress_data: np.ndarray,
    output_path: Path,
    fps: float,
    task_name: str = "",
    value_label: str = "Progress",
    value_min: float | None = 0.0,
    value_max: float | None = 1.0,
) -> Path:
    """Decode episode frames with PyAV, draw a scalar overlay, and write an MP4.

    PyAV provides software AV1 decoding on platforms where OpenCV's bundled
    decoder cannot open LeRobot dataset videos.

    Args:
        video_path: Path to the full source video file.
        from_timestamp: Start timestamp of the episode in seconds.
        to_timestamp: End timestamp of the episode in seconds.
        progress_data: (N, 2) array of (frame_index, progress).
        output_path: Path to write the output MP4.
        fps: Frames per second for the output video.
        task_name: Optional task name to display at the top of the video.
        value_label: Label displayed beside the current scalar value.
        value_min: Graph minimum. Uses the episode minimum when None.
        value_max: Graph maximum. Uses the episode maximum when None.

    Returns:
        Path to the written output file (MP4).
    """
    duration_seconds = to_timestamp - from_timestamp
    num_frames = int(round(duration_seconds * fps))
    frame_iterator = _iter_episode_frames_pyav(video_path, from_timestamp, num_frames, fps)
    try:
        first_frame = next(frame_iterator)
    except StopIteration as e:
        raise RuntimeError(
            f"PyAV could not decode any frames from {video_path} at {from_timestamp:.3f}s"
        ) from e

    frame_height, frame_width = first_frame.shape[:2]
    logging.info(
        "   Video: %dx%d, %d frames @ %.1f fps (%.2fs), decoder=pyav",
        frame_width,
        frame_height,
        num_frames,
        fps,
        duration_seconds,
    )

    progress_values = progress_data[:, 1].astype(float)
    finite_values = progress_values[np.isfinite(progress_values)]
    if not finite_values.size:
        raise ValueError("The selected value column contains no finite values")
    resolved_min = float(finite_values.min()) if value_min is None else value_min
    resolved_max = float(finite_values.max()) if value_max is None else value_max
    if resolved_max <= resolved_min:
        padding = max(abs(resolved_min) * 0.05, 1e-6)
        resolved_min -= padding
        resolved_max += padding

    pixel_coords = _precompute_pixel_coords(
        progress_data,
        num_frames,
        frame_width,
        frame_height,
        value_min=resolved_min,
        value_max=resolved_max,
    )
    y_ref = int(frame_height * GRAPH_Y_TOP_FRAC)

    fill_image = _prerender_fill_polygon(pixel_coords, frame_width, frame_height)

    ref_line_image = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
    cv2.line(
        ref_line_image,
        (0, y_ref),
        (frame_width - 1, y_ref),
        (200, 200, 200, int(255 * REF_ALPHA)),
        1,
        cv2.LINE_AA,
    )

    frame_indices = progress_data[:, 0].astype(int)

    logging.info("[3/4] Compositing %d frames ...", num_frames)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create output video: {output_path}")

    written_frames = 0
    try:
        for frame_idx, frame in enumerate(itertools.chain([first_frame], frame_iterator)):
            drawn_count = int(np.searchsorted(frame_indices, frame_idx, side="right"))
            x_current = (
                int(pixel_coords[min(drawn_count, len(pixel_coords)) - 1][0]) + 1 if drawn_count > 0 else 0
            )

            _alpha_composite_region(frame, ref_line_image, frame_width)
            _alpha_composite_region(frame, fill_image, x_current)

            if drawn_count >= 2:
                time_position = (drawn_count - 1) / max(len(progress_values) - 1, 1)
                line_color = _progress_color(time_position)
                points = pixel_coords[:drawn_count].reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(
                    frame,
                    [points],
                    isClosed=False,
                    color=(255, 255, 255),
                    thickness=SHADOW_THICKNESS,
                    lineType=cv2.LINE_AA,
                )
                cv2.polylines(
                    frame,
                    [points],
                    isClosed=False,
                    color=line_color,
                    thickness=LINE_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

            if drawn_count > 0:
                score = float(progress_values[min(drawn_count, len(progress_values)) - 1])
                score_text = f"{value_label}: {score:.2f}" if value_label else f"{score:.2f}"
                (text_width, _), _ = cv2.getTextSize(
                    score_text, cv2.FONT_HERSHEY_SIMPLEX, SCORE_FONT_SCALE, 2
                )
                score_x = frame_width - text_width - 12
                score_y = frame_height - 12
                time_position = (drawn_count - 1) / max(len(progress_values) - 1, 1)
                score_color = _progress_color(time_position)
                cv2.putText(
                    frame,
                    score_text,
                    (score_x, score_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    SCORE_FONT_SCALE,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    score_text,
                    (score_x, score_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    SCORE_FONT_SCALE,
                    score_color,
                    2,
                    cv2.LINE_AA,
                )

            _draw_text_outlined(
                frame,
                f"{resolved_max:.2f}",
                (4, max(y_ref + 18, 18)),
                TASK_FONT_SCALE,
            )
            _draw_text_outlined(
                frame,
                f"{resolved_min:.2f}",
                (4, int(frame_height * GRAPH_Y_BOT_FRAC) - 4),
                TASK_FONT_SCALE,
            )

            if task_name:
                (text_width, _), _ = cv2.getTextSize(task_name, cv2.FONT_HERSHEY_SIMPLEX, TASK_FONT_SCALE, 1)
                task_x = max((frame_width - text_width) // 2, 4)
                _draw_text_outlined(frame, task_name, (task_x, 22), TASK_FONT_SCALE)

            writer.write(frame)
            written_frames += 1
            if frame_idx % 100 == 0:
                logging.info("   Frame %d/%d ...", frame_idx, num_frames)
    finally:
        writer.release()

    if written_frames != num_frames:
        raise RuntimeError(
            f"Decoded {written_frames} of {num_frames} expected episode frames from {video_path}"
        )

    logging.info("   MP4 written: %s", output_path)
    return output_path


def convert_mp4_to_gif(mp4_path: Path) -> Path:
    """Convert an MP4 to an optimized GIF using ffmpeg palette generation.

    Args:
        mp4_path: Path to the source MP4 file.

    Returns:
        Path to the generated GIF file.
    """
    capture = cv2.VideoCapture(str(mp4_path))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()

    gif_path = mp4_path.with_suffix(".gif")
    palette_path = mp4_path.parent / "_palette.png"

    logging.info("[4/4] Converting to GIF ...")
    result_palette = subprocess.run(  # nosec B607
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            f"fps=10,scale={frame_width}:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff",
            "-update",
            "1",
            str(palette_path),
        ],
        capture_output=True,
        text=True,
    )
    if result_palette.returncode != 0:
        logging.warning("palettegen failed:\n%s", result_palette.stderr[-500:])

    result_gif = subprocess.run(  # nosec B607
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-i",
            str(palette_path),
            "-filter_complex",
            f"fps=10,scale={frame_width}:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3",
            str(gif_path),
        ],
        capture_output=True,
        text=True,
    )
    if result_gif.returncode != 0:
        logging.warning("GIF encode failed:\n%s", result_gif.stderr[-500:])

    palette_path.unlink(missing_ok=True)
    logging.info("   GIF written: %s", gif_path)
    return gif_path


def process_dataset(
    repo_id: str,
    episode: int,
    camera_key: str | None,
    output_dir: Path,
    create_gif: bool = False,
    progress_file: str = "sarm_progress.parquet",
    progress_path: Path | None = None,
    value_column: str | None = None,
    value_label: str | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
) -> Path | None:
    """Full pipeline: download, extract metadata, composite progress, write output.

    Args:
        repo_id: HuggingFace dataset repository ID.
        episode: Episode index.
        camera_key: Camera key to use, or None for auto-selection.
        output_dir: Directory to write output files.
        create_gif: If True, also generate a GIF from the MP4.
        progress_file: Filename of the per-frame progress parquet inside the
            dataset repo.
        progress_path: Optional local parquet path.
        value_column: Explicit scalar column to visualize.
        value_label: Label displayed beside the current value.
        value_min: Graph minimum, or None for automatic scaling.
        value_max: Graph maximum, or None for automatic scaling.

    Returns:
        Path to the final output file, or None on failure.
    """
    safe_name = repo_id.replace("/", "_")
    logging.info("Processing: %s  |  episode %d", repo_id, episode)

    local_path = download_episode_metadata(repo_id, episode, progress_file, progress_path)
    logging.info("   Local cache: %s", local_path)

    episode_meta = load_episode_meta(local_path, episode, camera_key)
    logging.info("   Episode meta: %s", episode_meta)

    video_path = download_video_file(repo_id, local_path, episode_meta["video_rel"])

    progress_data = load_progress_data(
        local_path,
        episode,
        progress_file,
        progress_path=progress_path,
        value_column=value_column,
    )
    if progress_data is None:
        logging.error("Could not load progress data from %s. Skipping overlay.", progress_file)
        return None

    logging.info("   Progress frames: %d", len(progress_data))

    output_label = value_column or "progress"
    safe_label = "".join(character if character.isalnum() else "_" for character in output_label).strip("_")
    camera_suffix = episode_meta["camera"].removeprefix("observation.images.").replace(".", "_")
    output_path = output_dir / f"{safe_name}_ep{episode}_{camera_suffix}_{safe_label}.mp4"
    if value_column is None:
        value_min = 0.0 if value_min is None else value_min
        value_max = 1.0 if value_max is None else value_max
    final_path = composite_progress_video(
        video_path=video_path,
        from_timestamp=episode_meta["from_ts"],
        to_timestamp=episode_meta["to_ts"],
        progress_data=progress_data,
        output_path=output_path,
        fps=episode_meta["fps"],
        task_name=episode_meta.get("task_name", ""),
        value_label=value_label or output_label,
        value_min=value_min,
        value_max=value_max,
    )

    if create_gif:
        final_path = convert_mp4_to_gif(final_path)

    logging.info("Done: %s", final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create MP4/GIF videos with per-frame progress overlay for dataset episodes."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID (e.g. 'lerobot-data-collection/level2_final_quality3').",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index to visualize.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=None,
        help="Camera observation key (e.g. 'observation.images.base'). Auto-selects first camera if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("progress_videos"),
        help="Directory to write output files (default: ./progress_videos).",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also generate a GIF from the MP4 output.",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default="sarm_progress.parquet",
        help=(
            "Filename of the per-frame progress parquet inside the dataset repo "
            "(default: 'sarm_progress.parquet')."
        ),
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        help="Local parquet path. Overrides --progress-file and does not upload or modify the dataset.",
    )
    parser.add_argument(
        "--value-column",
        help="Scalar parquet column to plot, for example remaining_time_s or potential.",
    )
    parser.add_argument(
        "--value-label",
        help="Display label for --value-column (default: the column name).",
    )
    parser.add_argument("--value-min", type=float, help="Graph minimum (default: episode minimum).")
    parser.add_argument("--value-max", type=float, help="Graph maximum (default: episode maximum).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = process_dataset(
        repo_id=args.repo_id,
        episode=args.episode,
        camera_key=args.camera_key,
        output_dir=args.output_dir,
        create_gif=args.gif,
        progress_file=args.progress_file,
        progress_path=args.progress_path,
        value_column=args.value_column,
        value_label=args.value_label,
        value_min=args.value_min,
        value_max=args.value_max,
    )

    if result:
        logging.info("Output: %s", result)


if __name__ == "__main__":
    main()
