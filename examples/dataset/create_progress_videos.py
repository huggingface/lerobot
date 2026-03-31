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
Create MP4 (or GIF) videos with sarm_progress overlay for specified episodes.

Downloads datasets from HuggingFace, seeks directly into the episode segment
of the source video, draws a progress line on each frame, and writes the result.

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
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

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


def download_episode_metadata(repo_id: str, episode: int) -> Path:
    """Download only the metadata and sarm_progress files for a dataset.

    Args:
        repo_id: HuggingFace dataset repository ID.
        episode: Episode index (used for logging only; all meta is fetched).

    Returns:
        Local cache path for the downloaded snapshot.
    """
    logging.info("[1/4] Downloading metadata for %s (episode %d) ...", repo_id, episode)
    local_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["meta/**", "sarm_progress.parquet"],
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


def load_progress_data(local_path: Path, episode: int) -> np.ndarray | None:
    """Load sarm_progress values for an episode.

    Args:
        local_path: Dataset cache root.
        episode: Episode index.

    Returns:
        Sorted (N, 2) array of (frame_index, progress), or None if unavailable.
    """
    parquet_path = local_path / "sarm_progress.parquet"
    if not parquet_path.exists():
        logging.warning("sarm_progress.parquet not found")
        return None
    df = pd.read_parquet(parquet_path)
    logging.info("   sarm_progress.parquet columns: %s", list(df.columns))
    episode_df = df[df["episode_index"] == episode].copy()
    if episode_df.empty:
        logging.warning("No sarm_progress rows for episode %d", episode)
        return None
    episode_df = episode_df.sort_values("frame_index")

    if "progress_dense" in episode_df.columns and episode_df["progress_dense"].notna().any():
        progress_column = "progress_dense"
    elif "progress_sparse" in episode_df.columns:
        progress_column = "progress_sparse"
    else:
        progress_columns = [c for c in episode_df.columns if "progress" in c.lower()]
        if not progress_columns:
            return None
        progress_column = progress_columns[0]

    logging.info("   Using progress column: '%s'", progress_column)
    return episode_df[["frame_index", progress_column]].rename(columns={progress_column: "progress"}).values


def _precompute_pixel_coords(
    progress_data: np.ndarray,
    num_frames: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Map progress samples to pixel coordinates for overlay drawing.

    Args:
        progress_data: (N, 2) array of (frame_index, progress).
        num_frames: Total number of video frames.
        frame_width: Video width in pixels.
        frame_height: Video height in pixels.

    Returns:
        (N, 2) array of (x, y) pixel coordinates.
    """
    frame_indices = progress_data[:, 0].astype(float)
    progress_values = np.clip(progress_data[:, 1].astype(float), 0.0, 1.0)

    y_top = int(frame_height * GRAPH_Y_TOP_FRAC)
    y_bot = int(frame_height * GRAPH_Y_BOT_FRAC)
    graph_height = y_bot - y_top

    x_coords = (frame_indices / (num_frames - 1) * (frame_width - 1)).astype(int)
    y_coords = (y_bot - progress_values * graph_height).astype(int)

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


def composite_progress_video(
    video_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    progress_data: np.ndarray,
    output_path: Path,
    fps: float,
    task_name: str = "",
) -> Path:
    """Read episode frames by seeking into the source video, draw progress overlay, write output.

    Uses cv2.CAP_PROP_POS_MSEC to seek directly into the source video,
    eliminating the need for an intermediate clip file.

    Args:
        video_path: Path to the full source video file.
        from_timestamp: Start timestamp of the episode in seconds.
        to_timestamp: End timestamp of the episode in seconds.
        progress_data: (N, 2) array of (frame_index, progress).
        output_path: Path to write the output MP4.
        fps: Frames per second for the output video.
        task_name: Optional task name to display at the top of the video.

    Returns:
        Path to the written output file (MP4).
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, from_timestamp * 1000)

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = to_timestamp - from_timestamp
        num_frames = int(round(duration_seconds * fps))

        logging.info(
            "   Video: %dx%d, %d frames @ %.1f fps (%.2fs)",
            frame_width,
            frame_height,
            num_frames,
            fps,
            duration_seconds,
        )

        pixel_coords = _precompute_pixel_coords(progress_data, num_frames, frame_width, frame_height)
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
        progress_values = progress_data[:, 1].astype(float)

        logging.info("[3/4] Compositing %d frames ...", num_frames)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

        for frame_idx in range(num_frames):
            ret, frame = capture.read()
            if not ret:
                break

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
                score_text = f"{score:.2f}"
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

            if task_name:
                (text_width, _), _ = cv2.getTextSize(task_name, cv2.FONT_HERSHEY_SIMPLEX, TASK_FONT_SCALE, 1)
                task_x = max((frame_width - text_width) // 2, 4)
                _draw_text_outlined(frame, task_name, (task_x, 22), TASK_FONT_SCALE)

            writer.write(frame)
            if frame_idx % 100 == 0:
                logging.info("   Frame %d/%d ...", frame_idx, num_frames)

        writer.release()
    finally:
        capture.release()

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
) -> Path | None:
    """Full pipeline: download, extract metadata, composite progress, write output.

    Args:
        repo_id: HuggingFace dataset repository ID.
        episode: Episode index.
        camera_key: Camera key to use, or None for auto-selection.
        output_dir: Directory to write output files.
        create_gif: If True, also generate a GIF from the MP4.

    Returns:
        Path to the final output file, or None on failure.
    """
    safe_name = repo_id.replace("/", "_")
    logging.info("Processing: %s  |  episode %d", repo_id, episode)

    local_path = download_episode_metadata(repo_id, episode)
    logging.info("   Local cache: %s", local_path)

    episode_meta = load_episode_meta(local_path, episode, camera_key)
    logging.info("   Episode meta: %s", episode_meta)

    video_path = download_video_file(repo_id, local_path, episode_meta["video_rel"])

    progress_data = load_progress_data(local_path, episode)
    if progress_data is None:
        logging.error("Could not load sarm_progress data. Skipping overlay.")
        return None

    logging.info("   Progress frames: %d", len(progress_data))

    output_path = output_dir / f"{safe_name}_ep{episode}_progress.mp4"
    final_path = composite_progress_video(
        video_path=video_path,
        from_timestamp=episode_meta["from_ts"],
        to_timestamp=episode_meta["to_ts"],
        progress_data=progress_data,
        output_path=output_path,
        fps=episode_meta["fps"],
        task_name=episode_meta.get("task_name", ""),
    )

    if create_gif:
        final_path = convert_mp4_to_gif(final_path)

    logging.info("Done: %s", final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create MP4/GIF videos with sarm_progress overlay for dataset episodes."
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = process_dataset(
        repo_id=args.repo_id,
        episode=args.episode,
        camera_key=args.camera_key,
        output_dir=args.output_dir,
        create_gif=args.gif,
    )

    if result:
        logging.info("Output: %s", result)


if __name__ == "__main__":
    main()
