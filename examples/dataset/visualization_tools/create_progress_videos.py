"""
Create MP4 videos with sarm_progress overlay for specified episodes.
Downloads datasets from HuggingFace, extracts episode video + progress data,
and draws the progress line directly on each frame (no panel, no axes).
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download


# ─────────────────────── Config ───────────────────────
DATASETS = [
    {"repo_id": "lerobot-data-collection/level2_final_quality3", "episode": 1100},
]
CAMERA_KEY = "observation.images.base"  # None = auto-select first camera, or set e.g. "observation.images.top"
OUTPUT_DIR = Path("/Users/pepijnkooijmans/Documents/GitHub_local/progress_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Progress line spans the full video height
GRAPH_Y_TOP_FRAC  = 0.01
GRAPH_Y_BOT_FRAC  = 0.99
LINE_THICKNESS    = 3
SHADOW_THICKNESS  = 6                # white edge thickness
REF_ALPHA         = 0.45             # opacity of the 1.0 reference line
FILL_ALPHA        = 0.55             # opacity of the grey fill under the line
SCORE_FONT_SCALE  = 0.8
TASK_FONT_SCALE   = 0.55


# ─────────────────────── Helpers ──────────────────────

def download_episode(repo_id: str, episode: int) -> Path:
    """Download only the files needed for this episode."""
    safe_ep = f"{episode:06d}"
    # We need: meta/, sarm_progress.parquet, and the relevant video/data chunks.
    # We'll download meta + sarm first, then figure out chunks.
    print(f"\n[1/5] Downloading metadata for {repo_id} …")
    local = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["meta/**", "sarm_progress.parquet"],
            ignore_patterns=["*.mp4"],
        )
    )
    return local


def load_episode_meta(local: Path, episode: int) -> dict:
    """Read info.json + episode-level parquet to get fps, video paths, timestamps."""
    info = json.loads((local / "meta" / "info.json").read_text())
    fps = info["fps"]
    features = info["features"]

    # Find video keys (keys whose dtype=="video")
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if not video_keys:
        raise RuntimeError("No video keys found in dataset features")
    if CAMERA_KEY is not None:
        if CAMERA_KEY not in video_keys:
            raise RuntimeError(f"CAMERA_KEY='{CAMERA_KEY}' not found. Available: {video_keys}")
        first_cam = CAMERA_KEY
    else:
        first_cam = video_keys[0]
    print(f"   fps={fps}  camera='{first_cam}'  all_cams={video_keys}")

    # Load all episode-meta parquet files and find our episode
    ep_rows = []
    for pq in sorted((local / "meta" / "episodes").glob("**/*.parquet")):
        df = pd.read_parquet(pq)
        ep_rows.append(df)
    ep_df = pd.concat(ep_rows, ignore_index=True)
    row = ep_df[ep_df["episode_index"] == episode]
    if row.empty:
        raise RuntimeError(f"Episode {episode} not found in episode metadata")
    row = row.iloc[0]

    # Extract video chunk/file index for first camera
    cam_key = first_cam.replace(".", "/")  # some datasets store as nested key
    # Try both dot and slash variants of the key
    chunk_col = f"videos/{first_cam}/chunk_index"
    file_col  = f"videos/{first_cam}/file_index"
    ts_col    = f"videos/{first_cam}/from_timestamp"
    to_col    = f"videos/{first_cam}/to_timestamp"

    # Some datasets use different column naming
    if chunk_col not in row.index:
        # Try without the 'videos/' prefix
        chunk_col = f"{first_cam}/chunk_index"
        file_col  = f"{first_cam}/file_index"
        ts_col    = f"{first_cam}/from_timestamp"
        to_col    = f"{first_cam}/to_timestamp"
    if chunk_col not in row.index:
        raise RuntimeError(f"Cannot find video metadata columns for {first_cam}.\nAvailable: {list(row.index)}")

    chunk_idx = int(row[chunk_col])
    file_idx  = int(row[file_col])
    from_ts   = float(row[ts_col])
    to_ts     = float(row[to_col])

    video_template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    video_rel = video_template.format(
        video_key=first_cam,
        chunk_index=chunk_idx,
        file_index=file_idx,
    )

    # Load task name for this episode
    # tasks.parquet uses the task string as the row index; task_index column holds the int id
    task_name = ""
    try:
        # Prefer the 'tasks' list directly on the episode row
        if "tasks" in row.index and row["tasks"] is not None:
            tasks_val = row["tasks"]
            if isinstance(tasks_val, (list, tuple, np.ndarray)) and len(tasks_val) > 0:
                task_name = str(tasks_val[0])
            else:
                task_name = str(tasks_val).strip("[]'")
        else:
            tasks_pq = local / "meta" / "tasks.parquet"
            if tasks_pq.exists():
                tasks_df = pd.read_parquet(tasks_pq)
                # Row index is the task string; task_index column is the int
                task_idx = int(row.get("task_index", 0)) if "task_index" in row.index else 0
                match = tasks_df[tasks_df["task_index"] == task_idx]
                if not match.empty:
                    task_name = str(match.index[0])
        print(f"   Task name: '{task_name}'")
    except Exception as e:
        print(f"   WARNING: could not load task name: {e}")

    return {
        "fps": fps,
        "first_cam": first_cam,
        "video_rel": video_rel,
        "chunk_index": chunk_idx,
        "file_index": file_idx,
        "from_ts": from_ts,
        "to_ts": to_ts,
        "task_name": task_name,
    }


def download_video(repo_id: str, local: Path, video_rel: str) -> Path:
    """Download the specific video file if not already present."""
    video_path = local / video_rel
    if video_path.exists():
        print(f"   Video already cached: {video_path}")
        return video_path
    print(f"[2/5] Downloading video file {video_rel} …")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local),
        allow_patterns=[video_rel],
    )
    if not video_path.exists():
        raise RuntimeError(f"Video not found after download: {video_path}")
    return video_path


def load_progress(local: Path, episode: int) -> np.ndarray | None:
    """Load sarm_progress values for this episode. Returns sorted array of (frame_index, progress)."""
    pq_path = local / "sarm_progress.parquet"
    if not pq_path.exists():
        print("   WARNING: sarm_progress.parquet not found, trying data parquet …")
        return None
    df = pd.read_parquet(pq_path)
    print(f"   sarm_progress.parquet columns: {list(df.columns)}")
    ep_df = df[df["episode_index"] == episode].copy()
    if ep_df.empty:
        print(f"   WARNING: No sarm_progress rows for episode {episode}")
        return None
    ep_df = ep_df.sort_values("frame_index")

    # Prefer dense, fall back to sparse
    if "progress_dense" in ep_df.columns and ep_df["progress_dense"].notna().any():
        prog_col = "progress_dense"
    elif "progress_sparse" in ep_df.columns:
        prog_col = "progress_sparse"
    else:
        # Last resort: any column with 'progress' in the name
        prog_cols = [c for c in ep_df.columns if "progress" in c.lower()]
        if not prog_cols:
            return None
        prog_col = prog_cols[0]

    print(f"   Using progress column: '{prog_col}'")
    return ep_df[["frame_index", prog_col]].rename(columns={prog_col: "progress"}).values


def extract_episode_clip(video_path: Path, from_ts: float, to_ts: float, out_path: Path) -> Path:
    """Use ffmpeg to cut the episode segment from the combined video file."""
    duration = to_ts - from_ts
    print(f"[3/5] Extracting clip [{from_ts:.3f}s → {to_ts:.3f}s] ({duration:.2f}s) …")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(from_ts),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-an",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip extraction failed:\n{result.stderr}")
    return out_path


def precompute_pixels(
    progress_data: np.ndarray,
    n_frames: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Map each progress sample to pixel coordinates.
    Returns array of shape (N, 2) with (x, y) in pixel space.
    x spans full video width; y maps progress [0,1] to graph band.
    """
    frame_indices = progress_data[:, 0].astype(float)
    progress_vals = np.clip(progress_data[:, 1].astype(float), 0.0, 1.0)
    n = len(frame_indices)

    y_top = int(frame_h * GRAPH_Y_TOP_FRAC)
    y_bot = int(frame_h * GRAPH_Y_BOT_FRAC)
    graph_h = y_bot - y_top

    xs = (frame_indices / (n_frames - 1) * (frame_w - 1)).astype(int)
    # progress=1 → y_top, progress=0 → y_bot
    ys = (y_bot - progress_vals * graph_h).astype(int)

    return np.stack([xs, ys], axis=1)  # (N, 2)


def progress_color(t: float) -> tuple[int, int, int]:
    """Interpolate BGR color red→green based on normalised position t in [0,1]."""
    r = int(255 * (1.0 - t))
    g = int(255 * t)
    return (0, g, r)  # BGR


def prerender_fill(
    pixels: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Pre-render the full grey fill polygon under the curve as a BGRA image."""
    y_bot = int(frame_h * GRAPH_Y_BOT_FRAC)
    fill_img = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    poly = np.concatenate([
        pixels,
        [[pixels[-1][0], y_bot], [pixels[0][0], y_bot]],
    ], axis=0).astype(np.int32)
    cv2.fillPoly(fill_img, [poly], color=(128, 128, 128, int(255 * FILL_ALPHA)))
    return fill_img


def alpha_composite(base: np.ndarray, overlay_bgra: np.ndarray, x_max: int) -> None:
    """Blend overlay onto base in-place, but only for x < x_max."""
    if x_max <= 0:
        return
    roi_b = base[:, :x_max]
    roi_o = overlay_bgra[:, :x_max]
    alpha = roi_o[:, :, 3:4].astype(np.float32) / 255.0
    roi_b[:] = np.clip(
        roi_o[:, :, :3].astype(np.float32) * alpha
        + roi_b.astype(np.float32) * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def draw_text_outlined(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    """Draw text with a dark outline for readability on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, pos, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def composite_video(
    clip_path: Path,
    progress_data: np.ndarray,
    out_path: Path,
    fps: float,
    frame_h: int,
    frame_w: int,
    task_name: str = "",
) -> Path:
    """Read clip frames, draw gradient progress line with fill + labels, export as GIF."""
    n_total = int(cv2.VideoCapture(str(clip_path)).get(cv2.CAP_PROP_FRAME_COUNT))
    pixels = precompute_pixels(progress_data, n_total, frame_w, frame_h)

    y_top = int(frame_h * GRAPH_Y_TOP_FRAC)
    y_bot = int(frame_h * GRAPH_Y_BOT_FRAC)
    y_ref = y_top

    # Pre-render fill polygon (line is drawn per-frame with live color)
    fill_img = prerender_fill(pixels, frame_w, frame_h)

    # 1.0 reference line overlay (full width, drawn once)
    ref_img = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    cv2.line(ref_img, (0, y_ref), (frame_w - 1, y_ref),
             (200, 200, 200, int(255 * REF_ALPHA)), 1, cv2.LINE_AA)

    frame_indices = progress_data[:, 0].astype(int)
    progress_vals = progress_data[:, 1].astype(float)

    print(f"[4/4] Compositing {n_total} frames …")
    cap = cv2.VideoCapture(str(clip_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = out_path.parent / (out_path.stem + "_tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (frame_w, frame_h))

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        n_drawn = int(np.searchsorted(frame_indices, fi, side="right"))
        x_cur = int(pixels[min(n_drawn, len(pixels)) - 1][0]) + 1 if n_drawn > 0 else 0

        # 1. reference line (full width, always)
        alpha_composite(frame, ref_img, frame_w)

        # 2. grey fill under curve up to current x
        alpha_composite(frame, fill_img, x_cur)

        # 3. progress line — single color that transitions red→green over time
        if n_drawn >= 2:
            t_cur = (n_drawn - 1) / max(len(progress_vals) - 1, 1)
            line_col = progress_color(t_cur)
            pts = pixels[:n_drawn].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=False,
                          color=(255, 255, 255), thickness=SHADOW_THICKNESS,
                          lineType=cv2.LINE_AA)
            cv2.polylines(frame, [pts], isClosed=False,
                          color=line_col, thickness=LINE_THICKNESS,
                          lineType=cv2.LINE_AA)

        # 4. score — bottom right
        if n_drawn > 0:
            score = float(progress_vals[min(n_drawn, len(progress_vals)) - 1])
            score_text = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          SCORE_FONT_SCALE, 2)
            sx = frame_w - tw - 12
            sy = frame_h - 12
            # coloured score matching current gradient position
            t_cur = (n_drawn - 1) / max(len(progress_vals) - 1, 1)
            score_col = progress_color(t_cur)
            cv2.putText(frame, score_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                        SCORE_FONT_SCALE, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, score_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                        SCORE_FONT_SCALE, score_col, 2, cv2.LINE_AA)

        # 5. task name — top centre
        if task_name:
            (tw, _), _ = cv2.getTextSize(task_name, cv2.FONT_HERSHEY_SIMPLEX,
                                         TASK_FONT_SCALE, 1)
            tx = max((frame_w - tw) // 2, 4)
            draw_text_outlined(frame, task_name, (tx, 22), TASK_FONT_SCALE)

        writer.write(frame)
        fi += 1
        if fi % 100 == 0:
            print(f"   Frame {fi}/{n_total} …", end="\r")

    cap.release()
    writer.release()
    print()

    # Convert to GIF: full resolution, 12fps, 128-color diff palette (<40MB)
    gif_path = out_path.with_suffix(".gif")
    palette = out_path.parent / "_palette.png"
    r1 = subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_path),
        "-vf", f"fps=10,scale={frame_w}:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff",
        "-update", "1",
        str(palette),
    ], capture_output=True, text=True)
    if r1.returncode != 0:
        print(f"   WARNING: palettegen failed:\n{r1.stderr[-500:]}")
    r2 = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(tmp_path), "-i", str(palette),
        "-filter_complex",
        f"fps=10,scale={frame_w}:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3",
        str(gif_path),
    ], capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"   WARNING: gif encode failed:\n{r2.stderr[-500:]}")
    tmp_path.unlink(missing_ok=True)
    palette.unlink(missing_ok=True)
    return gif_path


# ─────────────────────── Main ──────────────────────────

def process_dataset(repo_id: str, episode: int):
    safe_name = repo_id.replace("/", "_")
    print(f"\n{'='*60}")
    print(f"Processing: {repo_id}  |  episode {episode}")
    print(f"{'='*60}")

    # 1. Download metadata
    local = download_episode(repo_id, episode)
    print(f"   Local cache: {local}")

    # 2. Read episode metadata
    ep_meta = load_episode_meta(local, episode)
    print(f"   Episode meta: {ep_meta}")

    # 3. Download video file
    video_path = download_video(repo_id, local, ep_meta["video_rel"])

    # 4. Extract clip
    clip_path = OUTPUT_DIR / f"{safe_name}_ep{episode}_clip.mp4"
    extract_episode_clip(video_path, ep_meta["from_ts"], ep_meta["to_ts"], clip_path)

    # 5. Load progress data
    progress_data = load_progress(local, episode)
    if progress_data is None:
        print("   ERROR: Could not load sarm_progress data. Skipping overlay.")
        return

    n_progress = len(progress_data)
    print(f"   Progress frames: {n_progress}")

    # 6. Get clip dimensions
    cap = cv2.VideoCapture(str(clip_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or ep_meta["fps"]
    cap.release()
    print(f"   Clip: {frame_w}×{frame_h}  {n_frames} frames @ {actual_fps:.1f}fps")

    # 7. Composite (draw line directly on frames)
    out_path = OUTPUT_DIR / f"{safe_name}_ep{episode}_progress.mp4"
    final = composite_video(clip_path, progress_data, out_path, actual_fps, frame_h, frame_w,
                            task_name=ep_meta.get("task_name", ""))
    clip_path.unlink(missing_ok=True)
    print(f"\n✓ Done: {final}")
    return final


if __name__ == "__main__":
    results = []
    for cfg in DATASETS:
        try:
            out = process_dataset(cfg["repo_id"], cfg["episode"])
            if out:
                results.append(out)
        except Exception as e:
            print(f"\nERROR processing {cfg['repo_id']}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("Output files:")
    for r in results:
        print(f"  {r}")
