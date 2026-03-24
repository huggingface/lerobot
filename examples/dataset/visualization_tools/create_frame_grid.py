"""
Create a JPG grid of random frames sampled from a LeRobot video dataset.
Downloads metadata + video chunks from HuggingFace, picks random frames,
decodes them, and tiles into a single image.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "lerobot-data-collection/level2_final_quality3"
CAMERA_KEY = "observation.images.base"
GRID_COLS = 15
GRID_ROWS = 10
THUMB_WIDTH = 160
OUTPUT_DIR = Path("/Users/pepijnkooijmans/Documents/GitHub_local/progress_videos")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 1


def download_metadata(repo_id: str) -> Path:
    """Download only metadata (no videos yet)."""
    print(f"[1/3] Downloading metadata for {repo_id} …")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["meta/**"],
            ignore_patterns=["*.mp4"],
        )
    )


def load_video_info(local: Path) -> tuple[str, list[dict], int]:
    """Parse info.json and episode parquets. Returns (camera_key, episode_rows, fps)."""
    info = json.loads((local / "meta" / "info.json").read_text())
    fps = info["fps"]
    features = info["features"]

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if not video_keys:
        raise RuntimeError("No video keys found in dataset features")

    if CAMERA_KEY is not None:
        if CAMERA_KEY not in video_keys:
            raise RuntimeError(f"CAMERA_KEY='{CAMERA_KEY}' not found. Available: {video_keys}")
        cam = CAMERA_KEY
    else:
        cam = video_keys[0]
    print(f"   camera='{cam}'  all_cams={video_keys}  fps={fps}")

    ep_rows = []
    for pq in sorted((local / "meta" / "episodes").glob("**/*.parquet")):
        ep_rows.append(pd.read_parquet(pq))
    ep_df = pd.concat(ep_rows, ignore_index=True)

    video_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    chunk_col = f"videos/{cam}/chunk_index"
    file_col = f"videos/{cam}/file_index"
    ts_from = f"videos/{cam}/from_timestamp"
    ts_to = f"videos/{cam}/to_timestamp"
    if chunk_col not in ep_df.columns:
        chunk_col = f"{cam}/chunk_index"
        file_col = f"{cam}/file_index"
        ts_from = f"{cam}/from_timestamp"
        ts_to = f"{cam}/to_timestamp"

    episodes = []
    for _, row in ep_df.iterrows():
        ci = int(row[chunk_col])
        fi = int(row[file_col])
        episodes.append(
            {
                "episode_index": int(row["episode_index"]),
                "chunk_index": ci,
                "file_index": fi,
                "from_ts": float(row[ts_from]),
                "to_ts": float(row[ts_to]),
                "video_rel": video_template.format(video_key=cam, chunk_index=ci, file_index=fi),
            }
        )
    return cam, episodes, fps


def pick_random_frames(episodes: list[dict], fps: int, n: int, rng: random.Random) -> list[dict]:
    """Pick n random (episode, timestamp) pairs, return sorted by video file for efficient access."""
    picks = []
    for _ in range(n):
        ep = rng.choice(episodes)
        duration = ep["to_ts"] - ep["from_ts"]
        if duration <= 0:
            continue
        t = ep["from_ts"] + rng.random() * duration
        picks.append({**ep, "seek_ts": t})
    picks.sort(key=lambda p: (p["video_rel"], p["seek_ts"]))
    return picks


def download_video_files(repo_id: str, local: Path, picks: list[dict]) -> None:
    """Download only the video files we need."""
    needed = sorted({p["video_rel"] for p in picks})
    print(f"[2/3] Downloading {len(needed)} video file(s) …")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local),
        allow_patterns=needed,
    )


def extract_frame(video_path: Path, seek_ts: float) -> np.ndarray | None:
    """Decode a single frame at the given timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ts * 1000.0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def build_grid(frames: list[np.ndarray], cols: int, thumb_w: int) -> np.ndarray:
    """Resize frames to uniform thumbnails and tile into a grid."""
    if not frames:
        raise RuntimeError("No frames decoded")

    h0, w0 = frames[0].shape[:2]
    thumb_h = int(thumb_w * h0 / w0)

    thumbs = [cv2.resize(f, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for f in frames]

    rows = []
    for i in range(0, len(thumbs), cols):
        row_thumbs = thumbs[i : i + cols]
        while len(row_thumbs) < cols:
            row_thumbs.append(np.zeros_like(row_thumbs[0]))
        rows.append(np.hstack(row_thumbs))
    return np.vstack(rows)


def main() -> None:
    rng = random.Random(SEED)
    n_frames = GRID_COLS * GRID_ROWS

    local = download_metadata(REPO_ID)
    cam, episodes, fps = load_video_info(local)
    picks = pick_random_frames(episodes, fps, n_frames, rng)
    download_video_files(REPO_ID, local, picks)

    print(f"[3/3] Decoding {n_frames} frames …")
    frames: list[np.ndarray] = []
    for p in picks:
        vp = local / p["video_rel"]
        if not vp.exists():
            print(f"   SKIP: {p['video_rel']} not found")
            continue
        frame = extract_frame(vp, p["seek_ts"])
        if frame is not None:
            frames.append(frame)

    print(f"   Decoded {len(frames)}/{n_frames} frames")
    grid = build_grid(frames, GRID_COLS, THUMB_WIDTH)

    safe_name = REPO_ID.replace("/", "_")
    out_path = OUTPUT_DIR / f"{safe_name}_grid_{GRID_COLS}x{GRID_ROWS}.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n✓ Saved: {out_path}  ({grid.shape[1]}×{grid.shape[0]})")


if __name__ == "__main__":
    main()
