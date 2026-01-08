#!/usr/bin/env python
"""
Convert a LeRobot v3.0 dataset to a Lance table (one row per episode + video blob columns).

- Read the existing v3.0 dataset root (meta/episodes/*.parquet, data/*/*.parquet, and videos/*/*.mp4)
- Aggregate frame-level data per episode (timestamps, action, observation.state) into sequence columns
- For each camera video, use PyAV to cut the interval based on from_timestamp/to_timestamp, encode to in-memory bytes, and write to blob columns
- Build a PyArrow Table and write to the target .lance directory via lance.write_dataset(table, path, schema=...) (e.g. root/<root.name>.lance)

Dependencies:
- Requires pyarrow; lance and av are try-imported at runtime and will raise clear errors if not installed.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds

from lerobot.datasets.lance.lance_support import build_episode_row, build_lance_schema
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    load_episodes,
    load_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


def _list_camera_video_keys(features: Dict[str, Dict]) -> List[str]:
    return [k for k, ft in features.items() if ft.get("dtype") == "video"]


def _extract_video_segment_bytes(
    video_path: Path,
    from_ts: float,
    to_ts: float,
    fps: int,
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
) -> bytes:
    """Cut a video segment [from_ts, to_ts) with PyAV and encode to in-memory bytes."""
    try:
        import av  # type: ignore
    except Exception as e:
        raise ImportError("'av' is required to cut video segments. Please `pip install av`. ") from e

    from fractions import Fraction

    in_container = av.open(str(video_path), mode="r")
    if not in_container.streams.video:
        in_container.close()
        raise ValueError(f"No video stream found in {video_path}")
    v_in = in_container.streams.video[0]

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    fps_fraction = Fraction(fps).limit_denominator(1000)
    v_out = out.add_stream(vcodec, rate=fps_fraction)
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = pix_fmt
    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    # Compute seek start according to time_base
    start_pts = int(from_ts / float(v_in.time_base)) if v_in.time_base else 0
    in_container.seek(start_pts, any_frame=False, stream=v_in)

    frame_count = 0
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue
            t = float(frame.pts * frame.time_base) if frame.pts is not None else (frame_count / float(fps))
            if t >= to_ts:
                break
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)
            frame_count += 1
        if t >= to_ts:
            break

    for pkt in v_out.encode():
        out.mux(pkt)
    out.close()
    in_container.close()

    return buf.getvalue()


def _aggregate_episode_sequences(table: pa.Table, have_action: bool, have_obs_state: bool) -> Tuple[List[float], np.ndarray | None, np.ndarray | None]:
    """Aggregate a frame-level table into (timestamps, actions, obs_state)."""
    ts = [float(x) for x in table.column("timestamp").to_pylist()] if "timestamp" in table.schema.names else []

    actions = None
    if have_action and "action" in table.schema.names:
        # Each per-frame action is a list of length D; stack into (T, D)
        act_list = table.column("action").to_pylist()
        actions = np.asarray(act_list, dtype=np.float32)

    obs_state = None
    if have_obs_state and "observation.state" in table.schema.names:
        st_list = table.column("observation.state").to_pylist()
        obs_state = np.asarray(st_list, dtype=np.float32)

    return ts, actions, obs_state


def convert_dataset_v30_to_lance(root: Path, out_path: Path) -> None:
    """Convert a v3.0 dataset root to a Lance dataset (one row per episode + blobs)."""
    info = load_info(root)
    episodes_ds = load_episodes(root)

    camera_keys = _list_camera_video_keys(info["features"])  # only process video modalities
    schema = build_lance_schema(info["features"], camera_keys)

    # Build Arrow dataset from all parquet files under data for episode filtering
    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if len(data_paths) == 0:
        raise FileNotFoundError(f"No data/*/*.parquet found: {root / 'data'}")
    arrow_ds = pa_ds.dataset(data_paths, format="parquet")

    # Aggregate per-episode data, cut video segments, and build Arrow rows
    rows: List[Dict[str, object]] = []

    for i in range(len(episodes_ds)):
        ep = episodes_ds[i]
        ep_idx = int(ep["episode_index"]) if "episode_index" in ep else i
        task_idx = int(ep["task_index"]) if "task_index" in ep else 0
        fps = int(info["fps"])  # Use a unified fps value from info

        # Read frame range for this episode
        # Prefer filtering by episode_index; if data does not have episode_index, filter by index range
        filter_expr = pa_ds.field("episode_index") == ep_idx
        table = arrow_ds.to_table(filter=filter_expr)

        have_action = "action" in info["features"]
        have_obs_state = "observation.state" in info["features"]
        timestamps, actions, obs_state = _aggregate_episode_sequences(table, have_action, have_obs_state)

        # For each video key, cut segments according to episodes metadata
        camera_blobs: Dict[str, bytes] = {}
        camera_ranges: Dict[str, Tuple[float, float]] = {}

        for ck in camera_keys:
            from_key = f"videos/{ck}/from_timestamp"
            to_key = f"videos/{ck}/to_timestamp"
            chunk_key = f"videos/{ck}/chunk_index"
            file_key = f"videos/{ck}/file_index"
            if from_key in ep and to_key in ep and chunk_key in ep and file_key in ep:
                from_ts = float(ep[from_key])
                to_ts = float(ep[to_key])
                chunk_idx = int(ep[chunk_key])
                file_idx = int(ep[file_key])
        # Build video file path
                vpath_str = info["video_path"] or DEFAULT_VIDEO_PATH
                vpath = root / vpath_str.format(video_key=ck, chunk_index=chunk_idx, file_index=file_idx)
                blob = _extract_video_segment_bytes(vpath, from_ts, to_ts, fps)
                camera_blobs[ck] = blob
                camera_ranges[ck] = (from_ts, to_ts)

        row = build_episode_row(
            episode_index=ep_idx,
            task_index=task_idx,
            fps=fps,
            timestamps=timestamps,
            actions=actions,
            obs_state=obs_state,
            camera_blobs=camera_blobs,
            camera_ranges=camera_ranges,
        )
        rows.append(row)

    # Convert row list into column dict for from_pydict
    columns: Dict[str, List[object]] = {f.name: [] for f in schema}
    for r in rows:
        for k in columns.keys():
            columns[k].append(r.get(k))

    table = pa.Table.from_pydict(columns, schema=schema)

    # Write to the Lance dataset directory
    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance' package; please install with `pip install lance` to write Lance dataset.") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(out_path), schema=schema)
    print(f"Lance dataset written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert v3.0 dataset to Lance (one row per episode + blob columns)")
    parser.add_argument("--repo-id", type=str, help="HF dataset repo id, e.g., lerobot/pusht", required=False)
    parser.add_argument(
        "--root",
        type=str,
        help="Local v3.0 dataset root; if omitted and repo-id is provided, use the default cache directory",
        required=False,
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output Lance dataset directory (default: root/<root.name>.lance)",
        required=False,
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    elif args.repo_id:
        root = HF_LEROBOT_HOME / args.repo_id
    else:
        raise ValueError("You must provide --root or --repo-id to locate the v3.0 dataset root")

    out = Path(args.out) if args.out else (root / f"{root.name}.lance")
    convert_dataset_v30_to_lance(root, out)


if __name__ == "__main__":
    main()
