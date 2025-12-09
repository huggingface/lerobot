#!/usr/bin/env python
"""
Convert a LeRobot v3.0 dataset to Lance dual tables:
- Episodes table: one row per episode with blob columns and episode metadata
- Frames table: one row per frame with lightweight numeric columns

This converter:
- Reads meta/episodes/*.parquet and data/*/*.parquet
- Aggregates per-episode sequences and cuts video segments into blobs
- Writes episodes.lance and frames.lance under the dataset root by default

Dependencies:
    pip install lance av pyarrow
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds

from lerobot.datasets.lance.lance_support import (
    build_episode_row,
    build_lance_schema,
    build_frame_row,
    build_frames_schema,
)
from lerobot.datasets.utils import DEFAULT_VIDEO_PATH, load_episodes, load_info


def _list_camera_video_keys(features: Dict[str, Dict]) -> List[str]:
    return [k for k, ft in features.items() if ft.get("dtype") == "video"]


def _extract_video_segment_bytes(video_path: Path, from_ts: float, to_ts: float, fps: int) -> bytes:
    """Cut a video segment [from_ts, to_ts) with PyAV and encode to in-memory bytes (mp4)."""
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

    import io

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    fps_fraction = Fraction(fps).limit_denominator(1000)
    v_out = out.add_stream("libx264", rate=fps_fraction)
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = "yuv420p"
    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    start_pts = int(from_ts / float(v_in.time_base)) if v_in.time_base else 0
    in_container.seek(start_pts, any_frame=False, stream=v_in)

    frame_count = 0
    last_t = from_ts
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue
            t = float(frame.pts * frame.time_base) if frame.pts is not None else last_t
            if t >= to_ts:
                break
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)
            frame_count += 1
            last_t = t
        if last_t >= to_ts:
            break

    for pkt in v_out.encode():
        out.mux(pkt)
    out.close()
    in_container.close()
    return buf.getvalue()


def _aggregate_episode_sequences(table: pa.Table, have_action: bool, have_obs_state: bool) -> Tuple[List[float], np.ndarray | None, np.ndarray | None]:
    ts = [float(x) for x in table.column("timestamp").to_pylist()] if "timestamp" in table.schema.names else []

    actions = None
    if have_action and "action" in table.schema.names:
        act_list = table.column("action").to_pylist()
        actions = np.asarray(act_list, dtype=np.float32)

    obs_state = None
    if have_obs_state and "observation.state" in table.schema.names:
        st_list = table.column("observation.state").to_pylist()
        obs_state = np.asarray(st_list, dtype=np.float32)

    return ts, actions, obs_state


def convert_dataset_v30_to_lance_dual(root: Path, out_episodes: Path, out_frames: Path) -> None:
    info = load_info(root)
    episodes_ds = load_episodes(root)

    camera_keys = _list_camera_video_keys(info["features"])  # only process video modalities
    ep_schema = build_lance_schema(info["features"], camera_keys)
    fr_schema = build_frames_schema(info["features"])  # frame-level schema

    # Build Arrow dataset from all parquet files under data for per-episode filtering
    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if len(data_paths) == 0:
        raise FileNotFoundError(f"No data/*/*.parquet found: {root / 'data'}")
    arrow_ds = pa_ds.dataset(data_paths, format="parquet")

    # Aggregate per-episode rows and collect per-frame rows
    ep_rows: List[Dict[str, object]] = []
    fr_rows: List[Dict[str, object]] = []

    for i in range(len(episodes_ds)):
        ep = episodes_ds[i]
        ep_idx = int(ep["episode_index"]) if "episode_index" in ep else i
        task_idx = int(ep["task_index"]) if "task_index" in ep else 0
        fps = int(info["fps"])  # Use unified fps value

        # Filter frame-level table for this episode
        filter_expr = pa_ds.field("episode_index") == ep_idx
        table = arrow_ds.to_table(filter=filter_expr)

        have_action = "action" in info["features"]
        have_obs_state = "observation.state" in info["features"]
        timestamps, actions, obs_state = _aggregate_episode_sequences(table, have_action, have_obs_state)

        # Build frames rows: iterate over all frames in this episode
        # Retrieve other needed columns
        def col_or_none(name: str):
            return table.column(name).to_pylist() if name in table.schema.names else None
        frame_indices = col_or_none("frame_index") or list(range(len(timestamps)))
        global_indices = col_or_none("index") or list(range(len(timestamps)))
        task_indices = col_or_none("task_index") or [task_idx] * len(timestamps)
        actions_list = actions.tolist() if actions is not None else [None] * len(timestamps)
        obs_list = obs_state.tolist() if obs_state is not None else [None] * len(timestamps)

        for f_local, ts, gidx, tki in zip(frame_indices, timestamps, global_indices, task_indices, strict=False):
            a = actions_list[f_local] if actions is not None else None
            s = obs_list[f_local] if obs_state is not None else None
            fr_rows.append(
                build_frame_row(
                    episode_index=ep_idx,
                    frame_index=int(f_local),
                    timestamp=float(ts),
                    index=int(gidx),
                    task_index=int(tki),
                    action=a,
                    obs_state=s,
                )
            )

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
                vpath_str = info["video_path"] or DEFAULT_VIDEO_PATH
                vpath = root / vpath_str.format(video_key=ck, chunk_index=chunk_idx, file_index=file_idx)
                blob = _extract_video_segment_bytes(vpath, from_ts, to_ts, fps)
                camera_blobs[ck] = blob
                camera_ranges[ck] = (from_ts, to_ts)

        ep_rows.append(
            build_episode_row(
                episode_index=ep_idx,
                task_index=task_idx,
                fps=fps,
                timestamps=timestamps,
                actions=actions,
                obs_state=obs_state,
                camera_blobs=camera_blobs,
                camera_ranges=camera_ranges,
            )
        )

    # Convert rows to column dicts and write Lance datasets
    ep_cols: Dict[str, List[object]] = {f.name: [] for f in ep_schema}
    for r in ep_rows:
        for k in ep_cols.keys():
            ep_cols[k].append(r.get(k))
    fr_cols: Dict[str, List[object]] = {f.name: [] for f in fr_schema}
    for r in fr_rows:
        for k in fr_cols.keys():
            fr_cols[k].append(r.get(k))

    ep_table = pa.Table.from_pydict(ep_cols, schema=ep_schema)
    fr_table = pa.Table.from_pydict(fr_cols, schema=fr_schema)

    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("Missing 'lance' package; please install with `pip install lance` to write Lance dataset.") from e

    out_episodes.parent.mkdir(parents=True, exist_ok=True)
    out_frames.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(ep_table, str(out_episodes), schema=ep_schema)
    lance.write_dataset(fr_table, str(out_frames), schema=fr_schema)
    print(f"Lance episodes dataset written: {out_episodes}")
    print(f"Lance frames dataset written: {out_frames}")
