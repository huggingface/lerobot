#!/usr/bin/env python
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

pytest.importorskip("lance")
pytest.importorskip("av")

from lerobot.datasets.lance.lance_support import (
    build_lance_schema,
    build_episode_row,
    build_frames_schema,
    build_frame_row,
)
from lerobot.datasets.lance.lance_dual_dataset import LanceFramesTable


def _make_dummy_video_bytes(num_frames: int = 4, width: int = 64, height: int = 48, fps: int = 10) -> bytes:
    import av  # type: ignore
    from fractions import Fraction

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    v_out = out.add_stream("libx264", rate=Fraction(fps, 1))
    v_out.width = width
    v_out.height = height
    v_out.pix_fmt = "yuv420p"
    v_out.time_base = Fraction(1, fps)

    out.start_encoding()
    for i in range(num_frames):
        frame = av.VideoFrame(width, height, "rgb24")
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, :] = (i * 30) % 255
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for pkt in v_out.encode(frame):
            out.mux(pkt)
    for pkt in v_out.encode():
        out.mux(pkt)
    out.close()
    return buf.getvalue()


def test_dual_tables_len_item_window(tmp_path: Path):
    # Features and schemas
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a1", "a2"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["s1", "s2", "s3"]},
        "observation.images.cam": {"dtype": "video", "shape": (48, 64, 3), "names": ["height", "width", "channels"]},
    }
    camera_keys = ["observation.images.cam"]
    ep_schema = build_lance_schema(features, camera_keys)
    fr_schema = build_frames_schema(features)

    fps = 10
    timestamps = [0.0, 0.1, 0.2, 0.3]
    actions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    obs_state = np.array(
        [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3]],
        dtype=np.float32,
    )
    vid_bytes = _make_dummy_video_bytes(num_frames=4, width=64, height=48, fps=fps)

    # Episodes table with one episode
    ep_row = build_episode_row(
        episode_index=0,
        task_index=0,
        fps=fps,
        timestamps=timestamps,
        actions=actions,
        obs_state=obs_state,
        camera_blobs={"observation.images.cam": vid_bytes},
        camera_ranges={"observation.images.cam": (0.0, 0.4)},
    )
    ep_cols = {f.name: [ep_row.get(f.name)] for f in ep_schema}
    ep_table = pa.Table.from_pydict(ep_cols, schema=ep_schema)

    # Frames table with 4 rows
    fr_rows = [
        build_frame_row(0, i, timestamps[i], i, 0, actions[i].tolist(), obs_state[i].tolist()) for i in range(4)
    ]
    fr_cols = {f.name: [r.get(f.name) for r in fr_rows] for f in fr_schema}
    fr_table = pa.Table.from_pydict(fr_cols, schema=fr_schema)

    import lance  # type: ignore
    episodes_dir = tmp_path / "episodes.lance"
    frames_dir = tmp_path / "frames.lance"
    lance.write_dataset(ep_table, str(episodes_dir), schema=ep_schema)
    lance.write_dataset(fr_table, str(frames_dir), schema=fr_schema)

    ds = LanceFramesTable(frames_dir, episodes_dir, image_transforms=lambda x: x)
    assert len(ds) == 4

    # Verify __getitem__ keys/types
    item0 = ds[0]
    assert isinstance(item0["timestamp"], torch.Tensor) and item0["timestamp"].shape == (1,)
    assert isinstance(item0["frame_index"], torch.Tensor) and item0["frame_index"].item() == 0
    assert isinstance(item0["episode_index"], torch.Tensor) and item0["episode_index"].item() == 0
    assert isinstance(item0["task_index"], torch.Tensor)
    assert isinstance(item0["action"], torch.Tensor) and item0["action"].shape == (2,)
    assert isinstance(item0["observation.state"], torch.Tensor) and item0["observation.state"].shape == (3,)
    # Decoded image tensor exists
    assert "observation.images.cam" in item0
    img = item0["observation.images.cam"]
    assert isinstance(img, torch.Tensor) and img.ndim == 3

    # take (random sampling)
    items = ds.take([0, 2])
    assert len(items) == 2
    assert items[0]["frame_index"].item() == 0
    assert items[1]["frame_index"].item() == 2

    # window sampling (k=3)
    win = ds.sample_window(1, 3)
    assert len(win) == 3
    assert win[0]["frame_index"].item() == 1
    assert win[-1]["frame_index"].item() == 3

    # filtering by episode
    frame_ids = ds.filter_by_episode(0)
    assert frame_ids == [0, 1, 2, 3]

    # filtering by timestamp range
    ids_ts = ds.filter_by_ts_range(0.15, 0.35)
    assert ids_ts == [2, 3]
