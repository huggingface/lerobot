#!/usr/bin/env python
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

pytest.importorskip("lance")
pytest.importorskip("av")

from lerobot.datasets.lance.lance_support import build_lance_schema, build_episode_row
from lerobot.datasets.lance.lance_dataset import LanceFrameDataset


def _make_dummy_video_bytes(num_frames: int = 3, width: int = 64, height: int = 48, fps: int = 10) -> bytes:
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


def test_lance_frame_dataset_len_and_item(tmp_path: Path):
    # schema
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a1", "a2"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["s1", "s2", "s3"]},
        "observation.images.cam": {
            "dtype": "video",
            "shape": (48, 64, 3),
            "names": ["height", "width", "channels"],
        },
    }
    camera_keys = ["observation.images.cam"]
    schema = build_lance_schema(features, camera_keys)

    # episode content (3 frames)
    fps = 10
    timestamps = [0.0, 0.1, 0.2]
    actions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    obs_state = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], dtype=np.float32)
    vid_bytes = _make_dummy_video_bytes(num_frames=3, width=64, height=48, fps=fps)

    row = build_episode_row(
        episode_index=0,
        task_index=0,
        fps=fps,
        timestamps=timestamps,
        actions=actions,
        obs_state=obs_state,
        camera_blobs={"observation.images.cam": vid_bytes},
        camera_ranges={"observation.images.cam": (0.0, 0.3)},
    )

    columns = {f.name: [row.get(f.name)] for f in schema}
    table = pa.Table.from_pydict(columns, schema=schema)

    import lance  # type: ignore

    out_dir = tmp_path / "mini_frame.lance"
    lance.write_dataset(table, str(out_dir), schema=schema)

    # Set image_transforms to identity
    image_transforms = lambda x: x
    ds = LanceFrameDataset(out_dir, image_transforms=image_transforms)

    # __len__
    assert len(ds) == 3

    # Internal Lance dataset and schema should exist
    assert getattr(ds, "_ds", None) is not None
    assert getattr(ds, "_schema", None) is not None
    assert isinstance(ds._schema, pa.Schema)

    # __getitem__ fields and types
    item0 = ds[0]
    assert set(["timestamp", "frame_index", "episode_index", "index", "task_index"]).issubset(item0.keys())
    assert isinstance(item0["timestamp"], torch.Tensor)
    assert item0["timestamp"].shape == (1,)
    assert isinstance(item0["frame_index"], torch.Tensor) and item0["frame_index"].item() == 0
    # action / obs.state
    assert isinstance(item0["action"], torch.Tensor) and item0["action"].shape == (2,)
    assert isinstance(item0["observation.state"], torch.Tensor) and item0["observation.state"].shape == (3,)

    # video decode output exists and shape is correct
    assert "observation.images.cam" in item0
    img = item0["observation.images.cam"]
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 3 and img.shape[0] in (1, 3)  # C,H,W; likely 3 channels


def test_lance_frame_dataset_take(tmp_path: Path):
    # Reuse the previous construction to build a small Lance dataset with 1 episode and 3 frames
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a1", "a2"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["s1", "s2", "s3"]},
        "observation.images.cam": {
            "dtype": "video",
            "shape": (48, 64, 3),
            "names": ["height", "width", "channels"],
        },
    }
    camera_keys = ["observation.images.cam"]
    schema = build_lance_schema(features, camera_keys)

    fps = 10
    timestamps = [0.0, 0.1, 0.2]
    actions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    obs_state = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], dtype=np.float32)
    vid_bytes = _make_dummy_video_bytes(num_frames=3, width=64, height=48, fps=fps)

    row = build_episode_row(
        episode_index=0,
        task_index=0,
        fps=fps,
        timestamps=timestamps,
        actions=actions,
        obs_state=obs_state,
        camera_blobs={"observation.images.cam": vid_bytes},
        camera_ranges={"observation.images.cam": (0.0, 0.3)},
    )
    columns = {f.name: [row.get(f.name)] for f in schema}
    table = pa.Table.from_pydict(columns, schema=schema)

    import lance  # type: ignore
    out_dir = tmp_path / "mini_take.lance"
    lance.write_dataset(table, str(out_dir), schema=schema)

    ds = LanceFrameDataset(out_dir, image_transforms=lambda x: x)

    items = ds.take([0, 1])
    assert isinstance(items, list) and len(items) == 2
    # First item equivalent to ds[0] for core fields
    first = items[0]
    direct = ds[0]
    assert first["index"].item() == direct["index"].item()
    assert first["frame_index"].item() == direct["frame_index"].item()
    assert first["episode_index"].item() == direct["episode_index"].item()
    assert first["task_index"].item() == direct["task_index"].item()
    assert torch.isclose(first["timestamp"], direct["timestamp"]).all()
