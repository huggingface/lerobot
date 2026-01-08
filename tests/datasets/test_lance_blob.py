#!/usr/bin/env python
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("lance")
pytest.importorskip("av")

from lerobot.datasets.lance.lance_support import build_lance_schema, build_episode_row


def make_dummy_video_bytes(num_frames: int = 3, width: int = 64, height: int = 48, fps: int = 10) -> bytes:
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
    # Generate solid-color frames
        frame = av.VideoFrame(width, height, "rgb24")
    # Write pixel data (three channels)
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


def test_lance_blob_roundtrip(tmp_path: Path):
    # Assume features include one action vector, one obs.state vector, and one video key
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a1", "a2"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["s1", "s2", "s3"]},
        "observation.images.cam": {"dtype": "video", "shape": (48, 64, 3), "names": ["height", "width", "channels"]},
    }
    camera_keys = ["observation.images.cam"]
    schema = build_lance_schema(features, camera_keys)

    fps = 10
    timestamps = [0.0, 0.1, 0.2]
    actions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    obs_state = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]], dtype=np.float32)
    vid_bytes = make_dummy_video_bytes(num_frames=3, width=64, height=48, fps=fps)

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

    out_dir = tmp_path / "mini.lance"
    lance.write_dataset(table, str(out_dir), schema=schema)

    ds = lance.dataset(str(out_dir))
    # Validate blob column presence
    col = "video_observation_images_cam"
    assert col in ds.to_table().schema.names
    # take_blobs returns file-like
    blobs = ds.take_blobs(col, ids=[0])
    assert blobs and hasattr(blobs[0], "read")

    import av  # type: ignore

    with av.open(blobs[0], mode="r") as container:
        v = container.streams.video[0]
        frames = 0
        for pkt in container.demux(v):
            for frame in pkt.decode():
                frames += 1
        assert frames == 3

    # Validate sequence column types
    tbl = ds.to_table()
    assert tbl.schema.field_by_name("timestamps").type == pa.large_list(pa.float32())
    assert tbl.schema.field_by_name("actions").type == pa.large_list(pa.fixed_size_list(pa.float32(), 2))
    assert tbl.schema.field_by_name("obs_state").type == pa.large_list(pa.fixed_size_list(pa.float32(), 3))

