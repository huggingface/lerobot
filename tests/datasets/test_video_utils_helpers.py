# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import io

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.video_utils import (  # noqa: E402
    apply_rgb_transforms,
    convert_image_depth_units,
    extract_video_clip,
    normalize_rgb_frames,
)


@pytest.fixture(scope="module")
def b_frame_video(tmp_path_factory):
    av = pytest.importorskip("av")
    path = tmp_path_factory.mktemp("video-clip") / "source.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 6
        stream.options = {"sc_threshold": "0", "bf": "2"}
        for value in range(18):
            image = np.full((32, 32, 3), value * 10, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


@pytest.mark.parametrize("start,end", [(0, 6), (6, 12), (12, 18)])
def test_extract_video_clip_preserves_b_frames(b_frame_video, start, end):
    av = pytest.importorskip("av")
    with av.open(str(b_frame_video)) as source:
        assert source.streams.video[0].codec_context.has_b_frames
        original = [frame.to_ndarray(format="rgb24") for frame in source.decode(video=0)]
    clip = extract_video_clip(b_frame_video, start / 10, end / 10)
    with av.open(io.BytesIO(clip)) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == end - start
    assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
        np.arange(end - start) / 10
    )
    np.testing.assert_array_equal(
        np.stack([frame.to_ndarray(format="rgb24") for frame in frames]), np.stack(original[start:end])
    )


@pytest.mark.parametrize(
    "start,end", [(0.1, 0.6), (0, 0.3), (0, 3), (-1, 1), (1, 0), (np.nan, 1), (0, np.inf)]
)
def test_extract_video_clip_rejects_unsafe_or_invalid_cuts(b_frame_video, start, end):
    with pytest.raises(ValueError):
        extract_video_clip(b_frame_video, start, end)


def test_normalize_rgb_frames():
    frames = torch.tensor([[0, 255]], dtype=torch.uint8)
    assert normalize_rgb_frames(frames, return_uint8=True) is frames
    normalized = normalize_rgb_frames(frames, return_uint8=False)
    assert normalized.dtype == torch.float32
    assert normalized.tolist() == [[0.0, 1.0]]


def test_apply_rgb_transforms_skips_depth_and_missing_transforms():
    item = {"rgb": torch.ones(1), "depth": torch.ones(1)}
    apply_rgb_transforms(item, None, ["rgb", "depth"], ["depth"])
    assert item["rgb"].item() == 1
    apply_rgb_transforms(item, lambda image: image * 2, ["rgb", "depth"], ["depth"])
    assert item["rgb"].item() == 2
    assert item["depth"].item() == 1


def test_convert_image_depth_units():
    item = {
        "metres": torch.tensor([1.5]),
        "millimetres": torch.tensor([1500.0]),
        "unknown": torch.tensor([7.0]),
    }
    convert_image_depth_units(item, {"metres": "m", "millimetres": "mm", "unknown": None}, output_unit="mm")
    assert item["metres"].item() == 1500.0
    assert item["millimetres"].item() == 1500.0
    assert item["unknown"].item() == 7.0
