#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from lerobot.datasets.dataset_writer import _encode_video_worker
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH


def test_encode_video_worker_forwards_vcodec(tmp_path):
    """Test that _encode_video_worker correctly forwards the vcodec parameter to encode_video_frames."""
    # Create the expected directory structure
    video_key = "observation.images.laptop"
    episode_index = 0
    frame_index = 0

    fpath = DEFAULT_IMAGE_PATH.format(
        image_key=video_key, episode_index=episode_index, frame_index=frame_index
    )
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image file
    dummy_img = Image.new("RGB", (64, 64), color="red")
    dummy_img.save(img_dir / "frame-000000.png")

    # Track what vcodec was passed to encode_video_frames
    captured_kwargs = {}

    def mock_encode_video_frames(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        # Create a dummy output file so the worker doesn't fail
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode_video_frames):
        # Test with h264 codec
        _encode_video_worker(video_key, episode_index, tmp_path, fps=30, vcodec="h264")

    assert "vcodec" in captured_kwargs
    assert captured_kwargs["vcodec"] == "h264"


def test_encode_video_worker_default_vcodec(tmp_path):
    """Test that _encode_video_worker uses libsvtav1 as the default codec."""
    # Create the expected directory structure
    video_key = "observation.images.laptop"
    episode_index = 0
    frame_index = 0

    fpath = DEFAULT_IMAGE_PATH.format(
        image_key=video_key, episode_index=episode_index, frame_index=frame_index
    )
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image file
    dummy_img = Image.new("RGB", (64, 64), color="red")
    dummy_img.save(img_dir / "frame-000000.png")

    # Track what vcodec was passed to encode_video_frames
    captured_kwargs = {}

    def mock_encode_video_frames(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        # Create a dummy output file so the worker doesn't fail
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode_video_frames):
        # Test with default codec (no vcodec specified)
        _encode_video_worker(video_key, episode_index, tmp_path, fps=30)

    assert "vcodec" in captured_kwargs
    assert captured_kwargs["vcodec"] == "libsvtav1"
