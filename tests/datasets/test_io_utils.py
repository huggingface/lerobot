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
import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from PIL import Image

from lerobot.datasets.io_utils import hf_transform_to_torch, item_to_torch
from lerobot.datasets.language import LANGUAGE_COLUMNS
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID


def _pil_image(h=4, w=6):
    array = np.random.default_rng(0).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_item_to_torch_matches_hf_transform_for_images():
    # Streaming should produce the same image tensor as the non-streaming transform.
    img = _pil_image()
    streamed = item_to_torch({"observation.images.cam": img})["observation.images.cam"]
    expected = hf_transform_to_torch({"observation.images.cam": [img]})["observation.images.cam"][0]

    assert streamed.shape == (3, 4, 6)  # (C, H, W)
    assert streamed.dtype == torch.float32
    torch.testing.assert_close(streamed, expected)


def test_item_to_torch_preserves_other_types():
    item = {
        "observation.state": np.array([1.0, 2.0, 3.0]),
        "action": [4, 5, 6],
        "task": "pick up the cube",
        LANGUAGE_COLUMNS[0]: ["a", "b"],  # skip keys must be left alone
    }
    out = item_to_torch(item)

    assert torch.equal(out["observation.state"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out["action"], torch.tensor([4, 5, 6]))
    assert out["task"] == "pick up the cube"
    assert out[LANGUAGE_COLUMNS[0]] == ["a", "b"]


def test_streaming_image_dataset_yields_tensors(tmp_path, lerobot_dataset_factory):
    # Regression for #2366: image (non-video) datasets used to yield raw PIL images in
    # streaming mode, which crashed the default collate during training.
    root = tmp_path / "test"
    ref = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        use_videos=False,
    )
    assert len(ref.meta.video_keys) == 0

    frame = next(iter(StreamingLeRobotDataset(repo_id=DUMMY_REPO_ID, root=root, buffer_size=8)))
    for cam in ref.meta.camera_keys:
        img = frame[cam]
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.ndim == 3 and img.shape[0] in (1, 3)  # (C, H, W)
