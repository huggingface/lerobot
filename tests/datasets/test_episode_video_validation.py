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
"""Tests for the episode-video timestamp validation.

Regression tests for datasets whose aggregated video files are shorter than the
episode metadata declares (e.g. lerobot/berkeley_autolab_ur5 v3.0, see
huggingface/lerobot#5039): the last episode assigned to a video file points past
the end of the file, and decoding any of its frames fails with an opaque
backend error. Loading such a dataset must fail fast with a clear error naming
the affected episodes, while the intact episodes remain loadable.
"""

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
from lerobot.datasets.video_utils import encode_video_frames
from tests.fixtures.constants import DUMMY_REPO_ID


def _truncate_video_file(root: Path, video_key: str, num_frames: int, fps: int, img_array_factory) -> None:
    """Re-encode a video file keeping only its first ``num_frames`` frames."""
    video_path = root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=0, file_index=0)
    tmp_dir = root / f"tmp_truncate_{video_key.replace('.', '_')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ft_shape = (64, 96, 3)
    for frame_index in range(num_frames):
        img = img_array_factory(height=ft_shape[0], width=ft_shape[1], content=f"{video_key}-{frame_index}")
        PIL.Image.fromarray(img).save(tmp_dir / f"frame-{frame_index:06d}.png")
    encode_video_frames(tmp_dir, video_path, fps=fps, overwrite=True)


def test_validation_passes_on_intact_dataset(tmp_path, lerobot_dataset_factory):
    """A dataset whose videos match their metadata loads without errors."""
    dataset = lerobot_dataset_factory(root=tmp_path / "intact", total_episodes=2, total_frames=10)

    assert dataset.num_episodes == 2
    assert dataset[0] is not None


def test_validation_raises_for_truncated_video(tmp_path, lerobot_dataset_factory, img_array_factory):
    """A video file missing the frames of the last episode assigned to it is detected at load."""
    root = tmp_path / "truncated"
    dataset = lerobot_dataset_factory(root=root, total_episodes=2, total_frames=10)

    ep_lengths = [dataset.meta.episodes[ep]["length"] for ep in range(2)]
    # Drop the frames of the last episode assigned to the 'laptop' video file,
    # as happens when a dataset is truncated during upload/conversion.
    _truncate_video_file(
        root, "laptop", num_frames=ep_lengths[0], fps=dataset.meta.fps, img_array_factory=img_array_factory
    )

    with pytest.raises(RuntimeError, match=r"episode\(s\) reference timestamps past the end"):
        LeRobotDataset(repo_id=DUMMY_REPO_ID, root=root)

    # 'phone' is intact: restricting to the episodes whose 'laptop' frames exist loads fine.
    ds_subset = LeRobotDataset(repo_id=DUMMY_REPO_ID, root=root, episodes=[0])
    assert ds_subset.num_episodes == 1
    assert np.asarray(ds_subset[0]["laptop"]).size > 0
