#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.datasets import LeRobotDataset
from lerobot.rl.crop_dataset_roi import convert_lerobot_dataset_to_cropped_lerobot_dataset
from lerobot.utils.feature_utils import dataset_to_policy_features

IMAGE_KEY = "observation.images.front"


@pytest.mark.parametrize(
    ("source_shape", "names", "expected_shape"),
    [
        ((16, 20, 3), ["height", "width", "channels"], (8, 6, 3)),
        ((3, 16, 20), ["channels", "height", "width"], (3, 8, 6)),
    ],
    ids=["hwc", "chw"],
)
def test_cropped_image_feature_shape_follows_its_names(tmp_path, source_shape, names, expected_shape):
    """The cropped feature keeps its names, so its shape must keep the layout those names declare."""
    features = {
        IMAGE_KEY: {"dtype": "image", "shape": source_shape, "names": names},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    source = LeRobotDataset.create(
        repo_id="dummy/source", fps=10, features=features, root=tmp_path / "source", use_videos=False
    )
    for _ in range(3):
        source.add_frame(
            {
                IMAGE_KEY: np.zeros(source_shape, dtype=np.uint8),
                "action": np.zeros(2, dtype=np.float32),
                "task": "pick",
            }
        )
    source.save_episode()
    source.finalize()

    cropped = convert_lerobot_dataset_to_cropped_lerobot_dataset(
        LeRobotDataset(source.repo_id, root=source.root),
        {IMAGE_KEY: (2, 3, 10, 12)},
        new_repo_id="dummy/cropped",
        new_dataset_root=tmp_path / "cropped",
        resize_size=(8, 6),
    )

    feature = cropped.meta.info.features[IMAGE_KEY]
    assert tuple(feature["shape"]) == expected_shape
    assert feature["names"] == names
    assert dataset_to_policy_features({IMAGE_KEY: feature})[IMAGE_KEY].shape == (3, 8, 6)
