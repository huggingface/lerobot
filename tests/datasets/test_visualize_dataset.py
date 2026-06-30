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
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_dataset_viz import get_extra_scalar_keys, is_scalar_like, visualize_dataset
from lerobot.utils.constants import ACTION, OBS_STATE


class DummyMeta:
    camera_keys = ["observation.images.front"]


class DummyDataset:
    meta = DummyMeta()
    features = {
        "index": {"dtype": "int64", "shape": [1]},
        ACTION: {"dtype": "float32", "shape": [6]},
        OBS_STATE: {"dtype": "float32", "shape": [6]},
        "observation.images.front": {"dtype": "video", "shape": [3, 480, 640]},
        "q_target": {"dtype": "float32", "shape": [1]},
        "intervention": {"dtype": "bool", "shape": []},
        "embedding": {"dtype": "float32", "shape": [32]},
        "comment": {"dtype": "string"},
    }


def test_get_extra_scalar_keys_skips_known_metadata_and_non_scalars():
    assert get_extra_scalar_keys(DummyDataset()) == ["q_target", "intervention"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor(1.0), True),
        (torch.tensor([1.0]), True),
        (torch.tensor([1.0, 2.0]), False),
        (1.0, True),
        (True, True),
    ],
)
def test_is_scalar_like(value, expected):
    assert is_scalar_like(value) is expected


@pytest.mark.skip("TODO: add dummy videos")
def test_visualize_local_dataset(tmp_path, lerobot_dataset_factory):
    root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs"
    dataset = lerobot_dataset_factory(root=root)
    rrd_path = visualize_dataset(
        dataset,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=output_dir,
    )
    assert rrd_path.exists()
