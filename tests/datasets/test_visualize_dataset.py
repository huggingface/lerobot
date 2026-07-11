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

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_dataset_viz import visualize_dataset


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


def test_scalar_feature_helpers_for_dataset_viz():
    import numpy as np
    import torch

    from lerobot.scripts.lerobot_dataset_viz import (
        is_scalar_feature,
        is_scalar_like,
        scalar_to_float,
    )

    assert is_scalar_feature({"dtype": "float32", "shape": (1,)})
    assert is_scalar_feature({"dtype": "bool", "shape": []})
    assert not is_scalar_feature({"dtype": "float32", "shape": (3,)})
    assert is_scalar_like(torch.tensor(1.5))
    assert is_scalar_like(np.array(2.0))
    assert scalar_to_float(torch.tensor(3.0)) == 3.0
