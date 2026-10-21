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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import lerobot.policies.factory as policy_factory

DATASET_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}
RENAME_MAP = {
    "observation.images.image": "observation.images.front",
    "observation.images.image2": "observation.images.wrist",
}


def _make_policy_with_hook(monkeypatch, rename_map):
    seen = {}
    cfg = SimpleNamespace(
        type="mock",
        device="cpu",
        pretrained_path=None,
        use_peft=False,
        input_features={},
        output_features={},
        set_dataset_feature_metadata=lambda features: seen.update(features=features),
    )
    ds_meta = SimpleNamespace(features={k: dict(v) for k, v in DATASET_FEATURES.items()}, stats={})
    monkeypatch.setattr(
        policy_factory, "get_policy_class", lambda _: MagicMock(return_value=torch.nn.Linear(1, 1))
    )
    policy_factory.make_policy(cfg, ds_meta=ds_meta, rename_map=rename_map)
    return seen["features"], ds_meta


def test_dataset_feature_hook_sees_renamed_keys(monkeypatch):
    features, ds_meta = _make_policy_with_hook(monkeypatch, RENAME_MAP)

    assert sorted(k for k in features if "images" in k) == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert features["observation.images.front"] is ds_meta.features["observation.images.image"]
    # The dataset's own metadata keeps its keys: the loader still reads them.
    assert "observation.images.image" in ds_meta.features


@pytest.mark.parametrize("rename_map", [None, {}])
def test_dataset_feature_hook_without_rename_map_is_unchanged(monkeypatch, rename_map):
    features, ds_meta = _make_policy_with_hook(monkeypatch, rename_map)

    assert features is ds_meta.features
