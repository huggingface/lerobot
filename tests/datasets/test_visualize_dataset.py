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
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
from lerobot.utils import import_utils
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, SUCCESS, TRUNCATED
from lerobot.utils.dataset_visualization_utils import (
    get_extra_scalar_keys,
    is_scalar_feature,
    is_scalar_like,
    scalar_to_float,
)


class DummyMeta:
    camera_keys = ["observation.images.front"]


class DummyFeatureDataset:
    meta = DummyMeta()
    features = {
        "index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        ACTION: {"dtype": "float32", "shape": [6]},
        OBS_STATE: {"dtype": "float32", "shape": [6]},
        DONE: {"dtype": "bool", "shape": [1]},
        REWARD: {"dtype": "float32", "shape": [1]},
        SUCCESS: {"dtype": "bool", "shape": [1]},
        TRUNCATED: {"dtype": "bool", "shape": [1]},
        "observation.images.front": {"dtype": "video", "shape": [3, 480, 640]},
        "q_target": {"dtype": "float32", "shape": [1]},
        "quality": {"dtype": "double", "shape": [1]},
        "intervention": {"dtype": "bool", "shape": []},
        "embedding": {"dtype": "float32", "shape": [32]},
        "comment": {"dtype": "string", "shape": [1]},
    }


class DummyVizDataset:
    repo_id = "dummy/custom-scalars"
    depth_output_unit = "m"
    meta = SimpleNamespace(camera_keys=[], depth_keys=[], stats=None)
    features = {
        "index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        ACTION: {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
        "q_target": {"dtype": "float32", "shape": [1]},
        "intervention": {"dtype": "bool", "shape": [1]},
        "embedding": {"dtype": "float32", "shape": [2]},
    }

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "index": torch.tensor(index),
            "timestamp": torch.tensor(index / 10),
            ACTION: torch.tensor([index, index + 1], dtype=torch.float32),
            "q_target": torch.tensor([index + 0.5]),
            "intervention": torch.tensor([index == 0]),
            "embedding": torch.tensor([index, index + 1], dtype=torch.float32),
        }


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


def test_get_extra_scalar_keys_skips_known_metadata_and_non_scalars():
    dataset = DummyFeatureDataset()

    assert get_extra_scalar_keys(dataset) == [TRUNCATED, "q_target", "quality", "intervention"]
    assert get_extra_scalar_keys(dataset, additional_known_keys=(TRUNCATED,)) == [
        "q_target",
        "quality",
        "intervention",
    ]


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ({"dtype": "float32", "shape": (1,)}, True),
        ({"dtype": "double", "shape": [1]}, True),
        ({"dtype": "bool", "shape": []}, True),
        ({"dtype": "int64", "shape": 1}, True),
        ({"dtype": "float32", "shape": (3,)}, False),
        ({"dtype": "complex64", "shape": [1]}, False),
        ({"dtype": "datetime64[ns]", "shape": [1]}, False),
        ({"dtype": "string", "shape": [1]}, False),
        ({"shape": [1]}, False),
    ],
)
def test_is_scalar_feature(feature, expected):
    assert is_scalar_feature(feature) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor(1.5), True),
        (torch.tensor([1.5]), True),
        (torch.tensor([1.5, 2.5]), False),
        (np.array(2.0), True),
        (np.array([2.0]), True),
        (np.array([2.0, 3.0]), False),
        (True, True),
        ("not numeric", False),
    ],
)
def test_is_scalar_like(value, expected):
    assert is_scalar_like(value) is expected


def test_scalar_to_float():
    assert scalar_to_float(torch.tensor(3.0)) == 3.0
    assert scalar_to_float(np.array([4.0])) == 4.0


def test_visualize_dataset_logs_extra_scalars(monkeypatch):
    logged = []
    initialized = []

    dummy_rrb = SimpleNamespace(
        Spatial2DView=lambda origin=None, name=None: SimpleNamespace(
            kind="spatial", origin=origin, name=name
        ),
        TimeSeriesView=lambda origin=None, name=None, overrides=None: SimpleNamespace(
            kind="time_series", origin=origin, name=name, overrides=overrides
        ),
        Grid=lambda *views: SimpleNamespace(views=views),
        Blueprint=lambda root: SimpleNamespace(root=root),
    )
    dummy_rr = SimpleNamespace(
        SeriesLines=lambda names=None: SimpleNamespace(names=names),
        Scalars=lambda value: SimpleNamespace(value=value),
        init=lambda *args, **kwargs: initialized.append((args, kwargs)),
        log=lambda key, entity: logged.append((key, entity)),
        set_time=lambda *args, **kwargs: None,
        blueprint=dummy_rrb,
    )
    monkeypatch.setitem(sys.modules, "rerun", dummy_rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", dummy_rrb)
    monkeypatch.setattr(import_utils, "require_package", lambda *args, **kwargs: None)

    visualize_dataset(DummyVizDataset(), episode_index=0, batch_size=2)

    logged_keys = [key for key, _ in logged]
    assert logged_keys.count("q_target") == 2
    assert logged_keys.count("intervention") == 2
    assert "embedding" not in logged_keys

    blueprint = initialized[0][1]["default_blueprint"]
    time_series_origins = {view.origin for view in blueprint.root.views if view.kind == "time_series"}
    assert {"q_target", "intervention"} <= time_series_origins
    assert "embedding" not in time_series_origins
