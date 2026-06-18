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

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.feature_utils import (
    detect_legacy_scalar_features,
    get_hf_features_from_features,
    validate_feature_shape_and_names,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def test_scalar_shape_dispatches_to_value():
    features = {"episode_index": {"dtype": "int64", "shape": (), "names": None}}
    hf_features = get_hf_features_from_features(features)
    assert isinstance(hf_features["episode_index"], datasets.Value)


def test_length_one_vector_dispatches_to_fixed_sequence_not_value():
    features = {"observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    hf_features = get_hf_features_from_features(features)
    assert isinstance(hf_features["observation.state"], datasets.Sequence)
    assert hf_features["observation.state"].length == 1


def test_zero_length_vector_uses_variable_length_sequence():
    # A fixed-length Sequence(length=0, ...) is rejected by PyArrow, so shape=(0,)
    # must use a variable-length Sequence instead.
    features = {"observation.state": {"dtype": "float32", "shape": (0,), "names": []}}
    hf_features = get_hf_features_from_features(features)
    assert isinstance(hf_features["observation.state"], datasets.Sequence)
    assert hf_features["observation.state"].length == -1


def test_legacy_scalar_keys_squeezes_length_one_to_value():
    features = {"observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    hf_features = get_hf_features_from_features(features, legacy_scalar_keys=frozenset({"observation.state"}))
    assert isinstance(hf_features["observation.state"], datasets.Value)


def test_detect_legacy_scalar_features_reads_physical_schema(tmp_path):
    # A column physically stored as a scalar (the pre-fix convention) is detected as legacy...
    features = {"observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]}}
    legacy_table = pa.table({"observation.state": pa.array([1.0, 2.0], type=pa.float32())})
    legacy_dir = tmp_path / "legacy" / "chunk-000"
    legacy_dir.mkdir(parents=True)
    pq.write_table(legacy_table, legacy_dir / "file-000.parquet")
    assert detect_legacy_scalar_features(features, tmp_path / "legacy") == {"observation.state"}

    # ...but a column already stored as a length-1 list (the corrected convention) is not.
    new_table = pa.table({"observation.state": pa.array([[1.0], [2.0]], type=pa.list_(pa.float32(), 1))})
    new_dir = tmp_path / "new" / "chunk-000"
    new_dir.mkdir(parents=True)
    pq.write_table(new_table, new_dir / "file-000.parquet")
    assert detect_legacy_scalar_features(features, tmp_path / "new") == frozenset()


def test_detect_legacy_scalar_features_reports_bookkeeping_fields_too(tmp_path):
    # detect_legacy_scalar_features reports purely on physical encoding -- it doesn't know
    # or care that episode_index is a DEFAULT_FEATURES bookkeeping field. Callers that need
    # to treat bookkeeping fields differently (e.g. dataset_tools.py's copy/conversion
    # helpers, which must not wrap them into length-1 lists) exclude them themselves.
    features = {"episode_index": {"dtype": "int64", "shape": (1,), "names": None}}
    table = pa.table({"episode_index": pa.array([0, 0], type=pa.int64())})
    data_dir = tmp_path / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(table, data_dir / "file-000.parquet")
    assert detect_legacy_scalar_features(features, tmp_path) == {"episode_index"}


@pytest.mark.parametrize(
    "shape,names",
    [
        ((), None),
        ((1,), ["x"]),
        ((0,), []),
        ((3,), ["x", "y", "z"]),
    ],
)
def test_validate_feature_shape_and_names_accepts_valid_specs(shape, names):
    validate_feature_shape_and_names("f", {"dtype": "float32", "shape": shape, "names": names})


def test_validate_feature_shape_and_names_rejects_named_scalar():
    with pytest.raises(ValueError, match="scalar"):
        validate_feature_shape_and_names("f", {"dtype": "float32", "shape": (), "names": ["x"]})


def test_validate_feature_shape_and_names_accepts_grouped_dict_names():
    # Some bimanual datasets declare `names` as a dict of groups rather than a flat list.
    validate_feature_shape_and_names(
        "observation.state",
        {
            "dtype": "float32",
            "shape": (14,),
            "names": {"motors": [f"m{i}" for i in range(14)]},
        },
    )


def test_validate_feature_shape_and_names_rejects_length_mismatch():
    with pytest.raises(ValueError, match="expected 3"):
        validate_feature_shape_and_names("f", {"dtype": "float32", "shape": (3,), "names": ["x", "y"]})


@pytest.mark.parametrize("state_dim", [0, 1, 6])
def test_dataset_round_trip_preserves_declared_shape(tmp_path, state_dim):
    """The original bug: a shape=(N,) feature must never collapse to a bare scalar at
    read time, for any N -- including the historically-squeezed N=1 case and the N=0
    (camera-only) case."""
    names = [f"x{i}" for i in range(state_dim)]
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": names},
        "action": {"dtype": "float32", "shape": (1,), "names": ["a"]},
    }
    root = tmp_path / "ds"
    dataset = LeRobotDataset.create(repo_id="local/round_trip_test", fps=10, root=root, features=features)
    for _ep in range(2):
        for i in range(3):
            dataset.add_frame(
                {
                    "observation.state": np.full(state_dim, float(i), dtype=np.float32),
                    "action": np.array([float(i)], dtype=np.float32),
                    "task": "test task",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    reloaded = LeRobotDataset(repo_id="local/round_trip_test", root=root)
    item = reloaded[0]
    assert item["observation.state"].shape == (state_dim,)
    assert item["action"].shape == (1,)
    assert item["episode_index"].shape == ()

    batch = torch.utils.data.default_collate([reloaded[i] for i in range(4)])
    assert batch["observation.state"].shape == (4, state_dim)
    assert batch["action"].shape == (4, 1)
    assert batch["episode_index"].shape == (4,)
