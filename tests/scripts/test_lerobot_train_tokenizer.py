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

"""Keep FAST fitting inputs consistent with policy action preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("av")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.processor import NormalizerProcessorStep  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.scripts.lerobot_train_tokenizer import apply_normalization, process_episode  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("mode", list(NormalizationMode))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("eps", [1e-8, 1e-4])
def test_fitting_normalization_matches_policy_runtime(
    mode: NormalizationMode, dtype: type[np.floating], batched: bool, eps: float
) -> None:
    data = np.array([[-1, 1], [0.5, 1], [2, 1]], dtype=dtype)
    data = np.stack([data, data]) if batched else data[0]
    original = data.copy()
    stats = {
        name: np.array(values, dtype=dtype)
        for name, values in {
            "mean": [0.5, 1],
            "std": [0.5, 0],
            "min": [0, 1],
            "max": [1, 1],
            "q01": [0, 1],
            "q99": [1, 1],
            "q10": [0, 1],
            "q90": [1, 1],
        }.items()
    }
    tensor = torch.from_numpy(data)
    normalizer = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        norm_map={FeatureType.ACTION: mode},
        stats={ACTION: stats},
        dtype=tensor.dtype,
        eps=eps,
    )
    expected = normalizer(create_transition(action=tensor))[TransitionKey.ACTION]

    actual = apply_normalization(data, stats, mode, eps=eps)

    assert actual.shape == data.shape
    assert actual.dtype == data.dtype
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "mode,low,high",
    [(NormalizationMode.QUANTILES, "q01", "q99"), (NormalizationMode.QUANTILE10, "q10", "q90")],
)
def test_quantile_outliers_are_preserved(mode: NormalizationMode, low: str, high: str) -> None:
    actions = np.array([[[-1], [0.5], [2]]], dtype=np.float32)
    stats = {low: np.array([0], dtype=np.float32), high: np.array([1], dtype=np.float32)}

    actual = apply_normalization(actions, stats, mode)

    np.testing.assert_array_equal(actual, [[[-3], [0], [3]]])


@pytest.mark.parametrize("mode", [mode for mode in NormalizationMode if mode != NormalizationMode.IDENTITY])
def test_fitting_normalization_requires_stats(mode: NormalizationMode) -> None:
    with pytest.raises(ValueError, match="requires"):
        apply_normalization(np.zeros((1, 2, 1), dtype=np.float32), {}, mode)


def test_identity_normalization_needs_no_stats() -> None:
    data = np.ones((1, 2, 3), dtype=np.float32)

    assert apply_normalization(data, {}, NormalizationMode.IDENTITY) is data


@pytest.mark.parametrize(
    "mode,low,high",
    [
        (NormalizationMode.MIN_MAX, "min", "max"),
        (NormalizationMode.QUANTILES, "q01", "q99"),
        (NormalizationMode.QUANTILE10, "q10", "q90"),
    ],
)
def test_small_nonzero_ranges_match_runtime(mode: NormalizationMode, low: str, high: str) -> None:
    actions = np.array([0, 1e-10], dtype=np.float64)
    stats = {low: np.zeros(2), high: np.full(2, 1e-10)}

    np.testing.assert_array_equal(apply_normalization(actions, stats, mode), [-1, 1])


def test_integer_actions_are_normalized_as_floating_point() -> None:
    actions = np.array([0, 1, 2], dtype=np.int64)
    stats = {"min": np.zeros(3), "max": np.full(3, 2)}

    actual = apply_normalization(actions, stats, NormalizationMode.MIN_MAX)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [-1, 0, 1])


@pytest.mark.parametrize("use_relative_transform", [False, True])
def test_chunks_are_relative_to_their_first_state_for_listed_dims(use_relative_transform: bool) -> None:
    states = np.arange(12, dtype=np.float32).reshape(4, 3)
    actions = 100 + np.arange(12, dtype=np.float32).reshape(4, 3)
    frames = [
        {OBS_STATE: torch.from_numpy(states[i]), ACTION: torch.from_numpy(actions[i])} for i in range(4)
    ]
    dataset = SimpleNamespace(
        meta=SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 4}]),
        absolute_to_relative_idx=None,
        get_raw_item=lambda index: frames[index],
    )

    chunks = process_episode((dataset, 0, 2, [0, 2], 1.0, OBS_STATE, use_relative_transform))

    expected = np.stack([actions[0:2], actions[1:3], actions[2:4]])
    if use_relative_transform:
        # Dimensions 0 and 2 become offsets from the chunk's first state; dimension 1 stays absolute.
        expected[:, :, [0, 2]] -= states[:3, None, [0, 2]]
    np.testing.assert_array_equal(chunks, expected)
