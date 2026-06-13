# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.scripts.augment_dataset_quantile_stats import (
    compute_quantile_stats_for_dataset,
    has_quantile_stats,
)


def _numeric_keys(dataset):
    return [k for k, v in dataset.features.items() if v["dtype"] not in ("image", "video", "string")]


def _image_keys(dataset):
    return [k for k, v in dataset.features.items() if v["dtype"] in ("image", "video")]


def test_numeric_stats_are_unaffected_by_sampling(tmp_path, lerobot_dataset_factory):
    """Sampling only touches image/video frames; numeric features are read in
    full either way, so their stats must be identical with and without sampling."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=400, use_videos=False
    )

    exact = compute_quantile_stats_for_dataset(dataset, use_sampling=False)
    sampled = compute_quantile_stats_for_dataset(dataset, use_sampling=True)

    numeric_keys = _numeric_keys(dataset)
    assert numeric_keys, "fixture should expose numeric features"
    for key in numeric_keys:
        if key not in exact:
            continue
        for stat in ("mean", "std", "q01", "q50", "q99"):
            if stat in exact[key]:
                np.testing.assert_allclose(
                    sampled[key][stat],
                    exact[key][stat],
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"numeric feature '{key}' stat '{stat}' changed under sampling",
                )


def test_image_sampling_reduces_data_but_keeps_stats_close(tmp_path, lerobot_dataset_factory):
    """For images, sampling should reduce the number of samples considered while
    keeping the resulting statistics close to the exact ones."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=400, use_videos=False
    )

    exact = compute_quantile_stats_for_dataset(dataset, use_sampling=False)
    sampled = compute_quantile_stats_for_dataset(dataset, use_sampling=True)

    image_keys = _image_keys(dataset)
    assert image_keys, "fixture should expose at least one image feature"
    for key in image_keys:
        # sampling actually looked at fewer pixels
        assert sampled[key]["count"][0] < exact[key]["count"][0]
        # but per-channel mean stays close
        np.testing.assert_allclose(
            sampled[key]["mean"],
            exact[key]["mean"],
            rtol=0.15,
            err_msg=f"image feature '{key}' mean drifted too far under sampling",
        )


def test_short_episodes_use_all_frames(tmp_path, lerobot_dataset_factory):
    """With episodes shorter than the sampling floor, sampling is a no-op and
    must produce exactly the same stats as the exact path."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=40, use_videos=False
    )

    exact = compute_quantile_stats_for_dataset(dataset, use_sampling=False)
    sampled = compute_quantile_stats_for_dataset(dataset, use_sampling=True)

    for key in _image_keys(dataset):
        assert sampled[key]["count"][0] == exact[key]["count"][0]


def test_quantile_stats_present_after_compute(tmp_path, lerobot_dataset_factory):
    """The computed stats should contain quantile keys for the dataset."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=200, use_videos=False
    )
    stats = compute_quantile_stats_for_dataset(dataset, use_sampling=True)
    assert has_quantile_stats(stats)
