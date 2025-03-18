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

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_episode_stats


def sample_episode_video_frames(dataset: LeRobotDataset, episode_index: int, ft_key: str) -> np.ndarray:
    ep_len = dataset.meta.episodes[episode_index]["length"]
    sampled_indices = sample_indices(ep_len)
    query_timestamps = dataset._get_query_timestamps(0.0, {ft_key: sampled_indices})
    video_frames = dataset._query_videos(query_timestamps, episode_index)
    return video_frames[ft_key].numpy()


def convert_episode_stats(dataset: LeRobotDataset, ep_idx: int):
    ep_start_idx = dataset.episode_data_index["from"][ep_idx]
    ep_end_idx = dataset.episode_data_index["to"][ep_idx]
    ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    ep_stats = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            # We sample only for videos
            ep_ft_data = sample_episode_video_frames(dataset, ep_idx, key)
        else:
            ep_ft_data = np.array(ep_data[key])

        axes_to_reduce = (0, 2, 3) if ft["dtype"] in ["image", "video"] else 0
        keepdims = True if ft["dtype"] in ["image", "video"] else ep_ft_data.ndim == 1
        ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)

        if ft["dtype"] in ["image", "video"]:  # remove batch dim
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()
            }

    dataset.meta.episodes_stats[ep_idx] = ep_stats


def convert_stats(dataset: LeRobotDataset, num_workers: int = 0):
    assert dataset.episodes is None
    print("Computing episodes stats")
    total_episodes = dataset.meta.total_episodes
    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(convert_episode_stats, dataset, ep_idx): ep_idx
                for ep_idx in range(total_episodes)
            }
            for future in tqdm(as_completed(futures), total=total_episodes):
                future.result()
    else:
        for ep_idx in tqdm(range(total_episodes)):
            convert_episode_stats(dataset, ep_idx)

    for ep_idx in tqdm(range(total_episodes)):
        write_episode_stats(ep_idx, dataset.meta.episodes_stats[ep_idx], dataset.root)


def check_aggregate_stats(
    dataset: LeRobotDataset,
    reference_stats: dict[str, dict[str, np.ndarray]],
    video_rtol_atol: tuple[float] = (1e-2, 1e-2),
    default_rtol_atol: tuple[float] = (5e-6, 6e-5),
):
    """Verifies that the aggregated stats from episodes_stats are close to reference stats."""
    agg_stats = aggregate_stats(list(dataset.meta.episodes_stats.values()))
    for key, ft in dataset.features.items():
        # These values might need some fine-tuning
        if ft["dtype"] == "video":
            # to account for image sub-sampling
            rtol, atol = video_rtol_atol
        else:
            rtol, atol = default_rtol_atol

        for stat, val in agg_stats[key].items():
            if key in reference_stats and stat in reference_stats[key]:
                err_msg = f"feature='{key}' stats='{stat}'"
                np.testing.assert_allclose(
                    val, reference_stats[key][stat], rtol=rtol, atol=atol, err_msg=err_msg
                )
