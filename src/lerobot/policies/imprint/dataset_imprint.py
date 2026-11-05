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

"""Gather complete causal episode prefixes without decoding padded images."""

from bisect import bisect_right
from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_imprint import ImprintConfig
from .history import PREFIX_FRAMES, PREFIX_LENGTH, prefix_indices


class ImprintDataset(Dataset):
    """Adapt map-style rows to prefixes, padded only to this batch's maximum.

    No prefix is truncated. ``history_length`` bounds decoded frames per sample;
    an episode exceeding that capacity raises before training starts. The reader
    still handles episode boundaries, video decoding, and raw causal-action masks.
    """

    def __init__(
        self, dataset: LeRobotDataset, config: ImprintConfig, rename_map: dict[str, str] | None = None
    ) -> None:
        self.dataset = dataset
        self.config = config
        rename_map = rename_map or {}
        self.observation_keys = [
            key
            for key in dataset.features
            if rename_map.get(key, key) == OBS_STATE or key in dataset.meta.camera_keys
        ]
        self.action_key = next(key for key in dataset.features if rename_map.get(key, key) == ACTION)
        self.starts: list[int] = []
        self.lengths: list[int] = []
        mapping = dataset.absolute_to_relative_idx
        selected = dataset.episodes if dataset.episodes is not None else range(len(dataset.meta.episodes))
        episodes = sorted(
            (dataset.meta.episodes[index] for index in selected), key=lambda ep: ep["dataset_from_index"]
        )
        for episode in episodes:
            start = int(episode["dataset_from_index"])
            length = int(episode["length"])
            required = (length - 1 + config.history_downsample - 1) // config.history_downsample + 1
            if required > config.history_length:
                raise ValueError(
                    f"IMPRINT episode {episode['episode_index']} needs {required} prefix frames, "
                    f"above history_length={config.history_length}. Increase the capacity; "
                    "episode prefixes are never truncated."
                )
            if config.training_stage == "memory" and length > config.max_query_steps:
                raise ValueError("IMPRINT max_query_steps must cover every frame of each training episode.")
            self.starts.append(mapping[start] if mapping is not None else start)
            self.lengths.append(length)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        prefixes: list[list[int]] = []
        frame_indices: list[list[int]] = []
        for index in indices:
            episode = bisect_right(self.starts, index) - 1
            start = self.starts[episode]
            if episode < 0 or not 0 <= index - start < self.lengths[episode]:
                raise IndexError(index)
            frames = prefix_indices(index - start, self.config.history_downsample)
            frame_indices.append(frames)
            prefixes.append([start + frame for frame in frames])
        # Decode shared rows only once, and never read placeholder padding frames.
        union = sorted({index for prefix in prefixes for index in prefix})
        rows = dict(zip(union, self.dataset.__getitems__(union), strict=True))
        capacity = max(map(len, prefixes))
        items = []
        for index, prefix, frames in zip(indices, prefixes, frame_indices, strict=True):
            item = dict(rows[index])
            length = len(prefix)
            for key in self.observation_keys:
                values = torch.stack([rows[frame][key][0] for frame in prefix])
                item[key] = self._pad(values, capacity)
                item[f"{key}_is_pad"] = torch.arange(capacity) >= length
            key = self.action_key
            causal = torch.stack([rows[frame][key][0] for frame in prefix])
            causal_pad = torch.stack([rows[frame][f"{key}_is_pad"][0] for frame in prefix])
            item[key] = torch.cat([self._pad(causal, capacity), rows[index][key][1:]])
            item[f"{key}_is_pad"] = torch.cat(
                [self._pad(causal_pad, capacity, value=True), rows[index][f"{key}_is_pad"][1:]]
            )
            item[PREFIX_LENGTH] = torch.tensor(length)
            item[PREFIX_FRAMES] = self._pad(torch.tensor(frames), capacity)
            items.append(item)
        return items

    @staticmethod
    def _pad(values: torch.Tensor, capacity: int, value: bool | int = 0) -> torch.Tensor:
        padding = values.new_full((capacity - values.shape[0], *values.shape[1:]), value)
        return torch.cat([values, padding])
