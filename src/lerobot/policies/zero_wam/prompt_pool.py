# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample encoded demonstrations from compact, episode-disjoint training pools."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class PromptMember:
    episode_index: int
    start_frame: int
    end_frame: int
    latent_path: str


class _Pool:
    def __init__(self, members: list[PromptMember]):
        self.members: dict[int, list[PromptMember]] = {}
        for member in members:
            self.members.setdefault(member.episode_index, []).append(member)
        self.episodes = tuple(self.members)
        self.positions = {episode: i for i, episode in enumerate(self.episodes)}


class ICLPromptPools:
    """Index each member once instead of expanding every target/prompt combination.

    A row has ``pool_id``, ``split`` and ``members``. Members declare source ``episode_index``,
    ``start_frame``, ``end_frame`` and encoded ``latent_path``. All rows belong to one dataset.
    Select a split before indexing. A source episode cannot occur in another declared split.
    Sampling chooses an eligible pool, another episode, then one of that episode's segments.
    """

    def __init__(self, path: str | Path, split: str = "train"):
        self._by_episode: dict[int, list[_Pool]] = {}
        self.num_pools = 0
        self.num_members = 0
        seen_pools: set[str] = set()
        episode_splits: dict[int, str] = {}
        datasets: set[str] = set()
        with Path(path).open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_split = row.get("split")
                if not isinstance(row_split, str) or not row_split:
                    raise ValueError(f"Prompt pool at line {line_number} requires a split")
                dataset = row.get("dataset")
                if not isinstance(dataset, str) or not dataset:
                    raise ValueError("Prompt pools require one explicit dataset identifier")
                datasets.add(dataset)
                if len(datasets) != 1:
                    raise ValueError("Filter prompt pools to one dataset before training")
                members = []
                seen_members: set[tuple[int, int, int]] = set()
                for item in row["members"]:
                    fields = [item[k] for k in ("episode_index", "start_frame", "end_frame")]
                    if any(type(v) is not int for v in fields):
                        raise ValueError("Prompt episode IDs and frame boundaries must be integers")
                    episode, start, end = fields
                    if episode < 0 or not 0 <= start < end:
                        raise ValueError("Invalid prompt episode or frame interval")
                    previous = episode_splits.setdefault(episode, row_split)
                    if previous != row_split:
                        raise ValueError(f"Episode {episode} occurs in more than one prompt split")
                    identity = (episode, start, end)
                    if identity in seen_members:
                        raise ValueError("Duplicate member interval inside a prompt pool")
                    seen_members.add(identity)
                    if row_split != split:
                        continue
                    latent_path = item.get("latent_path")
                    if not isinstance(latent_path, str) or not latent_path:
                        raise ValueError("Encode every prompt pool member before training")
                    members.append(PromptMember(episode, start, end, latent_path))
                if row_split != split:
                    continue
                pool_id = row.get("pool_id")
                if not isinstance(pool_id, str) or not pool_id or pool_id in seen_pools:
                    raise ValueError("Selected prompt pools require unique nonempty pool IDs")
                seen_pools.add(pool_id)
                pool = _Pool(members)
                self.num_pools += 1
                self.num_members += len(members)
                if len(pool.episodes) < 2:
                    continue
                for episode in pool.episodes:
                    self._by_episode.setdefault(episode, []).append(pool)
        if not self.num_pools:
            raise ValueError(f"No prompt pools selected for split {split!r}")

    def sample(self, episode_index: int, first_frame: int, last_frame: int) -> str | None:
        """Return a prompt from another episode if the complete target window matches a pool."""
        if first_frame < 0 or last_frame < first_frame:
            raise ValueError("Invalid target training window")
        eligible = [
            pool
            for pool in self._by_episode.get(episode_index, [])
            if any(
                member.start_frame <= first_frame and last_frame < member.end_frame
                for member in pool.members[episode_index]
            )
        ]
        if not eligible:
            return None
        pool = eligible[int(torch.randint(len(eligible), (1,)).item())]
        # Skip the current episode's position without constructing a list of all alternatives.
        position = int(torch.randint(len(pool.episodes) - 1, (1,)).item())
        position += int(position >= pool.positions[episode_index])
        members = pool.members[pool.episodes[position]]
        return members[int(torch.randint(len(members), (1,)).item())].latent_path
