import bisect
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from lerobot.datasets.video_utils import decode_video_frames

from .libero_safety import REPO_ID, load_libero_safety_contract


def build_episode_safe_action_chunk(rows, local_index, chunk_size):
    """Return a physical action chunk padded strictly within one episode."""
    if not 0 <= local_index < len(rows):
        raise IndexError(local_index)
    valid = min(chunk_size, len(rows) - local_index)
    actions = [
        torch.as_tensor(rows[local_index + offset]["actions"], dtype=torch.float32) for offset in range(valid)
    ]
    actions.extend(torch.zeros_like(actions[0]) for _ in range(chunk_size - valid))
    return torch.stack(actions), torch.arange(chunk_size) >= valid


class LiberoSafetyV21Dataset:
    """Lazy per-episode reader; no full v2.1 conversion or dataset duplication."""

    def __init__(
        self,
        repo_id=REPO_ID,
        episodes=None,
        chunk_size=16,
        revision="main",
        cache_dir=None,
        task_indices=None,
    ):
        self.repo_id, self.revision, self.cache_dir = repo_id, revision, cache_dir
        self.contract = load_libero_safety_contract(repo_id, revision, cache_dir)
        self.chunk_size = chunk_size
        path = hf_hub_download(
            repo_id, "meta/episodes.jsonl", repo_type="dataset", revision=revision, cache_dir=cache_dir
        )
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        selected = set(range(len(rows)) if episodes is None else episodes)
        selected_tasks = (
            None if task_indices is None else {self.contract.tasks[index] for index in task_indices}
        )
        self.episode_rows = [
            row
            for row in rows
            if int(row["episode_index"]) in selected
            and (selected_tasks is None or row["tasks"][0] in selected_tasks)
        ]
        if not self.episode_rows:
            raise ValueError("LIBERO-Safety selection contains no episodes")
        self.selected_episode_indices = [int(row["episode_index"]) for row in self.episode_rows]
        self.ends, total = [], 0
        for row in self.episode_rows:
            total += int(row["length"])
            self.ends.append(total)
        features = dict(self.contract.features)
        features["action"] = features.pop("actions")
        stats = {
            key: {name: torch.tensor(value) for name, value in values.items()}
            for key, values in self.contract.stats.items()
        }
        stats["action"] = stats.pop("actions")
        starts = torch.tensor([0, *self.ends[:-1]])
        self.meta = SimpleNamespace(
            repo_id=repo_id,
            features=features,
            stats=stats,
            fps=self.contract.fps,
            total_episodes=len(self.episode_rows),
            total_frames=total,
            camera_keys=["observation.image", "observation.wrist_image"],
            has_language_columns=True,
            episodes={
                "dataset_from_index": starts,
                "dataset_to_index": torch.tensor(self.ends),
                "tasks": [row["tasks"] for row in self.episode_rows],
            },
        )
        self._cached_episode, self._cached_rows = None, None
        self._cached_video_paths = {}
        # `dataset_from_index`/`dataset_to_index` above are already relative to this
        # (possibly episode-filtered) instance's own compacted frame numbering, so from
        # `EpisodeAwareSampler`'s point of view this dataset *is* the whole world: every
        # row in its own from/to arrays is in play, and no absolute->relative remapping
        # is needed. This mirrors LeRobotDataset's behavior when constructed directly
        # with an `episodes=` subset (a fresh instance, not a filtered view).
        self.episodes = None
        self.absolute_to_relative_idx = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rows)

    def _rows(self, episode_index):
        if self._cached_episode != episode_index:
            chunk = episode_index // 1000
            path = hf_hub_download(
                self.repo_id,
                f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet",
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            self._cached_rows = pq.read_table(path).to_pylist()
            self._cached_episode = episode_index
        return self._cached_rows

    def _frame(self, episode_index, key, timestamp):
        chunk = episode_index // 1000
        cache_key = (episode_index, key)
        path = self._cached_video_paths.get(cache_key)
        if path is None:
            path = hf_hub_download(
                self.repo_id,
                f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4",
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            self._cached_video_paths[cache_key] = path
        return decode_video_frames(path, [timestamp], tolerance_s=1 / self.contract.fps, return_uint8=True)[0]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.ends, index)
        start = 0 if position == 0 else self.ends[position - 1]
        local = index - start
        episode = int(self.episode_rows[position]["episode_index"])
        rows = self._rows(episode)
        row = rows[local]
        actions, action_is_pad = build_episode_safe_action_chunk(rows, local, self.chunk_size)
        task_index, timestamp = int(row["task_index"]), float(row["timestamp"])
        return {
            "observation.image": self._frame(episode, "observation.image", timestamp),
            "observation.wrist_image": self._frame(episode, "observation.wrist_image", timestamp),
            "observation.state": torch.tensor(row["observation.state"]),
            "action": actions,
            "action_is_pad": action_is_pad,
            "task_index": torch.tensor(task_index),
            "task": self.contract.tasks[task_index],
            "timestamp": torch.tensor(timestamp),
            "frame_index": torch.tensor(row["frame_index"]),
            "episode_index": torch.tensor(episode),
        }
