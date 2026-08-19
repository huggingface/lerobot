from __future__ import annotations

import json
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "LIBERO-Safety/libero_safety"


@dataclass(frozen=True)
class LiberoSafetyContract:
    codebase_version: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    fps: int
    features: dict
    stats: dict
    tasks: dict[int, str]


def load_libero_safety_contract(repo_id=REPO_ID, revision=None, cache_dir=None):
    def download(filename):
        return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir)

    info = json.loads(Path(download("meta/info.json")).read_text())
    stats = json.loads(Path(download("meta/stats.json")).read_text())
    tasks = {}
    for line in Path(download("meta/tasks.jsonl")).read_text().splitlines():
        row = json.loads(line)
        tasks[int(row["task_index"])] = row["task"]
    required = {
        "observation.image",
        "observation.wrist_image",
        "observation.state",
        "actions",
        "task_index",
        "episode_index",
    }
    missing = required - info["features"].keys()
    if missing:
        raise ValueError(f"LIBERO-Safety schema missing required features: {sorted(missing)}")
    if info["features"]["observation.state"]["shape"] != [8]:
        raise ValueError("LIBERO-Safety state must have shape [8]")
    if info["features"]["actions"]["shape"] != [7]:
        raise ValueError("LIBERO-Safety action must have shape [7]")
    return LiberoSafetyContract(
        info["codebase_version"],
        info["total_episodes"],
        info["total_frames"],
        info["total_tasks"],
        info["fps"],
        info["features"],
        stats,
        tasks,
    )


class LiberoSafetySampleAdapter:
    """Zero-copy key/task adapter around a streaming or cached v2.1 sample source."""

    def __init__(self, dataset, contract: LiberoSafetyContract):
        self.dataset = dataset
        self.contract = contract

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["action"] = sample.pop("actions")
        task_index = int(torch.as_tensor(sample["task_index"]).item())
        try:
            sample["task"] = self.contract.tasks[task_index]
        except KeyError as error:
            raise KeyError(f"Unknown LIBERO-Safety task_index: {task_index}") from error
        return sample


class LiberoSafetyTrainingDataset(LiberoSafetySampleAdapter):
    """Training-compatible zero-copy view with canonical action metadata."""

    def __init__(self, dataset, contract):
        super().__init__(dataset, contract)
        self.meta = copy(dataset.meta)
        self.meta.features = dict(dataset.meta.features)
        self.meta.features["action"] = self.meta.features.pop("actions")
        self.meta.stats = dict(dataset.meta.stats)
        self.meta.stats["action"] = self.meta.stats.pop("actions")

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def episode_safe_window(samples, start: int, chunk_size: int):
    """Build a padded action chunk without ever crossing an episode boundary."""
    first = samples[start]
    episode = int(torch.as_tensor(first["episode_index"]).item())
    actions = []
    for index in range(start, min(start + chunk_size, len(samples))):
        sample = samples[index]
        if int(torch.as_tensor(sample["episode_index"]).item()) != episode:
            break
        actions.append(torch.as_tensor(sample.get("action", sample.get("actions"))))
    if not actions:
        raise ValueError("Cannot build an empty action chunk")
    valid = len(actions)
    actions.extend([torch.zeros_like(actions[0])] * (chunk_size - valid))
    padding = torch.arange(chunk_size) >= valid
    return torch.stack(actions), padding


def task_split(contract: LiberoSafetyContract, validation_task_indices=(13, 14)):
    validation = set(validation_task_indices)
    unknown = validation - contract.tasks.keys()
    if unknown:
        raise ValueError(f"Unknown validation tasks: {sorted(unknown)}")
    return {
        "train": [index for index in contract.tasks if index not in validation],
        "validation": sorted(validation),
    }
