# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Reviewed-only sampling must preserve source indices and action-chunk boundaries."""

import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import _training_dataset
from lerobot.datasets.language_task import RecipeTaskDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.datasets.sampler import compute_sampler_state
from lerobot.datasets.steering_commands import SteeringCommandDataset
from lerobot.scripts.lerobot_train import make_dataloaders


@pytest.fixture
def partial_dataset(monkeypatch, tmp_path):
    metadata = {
        2: {"dataset_from_index": 10, "length": 6},
        5: {"dataset_from_index": 30, "length": 7},
        8: {"dataset_from_index": 50, "length": 3},
    }
    rows = [
        {
            "index": origin + frame,
            "episode_index": episode,
            "frame_index": frame,
            "timestamp": frame / 30,
            "task": "source task",
            "action": torch.tensor([[frame + offset, origin] for offset in range(4)]),
            "action_is_pad": torch.tensor([frame + offset >= length for offset in range(4)]),
        }
        for episode, meta in metadata.items()
        for origin, length in [(meta["dataset_from_index"], meta["length"])]
        for frame in range(length)
    ]
    reader = SimpleNamespace(
        absolute_to_relative_idx={row["index"]: i for i, row in enumerate(rows)},
        num_frames=len(rows),
        get_item=lambda index: rows[index],
        get_items=lambda indices: [rows[i] for i in indices],
    )
    hf = Dataset.from_dict({"task_index": [0] * 6 + [1] * 10})

    def init(dataset, repo_id, **kwargs):
        dataset.repo_id = repo_id
        dataset.episodes = [2, 5, 8]
        dataset.delta_timestamps = {"action": [i / 30 for i in range(4)]}
        dataset.task_recipe = TrainingRecipe.from_dict(
            {"messages": [{"role": "user", "content": "overall task", "stream": "low_level"}]}
        )
        dataset.meta = SimpleNamespace(fps=30, episodes=metadata, features={}, has_language_columns=False)
        dataset.reader = reader

    monkeypatch.setattr(RecipeTaskDataset, "__init__", init)
    monkeypatch.setattr(LeRobotDataset, "_ensure_reader", lambda self: self.reader)
    monkeypatch.setattr(LeRobotDataset, "hf_dataset", property(lambda self: hf))
    manifest = {
        "version": 1,
        "source": {"repo_id": "test/source", "revision": "pinned"},
        "segments": [
            {
                "episode_index": ep,
                "start_frame": start,
                "end_frame": end,
                "review": {"verdict": "accepted", "reviewer": "fixture"},
                "commands": [{"style": "subtask", "text": text, "evidence": "fixture"}],
            }
            for ep, start, end, text in [(2, 1, 3, "reach"), (2, 4, 6, "release"), (5, 2, 5, "lift")]
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    def create(**kwargs):
        return SteeringCommandDataset(
            "test/source", revision="pinned", steering_manifest=str(path), task_probability=0, **kwargs
        )

    return create, rows, path


def test_partial_coverage_requires_explicit_opt_in_and_reports_exclusions(partial_dataset):
    create, rows, _ = partial_dataset
    with pytest.raises(ValueError, match="Missing reviewed steering coverage"):
        create()
    ds = create(skip_uncovered=True)
    assert ds.steering_coverage["covered_frames"] == 7
    assert ds.steering_coverage["excluded_frames"] == 9
    assert not ds.steering_coverage["complete"]
    assert len(ds) == len(rows) == 16
    # Direct access to an unreviewed anchor still fails, even when skipping is enabled.
    with pytest.raises(ValueError, match="Missing reviewed steering commands"):
        ds.__getitems__([0])


def test_sampler_preserves_noncontiguous_episode_row_mapping_and_tail_drop(partial_dataset):
    create, _, _ = partial_dataset
    ds = create(skip_uncovered=True)
    assert ds.make_steering_sampler().indices == [1, 2, 4, 5, 8, 9, 10]
    assert ds.make_steering_sampler(drop_n_last_frames=1).indices == [1, 2, 4, 8, 9, 10]
    with pytest.raises(ValueError, match="No reviewed steering frames remain"):
        ds.make_steering_sampler(drop_n_last_frames=99)
    with pytest.raises(ValueError, match="non-negative"):
        ds.make_steering_sampler(drop_n_last_frames=-1)
    expected = list(ds.make_steering_sampler(shuffle=True, seed=7))
    resumed = ds.make_steering_sampler(shuffle=True, seed=7)
    resumed.load_state_dict(compute_sampler_state(step=2, num_frames=7, batch_size=2, num_processes=1))
    assert list(resumed) == expected[4:]


@pytest.mark.parametrize(
    "max_eval_samples, expected_eval", [(0, [11, 12, 14, 15, 32, 33, 34]), (2, [11, 32])]
)
def test_real_loader_construction_filters_training_and_eval_without_reindexing(
    partial_dataset, max_eval_samples, expected_eval
):
    create, rows, _ = partial_dataset
    ds = create(skip_uncovered=True)
    cfg = SimpleNamespace(
        trainable_config=SimpleNamespace(drop_n_last_frames=0),
        dataset=SimpleNamespace(streaming=False),
        seed=7,
        resume=False,
        num_workers=0,
        batch_size=2,
        max_eval_samples=max_eval_samples,
        prefetch_factor=2,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
    )
    train, evaluation = make_dataloaders(cfg, ds, ds, 0, SimpleNamespace(device_type="cpu"))
    seen = []
    expected_commands = {
        11: "reach",
        12: "reach",
        14: "release",
        15: "release",
        32: "lift",
        33: "lift",
        34: "lift",
    }
    expected_valid_steps = {11: 2, 12: 1, 14: 2, 15: 1, 32: 3, 33: 2, 34: 1}
    by_index = {row["index"]: row for row in rows}
    for batch in train:
        for i, index in enumerate(batch["index"].tolist()):
            seen.append(index)
            assert batch["task"][i] == expected_commands[index]
            assert torch.equal(batch["action"][i], by_index[index]["action"])
            assert int((~batch["action_is_pad"][i]).sum()) == expected_valid_steps[index]
    assert sorted(seen) == sorted(expected_commands)
    assert [i for batch in evaluation for i in batch["index"].tolist()] == expected_eval


def test_skipping_does_not_bypass_empty_coverage_or_required_style_checks(partial_dataset):
    create, _, path = partial_dataset
    with pytest.raises(ValueError, match="Required steering styles absent"):
        create(skip_uncovered=True, required_styles=["trace"])
    manifest = json.loads(path.read_text())
    manifest["segments"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="No reviewed steering frames"):
        create(skip_uncovered=True)


def test_skip_config_requires_manifest_and_eval_does_not_require_training_styles():
    with pytest.raises(ValueError, match="requires a steering_manifest"):
        DatasetConfig(repo_id="test/source", steering_skip_uncovered=True)
    config = DatasetConfig(
        repo_id="test/source",
        steering_manifest="manifest.json",
        steering_skip_uncovered=True,
        steering_required_styles=["trace"],
        task_recipe={"messages": [{"role": "user", "content": "task", "stream": "low_level"}]},
    )
    cfg = SimpleNamespace(dataset=config)
    assert _training_dataset(cfg).keywords["required_styles"] == ["trace"]
    assert _training_dataset(cfg, evaluation=True).keywords["required_styles"] == []
    assert _training_dataset(cfg, evaluation=True).keywords["skip_uncovered"] is True
