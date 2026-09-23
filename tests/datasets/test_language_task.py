# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
from collections import Counter
from pathlib import Path

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.language_task import task_from_recipe
from lerobot.datasets.recipe import TrainingRecipe


@pytest.fixture
def recipe():
    return TrainingRecipe.from_yaml(Path(__file__).parents[2] / "examples/rebot_agent/steerable_80_20.yaml")


def sample(index=0, timestamp=1.0):
    return {
        "index": index,
        "timestamp": timestamp,
        "task": "put everything in the bin",
        "action": [1, 2],
        "language_persistent": [
            {"role": "assistant", "style": "subtask", "content": "pick the tape", "timestamp": 0.0},
            {"role": "assistant", "style": "subtask", "content": "pick the remote", "timestamp": 10.0},
        ],
    }


def test_recipe_blend_conditions_actions_at_both_abstraction_levels(recipe):
    counts = Counter(task_from_recipe(sample(i), recipe)["task"] for i in range(10000))
    assert 7800 < counts["pick the tape"] < 8200
    assert 1800 < counts["put everything in the bin"] < 2200
    assert "pick the remote" not in counts


def test_task_recipe_preserves_original_sample_and_actions(recipe):
    item = sample()
    result = task_from_recipe(item, recipe)
    assert item["task"] == "put everything in the bin"
    assert result["task"] == "pick the tape"
    assert result["action"] is item["action"]
    assert task_from_recipe(sample(timestamp=10), recipe)["task"] == "pick the remote"


def test_task_recipe_uses_existing_task_paraphrases():
    recipe = TrainingRecipe.from_dict(
        {"messages": [{"role": "user", "content": "${task}", "stream": "low_level"}]}
    )
    item = sample()
    assert task_from_recipe(item, recipe)["task"] == item["task"]
    paraphrases = {"collect the objects in the bin", "put all objects into the bin"}
    item["language_persistent"].extend(
        {"role": "user", "style": "task_aug", "content": text, "timestamp": 0.0}
        for text in sorted(paraphrases)
    )
    assert {task_from_recipe({**item, "index": i}, recipe)["task"] for i in range(50)} == paraphrases
    assert item["task"] == "put everything in the bin"


def test_missing_subtask_cannot_silently_train_generic_task(recipe):
    item = sample()
    item["language_persistent"] = []
    with pytest.raises(ValueError):
        task_from_recipe(item, recipe)


def test_recipe_streaming_is_explicitly_unsupported(recipe):
    with pytest.raises(ValueError, match="non-streaming"):
        DatasetConfig(repo_id="test/data", task_recipe=recipe, streaming=True)
