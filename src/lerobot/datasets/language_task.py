# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Use language recipes with policies whose language interface is a task string."""

from typing import Any

from lerobot.datasets.recipe import TrainingRecipe
from lerobot.utils.constants import MESSAGES_RENDERED

from .language_render import render_sample
from .lerobot_dataset import LeRobotDataset


def task_from_recipe(sample: dict[str, Any], recipe: TrainingRecipe) -> dict[str, Any]:
    """Render a weighted instruction recipe without changing action targets or source data."""
    rendered = render_sample(
        recipe=recipe,
        persistent=sample.get("language_persistent") or [],
        events=sample.get("language_events") or [],
        t=float(sample["timestamp"]),
        sample_idx=int(sample["index"]),
        task=sample.get("task"),
    )
    if rendered is None or rendered["target_message_indices"]:
        raise ValueError("task_recipe must render action conditioning, without text prediction targets")
    messages = [
        message
        for message, stream in zip(rendered[MESSAGES_RENDERED], rendered["message_streams"], strict=True)
        if stream == "low_level"
    ]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("task_recipe must render exactly one low_level user instruction")
    content = messages[0]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("task_recipe rendered an empty or non-text instruction; check annotation coverage")
    return {**sample, "task": content}


class RecipeTaskDataset(LeRobotDataset):
    """Adapt recipe-rendered instructions to existing task-conditioned action policies."""

    def __init__(self, *args, task_recipe: TrainingRecipe | dict, **kwargs):
        self.task_recipe = (
            TrainingRecipe.from_dict(task_recipe) if isinstance(task_recipe, dict) else task_recipe
        )
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return task_from_recipe(super().__getitem__(idx), self.task_recipe)
