#!/usr/bin/env python

"""Processor-factory behavior needed for language fine-tuning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.policies import factory


def test_language_finetuning_rebuilds_processors_from_active_config(monkeypatch):
    expected = (object(), object())
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(factory, "_make_processors_from_policy_config", build)
    config = SimpleNamespace(use_language_recipe=True, recipe_path=None)
    stats = {"observation.state": {"mean": 0.0}}

    result = factory.make_pre_post_processors(
        config,
        pretrained_path="old-checkpoint",
        dataset_stats=stats,
        rebuild_pretrained_processors=True,
    )

    assert result is expected
    assert calls == [{"config": config, "dataset_stats": stats, "dataset_meta": None}]


def test_language_rollout_loads_checkpoint_processors_even_when_dataset_stats_are_present(monkeypatch):
    saved = (SimpleNamespace(steps=[]), SimpleNamespace(steps=[]))
    load = MagicMock(side_effect=saved)
    rebuild = MagicMock()
    monkeypatch.setattr(factory.PolicyProcessorPipeline, "from_pretrained", load)
    monkeypatch.setattr(factory, "_make_processors_from_policy_config", rebuild)

    result = factory.make_pre_post_processors(
        SimpleNamespace(use_language_recipe=True, recipe_path="recipe.yaml"),
        pretrained_path="checkpoint",
        dataset_stats={"action": {"mean": 42.0}},
    )

    assert result == saved
    assert load.call_count == 2
    rebuild.assert_not_called()


def test_rebuilding_pretrained_processors_requires_training_stats():
    with pytest.raises(ValueError, match="non-empty training dataset statistics"):
        factory.make_pre_post_processors(
            SimpleNamespace(),
            pretrained_path="checkpoint",
            dataset_stats={},
            rebuild_pretrained_processors=True,
        )
