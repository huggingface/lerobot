#!/usr/bin/env python

"""Processor-factory behavior needed for language fine-tuning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.policies import factory
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)


def test_language_finetuning_rebuilds_processors_from_active_config(monkeypatch):
    expected = (object(), object())
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(factory, "_make_processors_from_policy_config", build)
    config = SimpleNamespace(use_language_recipe=True, recipe_path=None)
    stats = {"observation.state": {"mean": 0.0}}
    preprocessor_overrides = {"device_processor": {"device": "cpu"}}
    postprocessor_overrides = {"absolute_actions_processor": {"enabled": True}}

    result = factory.make_pre_post_processors(
        config,
        pretrained_path="old-checkpoint",
        dataset_stats=stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        rebuild_pretrained_processors=True,
    )

    assert result is expected
    assert calls == [
        {
            "config": config,
            "dataset_stats": stats,
            "dataset_meta": None,
            "preprocessor_overrides": preprocessor_overrides,
            "postprocessor_overrides": postprocessor_overrides,
        }
    ]


def test_rebuilt_pipeline_applies_overrides_and_reconnects_relative_actions():
    preprocessor = PolicyProcessorPipeline(
        steps=[
            DeviceProcessorStep(device="cpu"),
            RelativeActionsProcessorStep(),
        ]
    )
    postprocessor = PolicyProcessorPipeline(steps=[AbsoluteActionsProcessorStep()])

    preprocessor = factory._apply_processor_overrides(
        preprocessor,
        {
            "device_processor": {"device": "cpu", "float_dtype": "float64"},
            "relative_actions_processor": {
                "enabled": True,
                "exclude_joints": ["gripper"],
                "action_names": ["shoulder", "gripper"],
            },
        },
    )
    postprocessor = factory._apply_processor_overrides(
        postprocessor,
        {"absolute_actions_processor": {"enabled": True}},
    )
    factory._reconnect_relative_absolute_steps(preprocessor, postprocessor)

    device_step, relative_step = preprocessor.steps
    absolute_step = postprocessor.steps[0]
    assert device_step.device == "cpu"
    assert device_step.float_dtype == "float64"
    assert relative_step.enabled
    assert relative_step.exclude_joints == ["gripper"]
    assert relative_step.action_names == ["shoulder", "gripper"]
    assert absolute_step.enabled
    assert absolute_step.relative_step is relative_step


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
