#!/usr/bin/env python

"""Processor-factory behavior needed for language fine-tuning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.policies import factory
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)


def test_language_finetuning_rebuilds_processors_from_active_config(monkeypatch):
    expected = (SimpleNamespace(steps=[]), SimpleNamespace(steps=[]))
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
        pretrained_path=None,
        dataset_stats=stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        for_training=True,
    )

    assert result == expected
    assert calls == [
        {
            "config": config,
            "dataset_stats": stats,
            "dataset_meta": None,
            "preprocessor_overrides": preprocessor_overrides,
            "postprocessor_overrides": postprocessor_overrides,
        }
    ]


@pytest.mark.parametrize("rebuild_postprocessor", [False, True])
def test_rebuilt_pipeline_applies_overrides_and_reconnects_relative_actions(rebuild_postprocessor):
    preprocessor = PolicyProcessorPipeline(
        steps=[
            DeviceProcessorStep(device="cpu"),
            RelativeActionsProcessorStep(),
        ]
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=preprocessor.steps[1])]
    )

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
        {"absolute_actions_processor": {"enabled": True}} if rebuild_postprocessor else None,
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

    preprocessor({"observation.state": torch.tensor([[10.0, 20.0]])})
    output = postprocessor({"action": torch.tensor([[1.0, 2.0]])})
    torch.testing.assert_close(output["action"], torch.tensor([[11.0, 2.0]]))


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


def _act_config():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        device="cpu",
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(1,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
        normalization_mapping={"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD},
    )


def _stats(mean):
    return {
        key: {"mean": torch.tensor([mean]), "std": torch.tensor([2.0])}
        for key in ("observation.state", "action")
    }


@pytest.mark.parametrize("resume", [False, True])
def test_training_entrypoint_builds_config_or_restores_saved_stats(tmp_path, resume):
    from lerobot.scripts.lerobot_train import _make_training_processors

    config = _act_config()
    pre, post = factory.make_pre_post_processors(config, dataset_stats=_stats(10.0))
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    config.pretrained_path = str(tmp_path)
    cfg = SimpleNamespace(
        trainable_config=config, policy=config, resume=resume, rename_map={}, is_reward_model_training=False
    )
    pre, post = _make_training_processors(
        cfg, SimpleNamespace(config=config), SimpleNamespace(stats=_stats(20.0)), torch.device("cpu")
    )
    result = pre({"observation.state": torch.tensor([[22.0]])})
    torch.testing.assert_close(result["observation.state"], torch.tensor([[6.0 if resume else 1.0]]))
    torch.testing.assert_close(post(torch.tensor([[1.0]])), torch.tensor([[12.0 if resume else 22.0]]))


@pytest.mark.parametrize("for_training", [False, True])
def test_checkpoint_renderer_uses_saved_recipe_and_stats(tmp_path, for_training):
    from lerobot.language.recipe import MessageTurn, TrainingRecipe
    from lerobot.processor import RenderRuntimeMessagesStep, RenderTrainingMessagesStep

    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="Saved goal: ${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="high_level", target=True),
        ]
    )
    config = _act_config()
    pre, post = factory.make_pre_post_processors(config, dataset_stats=_stats(10.0))
    pre.steps = [RenderTrainingMessagesStep(recipe), *pre.steps]
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    # A runtime config must never replace the recipe stored with the processors.
    config.recipe = TrainingRecipe(messages=[MessageTurn(role="user", content="WRONG", stream="low_level")])
    loaded, _ = factory.make_pre_post_processors(
        config, pretrained_path=str(tmp_path), for_training=for_training, dataset_stats=_stats(99.0)
    )
    assert isinstance(
        loaded.steps[0], RenderTrainingMessagesStep if for_training else RenderRuntimeMessagesStep
    )
    assert loaded.steps[0].recipe == recipe
    batch = {"observation.state": torch.tensor([[12.0]])}
    if for_training:
        batch["task"] = "tidy"
    else:
        batch.update(query_kind="next_subtask", query_text="tidy")
    result = loaded(batch)
    torch.testing.assert_close(result["observation.state"], torch.tensor([[1.0]]))
    assert result["messages_rendered"] == [
        [{"role": "user", "content": "tidy" if for_training else "Saved goal: tidy"}]
    ]
    assert "messages" not in result
    if not for_training:
        assert "target_message_indices" not in result


def test_fresh_overrides_preserve_nonserialized_training_context():
    from lerobot.processor import RenderTrainingMessagesStep

    context = object()
    pipeline = PolicyProcessorPipeline(
        steps=[RenderTrainingMessagesStep(dataset_ctx=context), DeviceProcessorStep(device="cpu")]
    )
    configured = factory._apply_processor_overrides(pipeline, {"device_processor": {"device": "cpu"}})
    assert configured.steps[0].dataset_ctx is context


def test_finetuning_preserves_statistics_adapted_by_policy_factory(monkeypatch):
    from lerobot.policies.act import processor_act
    from lerobot.scripts.lerobot_train import _make_training_processors

    original_factory = processor_act.make_act_pre_post_processors

    def adapted_factory(config, dataset_stats):
        # Model a policy-specific normalization transform with real processor state.
        adapted = {key: {**stats, "mean": stats["mean"] + 10.0} for key, stats in dataset_stats.items()}
        return original_factory(config, adapted)

    monkeypatch.setattr(processor_act, "make_act_pre_post_processors", adapted_factory)
    config = _act_config()
    config.pretrained_path = "unused-model-weights-path"
    cfg = SimpleNamespace(
        trainable_config=config, policy=config, resume=False, rename_map={}, is_reward_model_training=False
    )
    pre, post = _make_training_processors(
        cfg, SimpleNamespace(config=config), SimpleNamespace(stats=_stats(20.0)), torch.device("cpu")
    )
    torch.testing.assert_close(
        pre({"observation.state": torch.tensor([[32.0]])})["observation.state"], torch.tensor([[1.0]])
    )
    torch.testing.assert_close(post(torch.tensor([[1.0]])), torch.tensor([[32.0]]))


def test_disabled_recipe_training_retains_runtime_only_renderer():
    from lerobot.language.recipe import MessageTurn, TrainingRecipe
    from lerobot.processor import RenderRuntimeMessagesStep

    recipe = TrainingRecipe(messages=[MessageTurn(role="user", content="${task}", stream="low_level")])
    step = RenderRuntimeMessagesStep(recipe)
    pre = PolicyProcessorPipeline(steps=[step])
    factory._set_message_rendering_mode(pre, for_training=True)
    assert pre.steps[0] is step
    assert "messages_rendered" not in pre({"task": "tidy", "language_events": []})
