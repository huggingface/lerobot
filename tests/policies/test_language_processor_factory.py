#!/usr/bin/env python

"""Processor-factory behavior needed for language fine-tuning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.policies import factory
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)


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


def _run_training_until_processors(monkeypatch, cfg, stats):
    """Exercise the real train entrypoint, stopping before optimizer/model training."""
    import inspect

    from lerobot.scripts import lerobot_train as trainer

    class ProcessorsReadyError(Exception):
        pass

    cfg.job = SimpleNamespace(is_remote=False)
    cfg.validate = lambda: None
    cfg.parallelism = None
    cfg.wandb = SimpleNamespace(enable=False)
    cfg.seed = None
    cfg.cudnn_deterministic = False
    cfg.checkpoint_format = SimpleNamespace(wants_dcp=False)
    cfg.peft = None
    accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"), wait_for_everyone=lambda: None)
    monkeypatch.setattr(trainer, "make_accelerator", lambda _: accelerator)
    monkeypatch.setattr(trainer.ParallelDims, "from_config", lambda *args: None)
    monkeypatch.setattr(trainer, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(trainer, "is_main_process", lambda: False)
    monkeypatch.setattr(
        trainer,
        "make_train_eval_datasets",
        lambda _: (SimpleNamespace(meta=SimpleNamespace(stats=stats)), None),
    )
    monkeypatch.setattr(trainer, "make_policy", lambda **kwargs: SimpleNamespace(config=cfg.policy))
    # Restore backend globals changed during training setup after each test.
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    pipelines = []

    def build(**kwargs):
        result = factory.make_pre_post_processors(**kwargs)
        pipelines.extend(result)
        return result

    def stop(*args):
        raise ProcessorsReadyError

    monkeypatch.setattr(trainer, "make_pre_post_processors", build)
    monkeypatch.setattr(trainer, "make_optimizer_and_scheduler", stop)
    with pytest.raises(ProcessorsReadyError):
        inspect.unwrap(trainer.train)(cfg)
    assert len(pipelines) == 2
    return pipelines


@pytest.mark.parametrize("resume", [False, True])
def test_training_entrypoint_builds_config_or_restores_saved_stats(tmp_path, resume, monkeypatch):
    config = _act_config()
    pre, post = factory.make_pre_post_processors(config, dataset_stats=_stats(10.0))
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    config.pretrained_path = str(tmp_path)
    cfg = SimpleNamespace(
        trainable_config=config, policy=config, resume=resume, rename_map={}, is_reward_model_training=False
    )
    pre, post = _run_training_until_processors(monkeypatch, cfg, _stats(20.0))
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
    pre.steps = [RenderRuntimeMessagesStep(recipe), RenderTrainingMessagesStep(recipe), *pre.steps]
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    # A runtime config must never replace the recipe stored with the processors.
    config.recipe = TrainingRecipe(messages=[MessageTurn(role="user", content="WRONG", stream="low_level")])
    loaded, _ = factory.make_pre_post_processors(
        config, pretrained_path=str(tmp_path), dataset_stats=_stats(99.0)
    )
    assert isinstance(loaded.steps[0], RenderRuntimeMessagesStep)
    assert isinstance(loaded.steps[1], RenderTrainingMessagesStep)
    assert loaded.steps[0].recipe == recipe
    batch = {"observation.state": torch.tensor([[12.0]])}
    if for_training:
        batch["task"] = "tidy"
        batch["action"] = torch.tensor([[12.0]])
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


def test_finetuning_preserves_statistics_adapted_by_policy_factory(monkeypatch):
    from lerobot.policies.act import processor_act

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
    pre, post = _run_training_until_processors(monkeypatch, cfg, _stats(20.0))
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
    assert pre.steps[0] is step
    assert "messages_rendered" not in pre({"task": "tidy", "language_events": []})


def test_fresh_training_preserves_relative_action_links_and_rename_map(monkeypatch):
    from lerobot.policies.act import processor_act
    from lerobot.processor import RenameObservationsProcessorStep

    original_factory = processor_act.make_act_pre_post_processors
    created = []

    def linked_factory(config, dataset_stats):
        pre, post = original_factory(config, dataset_stats)
        relative = RelativeActionsProcessorStep(enabled=True)
        absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)
        pre.steps = [relative, *pre.steps]
        post.steps = [*post.steps, absolute]
        created.extend([relative, absolute])
        return pre, post

    monkeypatch.setattr(processor_act, "make_act_pre_post_processors", linked_factory)
    config = _act_config()
    config.device = "unavailable-device"
    cfg = SimpleNamespace(
        trainable_config=config,
        policy=config,
        resume=False,
        rename_map={"observation.old_state": "observation.state"},
        is_reward_model_training=False,
    )
    pre, post = _run_training_until_processors(monkeypatch, cfg, _stats(0.0))
    assert pre.steps[0] is created[0]
    assert post.steps[-1] is created[1]
    assert post.steps[-1].relative_step is pre.steps[0]
    assert config.device == "unavailable-device"
    rename = next(step for step in pre.steps if isinstance(step, RenameObservationsProcessorStep))
    assert rename.rename_map == cfg.rename_map
    pre({"observation.state": torch.tensor([[10.0]])})
    torch.testing.assert_close(post(torch.tensor([[1.0]])), torch.tensor([[12.0]]))
