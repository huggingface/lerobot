#!/usr/bin/env python

"""Processor-factory behavior needed for language fine-tuning."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.configs.train import CheckpointFormat
from lerobot.policies import factory
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DataProcessorPipeline,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.transforms import ImageTransformsConfig


def test_language_rollout_loads_checkpoint_processors_even_when_dataset_stats_are_present(monkeypatch):
    saved = (PolicyProcessorPipeline(steps=[]), PolicyProcessorPipeline(steps=[]))
    load = MagicMock(side_effect=saved)
    rebuild = MagicMock()
    # Patch the concrete class so monkeypatch can restore the classmethod at teardown.
    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", load)
    monkeypatch.setattr(factory, "_make_processors_from_policy_config", rebuild)

    config = _act_config()
    config.recipe = {"messages": []}
    config.recipe_path = "recipe.yaml"
    result = factory.make_pre_post_processors(
        config,
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


@pytest.fixture
def dataset_dependencies():
    pytest.importorskip("datasets", reason="recipes require lerobot[dataset]")
    pytest.importorskip("av", reason="recipes require lerobot[dataset]")


@pytest.fixture
def training_dependencies(dataset_dependencies, restore_torch_backend_flags):
    pytest.importorskip("accelerate", reason="training setup requires lerobot[training]")


def _run_training_until_processors(monkeypatch, cfg, stats, *, main_process=False):
    """Exercise the real train entrypoint, stopping before optimizer/model training."""
    import inspect

    from lerobot.scripts import lerobot_train as trainer

    class ProcessorsReadyError(Exception):
        pass

    cfg.job = SimpleNamespace(is_remote=False)
    cfg.validate = lambda: None
    cfg.to_dict = lambda: {}
    cfg.parallelism = None
    cfg.trackio = SimpleNamespace(enable=False)
    cfg.seed = None
    cfg.cudnn_deterministic = False
    cfg.tf32 = True
    cfg.checkpoint_format = CheckpointFormat.SAFETENSORS
    cfg.peft = None
    cfg.dataset = SimpleNamespace(image_transforms=ImageTransformsConfig())
    accelerator = SimpleNamespace(
        num_processes=1, process_index=0, device=torch.device("cpu"), wait_for_everyone=lambda: None
    )
    monkeypatch.setattr(trainer, "make_accelerator", lambda _: accelerator)
    monkeypatch.setattr(trainer.ParallelDims, "from_config", lambda *args: None)
    monkeypatch.setattr(trainer, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(trainer, "is_main_process", lambda: main_process)
    monkeypatch.setattr(
        trainer,
        "make_train_eval_datasets",
        lambda _: (SimpleNamespace(meta=SimpleNamespace(stats=stats)), None),
    )
    monkeypatch.setattr(trainer, "make_policy", lambda **kwargs: SimpleNamespace(config=cfg.policy))
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


@pytest.mark.dataset
@pytest.mark.parametrize("recipe_mode", ["absent", "disabled", "builtin", "yaml"])
@pytest.mark.parametrize("main_process", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_training_entrypoint_only_rebuilds_for_language_finetuning(
    tmp_path, resume, recipe_mode, main_process, monkeypatch, caplog, training_dependencies
):
    config = _act_config()
    pre, post = factory.make_pre_post_processors(config, dataset_stats=_stats(10.0))
    checkpoint_dir = tmp_path / "pretrained_model"
    pre.save_pretrained(checkpoint_dir)
    post.save_pretrained(checkpoint_dir)
    config.pretrained_path = str(checkpoint_dir)
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe, resolve_recipe_override

    recipe = TrainingRecipe(messages=[MessageTurn(role="user", content="${task}", stream="low_level")])
    if recipe_mode == "yaml":
        recipe_path = tmp_path / "active-recipe.yaml"
        recipe_path.write_text("messages:\n  - {role: user, content: '${task}', stream: low_level}\n")
        config.recipe = resolve_recipe_override(None, recipe_path)
    elif recipe_mode != "absent":
        config.recipe = recipe if recipe_mode == "builtin" else None
    cfg = SimpleNamespace(
        trainable_config=config,
        policy=config,
        resume=resume,
        checkpoint_path=tmp_path if resume else None,
        rename_map={},
        is_reward_model_training=False,
    )
    load = MagicMock(wraps=factory.PolicyProcessorPipeline.from_pretrained)
    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", load)
    pre, post = _run_training_until_processors(monkeypatch, cfg, _stats(20.0), main_process=main_process)
    uses_checkpoint = resume or recipe_mode in ("absent", "disabled")
    warnings = [
        record.getMessage() for record in caplog.records if "saved processors from" in record.getMessage()
    ]
    assert len(warnings) == int(main_process and not uses_checkpoint)
    if warnings:
        assert str(tmp_path) in warnings[0]
    assert load.call_count == (2 if uses_checkpoint else 0)
    result = pre({"observation.state": torch.tensor([[22.0]])})
    torch.testing.assert_close(result["observation.state"], torch.tensor([[6.0 if resume else 1.0]]))
    torch.testing.assert_close(post(torch.tensor([[1.0]])), torch.tensor([[12.0 if resume else 22.0]]))


@pytest.mark.dataset
@pytest.mark.parametrize("for_training", [False, True])
def test_checkpoint_renderer_uses_saved_recipe_and_stats(tmp_path, for_training, dataset_dependencies):
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe
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


@pytest.mark.dataset
def test_finetuning_preserves_statistics_adapted_by_policy_factory(monkeypatch, training_dependencies):
    from lerobot.policies.act import processor_act

    original_factory = processor_act.make_act_pre_post_processors

    def adapted_factory(config, dataset_stats):
        # Model a policy-specific normalization transform with real processor state.
        adapted = {key: {**stats, "mean": stats["mean"] + 10.0} for key, stats in dataset_stats.items()}
        return original_factory(config, adapted)

    monkeypatch.setattr(processor_act, "make_act_pre_post_processors", adapted_factory)
    config = _act_config()
    config.pretrained_path = "unused-model-weights-path"
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe

    config.recipe = TrainingRecipe(messages=[MessageTurn(role="user", content="${task}", stream="low_level")])
    cfg = SimpleNamespace(
        trainable_config=config,
        policy=config,
        resume=False,
        checkpoint_path=None,
        rename_map={},
        is_reward_model_training=False,
    )
    pre, post = _run_training_until_processors(monkeypatch, cfg, _stats(20.0))
    torch.testing.assert_close(
        pre({"observation.state": torch.tensor([[32.0]])})["observation.state"], torch.tensor([[1.0]])
    )
    torch.testing.assert_close(post(torch.tensor([[1.0]])), torch.tensor([[32.0]]))


def test_pipeline_without_recipe_preserves_training_inputs_and_renders_vqa():
    from lerobot.processor import RenderRuntimeMessagesStep, RenderTrainingMessagesStep

    pre = PolicyProcessorPipeline(steps=[RenderRuntimeMessagesStep(), RenderTrainingMessagesStep()])
    training = {"task": "tidy", "language_events": [], "action": torch.tensor([[1.0]])}
    output = pre(training)
    assert "messages_rendered" not in output
    assert output["task"] == training["task"]
    torch.testing.assert_close(output["action"], training["action"])
    assert pre({"query_kind": "vqa", "query_text": "What is on the table?"})["messages_rendered"] == [
        {"role": "user", "content": "What is on the table?"}
    ]


@pytest.mark.dataset
def test_fresh_training_preserves_relative_action_links_and_batch_renaming(
    monkeypatch, caplog, training_dependencies
):
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe
    from lerobot.policies.act import processor_act
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

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
    config.recipe = TrainingRecipe(messages=[MessageTurn(role="user", content="${task}", stream="low_level")])
    cfg = SimpleNamespace(
        trainable_config=config,
        policy=config,
        resume=False,
        checkpoint_path=None,
        rename_map={"observation.old_state": "observation.state"},
        is_reward_model_training=False,
    )
    pre, post = _run_training_until_processors(monkeypatch, cfg, _stats(0.0), main_process=True)
    assert not any("saved processors from" in record.getMessage() for record in caplog.records)
    assert pre.steps[0] is created[0]
    assert post.steps[-1] is created[1]
    assert post.steps[-1].relative_step is pre.steps[0]
    processed = _preprocess_dataset_batch(
        {"observation.old_state": torch.tensor([[10.0]])}, [], cfg.rename_map, pre
    )
    torch.testing.assert_close(processed["observation.state"], torch.tensor([[5.0]]))
    torch.testing.assert_close(post(torch.tensor([[1.0]])), torch.tensor([[12.0]]))


@pytest.mark.parametrize("consent,override", [(False, None), (True, None), (False, True), (True, False)])
def test_policy_consent_reaches_both_loaded_processors(monkeypatch, consent, override):
    config = _act_config()
    config.trust_remote_code = consent
    load = MagicMock(side_effect=[SimpleNamespace(steps=[]), SimpleNamespace(steps=[])])
    monkeypatch.setattr(DataProcessorPipeline, "from_pretrained", load)
    kwargs = {} if override is None else {"trust_remote_code": override}
    factory.make_pre_post_processors(config, pretrained_path="user/checkpoint", **kwargs)
    assert load.call_count == 2
    assert config.trust_remote_code is consent
    assert all(
        call.kwargs["trust_remote_code"] is (consent if override is None else override)
        for call in load.call_args_list
    )


@pytest.mark.parametrize("consent,override", [(False, None), (True, None), (False, True), (True, False)])
@pytest.mark.parametrize("policy_type", ["pi0_fast", pytest.param("fineart_vla", marks=pytest.mark.dataset)])
def test_fresh_fast_processors_receive_consent(monkeypatch, consent, override, policy_type):
    if policy_type == "fineart_vla":
        pytest.importorskip("datasets")
    from lerobot.processor import tokenizer_processor

    monkeypatch.setattr(tokenizer_processor, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_processor, "AutoProcessor", MagicMock())
    tokenizer = MagicMock()
    monkeypatch.setattr(tokenizer_processor, "AutoTokenizer", tokenizer)
    config = factory.make_policy_config(policy_type, device="cpu", trust_remote_code=consent)
    config.validate_features()
    kwargs = {} if override is None else {"trust_remote_code": override}
    pre, _ = factory.make_pre_post_processors(config, **kwargs)
    step = next(step for step in pre.steps if type(step).__name__ == "ActionTokenizerProcessorStep")
    assert step.trust_remote_code is (consent if override is None else override)
    assert config.trust_remote_code is consent


@pytest.mark.parametrize("consent", [False, True])
def test_wallx_model_tokenizer_loaders_receive_consent(monkeypatch, consent):
    from lerobot.policies.wall_x import modeling_wall_x as wallx

    load = MagicMock(side_effect=[MagicMock(), RuntimeError("stop before model construction")])
    monkeypatch.setattr(wallx, "AutoProcessor", SimpleNamespace(from_pretrained=load))
    monkeypatch.setattr(wallx, "Qwen2_5_VLMoEModel", SimpleNamespace(_require_eager_attention=lambda _: None))
    with pytest.raises(RuntimeError, match="stop before model construction"):
        wallx.Qwen2_5_VLMoEForAction.from_pretrained(
            "user/model",
            config=SimpleNamespace(),
            action_tokenizer_path="user/fast@" + "a" * 40,
            trust_remote_code=consent,
        )
    assert load.call_count == 2
    assert all(call.kwargs["trust_remote_code"] is consent for call in load.call_args_list)
    assert load.call_args.kwargs["revision"] == "a" * 40
    assert load.call_args.kwargs["code_revision"] == "a" * 40


@pytest.mark.parametrize("consent", [False, True])
@pytest.mark.parametrize(
    "reference", ["physical-intelligence/fast", "physical-intelligence/fast@" + "a" * 40]
)
def test_fineart_fitter_fallback_requires_consent(monkeypatch, consent, reference):
    from lerobot.policies.fineart_vla import fit_fast_tokenizer as fitter

    loader = MagicMock(side_effect=ValueError("custom tokenizer code"))
    fallback = MagicMock()
    monkeypatch.setattr(fitter, "AutoProcessor", SimpleNamespace(from_pretrained=loader), raising=False)
    monkeypatch.setattr(fitter, "get_class_from_dynamic_module", fallback, raising=False)
    if consent:
        assert fitter._load_fast_fitter(reference, trust_remote_code=True) is fallback.return_value
    else:
        with pytest.raises(ValueError, match="custom tokenizer code"):
            fitter._load_fast_fitter(reference)
        fallback.assert_not_called()
    assert loader.call_args.kwargs["trust_remote_code"] is consent
    if "@" in reference:
        assert loader.call_args.kwargs["code_revision"] == "a" * 40
        if consent:
            assert fallback.call_args.kwargs["revision"] == "a" * 40
            assert fallback.call_args.kwargs["code_revision"] == "a" * 40


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_pi0_fast_pins_model_and_processor_tokenizers(monkeypatch, tmp_path, revision):
    pytest.importorskip("transformers")
    pytest.importorskip("scipy")
    from lerobot.policies.pi0_fast import modeling_pi0_fast as modeling
    from lerobot.processor import tokenizer_processor

    action_loader = MagicMock()
    text_loader = MagicMock()
    monkeypatch.setattr(modeling, "AutoProcessor", SimpleNamespace(from_pretrained=action_loader))
    monkeypatch.setattr(modeling, "AutoTokenizer", SimpleNamespace(from_pretrained=text_loader))
    monkeypatch.setattr(tokenizer_processor, "AutoProcessor", modeling.AutoProcessor)
    monkeypatch.setattr(tokenizer_processor, "AutoTokenizer", modeling.AutoTokenizer)
    monkeypatch.setattr(modeling, "PI0FastPytorch", MagicMock(side_effect=RuntimeError("stop before core")))
    config = factory.make_policy_config(
        "pi0_fast",
        device="cpu",
        trust_remote_code=True,
        action_tokenizer_name="user/fast" + (f"@{revision}" if revision else ""),
        text_tokenizer_name="user/text" + ("@" + "b" * 40 if revision else ""),
    )
    config.save_pretrained(tmp_path)
    config = type(config).from_pretrained(tmp_path, trust_remote_code=True)
    with pytest.raises(RuntimeError, match="stop before core"):
        modeling.PI0FastPolicy(config)
    factory.make_pre_post_processors(config)
    assert action_loader.call_count == 2
    assert text_loader.call_count == 3
    for loader, expected in [
        (action_loader, revision),
        (text_loader, None if revision is None else "b" * 40),
    ]:
        for call in loader.call_args_list:
            assert call.kwargs.get("revision") == expected
            assert call.kwargs.get("code_revision") == expected


def test_pi0_fast_pins_saved_processors_without_mutating_overrides(monkeypatch):
    from lerobot.policies.pi0_fast import processor_pi0_fast as processors

    config = factory.make_policy_config(
        "pi0_fast",
        device="cpu",
        action_tokenizer_name="user/fast@" + "a" * 40,
        text_tokenizer_name="user/text@" + "b" * 40,
    )
    load = MagicMock(return_value=(None, None))
    monkeypatch.setattr(processors, "load_pretrained_policy_processors", load)
    overrides = {"tokenizer_processor": {"max_length": 42}}
    factory.make_pre_post_processors(
        config, pretrained_path="user/checkpoint", preprocessor_overrides=overrides
    )
    assert load.call_args.kwargs["preprocessor_overrides"] == {
        "tokenizer_processor": {"max_length": 42, "tokenizer_name": "user/text@" + "b" * 40},
        "action_tokenizer_processor": {
            "action_tokenizer_name": "user/fast@" + "a" * 40,
            "paligemma_tokenizer_name": "user/text@" + "b" * 40,
        },
    }
    assert overrides == {"tokenizer_processor": {"max_length": 42}}


@pytest.mark.parametrize("policy_type", ["act", "pi0_fast"])
@pytest.mark.parametrize(
    "config_consent,consent", [(False, None), (True, None), (True, False), (False, True)]
)
def test_explicit_policy_consent_overrides_supplied_config(
    monkeypatch, tmp_path, policy_type, consent, config_consent
):
    if policy_type == "pi0_fast":
        pytest.importorskip("transformers")
        pytest.importorskip("scipy")
    model_cls = factory.get_policy_class(policy_type)
    config = factory.make_policy_config(policy_type, device="cpu", trust_remote_code=config_consent)

    class ConstructionReachedError(Exception):
        pass

    def construct(self, received, **kwargs):
        assert received.trust_remote_code is (config_consent if consent is None else consent)
        assert (received is config) is (consent is None)
        raise ConstructionReachedError

    monkeypatch.setattr(model_cls, "__init__", construct)
    kwargs = {} if consent is None else {"trust_remote_code": consent}
    with pytest.raises(ConstructionReachedError):
        model_cls.from_pretrained(tmp_path, config=config, **kwargs)
    assert config.trust_remote_code is config_consent
