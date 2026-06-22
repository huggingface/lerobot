#!/usr/bin/env python

"""Tests for PI05 Classifier-Free Guidance (CFG) inference."""

import pytest

pytest.importorskip("transformers", reason="transformers is required for PI05")

import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Config, make_pi05_pre_post_processors  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.rendered_messages_to_task import RenderedMessagesToTaskStep  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_UNCOND_ATTENTION_MASK,
    OBS_LANGUAGE_UNCOND_TOKENS,
)


class TestRenderedMessagesToTaskBaseTaskPreservation:
    """Tests that RenderedMessagesToTaskStep preserves base_task for CFG."""

    def test_preserves_string_base_task(self):
        transition = create_transition(
            complementary_data={
                "task": "pick up the cup",
                "messages": [
                    {"role": "user", "content": "pick up the cup, Advantage: positive"},
                ],
            }
        )
        step = RenderedMessagesToTaskStep()
        out = step(transition)
        data = out[TransitionKey.COMPLEMENTARY_DATA]

        assert data["base_task"] == "pick up the cup"
        assert data["task"] == "pick up the cup, Advantage: positive"

    def test_preserves_list_base_task(self):
        transition = create_transition(
            complementary_data={
                "task": ["task1", "task2"],
                "messages": [
                    {"role": "user", "content": "rendered with advantage"},
                ],
            }
        )
        step = RenderedMessagesToTaskStep()
        out = step(transition)
        data = out[TransitionKey.COMPLEMENTARY_DATA]

        assert data["base_task"] == ["task1", "task2"]

    def test_no_base_task_when_messages_absent(self):
        transition = create_transition(complementary_data={"task": "pick up the cup"})
        step = RenderedMessagesToTaskStep()
        out = step(transition)
        data = out[TransitionKey.COMPLEMENTARY_DATA]

        assert "base_task" not in data


class TestPi05PrepareStateTokenizerCfg:
    """Tests for Pi05PrepareStateTokenizerProcessorStep with cfg_enabled."""

    def _make_transition(self, task, base_task=None):
        complementary_data = {"task": task}
        if base_task is not None:
            complementary_data["base_task"] = base_task
        return create_transition(
            observation={"observation.state": torch.zeros(1, 14)},
            complementary_data=complementary_data,
        )

    def test_cfg_disabled_no_uncond_task(self):
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

        step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=14, cfg_enabled=False)
        transition = self._make_transition(task=["pick up the cup, Advantage: positive"])
        out = step(transition)
        data = out[TransitionKey.COMPLEMENTARY_DATA]

        assert "uncond_task" not in data

    def test_cfg_enabled_produces_uncond_task_from_base(self):
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

        step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=14, cfg_enabled=True)
        transition = self._make_transition(
            task=["pick up the cup, Advantage: positive"],
            base_task=["pick up the cup"],
        )
        out = step(transition)
        data = out[TransitionKey.COMPLEMENTARY_DATA]

        assert "uncond_task" in data
        assert len(data["uncond_task"]) == 1
        # Unconditional prompt uses base_task (no advantage)
        assert "Advantage" not in data["uncond_task"][0]
        assert "pick up the cup" in data["uncond_task"][0]
        assert "State:" in data["uncond_task"][0]

    def test_cfg_enabled_falls_back_to_task_when_no_base(self):
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

        step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=14, cfg_enabled=True)
        transition = self._make_transition(task=["pick up the cup"])
        out = step(transition)
        data = out[TransitionKey.COMPLEMENTARY_DATA]

        # Falls back to using task itself as unconditional
        assert "uncond_task" in data
        assert "pick up the cup" in data["uncond_task"][0]


class TestCfgPipelineConstruction:
    """Tests that the processor pipeline is constructed correctly for CFG."""

    def _make_config(self, cfg_beta=1.0, recipe_path=None):
        config = PI05Config(
            max_action_dim=7,
            max_state_dim=14,
            cfg_beta=cfg_beta,
            recipe_path=recipe_path,
            device="cpu",
        )
        config.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
        return config

    def _make_dataset_stats(self):
        return {
            "observation.state": {
                "mean": torch.zeros(14),
                "std": torch.ones(14),
                "min": torch.zeros(14),
                "max": torch.ones(14),
                "q01": torch.zeros(14),
                "q99": torch.ones(14),
            },
            "action": {
                "mean": torch.zeros(7),
                "std": torch.ones(7),
                "min": torch.zeros(7),
                "max": torch.ones(7),
                "q01": torch.zeros(7),
                "q99": torch.ones(7),
            },
            "observation.images.base_0_rgb": {
                "mean": torch.zeros(3, 224, 224),
                "std": torch.ones(3, 224, 224),
                "q01": torch.zeros(3, 224, 224),
                "q99": torch.ones(3, 224, 224),
            },
        }

    def test_no_uncond_tokenizer_when_cfg_disabled(self):
        from lerobot.processor import TokenizerProcessorStep

        config = self._make_config(cfg_beta=1.0)
        preprocessor, _ = make_pi05_pre_post_processors(config, self._make_dataset_stats())

        tokenizer_steps = [s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)]
        assert len(tokenizer_steps) == 1

    def test_uncond_tokenizer_added_when_cfg_enabled(self):
        from lerobot.processor import TokenizerProcessorStep

        config = self._make_config(cfg_beta=2.0)
        preprocessor, _ = make_pi05_pre_post_processors(config, self._make_dataset_stats())

        tokenizer_steps = [s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)]
        assert len(tokenizer_steps) == 2

        uncond_tokenizer = tokenizer_steps[1]
        assert uncond_tokenizer.task_key == "uncond_task"
        assert uncond_tokenizer.output_tokens_key == OBS_LANGUAGE_UNCOND_TOKENS
        assert uncond_tokenizer.output_mask_key == OBS_LANGUAGE_UNCOND_ATTENTION_MASK

    def test_cfg_pipeline_produces_both_token_sets(self):
        config = self._make_config(cfg_beta=2.0)
        preprocessor, _ = make_pi05_pre_post_processors(config, self._make_dataset_stats())

        batch = {
            "observation.state": torch.randn(14),
            "observation.images.base_0_rgb": torch.rand(3, 224, 224),
            "task": "pick up the cup",
        }
        processed = preprocessor(batch)

        assert OBS_LANGUAGE_TOKENS in processed
        assert OBS_LANGUAGE_ATTENTION_MASK in processed
        assert OBS_LANGUAGE_UNCOND_TOKENS in processed
        assert OBS_LANGUAGE_UNCOND_ATTENTION_MASK in processed

        # Both should be tensors with the same shape
        assert processed[OBS_LANGUAGE_TOKENS].shape == processed[OBS_LANGUAGE_UNCOND_TOKENS].shape
        assert (
            processed[OBS_LANGUAGE_ATTENTION_MASK].shape
            == processed[OBS_LANGUAGE_UNCOND_ATTENTION_MASK].shape
        )

    def test_cfg_beta_1_no_uncond_tokens_in_output(self):
        config = self._make_config(cfg_beta=1.0)
        preprocessor, _ = make_pi05_pre_post_processors(config, self._make_dataset_stats())

        batch = {
            "observation.state": torch.randn(14),
            "observation.images.base_0_rgb": torch.rand(3, 224, 224),
            "task": "pick up the cup",
        }
        processed = preprocessor(batch)

        assert OBS_LANGUAGE_TOKENS in processed
        assert OBS_LANGUAGE_UNCOND_TOKENS not in processed
