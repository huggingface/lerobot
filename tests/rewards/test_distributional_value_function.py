# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RECAP's distributional value function."""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.rewards.distributional_value_function.configuration_distributional_value_function import (
    DistributionalVFConfig,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_IMAGES
from tests.utils import skip_if_package_missing

BATCH_SIZE = 4
NUM_BINS = 201
IMAGE_KEY = f"{OBS_IMAGES}.top"
IMAGE_KEY_WRIST_LEFT = f"{OBS_IMAGES}.wrist_left"
IMAGE_KEY_WRIST_RIGHT = f"{OBS_IMAGES}.wrist_right"


def _make_config(**overrides) -> DistributionalVFConfig:
    defaults = {
        "device": "cpu",
        "image_resolution": (224, 224),
    }
    defaults.update(overrides)
    config = DistributionalVFConfig(**defaults)
    config.input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        IMAGE_KEY_WRIST_LEFT: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        IMAGE_KEY_WRIST_RIGHT: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {}
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
    }
    return config


def _make_model():
    from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import (
        DistributionalVFRewardModel,
    )

    return DistributionalVFRewardModel(_make_config())


def _make_batch(batch_size: int = BATCH_SIZE, device: str = "cpu") -> dict[str, torch.Tensor]:
    from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
        IMAGE_MASK_SUFFIX,
    )
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    return {
        IMAGE_KEY: torch.rand(batch_size, 3, 224, 224, device=device) * 2 - 1,
        IMAGE_KEY + IMAGE_MASK_SUFFIX: torch.ones(batch_size, dtype=torch.bool, device=device),
        IMAGE_KEY_WRIST_LEFT: torch.rand(batch_size, 3, 224, 224, device=device) * 2 - 1,
        IMAGE_KEY_WRIST_LEFT + IMAGE_MASK_SUFFIX: torch.ones(batch_size, dtype=torch.bool, device=device),
        IMAGE_KEY_WRIST_RIGHT: torch.rand(batch_size, 3, 224, 224, device=device) * 2 - 1,
        IMAGE_KEY_WRIST_RIGHT + IMAGE_MASK_SUFFIX: torch.ones(batch_size, dtype=torch.bool, device=device),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (batch_size, 200), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 200, dtype=torch.bool, device=device),
        "mc_return": torch.rand(batch_size, device=device) * -1.0,
        "is_terminal": torch.zeros(batch_size, dtype=torch.bool, device=device),
    }


# ------------------------------------------------------------------
# Config / registry tests
# ------------------------------------------------------------------


def test_config_registered_in_reward_model_registry():
    """DistributionalVFConfig is discoverable via RewardModelConfig registry."""
    known = RewardModelConfig.get_known_choices()
    assert "distributional_value_function" in known


def test_factory_returns_correct_class():
    """get_reward_model_class returns DistributionalVFRewardModel."""
    from lerobot.rewards.factory import get_reward_model_class

    cls = get_reward_model_class("distributional_value_function")
    from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import (
        DistributionalVFRewardModel,
    )

    assert cls is DistributionalVFRewardModel


def test_make_reward_model_config_factory():
    """make_reward_model_config creates DistributionalVFConfig with overrides."""
    from lerobot.rewards.factory import make_reward_model_config

    config = make_reward_model_config("distributional_value_function", num_value_bins=101)
    assert isinstance(config, DistributionalVFConfig)
    assert config.num_value_bins == 101


# ------------------------------------------------------------------
# Target distribution tests (HL-Gauss, Dirac delta, one-hot)
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_hl_gauss_sums_to_one():
    """HL-Gauss target distribution sums to 1 for each sample."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.1, -0.9, -0.0])
    dist = model.hl_gauss_target(targets)

    assert dist.shape == (4, NUM_BINS)
    torch.testing.assert_close(dist.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)


@skip_if_package_missing("transformers")
def test_hl_gauss_non_negative():
    """HL-Gauss target probabilities are all non-negative."""
    model = _make_model()
    targets = torch.linspace(-1.0, 0.0, 10)
    dist = model.hl_gauss_target(targets)

    assert (dist >= 0).all()


@skip_if_package_missing("transformers")
def test_hl_gauss_expected_value_matches():
    """E[V] under HL-Gauss distribution matches the target value."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.1, -0.9])
    dist = model.hl_gauss_target(targets)
    expected = (dist * model.value_head.bin_centers).sum(dim=-1)

    torch.testing.assert_close(expected, targets, atol=1e-4, rtol=0)


@skip_if_package_missing("transformers")
def test_hl_gauss_handles_2d_input():
    """HL-Gauss handles [batch_size, 1] shaped inputs correctly."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.3]).unsqueeze(-1)
    dist = model.hl_gauss_target(targets)

    assert dist.shape == (2, NUM_BINS)
    torch.testing.assert_close(dist.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=0)


@skip_if_package_missing("transformers")
def test_dirac_delta_sums_to_one():
    """Dirac delta target distribution sums to 1 for each sample."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.1, -0.9, -1.0, 0.0])
    dist = model.dirac_delta_target(targets)

    assert dist.shape == (5, NUM_BINS)
    torch.testing.assert_close(dist.sum(dim=-1), torch.ones(5), atol=1e-6, rtol=0)


@skip_if_package_missing("transformers")
def test_dirac_delta_at_most_two_nonzero():
    """Dirac delta places probability on at most two adjacent bins."""
    model = _make_model()
    targets = torch.tensor([-0.7523, -0.0013])
    dist = model.dirac_delta_target(targets)

    for i in range(2):
        assert (dist[i] > 0).sum() <= 2


@skip_if_package_missing("transformers")
def test_dirac_delta_expected_value_matches():
    """E[V] under Dirac delta distribution matches the target value."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.1, -0.9])
    dist = model.dirac_delta_target(targets)
    expected = (dist * model.value_head.bin_centers).sum(dim=-1)

    torch.testing.assert_close(expected, targets, atol=1e-5, rtol=0)


@skip_if_package_missing("transformers")
def test_dirac_delta_boundary_values_clamped():
    """Values outside support are clamped to boundary bins."""
    model = _make_model()
    targets = torch.tensor([-1.5, 0.5])
    dist = model.dirac_delta_target(targets)

    torch.testing.assert_close(dist.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0)
    assert dist[0, 0] == 1.0
    assert dist[1, -1] == 1.0


@skip_if_package_missing("transformers")
def test_one_hot_single_nonzero():
    """One-hot target has exactly one non-zero bin per sample."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.1, -1.0, 0.0])
    dist = model.one_hot_target(targets)

    assert dist.shape == (4, NUM_BINS)
    for i in range(4):
        assert (dist[i] > 0).sum() == 1
        assert dist[i].sum() == 1.0


@skip_if_package_missing("transformers")
def test_one_hot_nearest_bin():
    """One-hot target activates the bin closest to the target value."""
    model = _make_model()
    targets = torch.tensor([-0.5])
    dist = model.one_hot_target(targets)

    hot_idx = dist[0].argmax()
    assert model.value_head.bin_centers[hot_idx].item() == pytest.approx(-0.5, abs=0.003)


@skip_if_package_missing("transformers")
def test_terminal_gets_one_hot():
    """Terminal states receive one-hot targets; non-terminal get HL-Gauss."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.3, -0.7, -0.9])
    is_terminal = torch.tensor([False, True, False, True])

    dist = model.compute_target_distribution(
        targets, is_terminal, method="hl_gauss", use_one_hot_terminal=True
    )

    for i in range(4):
        assert dist[i].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (dist[1] > 0).sum() == 1
    assert (dist[3] > 0).sum() == 1
    assert (dist[0] > 0).sum() > 2
    assert (dist[2] > 0).sum() > 2


@skip_if_package_missing("transformers")
def test_no_terminal_override_when_disabled():
    """When use_one_hot_terminal=False, terminal states use the base method."""
    model = _make_model()
    targets = torch.tensor([-0.5, -0.3])
    is_terminal = torch.tensor([False, True])

    dist = model.compute_target_distribution(
        targets, is_terminal, method="hl_gauss", use_one_hot_terminal=False
    )

    assert (dist[1] > 0).sum() > 2


# ------------------------------------------------------------------
# Architecture / component tests
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_model_has_expected_components():
    """Model scaffold contains the SigLIP2+Gemma3+ValueHead components."""
    model = _make_model()

    assert hasattr(model, "vision_encoder")
    assert hasattr(model, "gemma3")
    assert hasattr(model, "image_proj")
    assert hasattr(model, "value_head")
    assert hasattr(model, "cls_embedding")
    assert hasattr(model.value_head, "mlp")
    assert hasattr(model.value_head, "bin_centers")


@skip_if_package_missing("transformers")
def test_model_bin_centers_shape():
    """Value head bin_centers buffer has shape (num_value_bins,)."""
    model = _make_model()
    assert model.value_head.bin_centers.shape == (NUM_BINS,)


@skip_if_package_missing("transformers")
def test_value_head_output_dim():
    """Value head linear projection outputs num_value_bins logits."""
    model = _make_model()
    assert model.value_head.mlp[-1].out_features == NUM_BINS


@skip_if_package_missing("transformers")
def test_cls_embedding_is_nn_embedding():
    """CLS is nn.Embedding (FSDP-safe) with correct shape."""
    model = _make_model()
    from torch import nn

    assert isinstance(model.cls_embedding, nn.Embedding)
    assert model.cls_embedding.num_embeddings == 1
    assert model.cls_embedding.embedding_dim == model.gemma3_hidden


@skip_if_package_missing("transformers")
def test_image_proj_dimensions():
    """Image projection maps SigLIP2 hidden to Gemma3 hidden."""
    model = _make_model()
    siglip_hidden = model.vision_encoder.config.hidden_size
    assert model.image_proj.in_features == siglip_hidden
    assert model.image_proj.out_features == model.gemma3_hidden


# ------------------------------------------------------------------
# Forward / inference tests
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_forward_returns_loss_and_dict():
    """Forward pass returns a finite scalar loss and output dict with expected keys."""
    model = _make_model()
    batch = _make_batch()

    loss, output_dict = model.forward(batch)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert "loss" in output_dict
    assert "predicted_value_mean" in output_dict
    assert "mc_return_mean" in output_dict
    assert "acc_best" in output_dict
    assert "acc_neighbor" in output_dict
    assert "mae" in output_dict


@skip_if_package_missing("transformers")
def test_forward_loss_is_positive():
    """Cross-entropy loss is strictly positive for random weights."""
    model = _make_model()
    batch = _make_batch()

    loss, _ = model.forward(batch)

    assert loss.item() > 0


@skip_if_package_missing("transformers")
def test_compute_reward_returns_correct_shape():
    """compute_reward returns [batch_size] tensor of finite float32 values."""
    model = _make_model()
    model.eval()
    batch = _make_batch(batch_size=3)

    with torch.no_grad():
        values = model.compute_reward(batch)

    assert values.shape == (3,)
    assert values.dtype == torch.float32
    assert torch.isfinite(values).all()


@skip_if_package_missing("transformers")
def test_compute_reward_values_in_support_range():
    """Predicted values lie within [value_support_min, value_support_max]."""
    model = _make_model()
    model.eval()
    batch = _make_batch(batch_size=8)

    with torch.no_grad():
        values = model.compute_reward(batch)

    assert (values >= -1.0 - 0.01).all()
    assert (values <= 0.0 + 0.01).all()


# ------------------------------------------------------------------
# Gradient flow tests
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_gradient_flows_through_value_head():
    """Backprop produces non-zero gradients on the value head projection."""
    model = _make_model()
    model.train()
    batch = _make_batch()

    loss, _ = model.forward(batch)
    loss.backward()

    assert model.value_head.mlp[-1].weight.grad is not None
    assert not torch.all(model.value_head.mlp[-1].weight.grad == 0)


@skip_if_package_missing("transformers")
def test_gradient_flows_through_cls_embedding():
    """Backprop produces non-zero gradients on the learned [CLS] embedding."""
    model = _make_model()
    model.train()
    batch = _make_batch()

    loss, _ = model.forward(batch)
    loss.backward()

    assert model.cls_embedding.weight.grad is not None
    assert not torch.all(model.cls_embedding.weight.grad == 0)


@skip_if_package_missing("transformers")
def test_gradient_flows_through_image_proj():
    """Backprop produces non-zero gradients on the image projection."""
    model = _make_model()
    model.train()
    batch = _make_batch()

    loss, _ = model.forward(batch)
    loss.backward()

    assert model.image_proj.weight.grad is not None
    assert not torch.all(model.image_proj.weight.grad == 0)


# ------------------------------------------------------------------
# Freeze / training infrastructure tests
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_freeze_vision_encoder():
    """freeze_vision_encoder disables requires_grad on SigLIP2."""
    model = _make_model()
    model.config.freeze_vision_encoder = True
    model._set_requires_grad()

    for p in model.vision_encoder.parameters():
        assert not p.requires_grad
    for p in model.value_head.parameters():
        assert p.requires_grad


@skip_if_package_missing("transformers")
def test_freeze_language_model():
    """freeze_language_model disables requires_grad on Gemma3."""
    model = _make_model()
    model.config.freeze_language_model = True
    model._set_requires_grad()

    for p in model.gemma3.parameters():
        assert not p.requires_grad
    for p in model.value_head.parameters():
        assert p.requires_grad


@skip_if_package_missing("transformers")
def test_stop_gradient_to_vlm_preserves_cls_grad():
    """With stop_gradient_to_vlm, CLS embedding still gets gradients."""
    config = _make_config(stop_gradient_to_vlm=True)
    from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import (
        DistributionalVFRewardModel,
    )

    model = DistributionalVFRewardModel(config)
    model.train()
    batch = _make_batch()

    loss, _ = model.forward(batch)
    loss.backward()

    assert model.cls_embedding.weight.grad is not None
    assert not torch.all(model.cls_embedding.weight.grad == 0)


# ------------------------------------------------------------------
# Config validation tests
# ------------------------------------------------------------------


def test_config_requires_visual_feature():
    """validate_features raises if no VISUAL feature is present."""
    config = DistributionalVFConfig()
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
    }

    with pytest.raises(ValueError, match="VISUAL"):
        config.validate_features()


def test_config_passes_with_visual_feature():
    """validate_features succeeds when a VISUAL feature is present."""
    config = _make_config()
    config.validate_features()


# ------------------------------------------------------------------
# Processor tests
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_processor_pipeline_produces_expected_keys():
    """Full preprocessor pipeline produces tokenized text, preprocessed images, and masks."""
    from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
        IMAGE_MASK_SUFFIX,
        make_distributional_vf_pre_post_processors,
    )
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    config = _make_config()
    preprocessor, _ = make_distributional_vf_pre_post_processors(config)

    raw_batch = {
        IMAGE_KEY: torch.rand(3, 224, 224),
        IMAGE_KEY_WRIST_LEFT: torch.rand(3, 224, 224),
        IMAGE_KEY_WRIST_RIGHT: torch.rand(3, 224, 224),
        "task": "pick up the cup",
    }

    processed = preprocessor(raw_batch)

    assert OBS_LANGUAGE_TOKENS in processed
    assert OBS_LANGUAGE_ATTENTION_MASK in processed
    assert IMAGE_KEY in processed
    assert IMAGE_KEY + IMAGE_MASK_SUFFIX in processed
    assert IMAGE_KEY_WRIST_LEFT + IMAGE_MASK_SUFFIX in processed
    assert IMAGE_KEY_WRIST_RIGHT + IMAGE_MASK_SUFFIX in processed

    img = processed[IMAGE_KEY]
    assert img.shape == (1, 3, 224, 224)
    assert img.min() >= -1.0 - 1e-5
    assert img.max() <= 1.0 + 1e-5


def test_task_prompt_formats_correctly():
    """Task prompt step builds 'Task: {task}.' format."""
    from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
        DistributionalVFPrepareTaskPromptStep,
    )

    step = DistributionalVFPrepareTaskPromptStep()

    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_up_the_cup"]},
    }

    result = step(transition)
    prompt = result[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt == "Task: pick up the cup."


def test_task_prompt_handles_string_input():
    """Task prompt step accepts a plain string task."""
    from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
        DistributionalVFPrepareTaskPromptStep,
    )

    step = DistributionalVFPrepareTaskPromptStep()

    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {"task": "open_drawer"},
    }

    result = step(transition)
    prompt = result[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt == "Task: open drawer."


def test_task_prompt_raises_on_missing_task():
    """Task prompt step raises ValueError when task key is absent."""
    from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
        DistributionalVFPrepareTaskPromptStep,
    )

    step = DistributionalVFPrepareTaskPromptStep()

    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    with pytest.raises(ValueError, match="No task found"):
        step(transition)


def test_image_preprocessor_resize_and_normalize():
    """Image preprocessor resizes, normalizes to [-1,1], and adds masks."""
    from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
        IMAGE_MASK_SUFFIX,
        DistributionalVFImagePreprocessorStep,
    )

    step = DistributionalVFImagePreprocessorStep(
        image_resolution=(224, 224),
        image_keys=(IMAGE_KEY,),
    )

    transition = {
        TransitionKey.OBSERVATION: {
            IMAGE_KEY: torch.rand(2, 3, 320, 240),  # non-square, [0, 1]
        }
    }

    result = step(transition)
    obs = result[TransitionKey.OBSERVATION]

    assert obs[IMAGE_KEY].shape == (2, 3, 224, 224)
    assert obs[IMAGE_KEY].min() >= -1.0 - 1e-5
    assert obs[IMAGE_KEY].max() <= 1.0 + 1e-5
    assert obs[IMAGE_KEY + IMAGE_MASK_SUFFIX].all()


def test_image_preprocessor_missing_camera_gets_placeholder():
    """Missing cameras get black placeholder and mask=False."""
    from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
        IMAGE_MASK_SUFFIX,
        DistributionalVFImagePreprocessorStep,
    )

    step = DistributionalVFImagePreprocessorStep(
        image_resolution=(224, 224),
        image_keys=(IMAGE_KEY, IMAGE_KEY_WRIST_LEFT),
    )

    transition = {
        TransitionKey.OBSERVATION: {
            IMAGE_KEY: torch.rand(2, 3, 224, 224),
        }
    }

    result = step(transition)
    obs = result[TransitionKey.OBSERVATION]

    assert obs[IMAGE_KEY + IMAGE_MASK_SUFFIX].all()
    assert not obs[IMAGE_KEY_WRIST_LEFT + IMAGE_MASK_SUFFIX].any()
    assert obs[IMAGE_KEY_WRIST_LEFT].shape == (2, 3, 224, 224)


# ------------------------------------------------------------------
# Save / load roundtrip
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_save_load_pretrained_roundtrip(tmp_path):
    """Saved model can be loaded back with identical weights."""
    from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import (
        DistributionalVFRewardModel,
    )

    model = _make_model()
    model._save_pretrained(tmp_path)

    loaded = DistributionalVFRewardModel.from_pretrained(str(tmp_path))

    orig_sd = model.state_dict()
    loaded_sd = loaded.state_dict()

    assert set(orig_sd.keys()) == set(loaded_sd.keys())
    for key in orig_sd:
        torch.testing.assert_close(orig_sd[key], loaded_sd[key], msg=f"Mismatch in {key}")


# ------------------------------------------------------------------
# Attention mask utility test
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_make_att_2d_masks():
    """Verify attention mask construction for prefix + CLS."""
    from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import (
        make_att_2d_masks,
    )

    pad = torch.ones(1, 4, dtype=torch.bool)
    att = torch.tensor([[0, 0, 0, 1]])
    mask = make_att_2d_masks(pad, att)[0]

    assert mask[0, 0]  # prefix sees prefix
    assert mask[1, 2]  # prefix sees prefix
    assert not mask[0, 3]  # prefix does NOT see CLS
    assert mask[3, 0]  # CLS sees prefix
    assert mask[3, 3]  # CLS sees itself


# ------------------------------------------------------------------
# Categorical metrics test
# ------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_categorical_metrics_perfect_prediction():
    """Metrics return acc_best=1 when logits peak at the correct bin."""
    model = _make_model()
    bin_centers = model.value_head.bin_centers
    target = bin_centers[100].unsqueeze(0)  # exact bin center

    batch = _make_batch(batch_size=1)
    batch["mc_return"] = target
    batch["is_terminal"] = torch.zeros(1, dtype=torch.bool)

    with torch.no_grad():
        _, output_dict = model.forward(batch)

    assert "acc_best" in output_dict
    assert "acc_neighbor" in output_dict
    assert "mae" in output_dict
    assert isinstance(output_dict["acc_best"], float)
    assert isinstance(output_dict["mae"], float)
