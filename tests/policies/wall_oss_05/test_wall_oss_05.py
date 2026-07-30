#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.wall_oss_05.configuration_wall_oss_05 import WallOSS05Config
from lerobot.policies.wall_oss_05.modeling_wall_oss_05 import (
    WallOSS05Model,
    WallOSS05Policy,
    WallOSS05VisionMLP,
)
from lerobot.policies.wall_oss_05.processor_wall_oss_05 import WallOSS05TaskPassthrough
from lerobot.utils.constants import ACTION, OBS_STATE


def _config(**overrides) -> WallOSS05Config:
    values = {
        "device": "cpu",
        "vlm_config": {"model_type": "qwen2_5_vl"},
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(26,)),
            "observation.images.face_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.right_wrist_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(26,))},
    }
    values.update(overrides)
    return WallOSS05Config(**values)


class _FakeAdapter:
    def __init__(self):
        self.received_tasks: list[str] = []
        self.received_observations = None

    def get_flow_prompt(self, task):
        self.received_tasks.append(task)
        return f"prefix<{task}>", "<|action|>"

    def construct_model_input(self, observations, prefix_list, postfix_list):
        self.received_observations = observations
        assert len(prefix_list) == len(postfix_list) == len(observations)
        return {"input_ids": torch.zeros((len(observations), 1), dtype=torch.long)}


class _TinyAuthorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, action_chunk, flow_loss_mask, **kwargs):
        del kwargs
        selected = action_chunk[flow_loss_mask]
        loss = (self.scale * selected).square().mean()
        return SimpleNamespace(loss=loss, flow_loss=loss)


class _TinyTiedAuthorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def _batch(batch_size=2):
    return {
        OBS_STATE: torch.arange(batch_size * 26, dtype=torch.float32).reshape(batch_size, 26),
        ACTION: torch.ones(batch_size, 32, 26),
        "observation.images.face_view": torch.rand(batch_size, 3, 16, 16),
        "observation.images.right_wrist_view": torch.rand(batch_size, 3, 16, 16),
        "task": ["pick the cup?!", "leave punctuation unchanged..."][:batch_size],
        "action_is_pad": torch.zeros(batch_size, 32, dtype=torch.bool),
    }


def _stats(q01: float = 0.0, q99: float = 2.0) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {"q01": torch.full((26,), q01), "q99": torch.full((26,), q99)},
        ACTION: {"q01": torch.full((26,), q01), "q99": torch.full((26,), q99)},
    }


def test_lerobot_processors_match_wall_quantile_normalization_and_clipping():
    preprocessor, postprocessor = make_pre_post_processors(_config(), dataset_stats=_stats())
    batch = _batch(batch_size=1)
    batch[OBS_STATE] = torch.tensor([[-1.0, 1.0, 3.0] + [1.0] * 23])
    batch[ACTION] = torch.full((1, 32, 26), 0.5)

    processed = preprocessor(batch)

    torch.testing.assert_close(processed[OBS_STATE][0, :3], torch.tensor([-1.0, 0.0, 1.0]))
    torch.testing.assert_close(processed[ACTION], torch.full((1, 32, 26), -0.5))
    torch.testing.assert_close(postprocessor(processed[ACTION]), batch[ACTION])


def test_native_prompt_and_image_token_expansion_preserve_the_contract():
    pytest.importorskip("transformers")

    policy = WallOSS05Policy(_config(), load_model=False)
    prefix, postfix = policy._get_flow_prompt("pick the cup?!")
    assert "Instruction: pick the cup?!" in prefix
    assert "front view:" in prefix
    assert "right wrist view:" in prefix
    assert postfix.count("<|action|>") == 32

    expanded = policy._expand_image_tokens(
        [prefix],
        torch.tensor([[1, 4, 4], [1, 8, 4]]),
        merge_size=2,
    )
    assert expanded[0].count("<|image_pad|>") == 12


def test_native_text_prompts_cover_subtask_and_vqa_contracts():
    pytest.importorskip("transformers")

    policy = WallOSS05Policy(_config(), load_model=False)

    subtask = policy._get_text_prompt("put the cup away", "subtask")
    vqa = policy._get_text_prompt("Where is the cup?", "vqa")

    assert "Instruction: put the cup away" in subtask
    assert "Predict the next action in language." in subtask
    assert "Question: Where is the cup?" in vqa
    assert "front view:" in vqa
    assert vqa.endswith("<|im_start|>assistant\n")


def test_policy_text_generation_decodes_native_model_tokens():
    pytest.importorskip("transformers")

    class Tokenizer:
        eos_token_id = 2

        @staticmethod
        def batch_decode(output_ids, **kwargs):
            assert kwargs["skip_special_tokens"]
            assert torch.equal(output_ids, torch.tensor([[4, 5]]))
            return ["  the cup is left  "]

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.kwargs = None

        def generate_text(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[4, 5]])

    policy = WallOSS05Policy(_config(), load_model=False)
    policy.processor = SimpleNamespace(tokenizer=Tokenizer())
    policy.model = TextModel()
    policy._build_text_model_inputs = lambda *args, **kwargs: {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
    }

    output = policy.generate_text(
        {"task": "locate cup"},
        kind="vqa",
        user_text="Where is it?",
        max_new_tokens=8,
    )

    assert output == ["the cup is left"]
    assert policy.model.kwargs["max_new_tokens"] == 8
    assert policy.model.kwargs["eos_token_id"] == 2


def test_native_causal_mask_excludes_padding_rows_and_columns():
    pytest.importorskip("transformers")

    attention_mask = torch.tensor([[0, 1, 1, 1]])
    mask = WallOSS05Model._causal_mask(attention_mask)
    expected = torch.tensor(
        [
            [
                [False, False, False, False],
                [False, True, False, False],
                [False, True, True, False],
                [False, True, True, True],
            ]
        ]
    )
    assert torch.equal(mask, expected)


def test_fused_vision_mlp_matches_split_swiglu_math():
    pytest.importorskip("transformers")

    mlp = WallOSS05VisionMLP(hidden_size=4, intermediate_size=6)
    inputs = torch.randn(3, 4)
    gate_weight, up_weight = mlp.gate_up_proj.weight.chunk(2)
    gate_bias, up_bias = mlp.gate_up_proj.bias.chunk(2)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(inputs, gate_weight, gate_bias))
        * torch.nn.functional.linear(inputs, up_weight, up_bias),
        mlp.down_proj.weight,
        mlp.down_proj.bias,
    )
    torch.testing.assert_close(mlp(inputs), expected)


def test_factory_uses_distinct_policy_type():
    config = make_policy_config("wall_oss_05")
    assert isinstance(config, WallOSS05Config)
    assert get_policy_class("wall_oss_05") is WallOSS05Policy
    assert config.type == "wall_oss_05"
    assert config.chunk_size == 32
    assert config.max_action_dim == 26
    assert config.normalization_mapping["STATE"] is NormalizationMode.QUANTILES
    assert config.normalization_mapping["ACTION"] is NormalizationMode.QUANTILES


def test_noncanonical_dimensions_are_rejected():
    config = _config(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.face_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.right_wrist_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
    )
    with pytest.raises(ValueError, match="canonical 26D"):
        WallOSS05Policy(config, load_model=False)


def test_task_passthrough_does_not_rewrite():
    step = WallOSS05TaskPassthrough()
    data = {"task": ["no period", "Keep?!  spaces"]}
    result = step.complementary_data(data)
    assert result is data
    assert result["task"] == ["no period", "Keep?!  spaces"]


def test_exact_selected_task_reaches_policy_formatter_unchanged():
    policy = WallOSS05Policy(_config(), load_model=False)
    adapter = _FakeAdapter()
    policy.processor = object()
    policy.model = _TinyAuthorModel()
    policy._get_flow_prompt = adapter.get_flow_prompt
    policy._construct_model_input = adapter.construct_model_input
    tasks = ["pick the cup?!", "leave punctuation unchanged..."]
    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=_stats(0, 52))
    policy._build_model_inputs(preprocessor(_batch()))
    assert adapter.received_tasks == tasks


def test_pre_and_postprocessors_are_serializable(tmp_path):
    config = _config()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=_stats())
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    assert (tmp_path / "policy_preprocessor.json").exists()
    assert (tmp_path / "policy_postprocessor.json").exists()


def test_training_dataset_stats_override_pretrained_processor_stats(tmp_path):
    config = _config()
    pretrained_stats = _stats(0, 2)
    training_stats = _stats(10, 20)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=pretrained_stats)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    features = {**config.input_features, **config.output_features}
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
        preprocessor_overrides={
            "normalizer_processor": {
                "stats": training_stats,
                "features": features,
                "norm_map": config.normalization_mapping,
            }
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": training_stats,
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
            }
        },
    )
    batch = _batch(batch_size=1)
    batch[OBS_STATE] = torch.full((1, 26), 15.0)
    batch[ACTION] = torch.full((1, 32, 26), 15.0)

    processed = loaded_preprocessor(batch)

    torch.testing.assert_close(processed[OBS_STATE], torch.zeros_like(processed[OBS_STATE]))
    torch.testing.assert_close(processed[ACTION], torch.zeros_like(processed[ACTION]))
    torch.testing.assert_close(
        loaded_postprocessor(torch.zeros(1, 32, 26)),
        torch.full((1, 32, 26), 15.0),
    )

    fine_tuned_checkpoint = tmp_path / "fine_tuned"
    loaded_preprocessor.save_pretrained(fine_tuned_checkpoint)
    loaded_postprocessor.save_pretrained(fine_tuned_checkpoint)
    reloaded_preprocessor, reloaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(fine_tuned_checkpoint),
    )
    torch.testing.assert_close(
        reloaded_preprocessor(batch)[OBS_STATE],
        torch.zeros_like(processed[OBS_STATE]),
    )
    torch.testing.assert_close(
        reloaded_postprocessor(torch.zeros(1, 32, 26)),
        torch.full((1, 32, 26), 15.0),
    )


def test_forward_backward_update_is_finite_and_tiny_batch_overfits():
    config = _config()
    policy = WallOSS05Policy(config, load_model=False)
    adapter = _FakeAdapter()
    policy.processor = object()
    policy.model = _TinyAuthorModel()
    policy._get_flow_prompt = adapter.get_flow_prompt
    policy._construct_model_input = adapter.construct_model_input
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=_stats(0, 52))
    batch = preprocessor(_batch())

    initial_loss = None
    for _ in range(8):
        optimizer.zero_grad()
        loss, output = policy(batch)
        if initial_loss is None:
            initial_loss = loss.item()
        assert torch.isfinite(loss)
        loss.backward()
        assert policy.model.scale.grad is not None
        assert torch.isfinite(policy.model.scale.grad)
        assert policy.model.scale.grad.abs().item() > 0
        optimizer.step()
        assert output["flow_loss"] >= 0

    final_loss, _ = policy(batch)
    assert final_loss.item() < initial_loss * 0.05


def test_save_preserves_tied_embedding_key_contract_and_vlm_config(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    for name in (
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (source / name).write_text("{}")

    policy = WallOSS05Policy(_config(), load_model=False)
    policy.model = _TinyTiedAuthorModel()
    policy._source_checkpoint = source
    policy.save_pretrained(output)

    with safe_open(output / "model.safetensors", framework="pt") as checkpoint:
        keys = set(checkpoint.keys())
    assert "model.embed_tokens.weight" in keys
    assert "lm_head.weight" not in keys
    assert not (output / "normalizer_action.pth").exists()
    assert not (output / "normalizer_propri.pth").exists()
    assert not (output / "author_config.json").exists()
    assert not (output / "author_train_config.yml").exists()
    saved_config = WallOSS05Config.from_pretrained(output)
    assert saved_config.vlm_config == {"model_type": "qwen2_5_vl"}
