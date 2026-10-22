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

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import (
    make_policy,
    make_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.wall_oss_05.configuration_wall_oss_05 import WallOSS05Config
from lerobot.policies.wall_oss_05.modeling_wall_oss_05 import (
    WallOSS05ActionProcessor,
    WallOSS05Model,
    WallOSS05Policy,
    WallOSS05VisionMLP,
)
from lerobot.policies.wall_oss_05.processor_wall_oss_05 import (
    WALL_OSS_PREDICT_ACTIONS,
    WallOSS05TokenizerStep,
)
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenderRuntimeMessagesStep,
    UnnormalizerProcessorStep,
    batch_to_transition,
    transition_to_batch,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.fixture
def dataset_dependencies():
    pytest.importorskip("datasets", reason="language recipes require lerobot[dataset]")
    pytest.importorskip("av", reason="language recipes require lerobot[dataset]")


def _config(**overrides) -> WallOSS05Config:
    values = {
        "device": "cpu",
        "recipe": None,
        "vlm_config": {"model_type": "qwen2_5_vl"},
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(26,)),
            "observation.images.face_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.right_wrist_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(26,))},
    }
    if overrides.get("use_language_recipe"):
        values.pop("recipe")
    values.update(overrides)
    return WallOSS05Config(**values)


@pytest.mark.parametrize("query_kind", ["vqa", "next_subtask"])
def test_runtime_query_reaches_tokenizer(query_kind, dataset_dependencies):
    config = _config(use_language_recipe=True)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=_stats())
    _stub_tokenizer(preprocessor)
    processed = preprocessor({**_batch(batch_size=1), "query_kind": query_kind, "query_text": "pick the cup"})
    expected = "pick the cup"
    if query_kind == "next_subtask":
        expected += "\nPredict the next action in language.\n"
    assert processed["messages_rendered"] == [[{"role": "user", "content": expected}]]
    assert not processed[WALL_OSS_PREDICT_ACTIONS].any()
    assert "query_kind" not in processed


def test_recipe_override_remains_portable_after_save(tmp_path, dataset_dependencies):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "messages:\n"
        '  - role: user\n    content: "${task}: choose a subtask"\n    stream: low_level\n'
        '  - role: assistant\n    content: "${subtask}"\n    stream: low_level\n'
        "    target: true\n    if_present: subtask\n"
    )
    config = _config(recipe_path=str(path))
    output = tmp_path / "checkpoint"
    output.mkdir()
    config._save_pretrained(output)
    path.unlink()

    restored = WallOSS05Config.from_pretrained(output)
    assert restored.recipe == config.recipe
    preprocessor, _ = make_pre_post_processors(restored, dataset_stats=_stats())
    _stub_tokenizer(preprocessor)
    result = preprocessor(
        {**_batch(batch_size=1), "query_kind": "next_subtask", "query_text": "clear the table"}
    )
    assert result["messages_rendered"] == [[{"role": "user", "content": "clear the table: choose a subtask"}]]


class _TinyTiedAuthorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.seen_texts: list[str] = []

    @staticmethod
    def convert_tokens_to_ids(token):
        assert token == "<|action|>"
        return 999

    def __call__(
        self,
        texts,
        *,
        return_tensors,
        padding,
        padding_side,
        truncation,
        max_length,
        return_offsets_mapping,
    ):
        del return_tensors, padding, truncation
        assert padding_side == "left"
        texts = [texts] if isinstance(texts, str) else texts
        self.seen_texts = list(texts)
        encoded = []
        offsets = []
        for text in texts:
            chars = text[:max_length]
            encoded.append([ord(char) % 251 + 1 for char in chars])
            offsets.append([(index, index + 1) for index in range(len(chars))])
        width = max(len(row) for row in encoded)
        attention_mask = []
        for index in range(len(encoded)):
            pad = width - len(encoded[index])
            encoded[index] = [self.pad_token_id] * pad + encoded[index]
            offsets[index] = [(0, 0)] * pad + offsets[index]
            attention_mask.append([0] * pad + [1] * (width - pad))
        output = {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if return_offsets_mapping:
            output["offset_mapping"] = torch.tensor(offsets, dtype=torch.long)
        return output


class _FakeImageProcessor:
    merge_size = 2

    @staticmethod
    def __call__(images, return_tensors):
        del return_tensors
        return {
            "pixel_values": torch.zeros(len(images), 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


def _tokenizer_step(config: WallOSS05Config | None = None) -> WallOSS05TokenizerStep:
    config = config or _config()
    step = WallOSS05TokenizerStep(
        processor_name="unused",
        camera_key_mapping=config.camera_key_mapping,
        chunk_size=config.chunk_size,
        max_state_dim=config.max_state_dim,
        tokenizer_max_length=config.tokenizer_max_length,
    )
    step._processor = SimpleNamespace(
        tokenizer=_CharTokenizer(),
        image_processor=_FakeImageProcessor(),
    )
    return step


def _stub_tokenizer(preprocessor: PolicyProcessorPipeline) -> WallOSS05TokenizerStep:
    step = next(step for step in preprocessor.steps if isinstance(step, WallOSS05TokenizerStep))
    fake = _tokenizer_step()
    step._processor = fake._processor
    return step


def _model_inputs(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros(batch_size, 3, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "pixel_values": torch.zeros(batch_size * 2, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 2, 2]] * (batch_size * 2)),
        "moe_token_types": torch.zeros(batch_size, 3, dtype=torch.bool),
        "dof_mask": torch.ones(batch_size, 32, 26, dtype=torch.bool),
    }


def _batch(batch_size=2):
    return {
        OBS_STATE: torch.arange(batch_size * 26, dtype=torch.float32).reshape(batch_size, 26),
        ACTION: torch.ones(batch_size, 32, 26),
        "observation.images.face_view": torch.rand(batch_size, 3, 16, 16),
        "observation.images.right_wrist_view": torch.rand(batch_size, 3, 16, 16),
        "task": ["pick the cup?!", "leave punctuation unchanged..."][:batch_size],
        "action_is_pad": torch.zeros(batch_size, 32, dtype=torch.bool),
    }


def _stats(
    q01: float = 0.0, q99: float = 2.0, *, state_dim: int = 26, action_dim: int = 26
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {"q01": torch.full((state_dim,), q01), "q99": torch.full((state_dim,), q99)},
        ACTION: {"q01": torch.full((action_dim,), q01), "q99": torch.full((action_dim,), q99)},
    }


def test_lerobot_processors_match_wall_quantile_normalization_and_clipping():
    preprocessor, postprocessor = make_pre_post_processors(_config(), dataset_stats=_stats())
    _stub_tokenizer(preprocessor)
    batch = _batch(batch_size=1)
    batch[OBS_STATE] = torch.tensor([[-1.0, 1.0, 3.0] + [1.0] * 23])
    batch[ACTION] = torch.full((1, 32, 26), 0.5)

    processed = preprocessor(batch)

    torch.testing.assert_close(processed[OBS_STATE][0, :3], torch.tensor([-1.0, 0.0, 1.0]))
    torch.testing.assert_close(processed[ACTION], torch.full((1, 32, 26), -0.5))
    torch.testing.assert_close(postprocessor(processed[ACTION]), batch[ACTION])


def test_recipe_processor_renders_joint_text_and_action_supervision(dataset_dependencies):
    config = _config(recipe=WallOSS05Config(device="cpu").recipe)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=_stats())
    _stub_tokenizer(preprocessor)
    batch = _batch(batch_size=1)
    batch.update(
        {
            "timestamp": torch.tensor([0.0]),
            "index": torch.tensor([7]),
            "language_persistent": [
                [
                    {
                        "role": "assistant",
                        "content": "pick up the red block",
                        "style": "subtask",
                        "timestamp": 0.0,
                        "camera": None,
                        "tool_calls": None,
                    }
                ]
            ],
            "language_events": [[]],
        }
    )

    processed = preprocessor(batch)

    assert isinstance(preprocessor.steps[0], RenderRuntimeMessagesStep)
    assert processed["messages_rendered"][0][-1]["content"] == "pick up the red block"
    assert processed["message_streams"] == [["low_level", "low_level"]]
    assert processed["target_message_indices"] == [[1]]


def test_recipe_prompt_marks_only_target_text_and_can_append_actions():
    pytest.importorskip("transformers")

    step = _tokenizer_step()
    prefix, postfix, predict_actions = step._recipe_prompt(
        [
            {"role": "user", "content": "sort the blocks"},
            {"role": "assistant", "content": "pick up the red block"},
        ],
        ["low_level", "low_level"],
        [1],
        "sort the blocks",
    )
    clean_prompt, spans = step._target_spans(prefix + postfix)

    assert predict_actions
    assert postfix.count("<|action|>") == step.chunk_size
    assert len(spans) == 1
    start, end = spans[0]
    assert clean_prompt[start:end] == "pick up the red block<|im_end|>"
    assert "Proprioception: <|propri|>" in clean_prompt


def test_native_input_builder_creates_assistant_only_text_labels():
    pytest.importorskip("transformers")

    step = _tokenizer_step(_config(tokenizer_max_length=2000))
    batch = _batch(batch_size=1)
    batch.update(
        {
            "messages_rendered": [
                [
                    {"role": "user", "content": "Where is the cup?"},
                    {"role": "assistant", "content": "on the left"},
                ]
            ],
            "message_streams": [["high_level", "high_level"]],
            "target_message_indices": [[1]],
        }
    )
    inputs = transition_to_batch(step(batch_to_transition(batch)))
    labels = inputs["text_labels"][0]
    supervised_ids = labels[labels != -100]
    expected = torch.tensor(
        [ord(char) % 251 + 1 for char in "on the left<|im_end|>"],
        dtype=torch.long,
    )

    torch.testing.assert_close(supervised_ids.cpu(), expected)
    assert not (inputs["input_ids"] == 999).any()


def test_masked_text_cross_entropy_uses_next_token_targets_only():
    pytest.importorskip("transformers")

    model = WallOSS05Model.__new__(WallOSS05Model)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(3, 5, bias=False)
    hidden_states = torch.randn(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[-100, 2, -100, 4]])

    loss = model._text_cross_entropy(hidden_states, labels)
    expected = torch.nn.functional.cross_entropy(
        model.lm_head(hidden_states[:, :-1][torch.tensor([[True, False, True]])]),
        torch.tensor([2, 4]),
    )

    assert loss is not None
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


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

    output = policy.generate_text(_model_inputs())

    assert output == "the cup is left"
    assert policy.model.kwargs["max_new_tokens"] == 100
    assert WallOSS05Policy.generate_text is not PreTrainedPolicy.generate_text
    assert WallOSS05Policy.supports_text_generation is not PreTrainedPolicy.supports_text_generation
    assert not hasattr(WallOSS05Policy, "generate_texts")
    assert policy.supports_text_generation()
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


@pytest.mark.parametrize("mask", ["none", "padding", "all"])
def test_flow_loss_averages_only_valid_action_elements(monkeypatch, mask):
    model = WallOSS05Model.__new__(WallOSS05Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_hidden_size=16)
    model.action_token_id = 31
    model.action_preprocessor = WallOSS05ActionProcessor(
        SimpleNamespace(
            action_hidden_size=16, text_config=SimpleNamespace(hidden_size=32), noise_scheduler={}
        )
    )
    monkeypatch.setattr(model, "_hidden_states", lambda **kwargs: torch.zeros(1, 2, 32))
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.zeros_like(tensor))
    dof_mask = torch.ones(1, 2, 26, dtype=torch.bool)
    row_mask = torch.ones(1, 2, dtype=torch.bool)
    if mask == "padding":
        dof_mask[..., 13:] = False
        row_mask[:, -1] = False
    elif mask == "all":
        row_mask[:] = False
    output = model(
        input_ids=torch.full((1, 2), 31),
        attention_mask=torch.ones(1, 2),
        moe_token_types=torch.ones(1, 2),
        action_chunk=torch.full((1, 2, 26), 2.0),
        dof_mask=dof_mask,
        sample_time=torch.tensor([0.5]),
        flow_loss_mask=row_mask,
    )
    # Zero noise and zero predicted velocity give x_prediction=1 against target=2.
    # Each valid action element has squared error one, regardless of padding.
    assert output.flow_loss.item() == pytest.approx(0.0 if mask == "all" else 1.0)


def _dimension_config(state_dim: int, action_dim: int):
    return _config(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.face_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.right_wrist_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )


def test_dimensions_exceeding_the_26d_contract_are_rejected():
    with pytest.raises(ValueError, match="cannot exceed the fixed 26D contract"):
        WallOSS05Policy(_dimension_config(state_dim=32, action_dim=32), load_model=False)


def test_exact_selected_task_reaches_policy_formatter_unchanged():
    tasks = ["pick the cup?!", "leave punctuation unchanged..."]
    preprocessor, _ = make_pre_post_processors(_config(), dataset_stats=_stats(0, 52))
    step = _stub_tokenizer(preprocessor)

    preprocessor(_batch())

    assert all(task in text for task, text in zip(tasks, step._processor.tokenizer.seen_texts, strict=True))


def test_finetuned_weight_revision_does_not_select_the_base_tokenizer_revision():
    config = _config(
        pretrained_revision="finetuned-weight-commit", tokenizer_revision="base-tokenizer-commit"
    )
    pre, _ = make_pre_post_processors(config, dataset_stats=_stats())
    tokenizer = next(step for step in pre.steps if isinstance(step, WallOSS05TokenizerStep))
    assert tokenizer.processor_name == "lerobot/wall-oss-0.5"
    assert tokenizer.revision == "base-tokenizer-commit"


@pytest.mark.parametrize(("state_dim", "action_dim"), [(26, 26), (6, 6), (8, 7)])
def test_training_dataset_stats_override_pretrained_processor_stats(tmp_path, state_dim, action_dim):
    config = _config()
    pretrained_stats = _stats(0, 2)
    training_stats = _stats(10, 20, state_dim=state_dim, action_dim=action_dim)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=pretrained_stats)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    config = _dimension_config(state_dim=state_dim, action_dim=action_dim)
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
    _stub_tokenizer(loaded_preprocessor)
    batch = _batch(batch_size=1)
    batch[OBS_STATE] = torch.full((1, state_dim), 15.0)
    batch[ACTION] = torch.full((1, 32, action_dim), 15.0)

    processed = loaded_preprocessor(batch)

    torch.testing.assert_close(processed[OBS_STATE], torch.zeros_like(processed[OBS_STATE]))
    torch.testing.assert_close(processed[ACTION], torch.zeros_like(processed[ACTION]))
    torch.testing.assert_close(
        loaded_postprocessor(torch.zeros(1, 32, 26)),
        torch.full((1, 32, action_dim), 15.0),
    )

    fine_tuned_checkpoint = tmp_path / "fine_tuned"
    loaded_preprocessor.save_pretrained(fine_tuned_checkpoint)
    loaded_postprocessor.save_pretrained(fine_tuned_checkpoint)
    reloaded_preprocessor, reloaded_postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(fine_tuned_checkpoint),
    )
    _stub_tokenizer(reloaded_preprocessor)
    torch.testing.assert_close(
        reloaded_preprocessor(batch)[OBS_STATE],
        torch.zeros_like(processed[OBS_STATE]),
    )
    torch.testing.assert_close(
        reloaded_postprocessor(torch.zeros(1, 32, 26)),
        torch.full((1, 32, action_dim), 15.0),
    )


@pytest.mark.parametrize(("state_dim", "action_dim"), [(6, 6), (8, 7)])
@pytest.mark.parametrize("reload", [False, True])
def test_processors_normalize_native_widths_and_mask_padded_joints(tmp_path, state_dim, action_dim, reload):
    config = _dimension_config(state_dim=state_dim, action_dim=action_dim)
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=_stats(state_dim=state_dim, action_dim=action_dim)
    )
    if reload:
        preprocessor.save_pretrained(tmp_path)
        postprocessor.save_pretrained(tmp_path)
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=tmp_path)

    _stub_tokenizer(preprocessor)
    batch = _batch(batch_size=1)
    batch[OBS_STATE] = torch.ones(1, state_dim)
    batch[ACTION] = torch.ones(1, 32, action_dim)
    processed = preprocessor(batch)

    torch.testing.assert_close(processed[OBS_STATE], torch.zeros(1, 26))
    torch.testing.assert_close(processed[ACTION], torch.zeros(1, 32, 26))
    assert processed["dof_mask"][..., :action_dim].all()
    assert not processed["dof_mask"][..., action_dim:].any()
    torch.testing.assert_close(postprocessor(torch.zeros(1, 32, 26)), torch.ones(1, 32, action_dim))
    batch.pop(ACTION)
    inference = preprocessor(batch)
    torch.testing.assert_close(inference["dof_mask"], processed["dof_mask"])


def _tiny_qwen_processor():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        "pick": 5,
        "up": 6,
        "the": 7,
        "red": 8,
        "cup": 9,
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<|propri|>": 12,
        "<|image_pad|>": 28,
        "<|vision_start|>": 29,
        "<|vision_end|>": 30,
        "<|action|>": 31,
    }
    # Fill unused IDs so additions preserve the model's explicit special-token IDs.
    vocabulary.update({f"unused_{i}": i for i in range(32) if i not in vocabulary.values()})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=[key for key in vocabulary if key.startswith("<|")],
    )
    image_processor = transformers.Qwen2VLImageProcessor(
        min_pixels=16 * 16,
        max_pixels=16 * 16,
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
    )
    return SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)


@pytest.mark.parametrize("saved_processors", [False, True])
def test_six_dimensional_dataset_trains_through_factory_and_saved_processors(
    tmp_path, dataset_dependencies, saved_processors
):
    """Use real dataset stats, tokenization, vision, decoder, flow loss and optimizer."""
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cameras = {
        "observation.images.top": "face_view",
        "observation.images.wrist": "right_wrist_view",
    }
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (6,), "names": None},
        ACTION: {"dtype": "float32", "shape": (6,), "names": None},
        **{
            key: {"dtype": "image", "shape": (3, 16, 16), "names": ["channels", "height", "width"]}
            for key in cameras
        },
    }
    root = tmp_path / "dataset"
    writer = LeRobotDataset.create(
        repo_id="test/wall-six-dim", fps=10, features=features, root=root, use_videos=False
    )
    for index in range(4):
        writer.add_frame(
            {
                OBS_STATE: torch.arange(6).float() + index,
                ACTION: torch.arange(6).float() - index,
                **{key: torch.full((3, 16, 16), index / 4) for key in cameras},
                "task": "pick up the red cup",
            }
        )
    writer.save_episode()
    writer.finalize()
    dataset = LeRobotDataset("test/wall-six-dim", root=root, delta_timestamps={ACTION: [0, 0.1]})
    assert len(dataset.meta.stats[ACTION]["q01"]) == 6
    processor = _tiny_qwen_processor()

    config = _config(
        chunk_size=2, n_action_steps=2, dtype=torch.float32, vlm_config=_tiny_native_config().to_dict()
    )
    # Start from a canonical 26D checkpoint with different camera names, just
    # like --policy.path plus --policy.camera_key_mapping on lerobot-train.
    base = tmp_path / "base"
    WallOSS05Policy(config).save_pretrained(base)
    base_pre, base_post = make_pre_post_processors(config, dataset_stats=_stats())
    base_pre.save_pretrained(base)
    base_post.save_pretrained(base)
    config = WallOSS05Config.from_pretrained(
        base, cli_overrides=[f"--camera_key_mapping={json.dumps(cameras)}"]
    )
    config.pretrained_path = base
    policy = make_policy(config, ds_meta=dataset.meta)
    assert config.input_features[OBS_STATE].shape == (6,)
    assert config.action_feature.shape == (6,)
    assert set(config.image_features) == set(cameras)
    assert all(feature.shape == (3, 16, 16) for feature in config.image_features.values())
    processor_kwargs = {"dataset_stats": dataset.meta.stats}
    if saved_processors:
        processor_kwargs.update(
            pretrained_path=base,
            preprocessor_overrides={
                "normalizer_processor": {
                    "features": {**config.input_features, **config.output_features},
                    "stats": dataset.meta.stats,
                }
            },
            postprocessor_overrides={
                "unnormalizer_processor": {"features": config.output_features, "stats": dataset.meta.stats}
            },
        )
    pre, post = make_pre_post_processors(config, **processor_kwargs)
    for step in pre.steps:
        if isinstance(step, WallOSS05TokenizerStep):
            step._processor = processor
    batch = pre(next(iter(DataLoader(dataset, batch_size=2))))
    assert not batch["dof_mask"][..., 6:].any()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert policy.model.visual.patch_embed.proj.weight.grad.norm() > 0
    optimizer.step()
    output = policy.predict_action_chunk(batch)
    assert post(output).shape == (2, 2, 6)
    pre.save_pretrained(tmp_path / "checkpoint")
    post.save_pretrained(tmp_path / "checkpoint")
    policy.save_pretrained(tmp_path / "checkpoint")
    restored = WallOSS05Policy.from_pretrained(tmp_path / "checkpoint", strict=True)
    torch.testing.assert_close(restored.state_dict(), policy.state_dict(), atol=0, rtol=0)
    pre2, post2 = make_pre_post_processors(config, pretrained_path=tmp_path / "checkpoint")
    for step in pre2.steps:
        if isinstance(step, WallOSS05TokenizerStep):
            step._processor = processor
    torch.testing.assert_close(pre2(next(iter(DataLoader(dataset, batch_size=2))))[ACTION], batch[ACTION])
    torch.testing.assert_close(post2(output), post(output))
    # Sharded resume constructs from config before DCP supplies the weights.
    restored.config.pretrained_path = tmp_path / "weights-restored-later"
    deferred = make_policy(restored.config, ds_meta=dataset.meta, defer_weight_load=True)
    assert deferred.model.model.embed_tokens.weight.shape == restored.model.model.embed_tokens.weight.shape


@pytest.mark.parametrize("reload", [False, True])
def test_constant_quantiles_match_wall_normalization_and_inverse(tmp_path, reload):
    config = _config()
    stats = _stats(0.5, 0.5)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    if reload:
        # Published processors used LeRobot's default epsilon before this fix.
        for pipeline in (preprocessor, postprocessor):
            for step in pipeline.steps:
                if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
                    step.eps = 1e-8
            pipeline.save_pretrained(tmp_path)
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=tmp_path)
    _stub_tokenizer(preprocessor)
    batch = _batch(batch_size=1)
    raw = torch.tensor([-0.5, 0.5, 0.75, 1.5] + [0.5] * 22)
    batch[OBS_STATE] = raw[None]
    batch[ACTION] = raw[None, None].expand(1, 32, -1)
    processed = preprocessor(batch)
    expected = torch.tensor([-1.0, -1.0, -0.5, 1.0] + [-1.0] * 22)
    torch.testing.assert_close(processed[OBS_STATE], expected[None], atol=0, rtol=0)
    torch.testing.assert_close(processed[ACTION], expected[None, None].expand(1, 32, -1), atol=0, rtol=0)
    torch.testing.assert_close(
        postprocessor(torch.ones(1, 32, 26)), torch.full((1, 32, 26), 0.5), atol=0, rtol=0
    )


def _tiny_native_config():
    transformers = pytest.importorskip("transformers")
    model_config = transformers.Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "eos_token_id": 11,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "window_size": 8,
            "fullatt_block_indexes": [0],
        },
        dim_inputs=[32, 16],
        experts=[{"hidden_size": 32, "intermediate_size": 64}, {"hidden_size": 16, "intermediate_size": 32}],
        action_hidden_size=16,
        action_token_id=31,
        noise_scheduler={"s": 0.999},
        image_token_id=28,
        vision_start_token_id=29,
        vision_end_token_id=30,
    )
    model_config.vision_config._attn_implementation = "sdpa"
    return model_config


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_vision_norms_apply_scale_before_casting_to_activation_dtype(dtype):
    """WALL-X scales its FP32 normalized activations before rounding back to BF16."""
    config = _tiny_native_config()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 31))
    model = WallOSS05Model(config, processor)
    norms = [model.visual.blocks[0].norm1, model.visual.blocks[0].norm2, model.visual.merger.ln_q]
    for norm in norms:
        norm.to(dtype)
        width = norm.weight.numel()
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.3, 1.7, width))
        inputs = torch.linspace(-3.7, 4.1, 9 * width).reshape(9, width).to(dtype)
        values = inputs.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.variance_epsilon)
        expected = (norm.weight.float() * normalized).to(dtype)
        torch.testing.assert_close(norm(inputs), expected, atol=0, rtol=0)


def _uncached_flow_sample(model, inputs, noise):
    """Independent full-sequence Euler reference retained from the original port."""
    actions = noise.clone()
    batch_size, horizon, width = actions.shape
    times = model.action_preprocessor.inference_times(10, actions.device)
    for index in range(10):
        embeddings = model.action_preprocessor.embed(
            times[index].expand(batch_size), actions, inputs["dof_mask"]
        )
        hidden = model._hidden_states(
            **{key: value for key, value in inputs.items() if key != "dof_mask"}, action_embeddings=embeddings
        )
        action_hidden = hidden[
            inputs["input_ids"] == model.action_token_id, : model.config.action_hidden_size
        ]
        velocity = model.action_preprocessor.action_proj_back(action_hidden.float()).reshape(
            batch_size, horizon, width
        )
        actions = actions + (times[index + 1] - times[index]) * velocity
    return actions


@pytest.mark.parametrize("batch_size", [1, 2])
def test_flow_sampling_caches_vision_and_prefix_with_action_parity(batch_size):
    processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 31))
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(19)
        model = WallOSS05Model(_tiny_native_config(), processor).eval()
        ids = torch.tensor([[0, 0, 4, 29, 28, 30, 29, 28, 30, 6, 31, 31]] * batch_size)
        if batch_size == 2:
            ids[1, :2] = torch.tensor([4, 5])
        inputs = {
            "input_ids": ids,
            "attention_mask": ids != 0,
            "pixel_values": torch.randn(8 * batch_size, 24),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * (2 * batch_size)),
            "moe_token_types": ids == 31,
            "dof_mask": (torch.arange(26) < 6).expand(batch_size, 2, -1).clone(),
        }
        noise = torch.randn(batch_size, 2, 26)
        expected = _uncached_flow_sample(model, inputs, noise)
        vision_calls, decoded_lengths = [], []
        vision_hook = model.visual.register_forward_pre_hook(lambda module, args: vision_calls.append(1))
        decoder_hook = model.model.layers[0].register_forward_pre_hook(
            lambda module, args: decoded_lengths.append(args[0].shape[1])
        )
        actual = model.generate_flow_action(
            **inputs, action_horizon=2, action_dim=26, num_inference_timesteps=10, noise=noise
        )["predict_action"]
        vision_hook.remove()
        decoder_hook.remove()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        assert len(vision_calls) == 1
        assert decoded_lengths == [ids.shape[1] - 2] + [2] * 10
        # A new observation must get its own prefix cache.
        inputs["pixel_values"] = inputs["pixel_values"] + 1
        expected = _uncached_flow_sample(model, inputs, noise)
        actual = model.generate_flow_action(
            **inputs, action_horizon=2, action_dim=26, num_inference_timesteps=10, noise=noise
        )["predict_action"]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("joint", [False, True], ids=["text-only", "joint-text-action"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_native_multimodal_model_trains_language_for_five_steps(joint, batch_size):
    """Exercise the real vision encoder, routed decoder, tied head and flow expert on CPU."""
    model_config = _tiny_native_config()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 31))
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = WallOSS05Model(model_config, processor)
        policy = WallOSS05Policy(
            _config(chunk_size=2, n_action_steps=2, flow_loss_weight=2.0, text_loss_weight=0.5),
            load_model=False,
        )
        policy.model = model
        ids = [4, 29, 28, 30, 29, 28, 30, 6, 7, 8] + ([31, 31] if joint else [])
        input_ids = torch.tensor([ids] * batch_size)
        labels = torch.full_like(input_ids, -100)
        labels[:, 8:10] = input_ids[:, 8:10]
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.randn(8 * batch_size, 24),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * (2 * batch_size)),
            "moe_token_types": input_ids == 31,
            "dof_mask": torch.ones(batch_size, 2, 26, dtype=torch.bool),
            "text_labels": labels,
            WALL_OSS_PREDICT_ACTIONS: torch.full((batch_size,), joint),
            ACTION: torch.zeros(batch_size, 2, 26),
        }
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
        text_losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss, metrics = policy(batch)
            loss.backward()
            assert torch.isfinite(loss)
            assert model.lm_head.weight.grad is not None
            assert model.lm_head.weight.grad.isfinite().all()
            assert model.lm_head.weight.grad.norm() > 0
            assert model.visual.patch_embed.proj.weight.grad is not None
            assert model.visual.patch_embed.proj.weight.grad.norm() > 0
            assert ("flow_loss" in metrics) is joint
            assert loss.item() == pytest.approx(
                0.5 * metrics["text_loss"] + 2.0 * metrics.get("flow_loss", 0.0), rel=1e-6
            )
            text_losses.append(metrics["text_loss"])
            optimizer.step()
        assert text_losses[-1] < text_losses[0]


def test_native_checkpoint_roundtrip_from_another_working_directory(tmp_path, monkeypatch):
    config = _config(vlm_config=_tiny_native_config().to_dict(), dtype=torch.float32)
    policy = WallOSS05Policy(config)
    output = tmp_path / "checkpoint"
    policy.save_pretrained(output)
    saved_config = WallOSS05Config.from_pretrained(output)
    assert saved_config.pretrained_name_or_path == "lerobot/wall-oss-0.5"
    assert saved_config.vlm_config == json.loads(json.dumps(config.vlm_config))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    restored = WallOSS05Policy.from_pretrained(output, strict=True)
    torch.testing.assert_close(restored.state_dict(), policy.state_dict(), atol=0, rtol=0)
    assert restored.model.lm_head.weight is restored.model.model.embed_tokens.weight
    with safe_open(output / "model.safetensors", framework="pt") as checkpoint:
        # The shared saver retains just one alias of the tied vocabulary matrix.
        assert len(set(checkpoint.keys()) & {"model.model.embed_tokens.weight", "model.lm_head.weight"}) == 1


@pytest.mark.parametrize("main_process", [False, True])
def test_save_uses_shared_collective_and_main_process_gate(tmp_path, monkeypatch, main_process):
    import lerobot.distributed.checkpoint as checkpoint
    import lerobot.distributed.utils as distributed_utils

    policy = WallOSS05Policy(_config(), load_model=False)
    policy.model = _TinyTiedAuthorModel()
    gathered = []

    def gather(module):
        gathered.append(module)
        # DDP has a full state dict even off-main, which must still not write.
        return module.state_dict()

    monkeypatch.setattr(checkpoint, "full_model_state_dict", gather)
    monkeypatch.setattr(distributed_utils, "is_main_process", lambda: main_process)
    policy.save_pretrained(tmp_path)
    assert gathered == [policy]
    assert (tmp_path / "model.safetensors").exists() is main_process
    assert (tmp_path / "config.json").exists() is main_process
