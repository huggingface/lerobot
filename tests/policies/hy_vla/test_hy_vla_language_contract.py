#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Exercise the normal Hy-VLA factory/pipeline with a real, local tokenizer."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("datasets")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from lerobot.datasets.recipe import MessageTurn, TrainingRecipe
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.hy_vla.configuration_hy_vla import HyVLAConfig
from lerobot.policies.hy_vla.modeling_hy_vla import HyVLAFlowMatching
from lerobot.policies.hy_vla.processor_hy_vla import (
    HY_VLA_ATTENTION_MASK,
    HY_VLA_INPUT_IDS,
    HY_VLA_PREDICT_ACTIONS,
    HyVLADecodeStep,
    HyVLAEncodeStep,
    HyVLATokenizerStep,
    make_hy_vla_pre_post_processors,
)
from lerobot.processor import RenderRuntimeMessagesStep, RenderTrainingMessagesStep
from tests.policies.hy_vla.test_processor_hy_vla import _batch, _identity_relative, _native_state, _stats
from tests.policies.hy_vla.test_training_smoke import _lightweight_policy, _text_batch


def _tokenizer():
    specials = ["[UNK]", "[PAD]", "[EOS]", "<｜hy_User｜>", "<｜hy_Assistant｜>"]
    words = ["Goal", ":", "tidy", "lift", "cup", "What", "is", "visible", "?"]
    tokenizer = Tokenizer(WordLevel({text: i for i, text in enumerate(specials + words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=specials[3:],
    )


def _recipe(stream="high_level"):
    return TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="Goal: ${task}", stream=stream),
            MessageTurn(
                role="assistant", content="${subtask}", stream=stream, target=True, if_present="subtask"
            ),
        ]
    )


def _pipeline(config):
    pre, post = make_hy_vla_pre_post_processors(config, norm_stats=_stats(50))
    tokenizer = _tokenizer()
    next(step for step in pre.steps if isinstance(step, HyVLATokenizerStep))._tokenizer = tokenizer
    return pre, post, tokenizer


def _annotated_batch():
    state = _native_state()
    return {
        **_batch(state, state.repeat(50, 1)),
        "task": "tidy",
        "timestamp": 0.0,
        "language_persistent": [
            {"timestamp": 0.0, "style": "subtask", "role": "assistant", "content": "lift cup"}
        ],
    }


@pytest.mark.parametrize("stream,predicts", [("high_level", False), ("low_level", True)])
def test_recipe_supervises_assistant_and_eos_only(stream, predicts):
    pre, _, tokenizer = _pipeline(HyVLAConfig(device="cpu", recipe=asdict(_recipe(stream))))
    processed = pre(_annotated_batch())
    assert isinstance(pre.steps[0], RenderRuntimeMessagesStep)
    assert isinstance(pre.steps[1], RenderTrainingMessagesStep)
    labels = processed["text_labels"]
    expected_ids = tokenizer("lift cup[EOS]", add_special_tokens=False)["input_ids"]
    assert labels[labels != -100].tolist() == expected_ids
    assert processed[HY_VLA_PREDICT_ACTIONS].tolist() == [predicts]
    assert (labels[~processed[HY_VLA_ATTENTION_MASK]] == -100).all()

    runtime = pre({**_batch(_native_state()), "query_kind": "next_subtask", "query_text": "tidy"})
    prompt_ids = runtime[HY_VLA_INPUT_IDS][runtime[HY_VLA_ATTENTION_MASK]].tolist()
    first_target = (labels[0] != -100).nonzero()[0].item()
    assert prompt_ids == processed[HY_VLA_INPUT_IDS][0, :first_target].tolist()
    assert "text_labels" not in runtime
    assert "query_kind" not in runtime and "query_text" not in runtime


def test_action_only_fallback_keeps_native_prompt_and_has_no_text_labels():
    pre, _, tokenizer = _pipeline(HyVLAConfig(device="cpu"))
    state = _native_state()
    output = pre({**_batch(state, state.repeat(50, 1)), "task": "tidy"})
    assert (
        output[HY_VLA_INPUT_IDS][output[HY_VLA_ATTENTION_MASK]].tolist()
        == tokenizer("tidy<｜hy_Assistant｜>", add_special_tokens=False)["input_ids"]
    )
    assert output[HY_VLA_PREDICT_ACTIONS].tolist() == [True]
    assert "text_labels" not in output


def test_training_recipe_can_be_disabled_and_truncation_reports_lost_targets():
    pre, _, _ = _pipeline(HyVLAConfig(device="cpu", use_language_recipe=False))
    assert "text_labels" not in pre(_annotated_batch())
    pre, _, _ = _pipeline(HyVLAConfig(device="cpu", tokenizer_max_length=1))
    with pytest.raises(ValueError, match="truncated every assistant target"):
        pre(_annotated_batch())


@pytest.mark.parametrize("legacy", ["current", "runtime_only", "released"])
def test_factory_reload_restores_geometry_link_and_language_recipe(tmp_path, legacy):
    import json

    config = HyVLAConfig(device="cpu", recipe=asdict(_recipe()))
    pre, post, tokenizer = _pipeline(config)
    if legacy != "current":
        pre.steps = [step for step in pre.steps if not isinstance(step, RenderTrainingMessagesStep)]
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path)
    if legacy == "released":
        filename = tmp_path / "policy_preprocessor.json"
        serialized = json.loads(filename.read_text())
        serialized["steps"][0]["registry_name"] = "render_messages_processor"
        serialized["steps"][0]["config"]["render_training"] = False
        filename.write_text(json.dumps(serialized))
    loaded_config = HyVLAConfig.from_pretrained(tmp_path)
    loaded_config.recipe = HyVLAConfig(device="cpu").recipe
    loaded_pre, loaded_post = make_pre_post_processors(loaded_config, pretrained_path=tmp_path)
    encoder = next(step for step in loaded_pre.steps if isinstance(step, HyVLAEncodeStep))
    decoder = next(step for step in loaded_post.steps if isinstance(step, HyVLADecodeStep))
    assert decoder.encoder is encoder
    assert len([step for step in loaded_pre.steps if isinstance(step, RenderRuntimeMessagesStep)]) == 1
    assert len([step for step in loaded_pre.steps if isinstance(step, RenderTrainingMessagesStep)]) == 1
    annotated = loaded_pre(_annotated_batch())
    if legacy == "released":
        assert "text_labels" not in annotated
    else:
        assert (annotated["text_labels"] != -100).any()
    runtime = loaded_pre({**_batch(_native_state()), "query_kind": "next_subtask", "query_text": "tidy"})
    assert runtime["messages_rendered"] == [[{"role": "user", "content": "Goal: tidy"}]]
    loaded_pre(_batch(_native_state()))
    torch.testing.assert_close(loaded_post(_identity_relative()), _native_state())


def test_fresh_recipe_training_preserves_checkpoint_normalization(tmp_path):
    import json

    config = HyVLAConfig(device="cpu")
    stats = _stats(50)
    stats["qpos_mean"].fill_(3.0)
    stats["action_std"].fill_(2.0)
    pre, post = make_hy_vla_pre_post_processors(config, norm_stats=stats)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    filename = tmp_path / "policy_preprocessor.json"
    serialized = json.loads(filename.read_text())
    next(step for step in serialized["steps"] if step["registry_name"] == "device_processor")["config"][
        "device"
    ] = "cuda"
    filename.write_text(json.dumps(serialized))
    config.pretrained_path = tmp_path
    config.recipe = asdict(_recipe())
    rebuilt, rebuilt_post = make_pre_post_processors(
        config, dataset_stats={"observation.state": {"mean": torch.zeros(16), "std": torch.ones(16)}}
    )
    encoder = next(step for step in rebuilt.steps if isinstance(step, HyVLAEncodeStep))
    decoder = next(step for step in rebuilt_post.steps if isinstance(step, HyVLADecodeStep))
    torch.testing.assert_close(encoder.qpos_mean, stats["qpos_mean"])
    torch.testing.assert_close(decoder.action_std, stats["action_std"])
    assert rebuilt.steps[0].recipe.messages[0].content == "Goal: ${task}"


def test_policy_save_uses_current_main_checkpoint_contract(tmp_path):
    from safetensors.torch import load_file

    policy = _lightweight_policy()
    policy.config.vlm_config_dict = {"model_type": "test"}
    policy.language_tokenizer = _tokenizer()
    policy.save_pretrained(tmp_path)
    assert "model.scale" in load_file(tmp_path / "model.safetensors")
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "tokenizer.json").is_file()


def test_recipe_override_remains_portable_after_source_yaml_is_removed(tmp_path):
    import yaml

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(asdict(_recipe())))
    config = HyVLAConfig(device="cpu", recipe_path=str(recipe_path))
    config.save_pretrained(tmp_path / "checkpoint")
    recipe_path.unlink()
    restored = HyVLAConfig.from_pretrained(tmp_path / "checkpoint")
    assert restored.recipe == config.recipe
    pre, _, _ = _pipeline(restored)
    output = pre({**_batch(_native_state()), "query_kind": "next_subtask", "query_text": "tidy"})
    assert output["messages_rendered"] == [[{"role": "user", "content": "Goal: tidy"}]]


def test_flow_reduction_excludes_high_level_rows_without_diluting_action_loss():
    policy = _lightweight_policy()
    batch = {HY_VLA_PREDICT_ACTIONS: torch.tensor([True, False])}
    losses = torch.stack([torch.ones(50, 32), torch.full((50, 32), 100.0)])
    assert policy._reduce_flow_loss(losses, batch, "mean").item() == 1.0
    torch.testing.assert_close(policy._reduce_flow_loss(losses, batch, "none"), torch.tensor([1.0, 0.0]))


def test_text_only_forward_skips_action_expert_and_backpropagates():
    model = object.__new__(HyVLAFlowMatching)
    nn.Module.__init__(model)
    head = nn.Linear(2, 6, bias=False)
    model.head = head
    hidden = torch.tensor([[[1.0, -0.5], [0.25, 1.0], [-0.5, 0.5]]])

    def forward(**kwargs):
        assert kwargs["inputs_embeds"][1] is None
        return (hidden, None), None, None, None

    model.dual_tower = SimpleNamespace(
        forward=forward, vlm=SimpleNamespace(language_model=SimpleNamespace(lm_head=head))
    )
    masks = torch.ones(1, 3, dtype=torch.bool)
    model.embed_prefix = lambda *args: (hidden, masks, masks, torch.zeros_like(masks), [], [])
    model.config = SimpleNamespace(visual_segment_isolation=False)
    tokens = torch.tensor([[1, 2, 3]])
    flow, text = model([], [], tokens, masks, None, None, text_labels=torch.tensor([[-100, 2, 3]]))
    assert flow is None
    expected = torch.nn.functional.cross_entropy(head(hidden[:, :-1]).reshape(-1, 6), torch.tensor([2, 3]))
    torch.testing.assert_close(text, expected)
    text.backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("with_action", [False, True])
def test_text_only_policy_rows_do_not_train_flow(with_action):
    class TextOnlyFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.0))

        def forward(self, images, masks, tokens, language_masks, state, actions, noise, time, labels):
            assert state is None and actions is None
            return None, self.scale.square()

    policy = _lightweight_policy()
    policy.model = TextOnlyFlow()
    policy.config.text_loss_weight = 0.25
    batch = _text_batch()
    batch[HY_VLA_PREDICT_ACTIONS] = torch.tensor([False])
    if not with_action:
        batch.pop("action")
    loss, metrics = policy(batch)
    assert loss.item() == 1.0
    assert "flow_loss" not in metrics
    loss.backward()
    assert policy.model.scale.grad.item() == 1.0


def test_text_generation_rejects_multiple_prompts_and_preserves_mem_state():
    policy = _lightweight_policy()
    policy.model.generate_text_tokens = lambda *args, **kwargs: torch.tensor([[5, 6], [5, 6]])
    with pytest.raises(ValueError, match="expected one Hy-VLA prompt"):
        policy.generate_text(_text_batch(batch_size=2))
    policy.config.use_video_encoder = True
    policy.config.img_history_size = 3
    policy.config.img_history_interval = 2
    policy.reset()
    queued = torch.ones(1, 20)
    policy._action_queue.append(queued)
    captured = {}

    def generate(images, *args, **kwargs):
        captured["images"] = images
        return torch.tensor([[5, 6]])

    policy.model.generate_text_tokens = generate
    assert policy.generate_text(_text_batch()) == "decoded"
    assert len(policy._action_queue) == 1 and policy._action_queue[0] is queued
    assert all(len(history) == 0 for history in policy._image_history.values())
    assert captured["images"][0].shape[1] == 3


def test_chunk_serving_does_not_advertise_incompatible_per_step_decoding():
    with pytest.raises(ValueError, match="requires synchronous inference"):
        _lightweight_policy().chunk_inference_spec()


def test_generation_uses_last_unpadded_token_and_masks_cached_padding():
    model = object.__new__(HyVLAFlowMatching)
    nn.Module.__init__(model)
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, -9.0]]])
    head = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        return (hidden if len(calls) == 1 else hidden[:, :1], None), None, None, None

    model.dual_tower = SimpleNamespace(
        forward=forward,
        vlm=SimpleNamespace(
            language_model=SimpleNamespace(
                lm_head=head, model=SimpleNamespace(embed_tokens=nn.Embedding(3, 2))
            )
        ),
    )
    masks = torch.tensor([[True, True, False]])
    model.embed_prefix = lambda *args: (hidden, masks, torch.ones_like(masks), masks, [], [])
    model.config = SimpleNamespace(visual_segment_isolation=False)
    generated = model.generate_text_tokens(
        [], [], torch.zeros(1, 3, dtype=torch.long), masks, max_new_tokens=2, eos_token_ids={0}
    )
    assert generated.tolist() == [[1, 2]]
    assert calls[1]["attention_mask"].tolist() == [[[True, True, False, True]]]
