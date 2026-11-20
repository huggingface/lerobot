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

"""Cross-episode prompt reuse with a tiny, local text encoder (no model downloads)."""

from copy import deepcopy
from unittest.mock import Mock
from weakref import ref

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.utils.constants import ACTION


@pytest.fixture
def policy() -> LingBotVAPolicy:
    config = LingBotVAConfig(
        device="cpu",
        dtype=torch.float32,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_attention_heads=2,
        attention_head_dim=24,
        num_layers=1,
        max_sequence_length=8,
        obs_cam_keys=["observation.images.image"],
    )
    config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = LingBotVAPolicy(config)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "pick": 2, "place": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    policy._frozen = {
        "tokenizer": PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
        ),
        "text_encoder": UMT5EncoderModel(
            UMT5Config(vocab_size=4, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2, dropout_rate=0)
        ).eval(),
        "streaming_vae": Mock(),
        "streaming_vae_half": Mock(),
    }
    return policy


def test_repeated_prompt_reuses_embeddings_after_reset(policy: LingBotVAPolicy) -> None:
    calls = Mock(return_value=None)
    policy._frozen["text_encoder"].register_forward_hook(calls)
    state_keys = set(policy.state_dict())
    policy._maybe_init_prompt({"task": "pick"})
    positive, negative = policy._prompt_embeds, policy._negative_prompt_embeds
    assert calls.call_count == 2
    assert torch.count_nonzero(positive[:, 1:]) == 0  # padding stays zero
    for task in (["pick"], ("pick",)):
        policy.reset()
        assert policy._prompt_embeds is None and policy._negative_prompt_embeds is None
        policy._maybe_init_prompt({"task": task})
        assert policy._prompt_embeds is positive
        assert policy._negative_prompt_embeds is negative
    assert calls.call_count == 2
    assert not positive.requires_grad and not negative.requires_grad
    with torch.no_grad():
        expected = policy._encode_prompt("pick")
    torch.testing.assert_close((positive, negative), expected, rtol=0, atol=0)
    assert set(policy.state_dict()) == state_keys


def test_changed_prompt_evicts_previous_embeddings(policy: LingBotVAPolicy) -> None:
    policy._maybe_init_prompt({"task": "pick"})
    original = policy._prompt_embeds.clone()
    previous = ref(policy._prompt_embeds)
    policy.reset()
    policy._maybe_init_prompt({"task": "place"})
    assert previous() is None  # no growing per-task dictionary
    assert not torch.equal(policy._prompt_embeds, original)
    policy.reset()
    policy._maybe_init_prompt({"task": "pick"})
    torch.testing.assert_close(policy._prompt_embeds, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "setting",
    [
        "length",
        "dtype",
        "cfg_off",
        "action_cfg",
        "encoder",
        "tokenizer",
        "encoder_dtype",
        "device",
        "encoder_device",
    ],
)
def test_encoding_changes_invalidate_cache(policy: LingBotVAPolicy, setting: str) -> None:
    if setting in ("device", "encoder_device") and not torch.cuda.is_available():
        pytest.skip("device change requires a CUDA GPU")
    if setting == "action_cfg":
        policy.config.guidance_scale = 1
        policy.reset()
    policy._maybe_init_prompt({"task": "pick"})
    previous = policy._prompt_embeds
    if setting == "length":
        policy.config.max_sequence_length = 4
    elif setting == "dtype":
        policy.dtype = torch.bfloat16
    elif setting == "cfg_off":
        policy.config.guidance_scale = 1
    elif setting == "action_cfg":
        policy.config.action_guidance_scale = 2
    elif setting in ("encoder", "tokenizer"):
        key = "text_encoder" if setting == "encoder" else setting
        old_resource = ref(policy._frozen[key])
        policy._frozen[key] = deepcopy(policy._frozen[key])
        assert old_resource() is None  # the cache must not keep replaced frozen modules alive
    elif setting == "encoder_dtype":
        policy._frozen["text_encoder"].to(torch.float64)
    elif setting == "device":
        policy.config.device = "cuda"
    elif setting == "encoder_device":
        policy._frozen["text_encoder"].to("cuda")
    policy.reset()
    policy._maybe_init_prompt({"task": "pick"})
    assert policy._prompt_embeds is not previous
    with torch.no_grad():
        expected = policy._encode_prompt("pick")
    torch.testing.assert_close(
        (policy._prompt_embeds, policy._negative_prompt_embeds), expected, rtol=0, atol=0
    )
    if setting == "cfg_off":
        assert policy._negative_prompt_embeds is None
    else:
        assert policy._negative_prompt_embeds is not None


@pytest.mark.parametrize("encoder_device, output_device", [("cpu", "cpu"), ("cpu", "cuda"), ("cuda", "cuda")])
@pytest.mark.parametrize(
    "initial_dtype, next_dtype",
    [(None, torch.bfloat16), (torch.bfloat16, None), (torch.bfloat16, torch.float16)],
)
def test_autocast_changes_invalidate_cache(
    policy: LingBotVAPolicy,
    initial_dtype: torch.dtype | None,
    next_dtype: torch.dtype | None,
    encoder_device: str,
    output_device: str,
) -> None:
    if output_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA output requires a GPU")
    policy.config.device = output_device
    policy._frozen["text_encoder"].to(encoder_device)
    with torch.autocast(encoder_device, dtype=initial_dtype, enabled=initial_dtype is not None):
        policy._maybe_init_prompt({"task": "pick"})
    previous = policy._prompt_embeds
    policy.reset()
    with torch.no_grad(), torch.autocast(encoder_device, dtype=next_dtype, enabled=next_dtype is not None):
        policy._maybe_init_prompt({"task": "pick"})
        assert policy._prompt_embeds is not previous
        expected = policy._encode_prompt("pick")
        torch.testing.assert_close(
            (policy._prompt_embeds, policy._negative_prompt_embeds), expected, rtol=0, atol=0
        )
        current = policy._prompt_embeds
        policy.reset()
        policy._maybe_init_prompt({"task": "pick"})
        assert policy._prompt_embeds is current


def test_reset_clears_episode_state_with_a_cached_prompt(policy: LingBotVAPolicy) -> None:
    policy._maybe_init_prompt({"task": "pick"})
    policy._action_queue.append(torch.ones(1, 7))
    policy._obs_buffer.append({"image": torch.ones(3, 32, 32)})
    policy._executed_actions = torch.ones(1, 7)
    policy._frame_st_id = 5
    policy._first_chunk = False
    policy._cache_initialised = True
    policy.last_predicted_frames = torch.ones(1)
    policy.last_predicted_latents = torch.ones(1)
    for block in policy.transformer.blocks:
        block.attn1.attn_caches["pos"] = {"k": torch.ones(1), "v": torch.ones(1)}
    policy.reset()
    assert not policy._action_queue and not policy._obs_buffer
    assert policy._executed_actions is None and policy._frame_st_id == 0
    assert policy._first_chunk and not policy._cache_initialised
    assert policy.last_predicted_frames is None and policy.last_predicted_latents is None
    assert all(block.attn1.attn_caches["pos"] is None for block in policy.transformer.blocks)
    policy._frozen["streaming_vae"].clear_cache.assert_called_once_with()
    policy._frozen["streaming_vae_half"].clear_cache.assert_called_once_with()


def test_missing_task_uses_empty_prompt(policy: LingBotVAPolicy) -> None:
    policy._maybe_init_prompt(None)
    assert policy._prompt_embeds is None
    policy._maybe_init_prompt({})
    previous = policy._prompt_embeds
    assert policy._prompt == ""
    policy.reset()
    policy._maybe_init_prompt({"task": ""})
    assert policy._prompt_embeds is previous
