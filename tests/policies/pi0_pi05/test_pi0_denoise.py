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

"""PI0's denoise loop with one read-only prefix cache shared by its steps."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers import GemmaConfig  # noqa: E402

from lerobot.policies.common.flow_matching import euler_integrate  # noqa: E402
from lerobot.policies.common.vla_utils import (  # noqa: E402
    clone_past_key_values,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from lerobot.policies.pi0 import PI0Config  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import PaliGemmaWithExpertModel, PI0Pytorch  # noqa: E402
from lerobot.policies.pi_gemma import PiGemmaForCausalLM, PiGemmaModel  # noqa: E402
from tests.utils import require_cuda  # noqa: E402


def _reference_sample_actions(model, images, img_masks, lang_tokens, lang_masks, state, noise, num_steps):
    """The loop before the read-only cache: every step clones the prefill's cache, lets the
    suffix append to the clone, and rebuilds the suffix masks from ``embed_suffix``."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(prefix_att_2d_masks),
        position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    def step(x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, _ = model.embed_suffix(state, x_t, timestep)
        batch_size, suffix_len = suffix_pad_masks.shape
        prefix_len = prefix_pad_masks.shape[1]
        full = torch.cat(
            [
                prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len),
                make_att_2d_masks(suffix_pad_masks, suffix_att_masks),
            ],
            dim=2,
        )
        position_ids = (
            torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        )
        out, _ = model.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(full),
            position_ids=position_ids,
            past_key_values=clone_past_key_values(cache),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None],
        )
        return model.action_out_proj(out[1][:, -model.config.chunk_size :].to(dtype=torch.float32))

    return euler_integrate(step, noise, num_steps)


def _inputs(config, batch_size, seed, device, n_tokens=12, n_cams=2):
    g = torch.Generator(device=device).manual_seed(seed)
    size = config.image_resolution
    images = [torch.rand(batch_size, 3, *size, device=device, generator=g) * 2 - 1 for _ in range(n_cams)]
    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(n_cams)]
    img_masks[-1][-1] = False  # a missing camera in the last row
    masks = torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device)
    for row in range(batch_size):
        masks[row, 9 - 2 * row :] = False  # padded prompt slots, a different count per row
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": torch.randint(0, 64, (batch_size, n_tokens), device=device, generator=g),
        "lang_masks": masks,
        "state": torch.randn(batch_size, config.max_state_dim, device=device, generator=g),
        "noise": torch.randn(
            batch_size, config.chunk_size, config.max_action_dim, device=device, generator=g
        ),
        "num_steps": config.num_inference_steps,
    }


def _tiny_gemma(hidden_size, vocab_size=64):
    # pi0's two towers share the attention layout (heads, one KV head, head size), only widths differ
    config = GemmaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        hidden_activation="gelu_pytorch_tanh",
    )
    config._attn_implementation = "eager"
    return config


class _TinyPaliGemmaWithExpert(PaliGemmaWithExpertModel):
    """Two-layer Gemma towers and a linear stand-in for SigLIP, so the loop runs on CPU."""

    def __init__(self, width, expert_width):
        nn.Module.__init__(self)
        self.freeze_vision_encoder = self.train_expert_only = False
        self.paligemma = nn.Module()
        self.paligemma.model = nn.Module()
        self.paligemma.model.language_model = PiGemmaModel(_tiny_gemma(width))
        self.gemma_expert = PiGemmaForCausalLM(_tiny_gemma(expert_width))
        self.gemma_expert.model.embed_tokens = None
        self.patch_proj = nn.Linear(3, width)

    def embed_image(self, image):
        pixels = image.flatten(2).transpose(1, 2).to(self.patch_proj.weight.dtype)
        return self.patch_proj(pixels)  # one token per pixel


class _TinyPI0(PI0Pytorch):
    def __init__(self, dtype, width=32, expert_width=16):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            chunk_size=4,
            max_action_dim=6,
            max_state_dim=5,
            num_inference_steps=3,
            min_period=4e-3,
            max_period=4.0,
            image_resolution=(2, 2),
            rtc_config=None,
        )
        self.rtc_processor = None
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert = _TinyPaliGemmaWithExpert(width, expert_width)
        # bf16 towers with fp32 norms, as PI0 builds them; the projections below stay fp32
        self.paligemma_with_expert.to_bfloat16_for_selected_params(dtype)
        self.action_in_proj = nn.Linear(6, expert_width)
        self.action_out_proj = nn.Linear(expert_width, 6)
        self.state_proj = nn.Linear(5, expert_width)
        self.action_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
        self.action_time_mlp_out = nn.Linear(expert_width, expert_width)


class _TinyOwnDenoiseStep(_TinyPI0):
    """A subclass on the old keyword call that manages the prefill's cache itself."""

    def denoise_step(self, *, state, prefix_pad_masks, past_key_values, x_t, timestep):
        assert not hasattr(past_key_values, "kv"), "expected the prefill's own cache"
        return super().denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_the_read_only_prefix_cache_matches_the_cloned_cache_on_a_tiny_pi0(dtype, batch_size):
    torch.manual_seed(0)
    model = _TinyPI0(dtype).eval()
    kw = _inputs(model.config, batch_size, seed=batch_size, device="cpu")
    assert torch.equal(model.sample_actions(**kw), _reference_sample_actions(model, **kw))


@torch.no_grad()
def test_a_subclass_with_its_own_denoise_step_keeps_its_contract():
    torch.manual_seed(0)
    model = _TinyPI0(torch.float32).eval()
    sub = _TinyOwnDenoiseStep(torch.float32).eval()
    sub.load_state_dict(model.state_dict())
    kw = _inputs(model.config, batch_size=2, seed=6, device="cpu")
    assert torch.equal(sub.sample_actions(**kw), _reference_sample_actions(model, **kw))


@require_cuda
@torch.no_grad()
def test_the_read_only_prefix_cache_matches_the_cloned_cache_at_full_size():
    torch.manual_seed(0)
    model = PI0Pytorch(PI0Config(device="cuda", dtype=torch.bfloat16)).cuda().eval()
    kw = _inputs(model.config, batch_size=2, seed=1, device="cuda")
    assert torch.equal(model.sample_actions(**kw), _reference_sample_actions(model, **kw))
