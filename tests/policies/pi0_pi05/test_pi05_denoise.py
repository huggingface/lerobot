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

"""PI0.5's denoise loop: the shared read-only prefix cache and the CUDA-graph replay."""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.common.flow_matching import euler_integrate  # noqa: E402
from lerobot.policies.common.vla_utils import clone_past_key_values, make_att_2d_masks  # noqa: E402
from lerobot.policies.pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="PI0.5 at full size needs a GPU")


def _build(cls=PI05Pytorch):
    torch.manual_seed(0)
    model = cls(PI05Config(device="cuda", dtype=torch.bfloat16)).cuda().eval()
    # Fresh AdaRMS conditioning layers are zero, which would make every step ignore its timestep
    # and let a stale timestep buffer go unnoticed.
    with torch.no_grad():
        for p in model.parameters():
            if not p.any():
                p.normal_(std=0.02)
    return model


@pytest.fixture(scope="module")
def model():
    return _build()


def _inputs(config, batch_size, seed, real_tokens=9):
    g = torch.Generator(device="cuda").manual_seed(seed)
    images = [torch.rand(batch_size, 3, *config.image_resolution, device="cuda", generator=g) * 2 - 1]
    tokens = torch.randint(0, 1024, (batch_size, 12), device="cuda", generator=g)
    masks = torch.ones(batch_size, 12, dtype=torch.bool, device="cuda")
    masks[:, real_tokens:] = False  # padded prompt slots, as the tokenizer pads
    return {
        "images": images,
        "img_masks": [torch.ones(batch_size, dtype=torch.bool, device="cuda")],
        "tokens": tokens,
        "masks": masks,
        "noise": torch.randn(
            batch_size, config.chunk_size, config.max_action_dim, device="cuda", generator=g
        ),
        "num_steps": config.num_inference_steps,
    }


def _reference_sample_actions(model, images, img_masks, tokens, masks, noise, num_steps):
    """The loop before the read-only cache: every step clones the prefill's cache, lets the
    suffix append to the clone, and rebuilds the suffix masks from ``embed_suffix``."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    mask_dtype = prefix_embs.dtype if model.use_typed_attention_masks else None
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=model._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=mask_dtype),
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    def step(x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, timestep)
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
            attention_mask=model._prepare_attention_masks_4d(full),
            position_ids=position_ids,
            past_key_values=clone_past_key_values(cache),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return model.action_out_proj(out[1][:, -model.config.chunk_size :].to(dtype=torch.float32))

    return euler_integrate(step, noise, num_steps)


@torch.no_grad()
def test_the_read_only_prefix_cache_matches_the_cloned_cache(model):
    model.config.denoise_cuda_graph = False
    kw = _inputs(model.config, batch_size=2, seed=1)
    assert torch.equal(model.sample_actions(**kw), _reference_sample_actions(model, **kw))


@torch.no_grad()
def test_the_cuda_graph_matches_eager_across_chunks_and_batch_sizes(model):
    model.__dict__.pop("_denoise_graphs", None)
    try:
        # the first two share a graph but differ in images, tokens, padding and noise
        for batch_size, seed, real_tokens in [(1, 2, 9), (1, 3, 5), (2, 4, 9)]:
            kw = _inputs(model.config, batch_size, seed, real_tokens)
            model.config.denoise_cuda_graph = False
            eager = model.sample_actions(**kw)
            model.config.denoise_cuda_graph = True
            graphed = model.sample_actions(**kw)
            torch.testing.assert_close(graphed, eager, rtol=0, atol=1e-5)
        # one capture per batch size, reused across chunks
        assert len(model._denoise_graphs) == 2
    finally:
        model.config.denoise_cuda_graph = False


@torch.no_grad()
def test_the_cuda_graph_follows_the_noise_and_can_be_reset(model):
    model.__dict__.pop("_denoise_graphs", None)
    model.config.denoise_cuda_graph = True
    try:
        kw = _inputs(model.config, batch_size=1, seed=7)
        first = model.sample_actions(**kw)
        # same observation, other noise: a replay that kept the captured noise would repeat itself
        other = torch.randn(
            kw["noise"].shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(8)
        )
        second = model.sample_actions(**{**kw, "noise": other})
        assert not torch.equal(first, second)
        assert len(model._denoise_graphs) == 1
        model.reset_denoise_graphs()
        assert not model.__dict__.get("_denoise_graphs")
    finally:
        model.config.denoise_cuda_graph = False


class _OwnDenoiseStep(PI05Pytorch):
    """A subclass on the four-argument contract that manages the prefill's cache itself, as
    FineART's does."""

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        assert not hasattr(past_key_values, "kv"), "expected the prefill's own cache"
        return super().denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)


@torch.no_grad()
def test_a_subclass_with_its_own_denoise_step_keeps_its_contract(model):
    sub = _OwnDenoiseStep(model.config).cuda().eval()
    sub.load_state_dict(model.state_dict())
    sub.config.denoise_cuda_graph = True
    try:
        kw = _inputs(model.config, batch_size=1, seed=6)
        assert torch.equal(sub.sample_actions(**kw), _reference_sample_actions(model, **kw))
        assert not sub.__dict__.get("_denoise_graphs")
    finally:
        sub.config.denoise_cuda_graph = False
        model.config.denoise_cuda_graph = False


def test_the_cuda_graph_and_compile_model_are_alternatives():
    with pytest.raises(ValueError, match="denoise_cuda_graph and compile_model"):
        PI05Config(denoise_cuda_graph=True, compile_model=True)
