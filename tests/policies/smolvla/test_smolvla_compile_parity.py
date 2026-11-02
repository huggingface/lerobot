#!/usr/bin/env python

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

import copy

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, skip_if_package_missing


def _build_tiny_smolvla_policy_and_batch(device: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    set_seed(42)

    config = SmolVLAConfig(
        max_action_dim=7,
        chunk_size=10,
        n_action_steps=10,
        num_steps=3,
        num_vlm_layers=2,
        use_cache=True,
        compile_denoise=False,
        device=device,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    dataset_stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    policy = SmolVLAPolicy(config).to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=None, dataset_stats=dataset_stats
    )

    batch = {
        "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["Pick up the object"],
    }
    batch = preprocessor(batch)
    noise = policy.model.sample_noise((1, config.chunk_size, 7), device)
    return policy, batch, noise


@skip_if_package_missing("transformers")
def test_smolvla_tuple_kv_parity():
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    policy, batch, noise = _build_tiny_smolvla_policy_and_batch(device="cpu")
    config = policy.config

    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, dynamic_kv = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        tuple_kv = tuple((layer.keys.clone(), layer.values.clone()) for layer in dynamic_kv.layers)
        tuple_kv_snapshot = tuple((k.clone(), v.clone()) for k, v in tuple_kv)

        prefix_len = prefix_pad_masks.shape[1]
        x_dyn = noise.clone()
        x_tup = noise.clone()
        dt = -1.0 / config.num_steps
        for step in range(config.num_steps):
            t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device="cpu").expand(1)
            dynamic_kv.crop(prefix_len)
            v_dyn = policy.model.denoise_step(prefix_pad_masks, dynamic_kv, x_dyn, t)
            v_tup = policy.model.denoise_step(prefix_pad_masks, tuple_kv, x_tup, t)
            assert torch.equal(v_dyn, v_tup), (
                f"Step {step} velocity mismatch between DynamicCache and tuple KV"
            )
            x_dyn = x_dyn + dt * v_dyn
            x_tup = x_tup + dt * v_tup

        assert torch.equal(x_dyn, x_tup), "Final denoised action mismatch between DynamicCache and tuple KV"
        for (k, v), (k_orig, v_orig) in zip(tuple_kv, tuple_kv_snapshot, strict=True):
            assert torch.equal(k, k_orig) and torch.equal(v, v_orig), "Tuple KV prefix was mutated"


@skip_if_package_missing("transformers")
@require_cuda
def test_smolvla_compile_parity():
    policy, batch, noise = _build_tiny_smolvla_policy_and_batch(device="cuda")

    with torch.no_grad():
        # 1. Eager mode with tuple KV cache
        policy.config.compile_denoise = False
        actions_eager = policy.predict_action_chunk(batch, noise=noise.clone())

        # 2. Compiled mode with dynamic=True
        policy.config.compile_denoise = True
        actions_compiled = policy.predict_action_chunk(batch, noise=noise.clone())

        # 3. Verify toggling compile_denoise back to False returns to eager exactly
        policy.config.compile_denoise = False
        actions_eager_retest = policy.predict_action_chunk(batch, noise=noise.clone())

        # 4. Verify copy.deepcopy after compilation uses the copied module weights
        policy.config.compile_denoise = True
        copied_policy = copy.deepcopy(policy)
        copied_policy.model.action_out_proj.weight.zero_()
        copied_policy.model.action_out_proj.bias.zero_()
        actions_zeroed = copied_policy.predict_action_chunk(batch, noise=noise.clone())

    # Eager toggling should be exact
    assert torch.equal(actions_eager, actions_eager_retest)

    # Zeroed action_out_proj on deep-copied policy must leave noise unchanged (v_t == 0)
    torch.testing.assert_close(actions_zeroed, noise[:, :, :7], atol=1e-6, rtol=1e-6)

    # Compiled vs eager should match within numerical tolerance
    torch.testing.assert_close(actions_eager, actions_compiled, atol=2e-2, rtol=2e-2)
    cosine_sim = torch.nn.functional.cosine_similarity(
        actions_eager.flatten(), actions_compiled.flatten(), dim=0
    ).item()
    assert cosine_sim > 0.9999, f"Expected high cosine similarity, got {cosine_sim}"
