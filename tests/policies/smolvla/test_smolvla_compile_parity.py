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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, skip_if_package_missing


@skip_if_package_missing("transformers")
@require_cuda
def test_smolvla_tuple_kv_and_compile_parity():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    set_seed(42)

    config = SmolVLAConfig(
        max_action_dim=7,
        chunk_size=50,
        n_action_steps=50,
        num_steps=3,
        use_cache=True,
        compile_denoise=False,
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

    device = config.device
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

    # Eager toggling should be exact
    torch.testing.assert_close(actions_eager, actions_eager_retest, atol=1e-5, rtol=1e-5)

    # Compiled vs eager should match within numerical tolerance
    torch.testing.assert_close(actions_eager, actions_compiled, atol=2e-2, rtol=2e-2)
    cosine_sim = torch.nn.functional.cosine_similarity(
        actions_eager.flatten(), actions_compiled.flatten(), dim=0
    ).item()
    assert cosine_sim > 0.9999, f"Expected high cosine similarity, got {cosine_sim}"
