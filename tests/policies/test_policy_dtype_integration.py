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

import draccus
import pytest
import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.dtype import get_dtype


@pytest.mark.parametrize("name", ["pi0", "pi05"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_pi_real_layers_initialize_run_and_reload_with_shared_precision(name, dtype, monkeypatch, tmp_path):
    """Use the actual PI attention/vision/action layers, with tiny dimensions and no downloads."""
    import importlib

    pytest.importorskip("transformers")
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    module = importlib.import_module(f"lerobot.policies.{name}.modeling_{name}")
    policy_cls = getattr(module, f"{name.upper()}Policy")
    monkeypatch.setattr(
        module,
        "get_gemma_config",
        lambda variant: module.GemmaConfig(
            width=32, depth=1, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=4
        ),
    )
    paligemma_cls = module.PaliGemmaForConditionalGenerationWithPiGemma
    expert_cls = module.PiGemmaForCausalLM

    def small_paligemma(config):
        config.text_config.vocab_size = 32
        config.image_token_index = 31
        vision = config.vision_config
        vision.hidden_size = 32
        vision.intermediate_size = 64
        vision.num_hidden_layers = 1
        vision.num_attention_heads = 4
        vision.patch_size = 8
        vision.projection_dim = 32
        return paligemma_cls(config)

    def small_expert(config):
        config.vocab_size = 32
        return expert_cls(config)

    monkeypatch.setattr(module, "PaliGemmaForConditionalGenerationWithPiGemma", small_paligemma)
    monkeypatch.setattr(module, "PiGemmaForCausalLM", small_expert)
    compile_dtypes = []

    def record_compile(fn, **kwargs):
        core = fn.__self__
        layer = core.paligemma_with_expert.gemma_expert.model.layers[0]
        compile_dtypes.append((layer.self_attn.q_proj.weight.dtype, core.action_out_proj.weight.dtype))
        return fn

    monkeypatch.setattr(torch, "compile", record_compile)
    config = policy_cls.config_class(
        dtype=dtype,
        device="cpu",
        image_resolution=(16, 16),
        max_action_dim=4,
        max_state_dim=4,
        chunk_size=2,
        n_action_steps=2,
        num_inference_steps=1,
        compile_model=True,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (4,)),
            "observation.images.camera": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (4,))},
    )
    policy = policy_cls(config)
    assert compile_dtypes == [(get_dtype(dtype), torch.float32)] * 2
    assert policy.dtype == get_dtype(dtype)
    assert policy.model.action_out_proj.weight.dtype == torch.float32
    backbone = policy.model.paligemma_with_expert
    assert backbone.paligemma.model.language_model.norm.weight.dtype == torch.float32
    for tower in (backbone.paligemma.model.language_model, backbone.gemma_expert.model):
        assert tower.rotary_emb.inv_freq.dtype == get_dtype(dtype)
        for layer in tower.layers:
            assert all(p.dtype == torch.float32 for p in layer.input_layernorm.parameters())
            assert all(p.dtype == torch.float32 for p in layer.post_attention_layernorm.parameters())
    assert all(p.dtype == torch.float32 for p in backbone.paligemma.model.multi_modal_projector.parameters())
    assert (
        backbone.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        == torch.float32
    )
    assert backbone.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype == get_dtype(dtype)
    batch = {
        OBS_STATE: torch.randn(1, 4),
        ACTION: torch.randn(1, 2, 4),
        "observation.images.camera": torch.rand(1, 3, 16, 16),
        "observation.language.tokens": torch.tensor([[1, 2, 3]]),
        "observation.language.attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.model.action_out_proj.weight.grad.dtype == torch.float32
    with torch.no_grad():
        assert torch.isfinite(policy.predict_action_chunk(batch)).all()
        policy.model.action_out_proj.weight.fill_(1.000123)
    policy.save_pretrained(tmp_path)
    restored = policy_cls.from_pretrained(tmp_path, dtype=dtype)
    assert restored.dtype == get_dtype(dtype)
    assert not restored.training
    assert torch.equal(policy.model.action_out_proj.weight, restored.model.action_out_proj.weight)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_act_inference_uses_parameter_dtype_for_generated_latents(dtype):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

    config = ACTConfig(
        device="cpu",
        dtype=dtype,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (4,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (4,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (3,))},
    )
    policy = ACTPolicy(config).eval()
    batch = {key: torch.zeros(1, 4, dtype=policy.dtype) for key in config.input_features}
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    assert actions.dtype == get_dtype(dtype)
    assert actions.shape == (1, 2, 3)
    assert torch.isfinite(actions).all()


@pytest.mark.parametrize("config_class", [PI0Config, PI05Config, PI0FastConfig])
def test_pi_configs_keep_explicit_float32_default(config_class):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        config = config_class(device="cpu")
        assert config.dtype == "float32"
        with draccus.config_type("json"):
            override = draccus.parse(config_class, args=["--device=cpu", "--dtype=bfloat16"])
        assert override.dtype == "bfloat16"
    finally:
        torch.set_default_dtype(previous)
