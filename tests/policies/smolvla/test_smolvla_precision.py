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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoModelForImageTextToText, PreTrainedModel, SmolVLMConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.policies.smolvla import smolvlm_with_expert
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@pytest.fixture
def tiny_smolvla_config(monkeypatch: pytest.MonkeyPatch) -> SmolVLAConfig:
    def vlm_config() -> SmolVLMConfig:
        return SmolVLMConfig(
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 16,
                "patch_size": 4,
            },
            text_config={
                "model_type": "llama",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 32,
                "pad_token_id": 0,
                # Match the inherited dtype in the released SmolVLM config.
                "dtype": "bfloat16",
            },
        )

    def load_vlm(*args: Any, **kwargs: Any) -> PreTrainedModel:
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return AutoModelForImageTextToText.from_config(vlm_config(), dtype=dtype)

    monkeypatch.setattr(smolvlm_with_expert.AutoConfig, "from_pretrained", lambda *a, **kw: vlm_config())
    monkeypatch.setattr(smolvlm_with_expert.AutoModelForImageTextToText, "from_pretrained", load_vlm)
    monkeypatch.setattr(
        smolvlm_with_expert.AutoProcessor,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(
            tokenizer=SimpleNamespace(fake_image_token_id=1, global_image_token_id=2)
        ),
    )
    return SmolVLAConfig(
        device="cpu",
        vlm_model_name="tiny-smolvlm",
        load_vlm_weights=True,
        num_vlm_layers=2,
        num_expert_layers=2,
        chunk_size=2,
        n_action_steps=2,
        max_state_dim=4,
        max_action_dim=4,
        resize_imgs_with_padding=(16, 16),
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.images.camera1": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
    )


@pytest.mark.parametrize("load_vlm_weights", [False, True])
@pytest.mark.parametrize("train_expert_only", [False, True])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.bfloat16])
def test_trainable_transformers_use_policy_dtype(
    tiny_smolvla_config: SmolVLAConfig,
    load_vlm_weights: bool,
    train_expert_only: bool,
    dtype: torch.dtype | None,
) -> None:
    tiny_smolvla_config.load_vlm_weights = load_vlm_weights
    tiny_smolvla_config.train_expert_only = train_expert_only
    tiny_smolvla_config.dtype = dtype
    policy = SmolVLAPolicy(tiny_smolvla_config)

    expected = dtype or torch.get_default_dtype()
    trainable = {
        name: param.dtype
        for name, param in policy.model.vlm_with_expert.named_parameters()
        if param.requires_grad
    }
    assert trainable
    assert set(trainable.values()) == {expected}, trainable
    assert policy.model.state_proj.weight.dtype == torch.float32
    assert policy.model.action_out_proj.weight.dtype == torch.float32
    if train_expert_only:
        assert {param.dtype for param in policy.model.vlm_with_expert.vlm.parameters()} == {torch.bfloat16}


def test_small_adamw_update_survives_writeback(tiny_smolvla_config: SmolVLAConfig) -> None:
    policy = SmolVLAPolicy(tiny_smolvla_config)
    weight = policy.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    with torch.no_grad():
        weight.fill_(1.0)
    weight.grad = torch.ones_like(weight)
    optimizer.step()

    # A 1e-4 update at 1.0 disappears on every step with bf16 parameter storage.
    assert torch.all(weight < 1.0)
    assert weight.dtype == torch.float32
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[weight]["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize("legacy_bf16, saved_dtype", [(True, None), (False, None), (False, torch.bfloat16)])
@pytest.mark.parametrize("load_vlm_weights", [False, True])
def test_checkpoint_resume_preserves_fp32_updates(
    tiny_smolvla_config: SmolVLAConfig,
    tmp_path: Path,
    legacy_bf16: bool,
    saved_dtype: torch.dtype | None,
    load_vlm_weights: bool,
) -> None:
    tiny_smolvla_config.dtype = saved_dtype
    tiny_smolvla_config.load_vlm_weights = load_vlm_weights
    policy = SmolVLAPolicy(tiny_smolvla_config)
    if legacy_bf16:
        # Before this fix the expert was bf16 while policy.dtype was unset.
        policy.model.vlm_with_expert.lm_expert.bfloat16()
    weight = policy.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    with torch.no_grad():
        weight.fill_(1.0)
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    expected = weight.detach().float().clone()
    policy.save_pretrained(tmp_path)
    state_dir = tmp_path / "training_state"
    state_dir.mkdir()
    save_optimizer_state(optimizer, state_dir)

    overrides = ["--dtype=float32"] if saved_dtype is not None else []
    config = SmolVLAConfig.from_pretrained(tmp_path, cli_overrides=overrides)
    resumed = SmolVLAPolicy.from_pretrained(tmp_path, config=config, strict=True)
    assert {param.dtype for param in resumed.parameters() if param.requires_grad} == {torch.float32}
    assert {param.dtype for param in resumed.model.vlm_with_expert.vlm.parameters()} == {torch.bfloat16}
    resumed_weight = resumed.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
    torch.testing.assert_close(resumed_weight, expected, rtol=0, atol=0)
    resumed_optimizer = resumed.config.get_optimizer_preset().build(resumed.get_optim_params())
    load_optimizer_state(resumed_optimizer, state_dir)
    assert resumed_optimizer.state[resumed_weight]["exp_avg"].dtype == torch.float32
    assert resumed_optimizer.state[resumed_weight]["exp_avg_sq"].dtype == torch.float32

    resumed_weight.grad = torch.ones_like(resumed_weight)
    resumed_optimizer.step()
    assert torch.all(resumed_weight < expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("train_expert_only", [False, True])
def test_training_and_inference_with_fp32_storage(
    tiny_smolvla_config: SmolVLAConfig, device: str, autocast: bool, train_expert_only: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "cuda" and autocast and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 is not supported")
    tiny_smolvla_config.train_expert_only = train_expert_only
    tiny_smolvla_config.num_steps = 2
    policy = SmolVLAPolicy(tiny_smolvla_config).to(device)
    batch = {
        "observation.images.camera1": torch.ones(1, 3, 16, 16, device=device),
        "observation.state": torch.ones(1, 4, device=device),
        "observation.language.tokens": torch.tensor([[1, 2, 3]], device=device),
        "observation.language.attention_mask": torch.ones(1, 3, dtype=torch.bool, device=device),
        "action": torch.ones(1, 2, 4, device=device),
    }
    noise = torch.zeros_like(batch["action"])
    time = torch.full((1,), 0.5, device=device)
    weight = policy.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
    before = weight.detach().clone()
    compute_dtypes: list[torch.dtype] = []
    hook = policy.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.register_forward_hook(
        lambda module, args, output: compute_dtypes.append(output.dtype)
    )
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        loss, _ = policy(batch, noise=noise, time=time)
    hook.remove()
    assert torch.isfinite(loss)
    assert set(compute_dtypes) == {torch.bfloat16 if autocast else torch.float32}
    loss.backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    optimizer.step()
    assert torch.any(weight != before)
    assert {param.dtype for param in policy.parameters() if param.requires_grad} == {torch.float32}
    if train_expert_only:
        assert all(param.grad is None for param in policy.model.vlm_with_expert.vlm.parameters())

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        actions = policy.predict_action_chunk(batch, noise=noise)
    assert actions.shape == (1, 2, 4)
    assert torch.isfinite(actions).all()
