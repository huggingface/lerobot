# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""PEFT/LoRA integration tests for lingbot_vla_v2.

The policy's default PEFT targets rely on PEFT's suffix matching hitting exactly the
attention projections of the Qwen3-VL LLM and the action expert (both named
q/k/v/o_proj), missing the fused-qkv vision tower, and full-training the tiny MoE
router via modules_to_save. These tests verify that matching behavior against a
synthetic module tree with the same naming, plus the hook contents — without
instantiating the 6B model (the full wrap/merge roundtrip runs on GPU in bench/).
"""

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")


def _targets():
    from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

    # The hook is self-contained (does not touch instance state).
    return LingbotVLAV2Policy._get_default_peft_targets(None)


def _synthetic_backbone():
    """Mimic lingbot_vla_v2's module naming: LLM + expert q/k/v/o_proj, fused-qkv
    vision tower, MoE router `mlp.gate`, shared-expert MLP, fused-expert Parameters."""
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)

    class VisionAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.qkv = nn.Linear(d, 3 * d)  # fused — must NOT be LoRA-targeted
            self.proj = nn.Linear(d, d)

    class MoeBlock(nn.Module):
        def __init__(self, d, n_experts):
            super().__init__()
            self.gate = nn.Linear(d, n_experts, bias=False)  # router — modules_to_save
            self.shared_expert = nn.Linear(d, d)
            # Routed experts as fused-storage plain Parameters (Qwen2FusedExperts style):
            # not nn.Linear, so PEFT must not touch them.
            self.experts = nn.Parameter(torch.zeros(n_experts, d, d))

    class Layer(nn.Module):
        def __init__(self, d, n_experts=4):
            super().__init__()
            self.self_attn = Attn(d)
            self.mlp = MoeBlock(d, n_experts)

    class Tiny(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.llm_layer = Layer(d)
            self.expert_layer = Layer(d)
            self.visual_attn = VisionAttn(d)

    return Tiny()


def test_default_targets_cover_attention_and_router_only():
    targets = _targets()
    assert set(targets["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert targets["modules_to_save"] == ["gate"]


def test_peft_matching_on_lingbot_naming():
    """Wrap the synthetic tree with PEFT using the policy defaults: LoRA must land on
    every q/k/v/o_proj (LLM + expert), the router goes to modules_to_save, and the
    vision tower / fused experts stay untouched."""
    from peft import LoraConfig, get_peft_model

    model = _synthetic_backbone()
    config = LoraConfig(r=4, lora_alpha=8, **_targets())
    wrapped = get_peft_model(model, config)

    # PeftModel prefixes everything with "base_model.model." — strip it.
    lora_modules = {name for name, _ in wrapped.named_modules() if "lora_" in name}
    lora_parents = {
        name.rsplit(".lora_", 1)[0].removeprefix("base_model.model.") for name in lora_modules
    }
    # LLM layer + expert layer, four projections each.
    for layer in ("llm_layer", "expert_layer"):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert f"{layer}.self_attn.{proj}" in lora_parents, f"missing LoRA on {layer}.{proj}"
    # Vision tower untouched (fused qkv must not match), experts Parameter untouched.
    assert not any("visual_attn" in name for name in lora_parents)
    assert not any("mlp" in name for name in lora_parents)
    # Router went to modules_to_save (full fine-tune, saved alongside adapters).
    modules_to_save = [name for name, _ in wrapped.named_modules() if "modules_to_save" in name]
    assert any(name.endswith("mlp.modules_to_save") or ".gate" in name for name in modules_to_save), modules_to_save

    # Trainable params = adapters + saved modules only.
    trainable = {name for name, p in wrapped.named_parameters() if p.requires_grad}
    assert any("lora_" in name for name in trainable)
    assert any("gate" in name and "modules_to_save" in name for name in trainable)
    assert not any("visual_attn" in name for name in trainable)
    assert not any(name.endswith("experts") for name in trainable)


def test_get_optim_params_filters_frozen():
    """get_optim_params must return only requires_grad params (optimizer state memory
    stays at the adapter scale, not the 6B scale), keyed by FQN so the optimizer
    preset can group by name (lingbot_adamw's MoE expert-LR scaling)."""
    from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            self.lin.bias.requires_grad_(False)

    fake = FakePolicy()
    params = LingbotVLAV2Policy.get_optim_params(fake)
    assert isinstance(params, dict)
    assert set(params) == {"lin.weight"}
    assert all(p.requires_grad for p in params.values())
    assert params["lin.weight"] is fake.lin.weight
