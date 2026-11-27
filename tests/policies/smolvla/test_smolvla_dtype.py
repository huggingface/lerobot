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
"""SmolVLA must apply `PreTrainedConfig.dtype` to its trainable action expert.

The expert is built from a copy of the VLM's text config, which carries bfloat16. Without
this wiring the trainable weights stay in bf16 no matter what `--policy.dtype` says, and
AdamW keeps no fp32 master copy to protect small updates (arXiv:2610.31859).
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoModelForImageTextToText, SmolVLMConfig  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla import smolvlm_with_expert  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402


def _tiny_vlm_config() -> SmolVLMConfig:
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
            # Mirrors the released SmolVLM config, so the expert inherits bf16 by default.
            "dtype": "bfloat16",
        },
    )


@pytest.fixture
def tiny_smolvla_config(monkeypatch: pytest.MonkeyPatch) -> SmolVLAConfig:
    def fake_from_pretrained(*_: Any, **kwargs: Any):
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        config = _tiny_vlm_config()
        return AutoModelForImageTextToText.from_config(config, **({"dtype": dtype} if dtype else {}))

    monkeypatch.setattr(smolvlm_with_expert.AutoConfig, "from_pretrained", lambda *a, **kw: _tiny_vlm_config())
    monkeypatch.setattr(smolvlm_with_expert.AutoModelForImageTextToText, "from_pretrained", fake_from_pretrained)
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


def test_unset_dtype_keeps_todays_inherited_bfloat16(tiny_smolvla_config: SmolVLAConfig) -> None:
    """`dtype=None` must not change existing behaviour: the expert inherits the VLM's bf16."""
    tiny_smolvla_config.dtype = None
    policy = SmolVLAPolicy(tiny_smolvla_config)

    weights = policy.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
    assert weights.dtype is torch.bfloat16


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"]
)
def test_explicit_dtype_applies_to_every_trainable_expert_param(
    tiny_smolvla_config: SmolVLAConfig, dtype: torch.dtype
) -> None:
    """When set, the dtype covers the whole expert -- including the cross-attention k/v
    projections, which are rebuilt outside the config and would otherwise stay fp32."""
    tiny_smolvla_config.dtype = dtype
    policy = SmolVLAPolicy(tiny_smolvla_config)

    trainable = [p for p in policy.model.vlm_with_expert.lm_expert.parameters() if p.requires_grad]
    assert trainable, "the action expert should carry the trainable weights"
    assert {p.dtype for p in trainable} == {dtype}


def test_fp32_dtype_keeps_optimizer_moments_in_fp32(tiny_smolvla_config: SmolVLAConfig) -> None:
    """The point of the fix: the update lands in fp32 storage, so it survives."""
    tiny_smolvla_config.dtype = torch.float32
    policy = SmolVLAPolicy(tiny_smolvla_config)

    weight = policy.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
    assert weight.dtype is torch.float32

    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    with torch.no_grad():
        weight.fill_(1.0)
    weight.grad = torch.ones_like(weight)
    optimizer.step()

    assert optimizer.state[weight]["exp_avg"].dtype is torch.float32
    assert torch.all(weight < 1.0), "a 1e-4-scale update must move a fp32 weight"
