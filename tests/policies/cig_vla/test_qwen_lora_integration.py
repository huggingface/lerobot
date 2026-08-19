import os

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from lerobot.policies.cig_vla.qwen3vl_backbone import Qwen3VLGroundingBackbone


def tiny_qwen():
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 100,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    )
    return Qwen3VLForConditionalGeneration(config)


def test_language_only_lora_targets_actual_qwen_module_tree():
    wrapper = Qwen3VLGroundingBackbone.__new__(Qwen3VLGroundingBackbone)
    nn.Module.__init__(wrapper)
    wrapper.model = tiny_qwen()
    wrapper.model.model.visual.requires_grad_(False)
    wrapper._apply_language_lora(4, 8, 0.0, "none")
    assert wrapper.lora_target_counts == {"q_proj": 1, "k_proj": 1, "v_proj": 1, "o_proj": 1}
    lora = [(name, parameter) for name, parameter in wrapper.named_parameters() if "lora_" in name]
    assert lora and all(parameter.requires_grad for _, parameter in lora)
    assert not any("visual" in name for name, _ in lora)
    assert all(
        not parameter.requires_grad for parameter in wrapper.model.base_model.model.model.visual.parameters()
    )
    output = wrapper.model(
        input_ids=torch.randint(0, 100, (1, 4)), attention_mask=torch.ones(1, 4, dtype=torch.long)
    )
    output.logits.square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for _, parameter in lora)
    assert all(
        parameter.grad is None for parameter in wrapper.model.base_model.model.model.visual.parameters()
    )


@pytest.mark.skipif(not os.getenv("CIG_RUN_REAL_QWEN"), reason="set CIG_RUN_REAL_QWEN=1 for 2B integration")
def test_real_qwen_initialization_and_targets():
    wrapper = Qwen3VLGroundingBackbone(
        "Qwen/Qwen3-VL-2B-Instruct", "bfloat16", True, True, 16, 32, 0.05, "none"
    )
    assert all(count > 0 for count in wrapper.lora_target_counts.values())
    visual = wrapper.model.base_model.model.model.visual
    assert all(not parameter.requires_grad for parameter in visual.parameters())
