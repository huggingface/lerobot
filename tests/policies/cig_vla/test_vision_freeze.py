from test_qwen_lora_integration import tiny_qwen
from torch import nn

from lerobot.policies.cig_vla.qwen3vl_backbone import Qwen3VLGroundingBackbone


def test_tiny_actual_qwen_vision_frozen_and_not_lora_targeted():
    wrapper = Qwen3VLGroundingBackbone.__new__(Qwen3VLGroundingBackbone)
    nn.Module.__init__(wrapper)
    wrapper.model = tiny_qwen()
    wrapper.freeze_vision_tower = True
    wrapper.model.model.visual.requires_grad_(False)
    wrapper._apply_language_lora(2, 4, 0.0, "none")
    wrapper.train()
    assert not wrapper._vision_module().training
    assert all(
        parameter.grad is None for parameter in wrapper.model.base_model.model.model.visual.parameters()
    )
    assert not any("visual" in name and "lora" in name for name, _ in wrapper.named_parameters())
