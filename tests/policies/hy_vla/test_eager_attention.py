from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from transformers import PretrainedConfig

from lerobot.policies.hy_vla.modeling.hunyuan_vl_mot.modeling_hunyuan_vl_mot import (
    _eager_attention_forward_mot,
)
from lerobot.policies.hy_vla.modeling.modeling_dual_tower import HyDualTowerConfig


def _run_attention(visual_segments, padding_mask=None):
    module = SimpleNamespace(num_key_value_groups=1, training=False)
    query = torch.zeros(1, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor([[[[1.0], [2.0], [4.0]]]])
    attention_mask = {
        "v_seqlens": torch.tensor(visual_segments, dtype=torch.long),
        "padding_mask": padding_mask,
    }
    output, _ = _eager_attention_forward_mot(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=1.0,
    )
    return output[0, :, 0, 0]


def test_eager_mot_attention_is_causal_for_text_tokens():
    output = _run_attention([])
    torch.testing.assert_close(output, torch.tensor([1.0, 1.5, 7.0 / 3.0]))


def test_eager_mot_attention_is_bidirectional_inside_visual_segments():
    output = _run_attention([[0, 2]])
    torch.testing.assert_close(output, torch.tensor([1.5, 1.5, 7.0 / 3.0]))


def test_eager_mot_attention_ignores_padding():
    output = _run_attention([], padding_mask=torch.tensor([[0, 1, 1]]))
    torch.testing.assert_close(output, torch.tensor([0.0, 2.0, 3.0]))


def test_eager_mot_attention_supports_backward_with_visual_segments():
    module = SimpleNamespace(num_key_value_groups=1, training=True)
    query = torch.randn(1, 1, 3, 2, requires_grad=True)
    key = torch.randn(1, 1, 3, 2, requires_grad=True)
    value = torch.randn(1, 1, 3, 2, requires_grad=True)
    output, _ = _eager_attention_forward_mot(
        module,
        query,
        key,
        value,
        {"v_seqlens": torch.tensor([[0, 2]]), "padding_mask": None},
        scaling=2**-0.5,
    )

    output.square().mean().backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_dual_tower_config_accepts_outer_policy_config():
    outer_config = object()
    config = HyDualTowerConfig(
        vlm_config=PretrainedConfig(),
        expert_config=PretrainedConfig(),
        config=outer_config,
    )
    assert config.config is outer_config
