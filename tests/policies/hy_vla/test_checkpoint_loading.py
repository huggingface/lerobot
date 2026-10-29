"""Strict streaming checkpoint loading, including tied parameters and real buffers."""

import pytest
import torch
from safetensors.torch import save_file, save_model
from torch import nn

pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from lerobot.policies.hy_vla.configuration_hy_vla import HyVLAConfig
from lerobot.policies.hy_vla.modeling_hy_vla import HyVLAPolicy


class TinyTiedPolicy(HyVLAPolicy):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embedding = nn.Embedding(8, 4)
        self.output = nn.Linear(4, 8, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("rotary", torch.arange(4).float(), persistent=False)
        self.to(dtype=torch.bfloat16)


def test_streaming_loader_restores_ties_and_nonpersistent_buffers(tmp_path):
    config = HyVLAConfig(device="cpu")
    original = TinyTiedPolicy(config).eval()
    save_model(original, tmp_path / "model.safetensors")
    loaded = TinyTiedPolicy.from_pretrained(tmp_path, config=config, low_cpu_mem_usage=True)
    assert loaded.embedding.weight is loaded.output.weight
    assert not any(parameter.is_meta for parameter in loaded.parameters())
    torch.testing.assert_close(loaded.rotary, original.rotary, atol=0, rtol=0)
    tokens = torch.tensor([1, 3, 5])
    torch.testing.assert_close(
        loaded.output(loaded.embedding(tokens)), original.output(original.embedding(tokens)), atol=0, rtol=0
    )


def test_streaming_loader_rejects_missing_keys(tmp_path):
    save_file({"embedding.weight": torch.zeros(8, 4)}, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="missing=.*output.weight"):
        TinyTiedPolicy.from_pretrained(tmp_path, config=HyVLAConfig(device="cpu"), low_cpu_mem_usage=True)


def test_streaming_loader_requires_strict_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="requires strict=True"):
        TinyTiedPolicy.from_pretrained(
            tmp_path, config=HyVLAConfig(device="cpu"), strict=False, low_cpu_mem_usage=True
        )
