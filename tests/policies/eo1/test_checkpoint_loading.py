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

from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")

from lerobot.policies.eo1.modeling_eo1 import EO1Policy
from tests.policies.eo1.test_eo1 import make_eo1_config, make_policy_batch


@pytest.fixture
def checkpoint(tmp_path):
    config = make_eo1_config()
    config.pretrained_path = tmp_path
    config.attn_implementation = "eager"
    config.vlm_config = {
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        "vision_config": {
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "fullatt_block_indexes": [0],
        },
    }
    torch.manual_seed(0)
    policy = EO1Policy(config).eval()
    policy.save_pretrained(tmp_path)
    return tmp_path, policy


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "auto"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
        ),
    ],
)
def test_checkpoint_restores_weights_dtypes_and_ties(checkpoint, dtype, device):
    path, original = checkpoint
    config = deepcopy(original.config)
    config.dtype = dtype
    config.device = device
    loaded = EO1Policy.from_pretrained(path, config=config, strict=True)
    backbone_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    for name, param in loaded.named_parameters():
        expected_dtype = backbone_dtype if name.startswith("model.vlm_backbone.") else torch.float32
        assert param.dtype == expected_dtype
        assert param.device.type == device
        torch.testing.assert_close(
            param.cpu(), original.state_dict()[name].to(expected_dtype), rtol=0, atol=0
        )
    backbone = loaded.model.vlm_backbone
    assert backbone.get_input_embeddings().weight is backbone.get_output_embeddings().weight
    assert not loaded.training
    assert not any(t.is_meta for t in loaded.buffers())
    for name, buffer in loaded.named_buffers():
        torch.testing.assert_close(buffer.cpu(), dict(original.named_buffers())[name], rtol=0, atol=0)


def test_checkpoint_roundtrip_inference_and_training(checkpoint, tmp_path):
    path, original = checkpoint
    loaded = EO1Policy.from_pretrained(path, strict=True)
    batch = make_policy_batch(include_action=False)
    batch.pop("pixel_values")
    batch.pop("image_grid_thw")
    torch.manual_seed(0)
    expected = original.predict_action_chunk(batch)
    torch.manual_seed(0)
    torch.testing.assert_close(loaded.predict_action_chunk(batch), expected, rtol=0, atol=0)
    loaded.config.gradient_checkpointing = True
    second_path = tmp_path / "roundtrip"
    loaded.save_pretrained(second_path)
    restored = EO1Policy.from_pretrained(second_path, strict=True).train()
    batch["action"] = torch.randn(1, 3, 3)
    original.train()
    torch.manual_seed(0)
    expected_loss, _ = original(batch)
    expected_loss.backward()
    torch.manual_seed(0)
    loss, _ = restored(batch)
    loss.backward()
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    for name, param in restored.named_parameters():
        expected_grad = dict(original.named_parameters())[name].grad
        if expected_grad is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, expected_grad, rtol=0, atol=0)
    assert torch.isfinite(loss)
    gradients = [p.grad for p in restored.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer = torch.optim.AdamW(restored.get_optim_params(), lr=1e-4)
    before = restored.model.action_out_proj[-1].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, restored.model.action_out_proj[-1].weight)


def test_complete_checkpoint_does_not_randomly_initialize_backbone(checkpoint, monkeypatch):
    path, original = checkpoint
    normal = torch.Tensor.normal_

    def normal_on_meta_only(tensor, *args, **kwargs):
        if not tensor.is_meta:
            pytest.fail("Checkpoint-backed Qwen weights should not be randomly initialized")
        return normal(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "normal_", normal_on_meta_only)
    EO1Policy.from_pretrained(path, config=deepcopy(original.config), strict=True)


@pytest.mark.parametrize(
    "missing_key",
    ["model.vlm_backbone.model.language_model.layers.0.self_attn.q_proj.weight", "model.state_proj.weight"],
)
def test_partial_checkpoint_initializes_missing_weights(checkpoint, missing_key):
    path, original = checkpoint
    weights = load_file(path / "model.safetensors")
    # Resolve the head's exact name from the real state dict, rather than mocking its module.
    if missing_key == "model.state_proj.weight":
        missing_key = next(k for k in weights if k.startswith("model.state_proj.") and k.endswith("weight"))
    removed = weights.pop(missing_key)
    weights["unexpected.weight"] = torch.ones(1)
    save_file(weights, path / "model.safetensors")
    loaded = EO1Policy.from_pretrained(path, config=deepcopy(original.config))
    initialized = loaded.state_dict()[missing_key]
    assert initialized.shape == removed.shape
    assert torch.isfinite(initialized).all() and initialized.count_nonzero() > 0
    assert not torch.equal(initialized, removed)
    for name, value in weights.items():
        if name != "unexpected.weight":
            torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Missing key|missing key"):
        EO1Policy.from_pretrained(path, config=deepcopy(original.config), strict=True)


def test_hub_options_reach_checkpoint_download(checkpoint, monkeypatch):
    path, original = checkpoint
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path / "model.safetensors")

    monkeypatch.setattr("lerobot.policies.eo1.modeling_eo1.hf_hub_download", download)
    EO1Policy.from_pretrained(
        "org/eo1",
        config=deepcopy(original.config),
        revision="fixed-revision",
        token=False,
        cache_dir=path,
        local_files_only=True,
        strict=True,
    )
    assert calls == [
        {
            "repo_id": "org/eo1",
            "filename": "model.safetensors",
            "revision": "fixed-revision",
            "token": False,
            "cache_dir": path,
            "local_files_only": True,
            "force_download": False,
            "resume_download": None,
            "proxies": None,
        }
    ]
