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

import json

import draccus
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.dtype import get_dtype


class TinyPolicy(PreTrainedPolicy):
    config_class = ACTConfig
    name = "dtype_test"
    _fp32_modules = ("head", "rotary_emb.inv_freq")

    def __init__(self, config):
        super().__init__(config)
        # A protected parameter comes first; it must not hide the main model dtype.
        self.head = nn.Linear(4, 4)
        self.backbone = nn.Linear(4, 4)
        self.head_extra = nn.Linear(4, 4)
        self.rotary_emb = nn.Module()
        self.rotary_emb.register_buffer("inv_freq", torch.tensor([1.000123, 0.500321]))
        self.register_buffer("indices", torch.arange(4))
        self.register_buffer("scale", torch.ones(4), persistent=False)
        self.head.weight.data.fill_(1.000123)
        self.post_init()

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        pass

    def forward(self, batch):
        x = self.backbone(batch.to(self.backbone.weight.dtype))
        return self.head(x.to(self.head.weight.dtype)).sum(), {}

    def predict_action_chunk(self, batch, **kwargs):
        return self.forward(batch)[0]

    def select_action(self, batch, **kwargs):
        return self.predict_action_chunk(batch)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_initialization_preserves_fp32_values_and_buffer_types(dtype):
    policy = TinyPolicy(ACTConfig(device="cpu", dtype=dtype))
    assert policy.dtype == get_dtype(dtype)
    assert policy.backbone.weight.dtype == get_dtype(dtype)
    assert policy.head_extra.weight.dtype == get_dtype(dtype)  # component boundaries, not substrings
    assert policy.scale.dtype == get_dtype(dtype)
    assert policy.indices.dtype == torch.int64
    assert policy.head.weight.dtype == torch.float32
    assert torch.equal(policy.head.weight, torch.full((4, 4), 1.000123))
    assert torch.equal(policy.rotary_emb.inv_freq, torch.tensor([1.000123, 0.500321]))

    loss, _ = policy(torch.ones(2, 4))
    loss.backward()
    assert policy.head.weight.grad.dtype == torch.float32
    assert policy.backbone.weight.grad.dtype == get_dtype(dtype)


@pytest.mark.parametrize("dtype", ["int64", "auto", "fp16", "float8_e4m3fn", torch.int64, 16, {}])
def test_invalid_config_dtype_fails_before_model_construction(dtype):
    with pytest.raises(ValueError, match="Invalid dtype"):
        ACTConfig(device="cpu", dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, "float32"])
def test_load_dtype_override_keeps_checkpoint_values_and_updates_supplied_config(tmp_path, dtype):
    original = TinyPolicy(ACTConfig(device="cpu", dtype="float32"))
    original.save_pretrained(tmp_path)
    config = ACTConfig(device="cpu", dtype="float64")
    policy = TinyPolicy.from_pretrained(tmp_path, config=config, dtype=dtype, strict=True)
    assert policy.config is config
    assert policy.dtype == get_dtype(dtype)
    assert policy.config.dtype == str(get_dtype(dtype)).removeprefix("torch.")
    assert config.pretrained_path == tmp_path
    assert not policy.training
    assert torch.equal(policy.head.weight, original.head.weight)
    assert torch.equal(policy.rotary_emb.inv_freq, original.rotary_emb.inv_freq)
    assert torch.equal(policy.backbone.weight, original.backbone.weight.to(get_dtype(dtype)))


def test_save_reload_uses_effective_dtype_after_explicit_to(tmp_path):
    policy = TinyPolicy(ACTConfig(device="cpu", dtype="float32"))
    assert type(policy).to is nn.Module.to
    policy.to(dtype=torch.bfloat16)
    assert policy.dtype == torch.bfloat16
    assert policy.head.weight.dtype == torch.bfloat16
    assert policy.rotary_emb.inv_freq.dtype == torch.bfloat16
    policy.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["dtype"] == "bfloat16"
    assert "torch_dtype" not in saved
    loaded = TinyPolicy.from_pretrained(tmp_path, strict=True)
    assert loaded.dtype == torch.bfloat16
    assert loaded.head.weight.dtype == torch.float32


@pytest.mark.parametrize("config_dtype, expected", [(None, torch.bfloat16), ("float32", torch.float32)])
def test_auto_uses_config_then_checkpoint_without_materializing_meta_tensors(
    tmp_path, config_dtype, expected
):
    config = ACTConfig(device="cpu", dtype=config_dtype)
    config.save_pretrained(tmp_path)
    weights = TinyPolicy(ACTConfig(device="cpu")).to(torch.bfloat16).state_dict()
    save_file(weights, tmp_path / "model.safetensors")
    policy = TinyPolicy.from_pretrained(tmp_path, dtype="auto", strict=True)
    assert policy.dtype == expected
    assert all(not tensor.is_meta for tensor in policy.parameters())


def test_config_cli_and_serialization_use_shared_dtype(tmp_path):
    with draccus.config_type("json"):
        config = draccus.parse(ACTConfig, args=["--dtype=bfloat16", "--device=cpu"])
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.dtype == "bfloat16"
    assert ACTConfig(dtype=torch.float16, device="cpu").dtype == "float16"


def test_post_init_preserves_parameter_identity_ties_and_gradients():
    policy = TinyPolicy(ACTConfig(device="cpu", dtype="float32"))
    policy.alias = nn.Linear(4, 4)
    policy.alias.weight = policy.head.weight
    param = policy.head.weight
    param.grad = torch.full_like(param, 1.000123)
    policy.config.dtype = "bfloat16"
    policy.post_init()
    assert policy.head.weight is policy.alias.weight is param
    assert {"head.weight", "alias.weight"} <= policy._fp32_tensor_names
    assert param.dtype == param.grad.dtype == torch.float32
    assert torch.equal(param.grad, torch.full_like(param, 1.000123))
    assert policy.alias.bias.dtype == torch.bfloat16


def test_default_dtype_is_resolved_without_changing_global_state():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        policy = TinyPolicy(ACTConfig(device="cpu"))
        assert policy.dtype == torch.float64
        assert policy.config.dtype == "float64"
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_unspecified_dtype_is_preserved_until_post_init_and_then_saved(tmp_path):
    class DeferredDtypePolicy(TinyPolicy):
        def post_init(self):
            assert self.config.dtype is None
            super().post_init()

    config = ACTConfig(device="cpu")
    assert config.dtype is None
    policy = DeferredDtypePolicy(config)
    assert policy.config is config
    assert config.dtype == str(policy.dtype).removeprefix("torch.")
    config.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["dtype"] == config.dtype


@pytest.mark.parametrize("from_checkpoint", [False, True])
def test_post_init_records_actual_dtype_when_all_tensors_are_fp32_exceptions(tmp_path, from_checkpoint):
    class FP32OnlyPolicy(TinyPolicy):
        _fp32_modules = ("head", "backbone", "head_extra", "rotary_emb", "scale")

    policy = FP32OnlyPolicy(ACTConfig(device="cpu", dtype="bfloat16"))
    if from_checkpoint:
        policy.save_pretrained(tmp_path / "checkpoint")
        policy = FP32OnlyPolicy.from_pretrained(tmp_path / "checkpoint", dtype=torch.bfloat16, strict=True)

    assert all(parameter.dtype == torch.float32 for parameter in policy.parameters())
    assert policy.dtype == torch.float32
    assert policy.config.dtype == "float32"
    policy.config.save_pretrained(tmp_path / "config_only")
    assert json.loads((tmp_path / "config_only" / "config.json").read_text())["dtype"] == "float32"


def test_empty_and_buffer_only_policy_dtype():
    policy = TinyPolicy(ACTConfig(device="cpu", dtype="bfloat16"))
    del policy.head, policy.backbone, policy.head_extra, policy.rotary_emb
    assert policy.dtype == torch.bfloat16
    del policy.scale
    assert policy.dtype == torch.bfloat16


def test_hub_loading_forwards_download_options(tmp_path, monkeypatch):
    TinyPolicy(ACTConfig(device="cpu")).save_pretrained(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr("lerobot.policies.pretrained.hf_hub_download", download)
    policy = TinyPolicy.from_pretrained(
        "owner/policy",
        config=ACTConfig(device="cpu"),
        dtype="float16",
        revision="commit",
        cache_dir=tmp_path,
        local_files_only=True,
        force_download=True,
    )
    assert policy.dtype == torch.float16
    assert calls[0]["revision"] == "commit"
    assert calls[0]["local_files_only"] is True
    assert calls[0]["force_download"] is True


def test_missing_weights_fail_instead_of_returning_random_model(tmp_path):
    ACTConfig(device="cpu").save_pretrained(tmp_path)
    with pytest.raises(FileNotFoundError):
        TinyPolicy.from_pretrained(tmp_path)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_fp32_paths_are_rooted_and_wildcards_match_one_level(dtype):
    class ScopedPolicy(TinyPolicy):
        _fp32_modules = (
            "model.wrapper.backbone.norm",
            "model.wrapper.backbone.layers.*.input_layernorm",
            "model.wrapper.backbone.inv_freq",
        )

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)
            self.norm.register_buffer("steps", torch.tensor(7))
            self.norm_extra = nn.LayerNorm(4)
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "input_layernorm": nn.LayerNorm(4),
                            "input_layernorm_extra": nn.LayerNorm(4),
                            "nested": nn.ModuleDict({"input_layernorm": nn.LayerNorm(4)}),
                        }
                    )
                ]
            )
            self.register_buffer("inv_freq", torch.tensor([1.000123]), persistent=False)
            self.register_buffer("inv_freq_extra", torch.tensor([1.000123]))

    policy = ScopedPolicy(ACTConfig(device="cpu", dtype="float32"))
    scoped = Block()
    policy.model = nn.ModuleDict({"wrapper": nn.ModuleDict({"backbone": scoped})})
    policy.norm = nn.LayerNorm(4)
    policy.modelXwrapperXbackbone = nn.ModuleDict({"norm": nn.LayerNorm(4)})
    prefix = "model.wrapper.backbone."
    assert policy._fp32_tensor_names == {
        prefix + name
        for name in (
            "norm.weight",
            "norm.bias",
            "layers.0.input_layernorm.weight",
            "layers.0.input_layernorm.bias",
            "inv_freq",
        )
    }
    assert prefix + "inv_freq" not in policy.state_dict()
    policy.config.dtype = dtype
    policy.post_init()
    assert scoped.norm.weight.dtype == torch.float32
    assert scoped.layers[0].input_layernorm.weight.dtype == torch.float32
    assert scoped.inv_freq.dtype == torch.float32
    assert torch.equal(scoped.inv_freq, torch.tensor([1.000123]))
    assert scoped.norm_extra.weight.dtype == get_dtype(dtype)
    assert scoped.layers[0].input_layernorm_extra.weight.dtype == get_dtype(dtype)
    assert scoped.layers[0].nested.input_layernorm.weight.dtype == get_dtype(dtype)
    assert scoped.inv_freq_extra.dtype == get_dtype(dtype)
    assert policy.norm.weight.dtype == get_dtype(dtype)
    assert policy.modelXwrapperXbackbone.norm.weight.dtype == get_dtype(dtype)
    assert scoped.norm.steps.dtype == torch.int64


def test_fp32_short_paths_only_match_at_the_policy_root():
    class NormPolicy(TinyPolicy):
        _fp32_modules = ("norm",)

    policy = NormPolicy(ACTConfig(device="cpu", dtype="float32"))
    policy.norm = nn.LayerNorm(4)
    policy.norm_extra = nn.LayerNorm(4)
    policy.model = nn.ModuleDict({"norm": nn.LayerNorm(4), "norm_extra": nn.LayerNorm(4)})
    policy.config.dtype = "bfloat16"
    policy.post_init()
    assert policy._fp32_tensor_names == {"norm.weight", "norm.bias"}
    assert policy.norm.weight.dtype == torch.float32
    assert policy.norm_extra.weight.dtype == torch.bfloat16
    assert policy.model.norm.weight.dtype == torch.bfloat16
    assert policy.model.norm_extra.weight.dtype == torch.bfloat16
    assert policy.backbone.weight.dtype == torch.bfloat16


def test_only_top_level_policy_fp32_declarations_are_used():
    class Block(nn.Module):
        _fp32_modules = ("norm",)

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)

    policy = TinyPolicy(ACTConfig(device="cpu", dtype="float32"))
    policy.model = Block()
    policy.config.dtype = "bfloat16"
    policy.post_init()
    assert policy.model.norm.weight.dtype == torch.bfloat16
    assert "model.norm.weight" not in policy._fp32_tensor_names


@pytest.mark.parametrize("matching_alias", ["head", "alias"])
def test_fp32_tensor_names_include_all_aliases_without_matching_equal_tensors(matching_alias):
    policy = TinyPolicy(ACTConfig(device="cpu", dtype="float32"))
    policy._fp32_modules = (f"{matching_alias}.weight", "inv_freq_alias")
    policy.alias = nn.Linear(4, 4)
    policy.alias.weight = policy.head.weight  # head is enumerated before the protected alias
    policy.head_alias = policy.head
    policy.rotary_alias = policy.rotary_emb
    policy.register_buffer("inv_freq_alias", policy.rotary_emb.inv_freq, persistent=False)
    policy.equal_weight = nn.Parameter(policy.head.weight.detach().clone())
    policy.equal_weight.grad = torch.full_like(policy.equal_weight, 1.000123)
    assert policy._fp32_tensor_names == {
        "head.weight",
        "alias.weight",
        "head_alias.weight",
        "rotary_emb.inv_freq",
        "rotary_alias.inv_freq",
        "inv_freq_alias",
    }
    original = policy.head.weight.detach().clone()
    policy.config.dtype = "bfloat16"
    policy.post_init()
    assert policy.alias.weight is policy.head.weight
    assert policy.head.weight.dtype == torch.float32
    assert torch.equal(policy.head.weight, original)
    assert policy.inv_freq_alias is policy.rotary_emb.inv_freq
    assert torch.equal(policy.inv_freq_alias, torch.tensor([1.000123, 0.500321]))
    assert policy.equal_weight.dtype == policy.equal_weight.grad.dtype == torch.bfloat16
    assert policy.alias.bias.dtype == torch.bfloat16
    assert policy.dtype == torch.bfloat16
    assert policy.config.dtype == "bfloat16"


def test_fp32_tensor_names_follow_current_registrations_without_casting():
    policy = TinyPolicy(ACTConfig(device="cpu", dtype="float32"))
    original_names = policy._fp32_tensor_names
    policy.alias = nn.Linear(4, 4)
    policy.alias.weight = policy.head.weight
    assert policy._fp32_tensor_names == original_names | {"alias.weight"}
    policy.alias.weight = nn.Parameter(policy.alias.weight.detach().clone())
    assert policy._fp32_tensor_names == original_names
    policy.head.register_buffer("new_buffer", torch.ones(4), persistent=False)
    assert policy._fp32_tensor_names == original_names | {"head.new_buffer"}
    policy.to(dtype=torch.bfloat16)
    assert policy._fp32_tensor_names == original_names | {"head.new_buffer"}
    assert policy.head.weight.dtype == torch.bfloat16  # Reading the property does not enforce FP32.
    assert policy.config.dtype == "float32"
    del policy.head
    assert policy._fp32_tensor_names == {"rotary_emb.inv_freq"}


@pytest.mark.parametrize("path", ["", ".norm", "model..norm", "model.**.norm", "model.norm*"])
def test_invalid_fp32_paths_fail_before_casting(path):
    policy = TinyPolicy(ACTConfig(device="cpu", dtype="float32"))
    policy._fp32_modules = (path,)
    policy.config.dtype = "bfloat16"
    with pytest.raises(ValueError, match="Invalid FP32 path"):
        policy.post_init()
    assert policy.backbone.weight.dtype == torch.float32
