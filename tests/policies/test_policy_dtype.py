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
"""Tests for the shared policy dtype contract.

The contract under test:

* `config.dtype` is a request. It is held as a `torch.dtype`, serialized as a name, and never
  written back to by the framework.
* `config.dtype is None` means "unspecified" and leaves the policy exactly as `__init__` built it.
* `_fp32_modules` declares the tensors that stay float32 regardless.
* `post_init()` enforces that layout, runs automatically, and is idempotent.
* `policy.dtype` observes the built policy and is never fed back into the config.
"""

import json
import re
import warnings
from pathlib import Path

import draccus
import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy


class TinyPolicy(PreTrainedPolicy):
    """Smallest policy that exercises all three `_fp32_modules` declaration forms."""

    config_class = ACTConfig
    name = "tiny_dtype"
    _fp32_modules = ("head", "norm", nn.LayerNorm, "blocks.*.scale")

    def __init__(self, config: ACTConfig, **kwargs):
        super().__init__(config)
        self.backbone = nn.Sequential(*[nn.Linear(8, 8) for _ in range(3)])
        self.head = nn.Linear(8, 4)
        self.norm = nn.Linear(8, 8)
        self.ln = nn.LayerNorm(8)
        self.blocks = nn.ModuleDict({"a": nn.ModuleDict({"scale": nn.Linear(4, 4)})})
        self.register_buffer("steps", torch.zeros(3, dtype=torch.long))
        self.register_buffer("cache", torch.zeros(3), persistent=False)
        self.tied = nn.Linear(8, 8)
        self.tied.weight = self.backbone[0].weight

    def get_optim_params(self):
        return {}

    def reset(self):
        pass

    def forward(self, batch):
        return torch.zeros(()), None

    def predict_action_chunk(self, batch, **kwargs):
        return torch.zeros(1)

    def select_action(self, batch, **kwargs):
        return torch.zeros(1)


def make_config(dtype=None) -> ACTConfig:
    return ACTConfig(device="cpu", dtype=dtype)


# --------------------------------------------------------------------------------------------
# config layer
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("torch.bfloat16", torch.bfloat16),
        (torch.float16, torch.float16),
        ("float64", torch.float64),
        (None, None),
    ],
)
def test_config_dtype_accepts_names_and_objects(value, expected):
    assert make_config(value).dtype is expected
    config = make_config()
    config.dtype = value
    assert config.dtype is expected


def test_config_dtype_is_never_a_string(tmp_path: Path):
    """transformers' field is polymorphic depending on the construction path; ours is not."""
    built = make_config("bfloat16")
    built._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assigned = make_config()
    assigned.dtype = "bfloat16"
    for config in (built, loaded, assigned):
        assert isinstance(config.dtype, torch.dtype)


@pytest.mark.parametrize("value", ["auto", "fp16", "int8", "bfloat", torch.int64, 16, {}])
def test_config_dtype_rejects_non_floating_requests(value):
    with pytest.raises((ValueError, TypeError)):
        make_config(value)


def test_config_dtype_has_no_allowlist():
    """Any floating dtype PyTorch exposes is legal; there is no hard-coded supported set."""
    assert make_config("float8_e4m3fn").dtype is torch.float8_e4m3fn


def test_config_dtype_round_trips_through_json(tmp_path: Path):
    config = make_config("bfloat16")
    config._save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    assert payload["dtype"] == "bfloat16"
    assert PreTrainedConfig.from_pretrained(tmp_path).dtype is torch.bfloat16


def test_unspecified_dtype_is_omitted_from_the_saved_config(tmp_path: Path):
    """An absent key and `"dtype": null` decode the same, so omitting it keeps old readers working."""
    make_config(None)._save_pretrained(tmp_path)
    assert "dtype" not in json.loads((tmp_path / "config.json").read_text())
    assert PreTrainedConfig.from_pretrained(tmp_path).dtype is None


def test_config_dtype_survives_replace_and_deepcopy():
    import copy
    import dataclasses

    config = make_config("bfloat16")
    assert dataclasses.replace(config, chunk_size=50, n_action_steps=50).dtype is torch.bfloat16
    assert copy.deepcopy(config).dtype is torch.bfloat16


def test_legacy_torch_dtype_key_is_migrated(tmp_path: Path):
    make_config("bfloat16")._save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["torch_dtype"] = payload.pop("dtype")
    (tmp_path / "config.json").write_text(json.dumps(payload))

    assert PreTrainedConfig.from_pretrained(tmp_path).dtype is torch.bfloat16


def test_migration_prefers_the_current_key_when_both_are_present(tmp_path: Path):
    make_config("bfloat16")._save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["torch_dtype"] = "float32"
    (tmp_path / "config.json").write_text(json.dumps(payload))

    assert PreTrainedConfig.from_pretrained(tmp_path).dtype is torch.bfloat16


def test_nested_dtype_keys_are_not_rewritten():
    """Our encoder dispatches on type, so a nested string spelled `dtype` is untouched.

    transformers' name-based `dict_dtype_to_str` corrupted Emu3's vocabulary for exactly this
    reason (huggingface/transformers#40766).
    """
    encoded = draccus.encode(make_config("bfloat16"), PreTrainedConfig)
    payload = json.loads(json.dumps({**encoded, "nested": {"dtype": "video"}}))
    assert payload["nested"]["dtype"] == "video"
    assert payload["dtype"] == "bfloat16"


# --------------------------------------------------------------------------------------------
# policy layer
# --------------------------------------------------------------------------------------------


def test_post_init_runs_without_the_policy_asking():
    policy = TinyPolicy(make_config("bfloat16"))
    assert policy.backbone[0].weight.dtype is torch.bfloat16


def test_declared_exceptions_stay_float32():
    policy = TinyPolicy(make_config("bfloat16"))
    assert policy.head.weight.dtype is torch.float32  # plain name
    assert policy.norm.weight.dtype is torch.float32  # plain name
    assert policy.ln.weight.dtype is torch.float32  # module class
    assert policy.blocks["a"]["scale"].weight.dtype is torch.float32  # wildcard path


def test_unspecified_dtype_changes_nothing():
    policy = TinyPolicy(make_config(None))
    assert policy.parameter_dtypes() == {torch.float32: sum(p.numel() for p in policy.parameters())}


def test_non_floating_buffers_are_untouched():
    policy = TinyPolicy(make_config("bfloat16"))
    assert policy.steps.dtype is torch.int64


def test_non_persistent_buffers_are_converted():
    policy = TinyPolicy(make_config("bfloat16"))
    assert policy.cache.dtype is torch.bfloat16


def test_tied_parameters_stay_tied():
    policy = TinyPolicy(make_config("bfloat16"))
    assert policy.tied.weight is policy.backbone[0].weight
    assert policy.tied.weight.dtype is torch.bfloat16


def test_config_is_not_rewritten_by_construction():
    config = make_config("bfloat16")
    TinyPolicy(config)
    assert config.dtype is torch.bfloat16


def test_post_init_is_idempotent():
    policy = TinyPolicy(make_config("bfloat16"))
    before = policy.parameter_dtypes()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a no-op re-run must not warn
        policy.post_init()
    assert policy.parameter_dtypes() == before


def test_post_init_is_a_no_op_when_init_already_set_the_precision():
    class AlreadyCorrect(TinyPolicy):
        name = "tiny_dtype_precast"

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.extra = nn.Linear(8, 8, dtype=config.dtype)

    policy = AlreadyCorrect(make_config("bfloat16"))
    assert policy.extra.weight.dtype is torch.bfloat16


def test_a_subclass_of_a_policy_is_finalized_once_and_completely():
    class Derived(TinyPolicy):
        name = "tiny_dtype_derived"

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.extra = nn.Linear(8, 8)

    policy = Derived(make_config("bfloat16"))
    assert policy.extra.weight.dtype is torch.bfloat16
    assert policy.head.weight.dtype is torch.float32


def test_dtype_reports_the_dominant_precision_not_the_first_parameter():
    class FirstParamIsProtected(TinyPolicy):
        name = "tiny_dtype_protected_first"
        _fp32_modules = ("head",)

        def __init__(self, config, **kwargs):
            nn.Module.__init__(self)
            PreTrainedPolicy.__init__(self, config)
            self.head = nn.Linear(8, 8)  # registered first, kept float32
            self.backbone = nn.Sequential(*[nn.Linear(64, 64) for _ in range(4)])

    policy = FirstParamIsProtected(make_config("bfloat16"))
    assert next(p.dtype for p in policy.parameters() if p.is_floating_point()) is torch.float32
    assert policy.dtype is torch.bfloat16


def test_dtype_breaks_ties_towards_the_request():
    policy = TinyPolicy(make_config("bfloat16"))
    counts = policy.parameter_dtypes()
    assert set(counts) == {torch.bfloat16, torch.float32}
    assert policy.dtype is torch.bfloat16


def test_policy_dtype_matches_the_request_after_construction():
    for name in ("bfloat16", "float32", "float16"):
        policy = TinyPolicy(make_config(name))
        assert policy.dtype is getattr(torch, name)


def test_casting_after_construction_warns_but_still_casts():
    policy = TinyPolicy(make_config("bfloat16"))
    with pytest.warns(UserWarning, match="keeps in float32"):
        policy.to(torch.float16)
    assert policy.head.weight.dtype is torch.float16


def test_moving_devices_does_not_warn():
    policy = TinyPolicy(make_config("bfloat16"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy.to("cpu")


def test_a_policy_without_declarations_never_warns_on_cast():
    class NoExceptions(TinyPolicy):
        name = "tiny_dtype_no_exceptions"
        _fp32_modules = ()

    policy = NoExceptions(make_config("bfloat16"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy.to(torch.float16)


def test_post_init_after_construction_warns_about_desynchronization():
    policy = TinyPolicy(make_config("bfloat16"))
    policy.to(torch.float32)
    with pytest.warns(UserWarning, match="after construction had finished"):
        policy.post_init()


# --------------------------------------------------------------------------------------------
# matching semantics
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "name", "matches"),
    [
        # a declared path matches under any wrapper prefix
        ("model.action_in_proj", "model.action_in_proj.weight", True),
        ("model.action_in_proj", "base_model.model.model.action_in_proj.weight", True),  # PEFT
        ("model.action_in_proj", "_orig_mod.model.action_in_proj.weight", True),  # torch.compile
        ("model.action_in_proj", "module.model.action_in_proj.weight", True),  # DDP
        # but it does not match a different module whose name merely starts the same
        ("model.action_in_proj", "model.action_in_proj_v2.weight", False),
        # a bare segment matches at any depth
        ("norm", "a.b.norm.weight", True),
        # and never part of a longer segment
        ("norm", "a.b.post_attention_layernorm.weight", False),
        ("norm", "a.b.layer_norm1.weight", False),
        # `*` is exactly one segment
        ("layers.*.input_layernorm", "m.layers.3.input_layernorm.weight", True),
        ("layers.*.input_layernorm", "m.layers.3.self_attn.input_layernorm.weight", False),
    ],
)
def test_fp32_path_matching(declaration, name, matches):
    from lerobot.policies.pretrained import _compile_fp32_paths

    pattern = _compile_fp32_paths((declaration,))
    assert pattern is not None
    assert bool(pattern.search(name)) is matches


@pytest.mark.parametrize("path", ["", ".norm", "model..norm", "model.norm*", "model.**.norm"])
def test_invalid_fp32_paths_are_rejected(path):
    from lerobot.policies.pretrained import _compile_fp32_paths

    with pytest.raises(ValueError, match="_fp32_modules"):
        _compile_fp32_paths((path,))


def test_protecting_one_alias_protects_every_alias():
    class SharedAcrossNames(TinyPolicy):
        name = "tiny_dtype_shared"
        _fp32_modules = ("keep",)

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.keep = nn.Linear(8, 8)
            self.mirror = nn.Linear(8, 8)
            self.mirror.weight = self.keep.weight

    policy = SharedAcrossNames(make_config("bfloat16"))
    assert policy.keep.weight.dtype is torch.float32
    assert policy.mirror.weight is policy.keep.weight


# --------------------------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------------------------


def test_loading_a_float32_checkpoint_into_a_bfloat16_policy_keeps_the_policy_layout(tmp_path: Path):
    source = TinyPolicy(make_config("float32"))
    target = TinyPolicy(make_config("bfloat16"))
    target.load_state_dict(source.state_dict(), strict=False)
    assert target.backbone[0].weight.dtype is torch.bfloat16
    assert target.head.weight.dtype is torch.float32


def test_the_load_path_never_assigns_parameters():
    """C5 (the checkpoint cannot change the policy's dtype) rests on `assign=False`."""
    source = Path("src/lerobot")
    offenders = [
        path
        for path in source.rglob("*.py")
        if re.search(r"load_state_dict\([^)]*assign\s*=\s*True", path.read_text(), re.S)
    ]
    assert offenders == []


def test_saving_warns_when_the_policy_no_longer_matches_its_request(tmp_path: Path, caplog):
    policy = TinyPolicy(make_config("bfloat16"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        policy.to(torch.float32)
    policy.save_pretrained(tmp_path)
    assert "config.dtype requests" in caplog.text
    assert json.loads((tmp_path / "config.json").read_text())["dtype"] == "bfloat16"
