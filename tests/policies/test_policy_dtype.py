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
"""Tests for `PreTrainedConfig.dtype`.

`dtype` holds a real `torch.dtype` at runtime and serializes to a plain name such as
`"bfloat16"`, so `config.json` keeps the exact same on-disk shape it always had. Two
parse paths must both work and are therefore covered separately for every case:

* ``PreTrainedConfig.from_pretrained`` parses a policy ``config.json`` directly.
* ``--resume`` hands the whole ``train_config.json`` to draccus, and the nested
  ``policy`` dict is decoded through ``decode_choice_class`` without ever calling
  ``from_pretrained``.
"""

import json
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import draccus
import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig

CPU = {"device": "cpu"}


@dataclass
class _TrainLikeConfig:
    """Minimal stand-in for `TrainPipelineConfig`'s nested polymorphic policy field."""

    policy: PreTrainedConfig | None = None


def _parse(config_cls, payload: dict, cli_args: list[str] | None = None):
    """Parse `payload` the way `PreTrainedConfig.from_pretrained` does."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file = Path(tmp_dir) / "config.json"
        config_file.write_text(json.dumps(payload))
        with draccus.config_type("json"):
            return draccus.parse(config_cls, str(config_file), args=cli_args or [])


def _parse_nested(policy_payload: dict):
    """Parse a policy dict nested inside a train config, as `--resume` does."""
    return _parse(_TrainLikeConfig, {"policy": policy_payload}).policy


@contextmanager
def _no_future_warning():
    """Assert the block emits no FutureWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    offenders = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not offenders, f"unexpected FutureWarning: {[str(w.message) for w in offenders]}"


# --------------------------------------------------------------------------------------
# Base field: a real torch.dtype in memory, a plain name on disk
# --------------------------------------------------------------------------------------


def test_dtype_defaults_to_none():
    assert ACTConfig(**CPU).dtype is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_dtype_accepts_torch_dtype(dtype):
    assert ACTConfig(**CPU, dtype=dtype).dtype is dtype


def test_dtype_rejects_a_string_in_python():
    # Names are a serialization detail; Python callers pass the torch.dtype itself.
    with pytest.raises(ValueError, match="must be a torch.dtype"):
        ACTConfig(**CPU, dtype="bfloat16")


def test_dtype_rejects_a_non_dtype_in_python():
    with pytest.raises(ValueError, match="must be a torch.dtype"):
        ACTConfig(**CPU, dtype=16)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("torch.bfloat16", torch.bfloat16),
        # Aliases come for free because names resolve against torch itself.
        ("half", torch.float16),
        ("double", torch.float64),
    ],
)
def test_dtype_decodes_names_from_config_json(name, expected):
    assert _parse(ACTConfig, {**CPU, "dtype": name}).dtype is expected


def test_dtype_decodes_names_from_the_cli():
    assert _parse(ACTConfig, CPU, ["--dtype=float16"]).dtype is torch.float16


def test_dtype_accepts_null_and_a_missing_key():
    assert _parse(ACTConfig, {**CPU, "dtype": None}).dtype is None
    assert _parse(ACTConfig, CPU).dtype is None


@pytest.mark.parametrize("value", ["nonsense", "auto", "nn", "zeros", "Tensor"])
def test_dtype_rejects_values_that_are_not_torch_dtypes(value):
    # `nn`/`zeros`/`Tensor` exist on `torch` but are not dtypes.
    with pytest.raises(Exception, match="Invalid dtype|not valid"):
        _parse(ACTConfig, {**CPU, "dtype": value})


def test_dtype_serializes_to_a_plain_name():
    encoded = draccus.encode(ACTConfig(**CPU, dtype=torch.bfloat16), PreTrainedConfig)
    assert encoded["dtype"] == "bfloat16"
    assert json.loads(json.dumps(encoded))["dtype"] == "bfloat16"


def test_dtype_serializes_none_as_null():
    assert draccus.encode(ACTConfig(**CPU), PreTrainedConfig)["dtype"] is None


def test_dtype_survives_a_save_load_round_trip():
    encoded = draccus.encode(ACTConfig(**CPU, dtype=torch.bfloat16), PreTrainedConfig)
    encoded.pop("type")  # `from_pretrained` strips the choice tag before parsing
    assert _parse(ACTConfig, encoded).dtype is torch.bfloat16


# --------------------------------------------------------------------------------------
# Per-policy dtype restrictions: kept, with the literals translated to torch dtypes
# --------------------------------------------------------------------------------------

RESTRICTED_POLICIES = [PI0Config, PI05Config, PI0FastConfig, XVLAConfig, MolmoAct2Config]


@pytest.mark.parametrize("config_cls", RESTRICTED_POLICIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_restricted_policies_accept_their_supported_dtypes(config_cls, dtype):
    assert config_cls(**CPU, dtype=dtype).dtype is dtype


@pytest.mark.parametrize("config_cls", RESTRICTED_POLICIES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_restricted_policies_reject_unsupported_dtypes(config_cls, dtype):
    with pytest.raises(ValueError, match="dtype"):
        config_cls(**CPU, dtype=dtype)


@pytest.mark.parametrize("config_cls", RESTRICTED_POLICIES)
def test_restricted_policies_reject_none(config_cls):
    # Matches the pre-existing behaviour: these policies require an explicit precision.
    with pytest.raises(ValueError, match="dtype"):
        config_cls(**CPU, dtype=None)


# --------------------------------------------------------------------------------------
# Backwards compatibility with config.json files written before the rename
# --------------------------------------------------------------------------------------

# (config class, legacy payload, expected dtype, deprecated key named in the warning)
LEGACY_CASES = [
    (FastWAMConfig, {"torch_dtype": "bfloat16"}, torch.bfloat16, "torch_dtype"),
    (FastWAMConfig, {"torch_dtype": "float32"}, torch.float32, "torch_dtype"),
    (VLAJEPAConfig, {"torch_dtype": "bfloat16"}, torch.bfloat16, "torch_dtype"),
    (Evo1Config, {"vlm_dtype": "bfloat16"}, torch.bfloat16, "vlm_dtype"),
    # `zuoxingdong/evo1_libero`, the checkpoint the EVO1 docs reproduce from, stores
    # float32 here — the old value must be carried over verbatim, never defaulted.
    (Evo1Config, {"vlm_dtype": "float32"}, torch.float32, "vlm_dtype"),
    (GrootConfig, {"model_params_fp32": True}, torch.float32, "model_params_fp32"),
    (GrootConfig, {"model_params_fp32": False}, torch.bfloat16, "model_params_fp32"),
    (EO1Config, {"dtype": "auto"}, torch.bfloat16, "auto"),
]


@pytest.mark.parametrize(("config_cls", "payload", "expected", "deprecated"), LEGACY_CASES)
def test_legacy_config_json_still_loads(config_cls, payload, expected, deprecated):
    with pytest.warns(FutureWarning, match=deprecated):
        config = _parse(config_cls, {**CPU, **payload})
    assert config.dtype is expected


@pytest.mark.parametrize(("config_cls", "payload", "expected", "deprecated"), LEGACY_CASES)
def test_legacy_config_still_loads_when_nested_in_a_train_config(config_cls, payload, expected, deprecated):
    # `--resume` never goes through `from_pretrained`, so the migration cannot live there.
    policy_type = PreTrainedConfig.get_choice_name(config_cls)
    with pytest.warns(FutureWarning, match=deprecated):
        config = _parse_nested({"type": policy_type, **CPU, **payload})
    assert config.dtype is expected


@pytest.mark.parametrize(("config_cls", "payload", "expected", "deprecated"), LEGACY_CASES)
def test_migrated_config_does_not_warn_again_after_being_re_saved(config_cls, payload, expected, deprecated):
    with pytest.warns(FutureWarning, match=deprecated):
        migrated = _parse(config_cls, {**CPU, **payload})

    encoded = draccus.encode(migrated, PreTrainedConfig)
    encoded.pop("type")

    with _no_future_warning():
        reloaded = _parse(config_cls, encoded)
    assert reloaded.dtype is expected


@pytest.mark.parametrize(
    ("config_cls", "new_payload", "expected"),
    [
        (FastWAMConfig, {"dtype": "bfloat16"}, torch.bfloat16),
        (VLAJEPAConfig, {"dtype": "float32"}, torch.float32),
        (Evo1Config, {"dtype": "bfloat16"}, torch.bfloat16),
        (GrootConfig, {"dtype": "float32"}, torch.float32),
        (EO1Config, {"dtype": "bfloat16"}, torch.bfloat16),
    ],
)
def test_new_config_json_does_not_warn(config_cls, new_payload, expected):
    with _no_future_warning():
        config = _parse(config_cls, {**CPU, **new_payload})
    assert config.dtype is expected


# --------------------------------------------------------------------------------------
# The EO1 "auto" sentinel is scoped to EO1 alone
# --------------------------------------------------------------------------------------


def test_eo1_defaults_to_bfloat16():
    # Qwen2.5-VL checkpoints are published in bf16, which is what "auto" always resolved to.
    assert EO1Config(**CPU).dtype is torch.bfloat16


def test_eo1_resolves_auto_when_constructed_directly():
    with pytest.warns(FutureWarning, match="auto"):
        assert EO1Config(**CPU, dtype="auto").dtype is torch.bfloat16


def test_eo1_still_rejects_names_that_are_not_dtypes():
    with pytest.raises(Exception, match="Invalid dtype|not valid"):
        _parse(EO1Config, {**CPU, "dtype": "nonsense"})


@pytest.mark.parametrize("config_cls", [ACTConfig, PI0Config, FastWAMConfig])
def test_auto_is_not_accepted_by_other_policies(config_cls):
    with pytest.raises(Exception, match="Invalid dtype|not valid"):
        _parse(config_cls, {**CPU, "dtype": "auto"})
