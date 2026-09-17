#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Regression tests for #4577: the pi0-family `from_pretrained` overrides used to
swallow every loading failure — missing checkpoint files and strict=True key
mismatches alike — and return the freshly initialized (untrained) model.

The tests drive the real `from_pretrained` implementations through tiny shell
subclasses that skip the multi-GB PaliGemma construction but keep the loading
path intact, so they run on CPU without any hub download.
"""

import json

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402


def _tiny_shell(policy_cls, fix_method_name: str, extra_fix_methods: tuple[str, ...] = ()):
    """Build a subclass of `policy_cls` whose __init__ skips the full model and
    whose key-remap hooks are passthroughs, keeping the loading path exercisable."""

    class _Shell(policy_cls):
        def __init__(self, config, **kwargs):
            PreTrainedPolicy.__init__(self, config)
            self.model = nn.Sequential(nn.Linear(4, 4))

        def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
            return state_dict

    setattr(_Shell, fix_method_name, lambda self, state_dict, model_config: state_dict)
    for name in extra_fix_methods:
        setattr(_Shell, name, lambda self, state_dict: state_dict)
    return _Shell


POLICIES = [
    (PI0Policy, "pi0", ("_fix_pytorch_state_dict_keys",)),
    (PI05Policy, "pi05", ("_fix_pytorch_state_dict_keys", "_prepare_pretrained_state_dict")),
    (PI0FastPolicy, "pi0_fast", ("_fix_pytorch_state_dict_keys",)),
]


def _make_checkpoint_dir(tmp_path, policy_type: str, with_weights: bool) -> str:
    checkpoint = tmp_path / f"fake_{policy_type}"
    checkpoint.mkdir()
    with open(checkpoint / "config.json", "w") as f:
        json.dump({"type": policy_type}, f)
    if with_weights:
        # Keys deliberately unrelated to the shell model: strict loading must
        # refuse them.
        from safetensors.torch import save_file

        save_file({"unrelated.key": torch.zeros(4, 4)}, str(checkpoint / "model.safetensors"))
    return str(checkpoint)


@pytest.mark.parametrize("policy_cls,policy_type,fixers", POLICIES)
def test_missing_checkpoint_raises(tmp_path, policy_cls, policy_type, fixers):
    shell = _tiny_shell(policy_cls, fixers[0], fixers[1:])
    checkpoint = _make_checkpoint_dir(tmp_path, policy_type, with_weights=False)

    with pytest.raises(Exception, match="Could not load model.safetensors"):
        shell.from_pretrained(checkpoint)


@pytest.mark.parametrize("policy_cls,policy_type,fixers", POLICIES)
def test_strict_mismatch_raises(tmp_path, policy_cls, policy_type, fixers):
    shell = _tiny_shell(policy_cls, fixers[0], fixers[1:])
    checkpoint = _make_checkpoint_dir(tmp_path, policy_type, with_weights=True)

    with pytest.raises(Exception, match="Could not load state dict"):
        shell.from_pretrained(checkpoint, strict=True)


@pytest.mark.parametrize("policy_cls,policy_type,fixers", POLICIES)
def test_non_strict_mismatch_returns_best_effort_model(tmp_path, policy_cls, policy_type, fixers):
    shell = _tiny_shell(policy_cls, fixers[0], fixers[1:])
    checkpoint = _make_checkpoint_dir(tmp_path, policy_type, with_weights=True)

    # strict=False is the explicit best-effort mode: mismatch is reported, not raised.
    model = shell.from_pretrained(checkpoint, strict=False)
    assert isinstance(model, policy_cls)
