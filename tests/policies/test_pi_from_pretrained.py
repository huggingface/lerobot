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

"""``from_pretrained`` of pi0, pi0_fast and pi0.5 on tiny Gemma widths.

A strict load builds the policy without random initialisation, so it must fill every
tensor from the checkpoint, leave the non-persistent buffers as ``__init__`` computed them,
and initialise what the checkpoint deliberately leaves out. A checkpoint that is missing,
unreadable or does not match raises, unless ``strict=False`` asks for a partial load.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import PreTrainedModel  # noqa: E402
from transformers.models.auto import CONFIG_MAPPING  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi0 import PI0Config, PI0Policy, modeling_pi0  # noqa: E402
from lerobot.policies.pi0_fast import PI0FastConfig, PI0FastPolicy, modeling_pi0_fast  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402

PROPRIO_KEYS = ("model.proprio_history_proj.weight", "model.proprio_history_proj.bias")
TORCH_INIT_FUNCTIONS = ("normal_", "uniform_", "kaiming_uniform_", "trunc_normal_", "xavier_uniform_")


@dataclass(frozen=True)
class PiVariant:
    name: str
    module: ModuleType
    policy_cls: type[PreTrainedPolicy]
    make_config: Callable[..., PreTrainedConfig]


VARIANTS = [
    PiVariant("pi0", modeling_pi0, PI0Policy, PI0Config),
    PiVariant("pi0_fast", modeling_pi0_fast, PI0FastPolicy, PI0FastConfig),
    PiVariant("pi05", modeling_pi05, PI05Policy, PI05Config),
]


def _tiny_paligemma_config():
    config = CONFIG_MAPPING["paligemma"]()
    config.vision_config.hidden_size = 16
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = 2
    return config


@pytest.fixture
def tiny_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the Gemma and SigLIP widths of every PI variant; the 257152 vocabulary stays."""
    for variant in VARIANTS:
        module = variant.module
        tiny = module.GemmaConfig(width=16, depth=1, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
        monkeypatch.setattr(module, "get_gemma_config", lambda _variant, tiny=tiny: tiny)
        monkeypatch.setattr(
            module, "CONFIG_MAPPING", {"paligemma": _tiny_paligemma_config, "gemma": CONFIG_MAPPING["gemma"]}
        )
    # pi0_fast loads its tokenizers from the Hub; the weights do not depend on them.
    monkeypatch.setattr(modeling_pi0_fast.AutoProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(modeling_pi0_fast.AutoTokenizer, "from_pretrained", lambda *a, **k: object())


def _config(variant: PiVariant, **overrides) -> PreTrainedConfig:
    config = variant.make_config(
        max_action_dim=4, max_state_dim=4, dtype=torch.float32, device="cpu", **overrides
    )
    config.input_features = {}
    config.output_features = {}
    return config


def _save(policy: PreTrainedPolicy, directory: Path) -> dict[str, torch.Tensor]:
    policy.save_pretrained(directory)
    return {key: value.clone() for key, value in policy.state_dict().items()}


def _non_persistent_buffers(policy: nn.Module) -> dict[str, torch.Tensor]:
    persistent = set(policy.state_dict())
    return {name: buf.clone() for name, buf in policy.named_buffers() if name not in persistent}


@pytest.fixture
def init_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record every torch.nn.init draw and every transformers weight initialisation."""
    calls: list[str] = []

    def spy(name: str, original: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return wrapper

    for name in TORCH_INIT_FUNCTIONS:
        monkeypatch.setattr(torch.nn.init, name, spy(name, getattr(torch.nn.init, name)))
    monkeypatch.setattr(
        PreTrainedModel, "initialize_weights", spy("initialize_weights", PreTrainedModel.initialize_weights)
    )
    yield calls


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_strict_load_restores_every_tensor_without_random_init(tiny_pi, init_calls, variant, tmp_path):
    torch.manual_seed(0)
    source = variant.policy_cls(_config(variant))
    saved = _save(source, tmp_path)
    buffers = _non_persistent_buffers(source)
    init_calls.clear()

    rng_before = torch.random.get_rng_state()
    loaded = variant.policy_cls.from_pretrained(tmp_path, config=_config(variant), strict=True)

    # Skipped init draws nothing from the global generator and calls no init function.
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert init_calls == []
    state = loaded.state_dict()
    assert state.keys() == saved.keys()
    for key, value in saved.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0, msg=key)
    # Non-persistent buffers are computed in __init__, not by the init functions skipped.
    assert buffers, "expected rotary/position buffers"
    loaded_buffers = _non_persistent_buffers(loaded)
    assert loaded_buffers.keys() == buffers.keys()
    for name, value in buffers.items():
        torch.testing.assert_close(loaded_buffers[name], value, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_non_strict_load_keeps_normal_init(tiny_pi, init_calls, variant, tmp_path):
    torch.manual_seed(0)
    source = variant.policy_cls(_config(variant))
    saved = _save(source, tmp_path)
    dropped = "model.action_out_proj.weight" if variant.name != "pi0_fast" else next(iter(saved))
    checkpoint = load_file(tmp_path / "model.safetensors")
    del checkpoint[dropped]
    save_file(checkpoint, tmp_path / "model.safetensors")

    torch.manual_seed(1)
    reference = variant.policy_cls(_config(variant)).state_dict()[dropped]
    init_calls.clear()
    torch.manual_seed(1)
    loaded = variant.policy_cls.from_pretrained(tmp_path, config=_config(variant), strict=False)

    assert init_calls, "a non-strict load must build with the normal random init"
    missing = loaded.state_dict()[dropped]
    assert torch.isfinite(missing).all()
    torch.testing.assert_close(missing, reference, rtol=0, atol=0)
    other = next(key for key in saved if key != dropped)
    torch.testing.assert_close(loaded.state_dict()[other], saved[other], rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_strict_load_raises_on_missing_key(tiny_pi, variant, tmp_path):
    source = variant.policy_cls(_config(variant))
    _save(source, tmp_path)
    checkpoint = load_file(tmp_path / "model.safetensors")
    dropped = sorted(checkpoint)[-1]
    del checkpoint[dropped]
    save_file(checkpoint, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError) as error:
        variant.policy_cls.from_pretrained(tmp_path, config=_config(variant), strict=True)
    message = str(error.value)
    assert dropped in message
    assert "strict=False" in message


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_strict_load_warns_on_unexpected_keys(tiny_pi, variant, tmp_path, caplog):
    # Published checkpoints such as lerobot/pi0_libero still carry the old normalisation buffers.
    saved = _save(variant.policy_cls(_config(variant)), tmp_path)
    checkpoint = load_file(tmp_path / "model.safetensors")
    checkpoint["normalize_inputs.buffer_observation_state.mean"] = torch.zeros(4)
    save_file(checkpoint, tmp_path / "model.safetensors")

    with caplog.at_level("WARNING"):
        loaded = variant.policy_cls.from_pretrained(tmp_path, config=_config(variant), strict=True)

    assert "normalize_inputs.buffer_observation_state.mean" in caplog.text
    state = loaded.state_dict()
    assert state.keys() == saved.keys()
    for key, value in saved.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0, msg=key)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_missing_checkpoint_file_raises(tiny_pi, variant, tmp_path):
    _config(variant).save_pretrained(tmp_path)  # config.json only, no weights

    with pytest.raises(OSError, match="model.safetensors"):
        variant.policy_cls.from_pretrained(tmp_path, config=_config(variant), local_files_only=True)


def test_corrupt_checkpoint_file_raises(tiny_pi, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file")

    with pytest.raises(Exception, match=r"(?i)header|deserializ"):
        PI05Policy.from_pretrained(tmp_path, config=_config(VARIANTS[2]), local_files_only=True)


def test_pi05_base_checkpoint_gets_fresh_proprio_memory_projection(tiny_pi, tmp_path):
    variant = VARIANTS[2]
    saved = _save(variant.policy_cls(_config(variant)), tmp_path)
    assert not set(PROPRIO_KEYS) & saved.keys()

    torch.manual_seed(3)
    loaded = PI05Policy.from_pretrained(
        tmp_path, config=_config(variant, use_proprioceptive_memory=True), strict=True
    )
    # The projection is the only random init a strict load runs, so it matches a fresh nn.Linear.
    torch.manual_seed(3)
    expected = nn.Linear(4, 16)

    projection = loaded.model.proprio_history_proj
    assert torch.isfinite(projection.weight).all() and torch.isfinite(projection.bias).all()
    assert projection.weight.abs().sum() > 0
    torch.testing.assert_close(projection.weight, expected.weight, rtol=0, atol=0)
    torch.testing.assert_close(projection.bias, expected.bias, rtol=0, atol=0)
    state = loaded.state_dict()
    for key, value in saved.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0, msg=key)


def test_pi05_non_strict_load_draws_the_same_random_numbers_as_a_normal_build(tiny_pi, tmp_path):
    variant = VARIANTS[2]
    _save(variant.policy_cls(_config(variant)), tmp_path)
    memory_config = _config(variant, use_proprioceptive_memory=True)

    torch.manual_seed(4)
    reference = PI05Policy(memory_config).model.proprio_history_proj
    rng_after_build = torch.random.get_rng_state()
    torch.manual_seed(4)
    loaded = PI05Policy.from_pretrained(tmp_path, config=memory_config, strict=False)

    # The projection keeps its construction-time init: nothing re-draws it after the build.
    assert torch.equal(torch.random.get_rng_state(), rng_after_build)
    projection = loaded.model.proprio_history_proj
    torch.testing.assert_close(projection.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(projection.bias, reference.bias, rtol=0, atol=0)


def test_pi05_memory_checkpoint_strict_load_draws_no_random_numbers(tiny_pi, tmp_path):
    memory_config = _config(VARIANTS[2], use_proprioceptive_memory=True)
    saved = _save(PI05Policy(memory_config), tmp_path)
    assert set(PROPRIO_KEYS) <= saved.keys()

    rng_before = torch.random.get_rng_state()
    loaded = PI05Policy.from_pretrained(tmp_path, config=memory_config, strict=True)

    # The checkpoint supplies the projection, so it is not re-initialised before the load.
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for key in PROPRIO_KEYS:
        torch.testing.assert_close(loaded.state_dict()[key], saved[key], rtol=0, atol=0)
