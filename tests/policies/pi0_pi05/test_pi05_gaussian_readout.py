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

"""CPU checks for the opt-in pi0.5 Gaussian readout. The default sampler stays Euler."""

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.gaussian_readout import (
    GaussianReadout,
    bind_processor_sha256,
    collapse_label,
    diagnose_two_mode_obstacle,
    flow_raw_action,
    load_gaussian_readout,
    make_epsilon_star,
    normalized_command,
    project_velocity,
    save_readout,
    solve_affine_residual,
    state_dict_sha256,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch


class _AttentionShell:
    def __init__(self) -> None:
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
            )
        )
        self.gemma_expert = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
        )

    def forward(self, **kwargs):
        embeds = kwargs["inputs_embeds"]
        if embeds[0] is None:
            return [None, embeds[1]], None
        return None, {"cached": True}


def _readout_config(**overrides):
    values = {
        "gaussian_readout_path": None,
        "num_inference_steps": 10,
        "chunk_size": 4,
        "max_action_dim": 8,
        "compile_model": False,
        "use_visual_memory": False,
        "use_proprioceptive_memory": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _EulerFake:
    """Enough of PI05Pytorch for the unchanged Euler branch. ``denoise_step`` is intentionally local."""

    def __init__(self) -> None:
        self.config = _readout_config()
        self.use_typed_attention_masks = False
        self.precompute_denoise_times = False
        self.rtc_processor = None
        self.paligemma_with_expert = _AttentionShell()

    def sample_noise(self, shape, device):
        return PI05Pytorch.sample_noise(self, shape, device)

    def embed_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        del images, img_masks, states, state_masks
        batch = tokens.shape[0]
        embeddings = torch.zeros(
            batch, 2, 4, dtype=masks.dtype if masks.dtype.is_floating_point else torch.float32
        )
        pad = torch.ones(batch, 2, dtype=torch.bool)
        attention = torch.zeros(batch, 2, dtype=torch.float32)
        return embeddings, pad, attention

    def _prepare_attention_masks_4d(self, masks, dtype=None):
        return PI05Pytorch._prepare_attention_masks_4d(self, masks, dtype)

    def _rtc_enabled(self) -> bool:
        return False

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        del prefix_pad_masks, past_key_values, timestep
        return -0.25 * x_t


class _ReadoutFake(nn.Module):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.config = _readout_config(gaussian_readout_path=path)
        self.use_typed_attention_masks = False
        self.precompute_denoise_times = False
        self.rtc_processor = None
        self.paligemma_with_expert = _AttentionShell()
        self.action_out_proj = nn.Linear(8, 8)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.fixed_features = torch.ones(1, 4, 8)

    def embed_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        del images, img_masks, states, state_masks
        batch = tokens.shape[0]
        embeddings = torch.zeros(batch, 2, 4)
        pad = torch.ones(batch, 2, dtype=torch.bool)
        attention = torch.zeros(batch, 2)
        return embeddings, pad, attention

    def _prepare_attention_masks_4d(self, masks, dtype=None):
        return PI05Pytorch._prepare_attention_masks_4d(self, masks, dtype)

    def _rtc_enabled(self) -> bool:
        return False

    def suffix_features(self, prefix_pad_masks, past_key_values, x_t, timestep):
        del prefix_pad_masks, past_key_values
        self.calls.append((x_t.detach().clone(), timestep.detach().clone()))
        return self.fixed_features.expand(x_t.shape[0], -1, -1).clone()


class _HelperFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=4)
        self.action_out_proj = nn.Linear(4, 4)
        self.paligemma_with_expert = _AttentionShell()
        self.calls = 0

    def embed_suffix(self, noisy_actions, timestep):
        del timestep
        batch, horizon, _ = noisy_actions.shape
        pad = torch.ones(batch, horizon, dtype=torch.bool)
        attention = torch.zeros(batch, horizon)
        attention[:, 0] = 1
        return noisy_actions, pad, attention, None

    def _prepare_attention_masks_4d(self, masks, dtype=None):
        return PI05Pytorch._prepare_attention_masks_4d(self, masks, dtype)

    def suffix_features(self, prefix_pad_masks, past_key_values, x_t, timestep):
        self.calls += 1
        return PI05Pytorch.suffix_features(self, prefix_pad_masks, past_key_values, x_t, timestep)


def _tokens():
    return torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool)


def _call_euler(model, **kwargs):
    tokens, masks = _tokens()
    return PI05Pytorch.sample_actions(model, [], [], tokens, masks, **kwargs)


def test_default_off_matches_explicit_step_count_and_rng():
    assert PI05Config().gaussian_readout_path is None
    model = _EulerFake()
    torch.manual_seed(4)
    default = _call_euler(model)
    torch.manual_seed(4)
    explicit = _call_euler(model, num_steps=10)
    torch.manual_seed(4)
    shorter = _call_euler(model, num_steps=3)
    assert torch.equal(default, explicit)
    assert default.shape == (1, 4, 8)
    assert not torch.equal(default, shorter)

    noise = torch.randn(1, 4, 8)
    torch.manual_seed(1)
    first = _call_euler(model, noise=noise)
    torch.manual_seed(2)
    second = _call_euler(model, noise=noise)
    assert torch.equal(first, second)

    torch.manual_seed(9)
    again = _call_euler(model)
    torch.manual_seed(9)
    repeated = _call_euler(model)
    assert torch.equal(again, repeated)


def test_time_one_sign_and_single_feature_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_pi05.euler_integrate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("euler")),
    )
    monkeypatch.setattr("lerobot.policies.pi05.modeling_pi05.clone_past_key_values", lambda past: past)

    path = tmp_path / "readout.safetensors"
    model = _ReadoutFake(str(path))
    bind_processor_sha256(model, "processor-hash")
    mean = torch.zeros(7)
    std = torch.ones(7)
    open_code = float(normalized_command(-1.0, mean[6], std[6]))
    close_code = float(normalized_command(1.0, mean[6], std[6]))
    readout = GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=open_code,
        close_normalized=close_code,
        action_mean=mean,
        action_std=std,
    )
    with torch.no_grad():
        readout.affine_weight.zero_()
        readout.affine_bias.zero_()
        readout.gripper_weight.zero_()
        readout.gripper_bias.fill_(-10)
        readout.var_in.weight.zero_()
        readout.var_in.bias.zero_()
        readout.var_out.weight.zero_()
        readout.var_out.bias.fill_(-20)
    save_readout(readout, path, teacher_sha256=state_dict_sha256(model), processor_sha256="processor-hash")

    _call_euler(model)
    assert len(model.calls) == 1
    model.calls.clear()
    torch.manual_seed(0)
    action = _call_euler(model)
    assert len(model.calls) == 1
    seen_noise, seen_time = model.calls[0]
    probe = make_epsilon_star(4, 8)
    assert torch.equal(seen_time, torch.ones(1))
    assert torch.equal(seen_noise, probe)
    features = model.fixed_features
    velocity = project_velocity(model, features)
    raw = flow_raw_action(probe, velocity)
    assert torch.allclose(raw, probe - velocity)
    assert not torch.allclose(raw, probe + velocity)
    generator = torch.Generator().manual_seed(0)
    motion_noise = torch.randn(1, 4, 6, generator=generator)
    command_noise = torch.randn(1, 4, generator=generator)
    scale = readout.variance(features).sqrt()
    expected_motion = raw[..., :6] + scale * motion_noise
    expected_gripper = torch.where(
        command_noise + (-10) >= 0, torch.tensor(close_code), torch.tensor(open_code)
    )
    assert torch.allclose(action[..., :6], expected_motion, atol=1e-6)
    assert torch.allclose(action[..., 6], expected_gripper)
    assert torch.equal(action[..., 7:], torch.zeros_like(action[..., 7:]))
    assert set(action[..., 6].reshape(-1).tolist()) <= {open_code, close_code}

    helper = _HelperFake()
    prefix_pad = torch.ones(1, 2, dtype=torch.bool)
    hidden = torch.randn(1, 4, 4)
    features = helper.suffix_features(prefix_pad, None, hidden, torch.ones(1))
    velocity = PI05Pytorch.denoise_step(helper, prefix_pad, None, hidden, torch.ones(1))
    assert helper.calls == 2
    assert torch.equal(velocity, helper.action_out_proj(features))

    rejected = _ReadoutFake(str(path))
    bind_processor_sha256(rejected, "processor-hash")
    rejected.config.compile_model = True
    torch.manual_seed(5)
    reference = torch.randn(3)
    torch.manual_seed(5)
    with pytest.raises(ValueError, match="compilation"):
        _call_euler(rejected)
    assert torch.equal(reference, torch.randn(3))
    rejected.config.compile_model = False
    rejected.config.use_visual_memory = True
    with pytest.raises(ValueError, match="memory"):
        _call_euler(rejected)
    rejected.config.use_visual_memory = False
    rejected.config.use_proprioceptive_memory = True
    with pytest.raises(ValueError, match="memory"):
        _call_euler(rejected)
    rejected.config.use_proprioceptive_memory = False
    with pytest.raises(ValueError, match="starting noise"):
        _call_euler(rejected, noise=torch.zeros(1, 4, 8))
    with pytest.raises(ValueError, match="step count"):
        _call_euler(rejected, num_steps=2)
    with pytest.raises(ValueError, match="real-time chunking"):
        _call_euler(rejected, inference_delay=1)

    class _Rtc(_ReadoutFake):
        def _rtc_enabled(self) -> bool:
            return True

    with pytest.raises(ValueError, match="real-time chunking"):
        _call_euler(_Rtc(str(path)))


def test_frozen_weights_and_readout_roundtrip(tmp_path):
    teacher = nn.Linear(4, 4)
    before = [parameter.detach().clone() for parameter in teacher.parameters()]
    count, horizon, width, motion = 20, 4, 8, 6
    features = torch.randn(count, horizon, width)
    weight = torch.randn(motion, width)
    bias = torch.randn(horizon, motion)
    raw = torch.randn(count, horizon, motion)
    teacher_mean = raw + torch.einsum("btw,mw->btm", features, weight) + bias
    solved_weight, solved_bias = solve_affine_residual(features, raw, teacher_mean)
    assert torch.allclose(solved_weight, weight, atol=1e-4, rtol=1e-4)
    assert torch.allclose(solved_bias, bias, atol=1e-4, rtol=1e-4)

    path = tmp_path / "roundtrip.safetensors"
    host = _ReadoutFake(str(path))
    bind_processor_sha256(host, "processor-hash")
    mean = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.05])
    std = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    open_code = normalized_command(-1.0, mean[6], std[6])
    close_code = normalized_command(1.0, mean[6], std[6])
    assert torch.allclose(open_code * std[6] + mean[6], torch.tensor(-1.0))
    assert torch.allclose(close_code * std[6] + mean[6], torch.tensor(1.0))
    readout = GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=float(open_code),
        close_normalized=float(close_code),
        action_mean=mean,
        action_std=std,
    )
    with torch.no_grad():
        readout.affine_weight.copy_(solved_weight)
        readout.affine_bias.copy_(solved_bias)
    original_hash = state_dict_sha256(host)
    save_readout(readout, path, teacher_sha256=original_hash, processor_sha256="processor-hash")
    loaded = load_gaussian_readout(path, host, "processor-hash")
    for key, value in readout.state_dict().items():
        assert torch.equal(value.cpu(), loaded.state_dict()[key].cpu()), key
    assert torch.equal(loaded.motion_mask[:6], torch.ones(6))
    assert torch.equal(loaded.motion_mask[6:], torch.zeros(2))
    assert torch.equal(loaded.pad_mask[7:], torch.ones(1))
    assert state_dict_sha256(host) == original_hash
    for parameter, saved in zip(teacher.parameters(), before, strict=True):
        assert torch.equal(parameter, saved)

    with pytest.raises(ValueError, match="teacher hash"):
        save_readout(readout, path, teacher_sha256="0" * 64, processor_sha256="processor-hash")
        load_gaussian_readout(path, host, "processor-hash")
    save_readout(readout, path, teacher_sha256=original_hash, processor_sha256="processor-hash")
    with pytest.raises(ValueError, match="processor hash"):
        load_gaussian_readout(path, host, "other-processor")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {}
        for key in handle.keys():  # noqa: SIM118
            tensors[key] = handle.get_tensor(key)
    tensors["epsilon_star"] = tensors["epsilon_star"] + 1
    tampered = tmp_path / "tampered.safetensors"
    save_file(tensors, str(tampered), metadata=metadata)
    with pytest.raises(ValueError, match="fixed noise"):
        load_gaussian_readout(tampered, host, "processor-hash")
    metadata["format"] = "other"
    save_file(tensors, str(tampered), metadata=metadata)
    with pytest.raises(ValueError, match="format"):
        load_gaussian_readout(tampered, host, "processor-hash")


def test_synthetic_two_mode_diagnostic():
    report = diagnose_two_mode_obstacle()
    assert report["teacher_modes"] == ["lower", "upper"]
    assert report["teacher_hits_obstacle"] is False
    assert report["mean_between_targets"] is True
    assert report["mean_mode"] == "neither"
    assert report["lost_modes"] == ["lower", "upper"]
    assert report["collapse_label"] == "collapse-observed"
    assert report["one_mode_label"] == "inconclusive"
    assert abs(report["mean_endpoint_y"]) < 1e-5
    assert collapse_label({"soup", "sauce"}, {"soup"}) == "collapse-observed"
    assert collapse_label({"soup"}, {"soup"}) == "inconclusive"
    assert collapse_label({"soup", "sauce"}, {"soup", "sauce"}) == "both-covered"
