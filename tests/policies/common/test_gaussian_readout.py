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

"""CPU checks for the shared opt-in Gaussian readout. The default sampler stays Euler."""

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.common.gaussian_readout import (
    GaussianReadout,
    bind_processor_sha256,
    flow_raw_action,
    load_gaussian_readout,
    make_epsilon_star,
    normalized_command,
    project_velocity,
    save_readout,
    solve_affine_residual,
    state_dict_sha256,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config
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


def _tokens():
    return torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool)


def _call_euler(model, **kwargs):
    tokens, masks = _tokens()
    return PI05Pytorch.sample_actions(model, [], [], tokens, masks, **kwargs)


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


def test_readout_load_preserves_cpu_rng(tmp_path):
    path = tmp_path / "rng.safetensors"
    host = _ReadoutFake(str(path))
    readout = GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=-1.0,
        close_normalized=1.0,
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )
    save_readout(readout, path, teacher_sha256=state_dict_sha256(host), processor_sha256="processor-hash")
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    load_gaussian_readout(path, host, "processor-hash")
    assert torch.equal(torch.random.get_rng_state(), before)


class _TinyBackbone(nn.Module):
    """Random CPU attention replaces the large VLM; policy embeddings/sampling are real."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
        self.expert_hidden_size = 8
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.paligemma = _AttentionShell().paligemma
        self.gemma_expert = _AttentionShell().gemma_expert
        self.suffix_calls = 0

    def embed_language_tokens(self, tokens):
        return self.embedding(tokens)

    def forward(self, *, inputs_embeds, past_key_values=None, attention_mask=None, **kwargs):
        prefix, suffix = inputs_embeds
        if suffix is None:
            return [prefix, None], SimpleNamespace(prefix=prefix, crop=lambda length: None)
        self.suffix_calls += 1
        condition = kwargs.get("adarms_cond", [None, None])[1]
        if condition is not None:
            suffix = suffix + condition[:, None, :]
        context = torch.cat((past_key_values.prefix, suffix), dim=1)
        mask = attention_mask[:, 0] if attention_mask.ndim == 4 else attention_mask
        if mask.dtype == torch.bool:
            mask = ~mask
        result, _ = self.attention(suffix, context, context, attn_mask=mask.repeat_interleave(2, dim=0))
        return [None, result], past_key_values


@pytest.fixture(params=["pi0", "pi05", "smolvla", "fineart_vla"])
def tiny_flow(request, monkeypatch):
    from copy import deepcopy

    from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPytorch
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    classes = {
        "pi0": (PI0Config, PI0Pytorch),
        "pi05": (PI05Config, PI05Pytorch),
        "smolvla": (SmolVLAConfig, VLAFlowMatching),
        "fineart_vla": (FineARTVLAConfig, FineARTVLAPytorch),
    }
    config_cls, core_cls = classes[request.param]
    model = core_cls.__new__(core_cls)
    nn.Module.__init__(model)
    model.config = config_cls(chunk_size=4, n_action_steps=4, max_action_dim=8, max_state_dim=8, device="cpu")
    model.rtc_processor = None
    model.gradient_checkpointing_enabled = False
    for name, inputs in (
        ("action_in_proj", 8),
        ("action_out_proj", 8),
        ("state_proj", 8),
        ("time_mlp_in", 8),
        ("time_mlp_out", 8),
        ("action_time_mlp_in", 16),
        ("action_time_mlp_out", 8),
    ):
        setattr(model, name, nn.Linear(inputs, 8))
    backbone = _TinyBackbone()
    if request.param == "smolvla":
        model.vlm_with_expert = backbone
        model.add_image_special_tokens = False
        model.prefix_length = -1
    else:
        model.paligemma_with_expert = backbone
        monkeypatch.setattr(f"{core_cls.__module__}.clone_past_key_values", deepcopy, raising=False)
    model.eval()
    return model, backbone


def _sample_tiny(model, tokens, masks, state, **kwargs):
    if model.config.type in ("pi0", "smolvla"):
        return model.sample_actions([], [], tokens, masks, state, **kwargs)
    return model.sample_actions([], [], tokens, masks, **kwargs)


def test_small_cpu_fit_and_inference(tiny_flow, tmp_path):
    from lerobot.policies.common.gaussian_readout import (
        evaluate_time_one_features,
        fit_gripper_head,
        fit_variance_head,
    )

    model, backbone = tiny_flow
    assert model.config.gaussian_readout_path is None
    tokens = torch.randint(0, 16, (6, 2))
    masks, state = torch.ones_like(tokens, dtype=torch.bool), torch.randn(6, 8)
    before = state_dict_sha256(model)
    probe = make_epsilon_star(4, 8)
    with torch.no_grad():
        samples = torch.stack([_sample_tiny(model, tokens, masks, state) for _ in range(3)])
        backbone.suffix_calls = 0
        features, raw = evaluate_time_one_features(model, [], [], tokens, masks, state, None, probe)
    assert backbone.suffix_calls == 1
    readout = GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=-1,
        close_normalized=1,
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )
    weight, bias = solve_affine_residual(features, raw[..., :6], samples[..., :6].mean(dim=0))
    with torch.no_grad():
        readout.affine_weight.copy_(weight)
        readout.affine_bias.copy_(bias)
    variance = samples[..., :6].var(dim=0, unbiased=False).clamp_min(1e-6)
    frequency = (samples[..., 6] >= 0).float().mean(dim=0)
    old_var = readout.var_out.weight.detach().clone()
    old_gripper = readout.gripper_weight.detach().clone()
    fit_variance_head(readout, features, variance, features, variance, updates=3, batch_size=6)
    fit_gripper_head(readout, features, frequency, features, frequency, updates=3, batch_size=6)
    assert not torch.equal(readout.var_out.weight, old_var)
    assert not torch.equal(readout.gripper_weight, old_gripper)
    assert state_dict_sha256(model) == before
    path = tmp_path / "tiny.safetensors"
    save_readout(readout, path, teacher_sha256=before, processor_sha256="tiny-processors")
    bind_processor_sha256(model, "tiny-processors")
    model.config.gaussian_readout_path = str(path)
    backbone.suffix_calls = 0
    with torch.no_grad():
        for _ in range(3):
            action = _sample_tiny(model, tokens, masks, state)
            assert action.shape == (6, 4, 8)
            assert torch.isfinite(action).all()
            assert (action[..., 7] == 0).all()
            assert ((action[..., 6] == -1) | (action[..., 6] == 1)).all()
            assert torch.isfinite(readout.log_prob(features, raw, action)).all()
    assert backbone.suffix_calls == 3
    assert state_dict_sha256(model) == before
    with pytest.raises(ValueError, match="starting noise"):
        _sample_tiny(model, tokens, masks, state, noise=probe)


def test_mixed_log_prob_matches_distributions():
    readout = GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=-1,
        close_normalized=1,
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )
    features, raw = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    action = readout.sample_action(features, raw)
    normal = torch.distributions.Normal(
        readout.continuous_mean(features, raw), readout.variance(features).sqrt()
    )
    binary = torch.distributions.Bernoulli(probs=torch.special.ndtr(readout.gripper_latent(features)))
    expected = normal.log_prob(action[..., :6]).sum((-2, -1)) + binary.log_prob(
        (action[..., 6] == 1).float()
    ).sum(-1)
    torch.testing.assert_close(readout.log_prob(features, raw, action), expected)
    action[0, 0, 6] = 0
    action[1, 0, 7] = 1
    assert readout.log_prob(features, raw, action).isneginf().all()


def test_default_off_preserves_sampling_and_rng(tiny_flow, monkeypatch):
    model, _ = tiny_flow
    assert model.config.gaussian_readout_path is None
    monkeypatch.setattr(
        "lerobot.policies.common.gaussian_readout._cached_readout",
        lambda *args: pytest.fail("default sampler loaded a readout"),
    )
    tokens, masks = _tokens()
    state = torch.randn(1, 8)

    def draw(**kwargs):
        return _sample_tiny(model, tokens, masks, state, **kwargs)

    torch.manual_seed(4)
    expected = draw()
    after = torch.random.get_rng_state()
    torch.manual_seed(4)
    assert torch.equal(draw(), expected)
    assert torch.equal(torch.random.get_rng_state(), after)
    field = "num_steps" if model.config.type == "smolvla" else "num_inference_steps"
    setattr(model.config, field, 3)
    torch.manual_seed(4)
    assert not torch.equal(draw(), expected)
    noise = torch.randn(1, 4, 8)
    torch.manual_seed(1)
    expected = draw(noise=noise)
    torch.manual_seed(2)
    assert torch.equal(draw(noise=noise), expected)
