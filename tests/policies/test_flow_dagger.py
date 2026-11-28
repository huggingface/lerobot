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

"""Tests for FlowDAgger: action inversion, sample selection, the noise policy and its training."""

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.common.flow_matching import FlowConvention, euler_integrate, invert_euler
from lerobot.policies.flow_dagger import (
    NoisePolicy,
    NoisePolicyConfig,
    NoisePolicyHolder,
    check_noise_policy_compatible,
    masked_mean_pool,
)
from lerobot.policies.flow_dagger.inversion import (
    InversionResult,
    SampleSelection,
    frames_to_next_intervention,
    hold_last_beyond,
    invert_dataset,
    segment_run_lengths,
    select_samples,
    summarize_inversion,
)
from lerobot.policies.flow_dagger.training import (
    BalancedSampler,
    NoisePolicyTrainConfig,
    train_noise_policy,
)

# ---------------------------------------------------------------------------
# invert_euler
# ---------------------------------------------------------------------------


def _nonlinear_velocity(seed: int = 0, dim: int = 3):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, generator=gen) * 0.5
    b = torch.randn(dim, generator=gen)

    def v(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ a) + b * t[:, None, None]

    return v


@pytest.mark.parametrize("convention", list(FlowConvention))
def test_invert_euler_recovers_noise(convention):
    v = _nonlinear_velocity()
    noise = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(1))
    actions = euler_integrate(v, noise, num_steps=10, convention=convention)

    recovered = invert_euler(v, actions, 10, num_fixed_point_iters=30, convention=convention)

    torch.testing.assert_close(recovered, noise, atol=1e-5, rtol=0)
    torch.testing.assert_close(euler_integrate(v, recovered, num_steps=10, convention=convention), actions)


def test_invert_euler_fixed_point_beats_explicit_reverse_step():
    v = _nonlinear_velocity(seed=3)
    noise = torch.randn(8, 5, 3, generator=torch.Generator().manual_seed(2))
    actions = euler_integrate(v, noise, num_steps=10)

    err_explicit = (invert_euler(v, actions, 10, num_fixed_point_iters=1) - noise).abs().max()
    err_fixed_point = (invert_euler(v, actions, 10, num_fixed_point_iters=5) - noise).abs().max()

    assert err_fixed_point < err_explicit / 10


def test_invert_euler_rejects_bad_arguments():
    v = _nonlinear_velocity()
    with pytest.raises(ValueError, match="num_fixed_point_iters"):
        invert_euler(v, torch.zeros(1, 2, 3), 10, num_fixed_point_iters=0)
    with pytest.raises(ValueError, match="num_steps"):
        invert_euler(v, torch.zeros(1, 2, 3), 0)


# ---------------------------------------------------------------------------
# Model-level hooks (pi05 / smolvla) with a stubbed VLM and action expert
# ---------------------------------------------------------------------------

_B, _L, _D, _H, _A = 2, 6, 16, 4, 3


def _stub_prefix_and_expert(model: nn.Module):
    gen = torch.Generator().manual_seed(0)
    prefix_out = torch.randn(_B, _L, _D, generator=gen)
    pad_mask = torch.ones(_B, _L, dtype=torch.bool)
    pad_mask[1, -2:] = False
    proj = torch.randn(_D, _A, generator=gen) * 0.3
    mix = torch.randn(_A, _A, generator=gen) * 0.4

    def compute_prefix(*args, **kwargs):
        return prefix_out, pad_mask, "kv-cache"

    def denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        assert past_key_values == "kv-cache"
        cond = masked_mean_pool(prefix_out, prefix_pad_masks) @ proj  # (B, A)
        return torch.tanh(x_t @ mix) - x_t + cond[:, None, :] * timestep[:, None, None]

    model._compute_prefix = compute_prefix
    model.denoise_step = denoise_step
    return masked_mean_pool(prefix_out, pad_mask)


def _make_pi05_model():
    pytest.importorskip("transformers")
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = PI05Config(chunk_size=_H, n_action_steps=_H, max_action_dim=_A, num_inference_steps=10)
    model.rtc_processor = None
    features = _stub_prefix_and_expert(model)

    def sample(noise=None, noise_fn=None):
        tokens = torch.zeros(_B, 1, dtype=torch.long)
        return model.sample_actions(None, None, tokens, None, noise=noise, noise_fn=noise_fn)

    def invert(actions):
        tokens = torch.zeros(_B, 1, dtype=torch.long)
        return model.invert_actions(None, None, tokens, None, actions, num_fixed_point_iters=20)

    return sample, invert, features


def _make_smolvla_model():
    pytest.importorskip("transformers")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SmolVLAConfig(chunk_size=_H, n_action_steps=_H, max_action_dim=_A, num_steps=10)
    model.rtc_processor = None
    features = _stub_prefix_and_expert(model)
    state = torch.zeros(_B, 2)

    def sample(noise=None, noise_fn=None):
        return model.sample_actions(None, None, None, None, state, noise=noise, noise_fn=noise_fn)

    def invert(actions):
        return model.invert_actions(None, None, None, None, state, actions, num_fixed_point_iters=20)

    return sample, invert, features


@pytest.mark.parametrize("make_model", [_make_pi05_model, _make_smolvla_model], ids=["pi05", "smolvla"])
def test_policy_inversion_round_trip(make_model):
    sample, invert, expected_features = make_model()
    expert_actions = torch.randn(_B, _H, _A, generator=torch.Generator().manual_seed(5))

    noise, features = invert(expert_actions)

    torch.testing.assert_close(features, expected_features)
    torch.testing.assert_close(sample(noise=noise), expert_actions, atol=1e-5, rtol=0)


@pytest.mark.parametrize("make_model", [_make_pi05_model, _make_smolvla_model], ids=["pi05", "smolvla"])
def test_policy_noise_fn_seeds_denoising(make_model):
    sample, _, expected_features = make_model()
    fixed = torch.randn(_B, _H, _A, generator=torch.Generator().manual_seed(6))
    seen = []

    def noise_fn(features):
        seen.append(features)
        return fixed

    out = sample(noise_fn=noise_fn)

    torch.testing.assert_close(seen[0], expected_features)
    torch.testing.assert_close(out, sample(noise=fixed))
    # An explicit noise wins over the noise policy.
    other = torch.zeros_like(fixed)
    torch.testing.assert_close(sample(noise=other, noise_fn=noise_fn), sample(noise=other))
    assert len(seen) == 1


def test_smolvla_invert_action_chunk_leaves_caller_batch_untouched():
    pytest.importorskip("transformers")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SmolVLAConfig(adapt_to_pi_aloha=True, chunk_size=_H, n_action_steps=_H)
    seen = {}
    policy.prepare_images = lambda batch: (None, None)
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: seen.setdefault("action", batch[ACTION])

    class _Model(nn.Module):
        def invert_actions(self, *args, **kwargs):
            return torch.zeros(1), torch.zeros(1)

    policy.model = _Model()
    gen = torch.Generator().manual_seed(0)
    batch = {
        OBS_STATE: torch.rand(_B, 14, generator=gen),
        ACTION: torch.rand(_B, _H, 14, generator=gen),
        OBS_LANGUAGE_TOKENS: torch.zeros(_B, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(_B, 1, dtype=torch.bool),
    }
    original = {k: v.clone() for k, v in batch.items()}

    policy.invert_action_chunk(batch)

    for key, value in original.items():
        torch.testing.assert_close(batch[key], value)
    assert not torch.equal(seen["action"], original[ACTION])  # the transform did apply internally


class _RecordingPolicy:
    """Minimal InvertiblePolicy that records the preprocessed batches it is given."""

    def __init__(self, chunk_size: int, action_dim: int):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.utils.constants import ACTION

        self.config = type(
            "Cfg",
            (),
            {
                "chunk_size": chunk_size,
                "max_action_dim": action_dim,
                "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
            },
        )()
        self.batches: list[dict] = []

    def invert_action_chunk(self, batch, *, num_fixed_point_iters=5):
        from lerobot.utils.constants import ACTION

        self.batches.append(batch)
        return batch[ACTION].clone(), torch.zeros(batch[ACTION].shape[0], 2)

    def predict_action_chunk(self, batch, *, noise=None):
        return noise

    def reset(self):
        pass


def test_invert_dataset_applies_rename_map_before_cropping():
    from lerobot.utils.constants import ACTION

    chunk, action_dim, n = 3, 2, 4
    cam_raw, cam = "observation.images.top_raw", "observation.images.top"
    items = [
        {
            "actions": torch.full((chunk, action_dim), float(i)),
            "actions_is_pad": torch.zeros(chunk, dtype=torch.bool),
            cam_raw: torch.full((3, 2, 2), 255, dtype=torch.uint8),
        }
        for i in range(n)
    ]
    for item in items:
        item["actions"][-1] = 99.0  # beyond valid_len: must be replaced by the last valid action
    selection = SampleSelection(
        intervention_positions=np.array([0, 1]),
        autonomous_positions=np.array([2, 3]),
        valid_len=np.full(n, 2),
    )
    policy = _RecordingPolicy(chunk, action_dim)

    result = invert_dataset(
        policy,
        lambda batch: batch,
        items,
        selection,
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        rename_map={"actions": ACTION, cam_raw: cam},
        camera_keys=[cam_raw],
        progress=False,
    )

    assert len(result) == n
    for batch in policy.batches:
        assert "actions" not in batch and f"{ACTION}_is_pad" not in batch
        assert batch[cam].dtype == torch.float32 and torch.all(batch[cam] == 1.0)
        torch.testing.assert_close(batch[ACTION][:, -1], batch[ACTION][:, 1])
    torch.testing.assert_close(result.reconstruction_mse, torch.zeros(n))


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def test_segment_run_lengths_and_next_intervention():
    ep = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    iv = np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)

    assert segment_run_lengths(ep, iv).tolist() == [2, 1, 2, 1, 1, 2, 1, 1]
    assert frames_to_next_intervention(ep, iv).tolist() == [2, 1, 0, 0, np.inf, 2, 1, 0]


def test_select_samples():
    # episode 0: 6 autonomous frames, 3 intervention frames, 2 autonomous; episode 1: 4 autonomous.
    ep = np.array([0] * 11 + [1] * 4)
    iv = np.array([0] * 6 + [1] * 3 + [0] * 2 + [0] * 4, dtype=bool)

    sel = select_samples(ep, iv, chunk_size=4, min_valid_steps=2, exclude_pre_intervention_frames=2)

    # Interventions with >= 2 steps left in their segment: rows 6, 7 (row 8 has 1).
    assert sel.intervention_positions.tolist() == [6, 7]
    # Autonomous rows: > 2 frames before the intervention (0..3), plus segments long enough.
    assert sel.autonomous_positions.tolist() == [0, 1, 2, 3, 9, 11, 12, 13]
    assert sel.valid_len.tolist()[:6] == [4, 4, 4, 3, 2, 1]
    assert sel.is_intervention.tolist() == [True, True] + [False] * 8

    strided = select_samples(
        ep, iv, chunk_size=4, min_valid_steps=2, exclude_pre_intervention_frames=2, autonomous_stride=3
    )
    assert strided.autonomous_positions.tolist() == [0, 3, 12]

    only_interventions = select_samples(
        ep, iv, chunk_size=4, min_valid_steps=2, exclude_pre_intervention_frames=2, use_autonomous=False
    )
    assert len(only_interventions.autonomous_positions) == 0


def test_hold_last_beyond():
    actions = torch.arange(2 * 4 * 1, dtype=torch.float32).view(2, 4, 1)
    out = hold_last_beyond(actions, torch.tensor([2, 4]))
    assert out[0, :, 0].tolist() == [0, 1, 1, 1]
    assert out[1, :, 0].tolist() == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Noise policy
# ---------------------------------------------------------------------------


def _noise_cfg(**overrides) -> NoisePolicyConfig:
    kwargs = {
        "feature_dim": 8,
        "chunk_size": 4,
        "max_action_dim": 3,
        "base_policy_type": "smolvla",
        "num_inference_steps": 10,
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return NoisePolicyConfig(**kwargs)


def test_noise_policy_starts_at_zero_and_round_trips(tmp_path):
    policy = NoisePolicy(_noise_cfg())
    features = torch.randn(5, 8)
    assert torch.equal(policy(features), torch.zeros(5, 4, 3))

    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    policy.save_pretrained(tmp_path / "np")
    loaded = NoisePolicy.from_pretrained(tmp_path / "np")

    assert loaded.config == policy.config
    torch.testing.assert_close(loaded.eval()(features), policy.eval()(features))


def test_check_noise_policy_compatible():
    cfg = _noise_cfg()
    check_noise_policy_compatible(cfg, policy_type="smolvla", chunk_size=4, max_action_dim=3)
    with pytest.raises(ValueError, match="chunk_size"):
        check_noise_policy_compatible(cfg, policy_type="smolvla", chunk_size=50, max_action_dim=3)
    with pytest.raises(ValueError, match="policy type"):
        check_noise_policy_compatible(cfg, policy_type="pi05", chunk_size=4, max_action_dim=3)


def test_noise_policy_holder(tmp_path):
    NoisePolicy(_noise_cfg()).save_pretrained(tmp_path / "np")
    holder = NoisePolicyHolder()
    kwargs = {
        "policy_type": "smolvla",
        "chunk_size": 4,
        "max_action_dim": 3,
        "num_inference_steps": 10,
        "device": torch.device("cpu"),
    }

    assert holder.get(None, **kwargs) is None
    loaded = holder.get(str(tmp_path / "np"), **kwargs)
    assert isinstance(loaded, NoisePolicy) and not loaded.training
    assert holder.get(str(tmp_path / "np"), **kwargs) is loaded  # cached

    in_memory = NoisePolicy(_noise_cfg(dropout=0.5))
    assert in_memory.training
    holder.set(in_memory)
    assert holder.get(None, **kwargs) is in_memory
    # Attached in memory, it is frozen exactly like a path-loaded noise policy (dropout off).
    assert not in_memory.training
    assert not any(p.requires_grad for p in in_memory.parameters())
    with pytest.raises(ValueError, match="chunk_size"):
        NoisePolicyHolder().get(str(tmp_path / "np"), **{**kwargs, "chunk_size": 50})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_balanced_sampler_fractions():
    gen = torch.Generator().manual_seed(0)
    interventions, autonomous = torch.arange(0, 5), torch.arange(100, 200)
    batch = BalancedSampler(interventions, autonomous, 64, 0.5, gen).sample()
    assert (batch < 100).sum() == 32 and (batch >= 100).sum() == 32

    only_interventions = BalancedSampler(interventions, torch.zeros(0, dtype=torch.long), 64, 0.5, gen)
    assert (only_interventions.sample() < 100).all()
    with pytest.raises(ValueError, match="empty"):
        BalancedSampler(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), 8, 0.5, gen)


def test_train_noise_policy_fits_targets():
    gen = torch.Generator().manual_seed(0)
    n = 512
    features = torch.randn(n, 8, generator=gen)
    mapping = torch.randn(8, 12, generator=gen) * 0.5
    noise = (features @ mapping).view(n, 4, 3)
    data = InversionResult(
        features=features,
        noise=noise,
        is_intervention=torch.arange(n) < 64,
        reconstruction_mse=torch.full((n,), float("nan")),
    )
    policy = NoisePolicy(_noise_cfg())
    cfg = NoisePolicyTrainConfig(steps=400, batch_size=64, lr=3e-3, warmup_steps=10, log_freq=400)

    metrics = train_noise_policy(policy, data, cfg, torch.device("cpu"), seed=0)

    baseline = float((noise**2).mean())
    assert metrics["val/intervention_mse"] < 0.1 * baseline
    assert metrics["val/autonomous_mse"] < 0.1 * baseline
    stats = summarize_inversion(data)
    assert stats["intervention/num_samples"] == 64 and stats["autonomous/num_samples"] == n - 64


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------


def test_checkpoint_fingerprint_tracks_local_changes(tmp_path):
    from lerobot.scripts.lerobot_train_flow_dagger import _checkpoint_fingerprint

    (tmp_path / "model.safetensors").write_bytes(b"weights-v1")
    (tmp_path / "policy_preprocessor.json").write_text("{}")
    first = _checkpoint_fingerprint(tmp_path, None)
    assert first is not None and first == _checkpoint_fingerprint(tmp_path, None)

    (tmp_path / "model.safetensors").write_bytes(b"weights-v2-longer")
    assert _checkpoint_fingerprint(tmp_path, None) != first
