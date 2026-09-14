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

# ruff: noqa: E402

"""Tests for Real-Time Chunking (RTC) inference support in Multi-Task DiT.

To run tests locally:
    uv run pytest tests/policies/multi_task_dit/test_multi_task_dit_rtc.py -v
"""

import inspect
import logging
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from transformers import CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel

from lerobot.configs import RTCAttentionSchedule
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.policies.rtc import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

CAM_KEY = "observation.images.cam"
STATE_DIM = 4
ACTION_DIM = 4
CHUNK_LEN = 8
HORIZON = 16
N_OBS_STEPS = 2


def _make_config(**overrides) -> MultiTaskDiTConfig:
    kwargs = {
        "n_obs_steps": N_OBS_STEPS,
        "horizon": HORIZON,
        "n_action_steps": CHUNK_LEN,
        "objective": "diffusion",
        "noise_scheduler_type": "DDIM",
        "num_train_timesteps": 50,
        "num_inference_steps": 5,
        "hidden_dim": 64,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "image_resize_shape": (16, 16),
        "image_crop_shape": (16, 16),
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            CAM_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        "output_features": {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        "device": "cpu",
    }
    kwargs.update(overrides)
    return MultiTaskDiTConfig(**kwargs)


@pytest.fixture(scope="module", autouse=True)
def local_clip_models():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            CLIPVisionModel,
            "from_pretrained",
            lambda *a, **kw: CLIPVisionModel(
                CLIPVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=16,
                    patch_size=8,
                )
            ),
        )
        patch.setattr(
            CLIPTextModel,
            "from_pretrained",
            lambda *a, **kw: CLIPTextModel(
                CLIPTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    max_position_embeddings=8,
                    bos_token_id=0,
                    eos_token_id=2,
                    pad_token_id=1,
                )
            ),
        )
        yield


def _activate_conditioning(policy: MultiTaskDiTPolicy) -> None:
    generator = torch.Generator().manual_seed(123)
    for block in policy.noise_predictor.transformer_blocks:
        modulation = block.adaLN_modulation[-1]
        torch.nn.init.normal_(modulation.weight, std=0.02, generator=generator)
        torch.nn.init.constant_(modulation.bias, 0.1)


@pytest.fixture(scope="module")
def diffusion_policy() -> MultiTaskDiTPolicy:
    torch.manual_seed(42)
    policy = MultiTaskDiTPolicy(_make_config())
    _activate_conditioning(policy)
    policy.eval()
    return policy


@pytest.fixture(scope="module")
def flow_policy() -> MultiTaskDiTPolicy:
    torch.manual_seed(42)
    policy = MultiTaskDiTPolicy(
        _make_config(
            objective="flow_matching",
            num_integration_steps=10,
            integration_method="euler",
        )
    )
    _activate_conditioning(policy)
    policy.eval()
    return policy


@pytest.fixture(scope="module")
def ddpm_policy() -> MultiTaskDiTPolicy:
    torch.manual_seed(42)
    policy = MultiTaskDiTPolicy(_make_config(noise_scheduler_type="DDPM"))
    _activate_conditioning(policy)
    policy.eval()
    return policy


def _make_engine_batch() -> dict[str, torch.Tensor]:
    """Distinct consecutive observations in the engine temporal input format."""
    g = torch.Generator().manual_seed(7)
    return {
        OBS_STATE: torch.randn(1, N_OBS_STEPS, STATE_DIM, generator=g),
        CAM_KEY: torch.rand(1, N_OBS_STEPS, 3, 16, 16, generator=g),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 32, (1, 8), generator=g),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 8, dtype=torch.long),
    }


@contextmanager
def _rtc(policy, **rtc_overrides):
    """Enable RTC on a (module-scoped) policy fixture, restoring on exit.

    Mirrors the rollout-context wiring: set config.rtc_config, then
    init_rtc_processor (lerobot/rollout/context.py).
    """
    try:
        policy.config.rtc_config = RTCConfig(**rtc_overrides)
        policy.init_rtc_processor()
        yield
    finally:
        policy.config.rtc_config = None
        policy.init_rtc_processor()


def _plain_chunk(policy, batch, seed: int) -> torch.Tensor:
    """Populate synchronous queues from the same distinct temporal observations."""
    policy.reset()
    for index in range(policy.config.n_obs_steps):
        frame = dict(batch)
        frame[OBS_STATE] = batch[OBS_STATE][:, index]
        frame[CAM_KEY] = batch[CAM_KEY][:, index]
        prepared = policy._prepare_batch(frame)
        populate_queues(policy._queues, prepared)
    torch.manual_seed(seed)
    return policy.predict_action_chunk(dict(prepared))


# --- Engine-compatibility gate -------------------------------------------------


def test_supports_rtc(diffusion_policy):
    assert diffusion_policy.supports_rtc() is True


def test_signature_bindable_like_engine_gate(diffusion_policy):
    # Exactly what supports_rtc_inference() checks in lerobot/rollout/inference/rtc.py.
    inspect.signature(diffusion_policy.predict_action_chunk).bind(
        object(),
        inference_delay=0,
        prev_chunk_left_over=None,
    )


def test_real_engine_gate_returns_true(diffusion_policy):
    rtc_engine = pytest.importorskip("lerobot.rollout.inference.rtc")
    assert rtc_engine.supports_rtc_inference(diffusion_policy) is True


def test_supports_rtc_false_for_flow_rk4():
    """RTC guidance is not implemented for flow matching with rk4 integration;
    supports_rtc() must say so up front so the rollout engine rejects the
    combination at startup instead of raising mid-episode on the first guided
    chunk."""
    torch.manual_seed(42)
    policy = MultiTaskDiTPolicy(
        _make_config(
            objective="flow_matching",
            num_integration_steps=10,
            integration_method="rk4",
        )
    )
    assert policy.supports_rtc() is False

    rtc_engine = pytest.importorskip("lerobot.rollout.inference.rtc")
    assert rtc_engine.supports_rtc_inference(policy) is False
    with _rtc(policy), pytest.raises(ValueError, match="integration_method='euler'"):
        policy.predict_action_chunk(_make_engine_batch())


def test_init_rtc_processor_wiring(diffusion_policy):
    diffusion_policy.config.rtc_config = RTCConfig(execution_horizon=6)
    try:
        diffusion_policy.init_rtc_processor()
        assert diffusion_policy.rtc_processor is not None
        assert diffusion_policy._rtc_enabled() is True
        assert diffusion_policy.rtc_processor.rtc_config.execution_horizon == 6
    finally:
        diffusion_policy.config.rtc_config = None
        diffusion_policy.init_rtc_processor()


# --- Behavior preservation -----------------------------------------------------


def test_rtc_no_prefix_matches_plain(diffusion_policy):
    """RTC-enabled call with prev_chunk_left_over=None must sample identically
    to the plain queue-stacked path."""
    batch = _make_engine_batch()
    expected = _plain_chunk(diffusion_policy, batch, seed=321)
    expected_rng = torch.get_rng_state()

    with _rtc(diffusion_policy):
        torch.manual_seed(321)
        actual = diffusion_policy.predict_action_chunk(
            dict(batch), inference_delay=0, prev_chunk_left_over=None
        )
    assert actual.shape == (1, CHUNK_LEN, ACTION_DIM)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_rtc_no_prefix_matches_plain_flow(flow_policy):
    batch = _make_engine_batch()
    expected = _plain_chunk(flow_policy, batch, seed=321)
    expected_rng = torch.get_rng_state()

    with _rtc(flow_policy):
        torch.manual_seed(321)
        actual = flow_policy.predict_action_chunk(dict(batch), inference_delay=0, prev_chunk_left_over=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


# --- Guidance actually guides --------------------------------------------------


def test_rtc_prefix_pins_first_delay_actions(diffusion_policy):
    """DDIM inpainting: the first inference_delay actions (weights == 1) are
    pinned to the previous chunk exactly by the final clean-level blend."""
    delay = 3
    prev = torch.linspace(-0.5, 0.5, CHUNK_LEN * ACTION_DIM).reshape(CHUNK_LEN, ACTION_DIM)

    with _rtc(diffusion_policy):
        torch.manual_seed(9)
        out = diffusion_policy.predict_action_chunk(
            _make_engine_batch(), inference_delay=delay, prev_chunk_left_over=prev
        )

    assert out.shape == (1, CHUNK_LEN, ACTION_DIM)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, :delay], prev[:delay], rtol=0, atol=1e-5)
    # The tail must NOT be a copy of the previous chunk (weights decay to 0).
    assert not torch.allclose(out[0, delay:], prev[delay:], atol=1e-3)


def test_rtc_prefix_reduces_boundary_mismatch_flow(flow_policy):
    """Flow guidance is soft (no exact pinning); guided samples must land much
    closer to the prefix than unguided ones."""
    delay = 2
    prev = torch.full((CHUNK_LEN, ACTION_DIM), 0.3)
    batch = _make_engine_batch()

    with _rtc(flow_policy):
        torch.manual_seed(11)
        unguided = flow_policy.predict_action_chunk(dict(batch), inference_delay=0, prev_chunk_left_over=None)
        torch.manual_seed(11)
        guided = flow_policy.predict_action_chunk(
            dict(batch), inference_delay=delay, prev_chunk_left_over=prev
        )

    err_guided = (guided[0, :delay] - prev[:delay]).abs().mean()
    err_unguided = (unguided[0, :delay] - prev[:delay]).abs().mean()
    assert torch.isfinite(guided).all()
    assert err_guided < err_unguided


def test_rtc_no_prefix_matches_plain_ddpm(ddpm_policy):
    batch = _make_engine_batch()
    expected = _plain_chunk(ddpm_policy, batch, seed=321)
    expected_rng = torch.get_rng_state()

    with _rtc(ddpm_policy):
        torch.manual_seed(321)
        actual = ddpm_policy.predict_action_chunk(dict(batch), inference_delay=0, prev_chunk_left_over=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_rtc_prefix_pins_first_delay_actions_ddpm(ddpm_policy):
    """The inpainting path is scheduler-generic (add_noise/step); verify the
    DDPM branch pins and blends like DDIM."""
    delay = 3
    prev = torch.linspace(-0.5, 0.5, CHUNK_LEN * ACTION_DIM).reshape(CHUNK_LEN, ACTION_DIM)

    with _rtc(ddpm_policy):
        torch.manual_seed(9)
        out = ddpm_policy.predict_action_chunk(
            _make_engine_batch(), inference_delay=delay, prev_chunk_left_over=prev
        )

    assert out.shape == (1, CHUNK_LEN, ACTION_DIM)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, :delay], prev[:delay], rtol=0, atol=1e-5)
    assert not torch.allclose(out[0, delay:], prev[delay:], atol=1e-3)


@pytest.mark.parametrize("policy_name", ["diffusion_policy", "ddpm_policy", "flow_policy"])
def test_disabled_rtc_consumes_engine_history_without_queues_or_prefix_guidance(request, policy_name):
    policy = request.getfixturevalue(policy_name)
    batch = _make_engine_batch()
    prev = torch.full((CHUNK_LEN, ACTION_DIM), 0.4)
    expected = _plain_chunk(policy, batch, seed=77)
    expected_rng = torch.get_rng_state()
    policy.reset()
    with _rtc(policy, enabled=False):
        torch.manual_seed(77)
        actual = policy.predict_action_chunk(batch, inference_delay=3, prev_chunk_left_over=prev)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not policy._queues[OBS_STATE]


def test_select_action_asserts_when_rtc_enabled(diffusion_policy):
    """select_action is not RTC-aware; it must refuse to run with RTC enabled
    (pi0/molmoact2 behavior)."""
    with _rtc(diffusion_policy), pytest.raises(AssertionError, match="RTC is not supported"):
        diffusion_policy.select_action(_make_engine_batch())


def test_execution_horizon_kwarg_accepted(diffusion_policy):
    prev = torch.zeros(CHUNK_LEN, ACTION_DIM)
    with _rtc(diffusion_policy):
        out = diffusion_policy.predict_action_chunk(
            _make_engine_batch(),
            inference_delay=2,
            prev_chunk_left_over=prev,
            execution_horizon=4,
        )
    assert out.shape == (1, CHUNK_LEN, ACTION_DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("missing_key", [OBS_STATE, CAM_KEY])
def test_rtc_rejects_single_frame_instead_of_inventing_history(diffusion_policy, missing_key):
    batch = _make_engine_batch()
    batch[missing_key] = batch[missing_key][:, -1]
    with _rtc(diffusion_policy), pytest.raises(ValueError, match="temporal history"):
        diffusion_policy.predict_action_chunk(batch, inference_delay=0, prev_chunk_left_over=None)


def test_disabled_rtc_single_frame_with_rtc_kwargs_still_uses_the_queues(diffusion_policy):
    """The dispatch keys on the batch's temporal axis, not on whether a caller
    happened to forward the RTC keyword arguments."""
    batch = _make_engine_batch()
    expected = _plain_chunk(diffusion_policy, batch, seed=5)
    frame = dict(batch)
    for key in (OBS_STATE, CAM_KEY):
        frame[key] = batch[key][:, -1]
    prepared = diffusion_policy._prepare_batch(frame)
    with _rtc(diffusion_policy, enabled=False):
        torch.manual_seed(5)
        actual = diffusion_policy.predict_action_chunk(prepared, inference_delay=0, prev_chunk_left_over=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rtc_preserves_distinct_history_and_does_not_advance_queues(diffusion_policy):
    batch = _make_engine_batch()
    before = {key: value.clone() for key, value in batch.items()}
    expected = _plain_chunk(diffusion_policy, batch, seed=17)
    diffusion_policy.reset()
    with _rtc(diffusion_policy):
        torch.manual_seed(17)
        actual = diffusion_policy.predict_action_chunk(batch)
        torch.manual_seed(17)
        repeated = diffusion_policy.predict_action_chunk(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    for key in batch:
        torch.testing.assert_close(batch[key], before[key])
    assert not diffusion_policy._queues[OBS_STATE]


def test_sigma_min_guidance_warning_is_logged_once_per_wiring(caplog):
    torch.manual_seed(42)
    policy = MultiTaskDiTPolicy(
        _make_config(
            objective="flow_matching",
            num_integration_steps=4,
            integration_method="euler",
            sigma_min=0.01,
        )
    ).eval()
    prev = torch.full((CHUNK_LEN, ACTION_DIM), 0.3)
    logger_name = "lerobot.policies.multi_task_dit.modeling_multi_task_dit"
    with caplog.at_level(logging.WARNING, logger=logger_name), _rtc(policy):
        for _ in range(3):
            policy.predict_action_chunk(_make_engine_batch(), inference_delay=1, prev_chunk_left_over=prev)
    assert sum("sigma_min" in record.message for record in caplog.records) == 1


def test_rtc_rejects_trained_mode_without_a_trained_checkpoint(diffusion_policy):
    with pytest.raises(ValueError, match="trained"), _rtc(diffusion_policy, mode="trained"):
        pass


WINDOW = slice(N_OBS_STEPS - 1, N_OBS_STEPS - 1 + CHUNK_LEN)


def _reference_flow_chunk(policy, batch, previous, delay, execution_horizon, seed):
    """Integrate MTDiT's euler sampler by hand, routing the action window through
    the unmodified pi0-convention ``RTCProcessor.denoise_step``.

    pi0 integrates data -> noise with ``time`` running 1 -> 0, so the reference
    receives ``1 - t`` and a negated velocity; the guided velocity is negated back.
    """
    conditioning = policy.observation_encoder.encode(policy._prepare_batch(batch))
    reference = RTCProcessor(policy.config.rtc_config)
    torch.manual_seed(seed)
    x = torch.randn(1, HORIZON, ACTION_DIM)
    grid = torch.linspace(0, 1, policy.config.num_integration_steps + 1)
    for t, next_t in zip(grid[:-1], grid[1:], strict=True):
        time = t.item()
        t_batch = torch.full((1,), time)

        def base(chunk, latent=x, t_batch=t_batch):
            full = latent.clone()
            full[:, WINDOW] = chunk
            return -policy.noise_predictor(full, t_batch, conditioning_vec=conditioning)[:, WINDOW]

        with torch.no_grad():
            velocity = policy.noise_predictor(x, t_batch, conditioning_vec=conditioning).clone()
            guided = reference.denoise_step(
                x[:, WINDOW],
                previous[:CHUNK_LEN].unsqueeze(0),
                delay,
                1 - time,
                base,
                execution_horizon=execution_horizon,
            )
            velocity[:, WINDOW] = -guided
            x = x + (next_t - t).item() * velocity
    return x[:, WINDOW]


@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("prefix_len", [3, CHUNK_LEN, 10])
@pytest.mark.parametrize("strength", [0.5, 4.0])
def test_flow_matches_the_shared_processor_on_the_action_window_with_one_model_call_per_step(
    flow_policy, schedule, prefix_len, strength
):
    batch = _make_engine_batch()
    previous = torch.linspace(-0.4, 0.5, prefix_len * ACTION_DIM).reshape(prefix_len, ACTION_DIM)
    delay = 1
    execution_horizon = min(prefix_len, 6)
    with _rtc(
        flow_policy,
        prefix_attention_schedule=schedule,
        max_guidance_weight=strength,
        execution_horizon=6,
        debug=True,
    ):
        expected = _reference_flow_chunk(flow_policy, batch, previous, delay, execution_horizon, seed=28)
        calls = []
        handle = flow_policy.noise_predictor.register_forward_hook(lambda *args: calls.append(1))
        try:
            torch.manual_seed(28)
            actual = flow_policy.predict_action_chunk(
                batch,
                inference_delay=delay,
                prev_chunk_left_over=previous,
                execution_horizon=execution_horizon,
            )
        finally:
            handle.remove()
        processor = flow_policy.rtc_processor
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    assert len(calls) == flow_policy.config.num_integration_steps
    steps = processor.get_all_debug_steps()
    assert len(steps) == flow_policy.config.num_integration_steps
    reference_weights = processor.get_prefix_weights(delay, execution_horizon, CHUNK_LEN)
    for step in steps:
        assert step.weights.shape == (1, CHUNK_LEN, 1)
        torch.testing.assert_close(step.weights.flatten(), reference_weights, rtol=0, atol=1e-7)
        assert step.inference_delay == delay
        assert step.execution_horizon == execution_horizon


class _LinearOracle(nn.Module):
    """Velocity field whose x0 estimate is exactly ``target`` from any latent.

    Along the flow-matching path x_t = (1 - t) noise + t data the true velocity
    is (data - x_t) / (1 - t); with it, ``x_t + (1 - t) v`` recovers ``data`` and
    the euler integrator is exact, so guidance effects have a closed form.
    """

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.target = target

    def forward(self, x, t, conditioning_vec=None):
        return (self.target - x) / (1 - t.view(-1, 1, 1))


@pytest.mark.parametrize("schedule", [RTCAttentionSchedule.ONES, RTCAttentionSchedule.LINEAR])
@pytest.mark.parametrize("strength", [0.1, 0.25])
def test_flow_guidance_has_the_closed_form_shift_under_a_linear_oracle(
    flow_policy, monkeypatch, schedule, strength
):
    """Independent of any implementation convention.

    The oracle's velocity restores the straight path toward ``target`` at every
    step, so a shift injected before the last euler step is multiplied by
    ``1 - dt / (1 - t)``, which is zero at the last step ``t = 1 - dt``. Only the
    final step's guidance kick survives; with a cap that binds there the guided
    window is exactly ``target + dt * strength * weights * (prev - target)``.

    A wrong velocity sign flips the shift, a wrong ``time`` argument corrupts the
    x0 estimate that the error term is built from, and a wrong window offset moves
    the shift to the wrong actions.
    """
    target = torch.linspace(-0.8, 0.8, HORIZON * ACTION_DIM).reshape(1, HORIZON, ACTION_DIM)
    monkeypatch.setattr(flow_policy, "noise_predictor", _LinearOracle(target))
    previous = torch.linspace(0.5, -0.5, CHUNK_LEN * ACTION_DIM).reshape(CHUNK_LEN, ACTION_DIM)
    delay = 2
    dt = 1 / flow_policy.config.num_integration_steps
    batch = _make_engine_batch()
    with _rtc(
        flow_policy,
        prefix_attention_schedule=schedule,
        max_guidance_weight=strength,
        execution_horizon=CHUNK_LEN,
    ):
        unguided = flow_policy.predict_action_chunk(batch)
        guided = flow_policy.predict_action_chunk(batch, inference_delay=delay, prev_chunk_left_over=previous)
        weights = flow_policy.rtc_processor.get_prefix_weights(delay, CHUNK_LEN, CHUNK_LEN).view(1, -1, 1)
    torch.testing.assert_close(unguided, target[:, WINDOW], rtol=0, atol=1e-5)
    expected = target[:, WINDOW] + dt * strength * weights * (previous.unsqueeze(0) - target[:, WINDOW])
    torch.testing.assert_close(guided, expected, rtol=0, atol=1e-5)
    assert not torch.allclose(guided, unguided, atol=1e-3)


@pytest.mark.parametrize("policy_name", ["diffusion_policy", "ddpm_policy", "flow_policy"])
def test_padding_beyond_the_true_leftover_length_has_no_effect(request, policy_name):
    """The engine zero-pads the prefix to ``execution_horizon`` for stable compiled
    inference and sends the true length separately; the padding must never be
    treated as committed actions."""
    policy = request.getfixturevalue(policy_name)
    valid = 3
    real = torch.linspace(-0.5, 0.5, valid * ACTION_DIM).reshape(valid, ACTION_DIM)
    padded = torch.zeros(10, ACTION_DIM)
    padded[:valid] = real
    batch = _make_engine_batch()
    with _rtc(policy):
        torch.manual_seed(3)
        from_real = policy.predict_action_chunk(
            batch, inference_delay=1, prev_chunk_left_over=real, execution_horizon=valid
        )
        torch.manual_seed(3)
        from_padded = policy.predict_action_chunk(
            batch, inference_delay=1, prev_chunk_left_over=padded, execution_horizon=valid
        )
    torch.testing.assert_close(from_real, from_padded, rtol=0, atol=0)
    assert not torch.allclose(from_padded[0, valid:], torch.zeros(CHUNK_LEN - valid, ACTION_DIM), atol=1e-2)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("missing_key", [OBS_STATE, CAM_KEY])
def test_rtc_rejects_incomplete_temporal_axis(diffusion_policy, missing_key, enabled):
    batch = _make_engine_batch()
    batch[missing_key] = batch[missing_key][:, :1]
    with _rtc(diffusion_policy, enabled=enabled), pytest.raises(ValueError, match="temporal history"):
        diffusion_policy.predict_action_chunk(batch, inference_delay=0, prev_chunk_left_over=None)


def test_single_observation_policy_accepts_a_frame_or_singleton_history():
    policy = MultiTaskDiTPolicy(_make_config(n_obs_steps=1)).eval()
    temporal = _make_engine_batch()
    frame = dict(temporal)
    for key in (OBS_STATE, CAM_KEY):
        frame[key] = temporal[key][:, -1]
        temporal[key] = frame[key].unsqueeze(1)
    expected = _plain_chunk(policy, temporal, seed=41)
    with _rtc(policy):
        torch.manual_seed(41)
        actual = policy.predict_action_chunk(frame)
        torch.manual_seed(41)
        stacked = policy.predict_action_chunk(temporal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, stacked, rtol=0, atol=0)


@pytest.mark.parametrize("policy_name", ["diffusion_policy", "ddpm_policy"])
@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("prefix_len", [1, 3])
def test_diffusion_short_prefix_pins_only_available_delay_actions(request, policy_name, schedule, prefix_len):
    policy = request.getfixturevalue(policy_name)
    prev = torch.full((prefix_len, ACTION_DIM), 0.25)
    with _rtc(policy, prefix_attention_schedule=schedule, execution_horizon=CHUNK_LEN):
        torch.manual_seed(45)
        actual = policy.predict_action_chunk(
            _make_engine_batch(), inference_delay=4, prev_chunk_left_over=prev
        )
    torch.testing.assert_close(actual[0, :prefix_len], prev, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert not torch.allclose(actual[0, prefix_len:], torch.full_like(actual[0, prefix_len:], 0.25))


@pytest.mark.parametrize("policy_name", ["diffusion_policy", "ddpm_policy", "flow_policy"])
def test_empty_leftover_preserves_no_prefix_result_and_rng(request, policy_name):
    policy = request.getfixturevalue(policy_name)
    batch = _make_engine_batch()
    with _rtc(policy):
        torch.manual_seed(9)
        expected = policy.predict_action_chunk(batch)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(9)
        actual = policy.predict_action_chunk(batch, prev_chunk_left_over=torch.empty(0, ACTION_DIM))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("policy_name", ["diffusion_policy", "ddpm_policy", "flow_policy"])
@pytest.mark.parametrize("history_key", [OBS_STATE, CAM_KEY])
def test_changing_only_an_older_observation_changes_conditioned_actions(request, policy_name, history_key):
    policy = request.getfixturevalue(policy_name)
    batch = _make_engine_batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed[history_key][:, 0] += 0.5
    with _rtc(policy):
        torch.manual_seed(13)
        original = policy.predict_action_chunk(batch)
        torch.manual_seed(13)
        actual = policy.predict_action_chunk(changed)
    assert not torch.allclose(actual, original, rtol=0, atol=1e-6)


def test_flow_fixture_denoiser_responds_to_time(flow_policy):
    batch = _make_engine_batch()
    conditioning = flow_policy.observation_encoder.encode(flow_policy._prepare_batch(batch))
    latent = torch.linspace(-1, 1, HORIZON * ACTION_DIM).reshape(1, HORIZON, ACTION_DIM)
    at_noise = flow_policy.noise_predictor(latent, torch.tensor([0.0]), conditioning)
    at_data = flow_policy.noise_predictor(latent, torch.tensor([1.0]), conditioning)
    assert not torch.allclose(at_noise, at_data, rtol=0, atol=1e-6)


@pytest.mark.parametrize("policy_name", ["diffusion_policy", "ddpm_policy", "flow_policy"])
def test_disabled_rtc_keeps_synchronous_select_action_queue_behavior(request, policy_name):
    policy = request.getfixturevalue(policy_name)
    batch = _make_engine_batch()
    frame = dict(batch)
    repeated = dict(batch)
    for key in (OBS_STATE, CAM_KEY):
        frame[key] = batch[key][:, -1]
        repeated[key] = frame[key].unsqueeze(1).expand_as(batch[key])
    expected = _plain_chunk(policy, repeated, seed=28)
    policy.reset()
    with _rtc(policy, enabled=False):
        torch.manual_seed(28)
        first = policy.select_action(dict(frame))
        next_frame = dict(frame)
        next_frame[OBS_STATE] = frame[OBS_STATE] + 1
        second = policy.select_action(next_frame)
    torch.testing.assert_close(policy._queues[OBS_STATE][-1], next_frame[OBS_STATE])
    torch.testing.assert_close(first, expected[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(second, expected[:, 1], rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [True, False])
def test_actual_rtc_engine_consumes_two_history_ticks_with_guidance_enabled_or_disabled(
    diffusion_policy, enabled
):
    rtc_engine = pytest.importorskip("lerobot.rollout.inference.rtc")

    tokens = _make_engine_batch()

    class TokenPreprocessor:
        steps = ()

        def __init__(self):
            self.states = []

        def __call__(self, batch):
            self.states.append(float(batch[OBS_STATE][0, 0]))
            batch[OBS_LANGUAGE_TOKENS] = tokens[OBS_LANGUAGE_TOKENS]
            batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens[OBS_LANGUAGE_ATTENTION_MASK]
            if len(self.states) == 2:
                engine._shutdown_event.set()
            return batch

    preprocessor = TokenPreprocessor()
    names = [f"joint{index}.pos" for index in range(STATE_DIM)]
    hw_features = {
        OBS_STATE: {"dtype": "float32", "shape": (STATE_DIM,), "names": names},
        CAM_KEY: {"dtype": "image", "shape": (16, 16, 3), "names": ["height", "width", "channels"]},
    }
    diffusion_policy.reset()
    with _rtc(diffusion_policy, enabled=enabled):
        engine = rtc_engine.RTCInferenceEngine(
            diffusion_policy,
            preprocessor,
            lambda actions: actions,
            SimpleNamespace(robot_type="test", action_features=dict.fromkeys(names, float)),
            diffusion_policy.config.rtc_config,
            hw_features,
            "test task",
            1,
            "cpu",
        )
        engine._action_queue = ActionQueue(diffusion_policy.config.rtc_config)
        engine._policy_active.set()
        for value in (1, 2):
            engine.notify_observation(
                {**dict.fromkeys(names, value), "cam": np.full((16, 16, 3), value, dtype=np.uint8)}
            )
        engine._rtc_loop()
    assert preprocessor.states == [1, 2]
    assert not engine.failed, engine.failure_traceback
    assert engine._action_queue.qsize() > 0
    assert not diffusion_policy._queues[OBS_STATE]
