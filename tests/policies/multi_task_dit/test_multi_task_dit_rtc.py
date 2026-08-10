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
    python -m pytest tests/policies/multi_task_dit/test_multi_task_dit_rtc.py -v
"""

import inspect
import os
from contextlib import contextmanager

import pytest
import torch

pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local transformers installation and is not meant for CI",
)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
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
        "image_resize_shape": (224, 224),
        "image_crop_shape": (224, 224),
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            CAM_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        "output_features": {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        "device": "cpu",
    }
    kwargs.update(overrides)
    return MultiTaskDiTConfig(**kwargs)


@pytest.fixture(scope="module")
def diffusion_policy() -> MultiTaskDiTPolicy:
    torch.manual_seed(42)
    policy = MultiTaskDiTPolicy(_make_config())
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
    policy.eval()
    return policy


@pytest.fixture(scope="module")
def ddpm_policy() -> MultiTaskDiTPolicy:
    torch.manual_seed(42)
    policy = MultiTaskDiTPolicy(_make_config(noise_scheduler_type="DDPM"))
    policy.eval()
    return policy


def _make_engine_batch() -> dict[str, torch.Tensor]:
    """A single preprocessed frame, shaped as the RTC engine hands it over."""
    g = torch.Generator().manual_seed(7)
    return {
        OBS_STATE: torch.randn(1, STATE_DIM, generator=g),
        CAM_KEY: torch.rand(1, 3, 96, 96, generator=g),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (1, 77), generator=g),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 77, dtype=torch.long),
    }


@contextmanager
def _rtc(policy, **rtc_overrides):
    """Enable RTC on a (module-scoped) policy fixture, restoring on exit.

    Mirrors the rollout-context wiring: set config.rtc_config, then
    init_rtc_processor (lerobot/rollout/context.py).
    """
    policy.config.rtc_config = RTCConfig(**rtc_overrides)
    policy.init_rtc_processor()
    try:
        yield
    finally:
        policy.config.rtc_config = None
        policy.init_rtc_processor()


def _plain_chunk(policy, batch, seed: int) -> torch.Tensor:
    """The plain (non-RTC) predict_action_chunk path, with the queue population
    select_action would normally have done (first tick: queue filled with
    copies of the current frame, matching the RTC path's duplication)."""
    prepared = policy._prepare_batch(dict(batch))
    policy.reset()
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

    with _rtc(diffusion_policy):
        torch.manual_seed(321)
        actual = diffusion_policy.predict_action_chunk(
            dict(batch), inference_delay=0, prev_chunk_left_over=None
        )
    assert actual.shape == (1, CHUNK_LEN, ACTION_DIM)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_rtc_no_prefix_matches_plain_flow(flow_policy):
    batch = _make_engine_batch()
    expected = _plain_chunk(flow_policy, batch, seed=321)

    with _rtc(flow_policy):
        torch.manual_seed(321)
        actual = flow_policy.predict_action_chunk(dict(batch), inference_delay=0, prev_chunk_left_over=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


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

    with _rtc(ddpm_policy):
        torch.manual_seed(321)
        actual = ddpm_policy.predict_action_chunk(dict(batch), inference_delay=0, prev_chunk_left_over=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


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


def test_disabled_config_keeps_plain_path(diffusion_policy):
    """rtc_config.enabled=False must dispatch to the plain queue-stacked path,
    ignoring any RTC kwargs (mirrors pi0/molmoact2's _rtc_enabled gate)."""
    batch = _make_engine_batch()
    prev = torch.full((CHUNK_LEN, ACTION_DIM), 0.4)
    expected = _plain_chunk(diffusion_policy, batch, seed=77)

    diffusion_policy.config.rtc_config = RTCConfig(enabled=False)
    try:
        diffusion_policy.init_rtc_processor()
        assert diffusion_policy._rtc_enabled() is False
        prepared = diffusion_policy._prepare_batch(dict(batch))
        diffusion_policy.reset()
        populate_queues(diffusion_policy._queues, prepared)
        torch.manual_seed(77)
        disabled = diffusion_policy.predict_action_chunk(
            dict(prepared), inference_delay=3, prev_chunk_left_over=prev
        )
    finally:
        diffusion_policy.config.rtc_config = None
        diffusion_policy.init_rtc_processor()

    torch.testing.assert_close(disabled, expected, rtol=0, atol=1e-6)


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
