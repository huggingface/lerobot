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

"""Behavior-pinning tests for the shared flow-matching sampling primitives.

``euler_integrate`` is compared against a verbatim copy of the historical pi0/pi05/
smolvla sampling loop (including its RTC hook semantics): any divergence from that
reference is a behavior change for released checkpoints.

The sampler tests do the same per policy: each ``_historical_*`` helper below is a copy of
the expression a policy used before it adopted the shared primitives, and every recipe is
asserted bit-identical (``torch.equal`` / ``atol=0``) to it on the same RNG stream.
"""

import pytest
import torch

from lerobot.policies.common.flow_matching import (
    _beta_distribution,
    device_beta_sampler,
    euler_integrate,
    sample_beta,
    sample_noise,
    sample_time_beta,
)


def test_sample_beta_range_dtype_and_reproducibility():
    torch.manual_seed(0)
    s1 = sample_beta(1.5, 1.0, 4096, "cpu")
    torch.manual_seed(0)
    s2 = sample_beta(1.5, 1.0, 4096, "cpu")
    assert torch.equal(s1, s2)
    assert s1.shape == (4096,) and s1.dtype == torch.float32
    assert s1.min() >= 0.0 and s1.max() <= 1.0
    # Beta(1.5, 1.0) mean is 1.5/2.5 = 0.6.
    assert abs(s1.mean().item() - 0.6) < 0.02


def test_sample_time_beta_openpi_convention():
    torch.manual_seed(1)
    time = sample_time_beta(4096, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    assert time.dtype == torch.float32
    assert time.min() >= 0.001 and time.max() <= 1.0
    # Exact composition: Beta sample * scale + offset, same RNG stream.
    torch.manual_seed(1)
    expected = sample_beta(1.5, 1.0, 4096, "cpu") * 0.999 + 0.001
    torch.testing.assert_close(time, expected, rtol=0, atol=0)


def test_sample_time_beta_defaults_to_the_identity_transform():
    # The .999/.001 endpoint offsets are the pi-family recipe, not a helper default.
    torch.manual_seed(1)
    time = sample_time_beta(512, "cpu", alpha=1.5, beta=1.0)
    torch.manual_seed(1)
    expected = sample_beta(1.5, 1.0, 512, "cpu")
    assert torch.equal(time, expected)

    torch.manual_seed(1)
    explicit_identity = sample_time_beta(512, "cpu", alpha=1.5, beta=1.0, scale=1.0, offset=0.0)
    assert torch.equal(explicit_identity, expected)


def test_sample_noise_seeded():
    torch.manual_seed(2)
    n1 = sample_noise((2, 8, 4), "cpu")
    torch.manual_seed(2)
    n2 = sample_noise((2, 8, 4), "cpu")
    assert torch.equal(n1, n2)
    assert n1.dtype == torch.float32 and n1.shape == (2, 8, 4)


# --- Cached Beta concentrations must stay on CPU ------------------------------------------


def test_beta_concentrations_are_cpu_under_a_non_cpu_default_device():
    # Regression: the cache means one construction under an ambient default device would be
    # reused forever, and Beta's _sample_dirichlet has no meta (or MPS) kernel.
    alpha, beta = 1.7, 1.3
    _beta_distribution.cache_clear()
    try:
        with torch.device("meta"):
            dist = _beta_distribution(alpha, beta)
        assert dist.concentration1.device.type == "cpu"
        assert dist.concentration0.device.type == "cpu"

        with torch.device("meta"):
            sample = sample_beta(alpha, beta, 16, "cpu")
        assert sample.device.type == "cpu"
        assert sample.shape == (16,) and sample.dtype == torch.float32
        assert sample.min() >= 0.0 and sample.max() <= 1.0
    finally:
        _beta_distribution.cache_clear()


def test_unpinned_concentrations_would_have_broken_under_meta():
    # Pins *why* the explicit device="cpu" above is required rather than incidental.
    with torch.device("meta"):
        unpinned = torch.distributions.Beta(torch.tensor(1.7), torch.tensor(1.3), validate_args=False)
    assert unpinned.concentration1.device.type == "meta"
    with pytest.raises(NotImplementedError):
        unpinned.sample((4,))


def test_beta_distribution_is_cached_per_concentration_pair():
    _beta_distribution.cache_clear()
    try:
        assert _beta_distribution(1.5, 1.0) is _beta_distribution(1.5, 1.0)
        assert _beta_distribution(1.5, 1.0) is not _beta_distribution(2.0, 2.0)
    finally:
        _beta_distribution.cache_clear()


# --- sample_noise dtype and distribution ---------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_sample_noise_normal_honors_dtype_and_matches_randn(dtype):
    torch.manual_seed(6)
    noise = sample_noise((3, 5, 4), "cpu", dtype=dtype)
    # Historical groot/wall_x expression: torch.randn(shape, device=..., dtype=...).
    torch.manual_seed(6)
    expected = torch.randn((3, 5, 4), device="cpu", dtype=dtype)
    assert noise.dtype == dtype
    assert torch.equal(noise, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_sample_noise_uniform_matches_evo1_expression_and_spans_pm_one(dtype):
    reference_actions = torch.zeros(4, 7, 3, dtype=dtype)
    torch.manual_seed(7)
    noise = sample_noise(
        reference_actions.shape, reference_actions.device, dtype=dtype, distribution="uniform"
    )
    # Historical evo1 expression: torch.rand_like(actions_gt) * 2 - 1.
    torch.manual_seed(7)
    expected = torch.rand_like(reference_actions) * 2 - 1
    assert noise.dtype == dtype
    assert torch.equal(noise, expected)
    assert noise.min() >= -1.0 and noise.max() < 1.0


def test_sample_noise_defaults_to_float32_normal():
    torch.manual_seed(8)
    default = sample_noise((2, 3), "cpu")
    torch.manual_seed(8)
    explicit = sample_noise((2, 3), "cpu", dtype=torch.float32, distribution="normal")
    assert default.dtype == torch.float32
    assert torch.equal(default, explicit)


def test_sample_noise_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="Unknown noise distribution"):
        sample_noise((2, 3), "cpu", distribution="beta")


# --- Per-policy recipes, pinned against their historical expressions -----------------------


def _historical_pi_family_sample_time(bsize, device, alpha, beta, scale, offset):
    """pi0 / pi05 / smolvla / eo1, pre-adoption."""
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    time_beta = dist.sample((bsize,)).to(device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


def _historical_groot_sample_time(bsize, device, dtype, alpha, beta, noise_s):
    """groot_n1_7 GR00T N1.7 action head, pre-adoption."""
    beta_alpha = torch.tensor(alpha, device="cpu", dtype=torch.float32)
    beta_beta = torch.tensor(beta, device="cpu", dtype=torch.float32)
    dist = torch.distributions.Beta(beta_alpha, beta_beta, validate_args=False)
    sample = dist.sample([bsize]).to(device, dtype=dtype)
    return (1 - sample) * noise_s


def _historical_evo1_sample_time(bsize, device, dtype):
    """evo1 flow-matching head, pre-adoption."""
    return torch.distributions.Beta(2, 2).sample((bsize,)).clamp(0.02, 0.98).to(device).to(dtype=dtype)


def _historical_wall_x_sample_time(bsize, device, alpha, beta, s):
    """wall_x action-embedding head, pre-adoption (concentrations and draw on `device`)."""
    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32, device=device),
        torch.tensor(beta, dtype=torch.float32, device=device),
    )
    sample = beta_dist.sample([bsize])
    return (1 - sample) * s


def test_pi_family_recipe_matches_historical_endpoint_offsets():
    torch.manual_seed(10)
    time = sample_time_beta(2048, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    torch.manual_seed(10)
    expected = _historical_pi_family_sample_time(2048, "cpu", 1.5, 1.0, 0.999, 0.001)
    torch.testing.assert_close(time, expected, rtol=0, atol=0)
    assert time.dtype == torch.float32
    # Endpoint offsets keep t strictly inside (0, 1].
    assert time.min() >= 0.001 and time.max() <= 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_groot_recipe_casts_before_complement_and_scale(dtype):
    alpha, beta, noise_s, bsize = 1.5, 1.0, 0.999, 1024

    torch.manual_seed(11)
    sample = sample_beta(alpha, beta, bsize, "cpu", dtype=dtype)
    time = (1 - sample) * noise_s

    torch.manual_seed(11)
    expected = _historical_groot_sample_time(bsize, "cpu", dtype, alpha, beta, noise_s)

    assert time.dtype == dtype
    assert torch.equal(time, expected)
    # GR00T's buckets are read off the timestep with no clamp (num_timestep_buckets=1000).
    buckets = (time * 1000).long()
    assert torch.equal(buckets, (expected * 1000).long())
    assert buckets.min() >= 0 and buckets.max() <= 1000


def test_groot_output_cast_in_the_helper_would_not_reproduce_the_recipe():
    # Pins the reason the cast stays before the transform: folding it to the end of a
    # generalized helper double-rounds and gives different bf16 timesteps.
    alpha, beta, noise_s, bsize = 1.5, 1.0, 0.999, 1024

    torch.manual_seed(12)
    recipe = (1 - sample_beta(alpha, beta, bsize, "cpu", dtype=torch.bfloat16)) * noise_s

    torch.manual_seed(12)
    cast_at_the_end = ((1 - sample_beta(alpha, beta, bsize, "cpu")) * noise_s).to(torch.bfloat16)

    assert not torch.equal(recipe, cast_at_the_end)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_evo1_recipe_clamps_in_float32_before_converting_dtype(dtype):
    bsize = 4096

    torch.manual_seed(13)
    t = sample_beta(2.0, 2.0, bsize, "cpu").clamp(0.02, 0.98).to(dtype=dtype)

    torch.manual_seed(13)
    expected = _historical_evo1_sample_time(bsize, "cpu", dtype)

    assert t.dtype == dtype
    assert torch.equal(t, expected)
    assert t.min() >= torch.tensor(0.02, dtype=dtype) and t.max() <= torch.tensor(0.98, dtype=dtype)

    time_index = (t * 999).long().clamp_(0, 999)
    assert torch.equal(time_index, (expected * 999).long().clamp_(0, 999))
    assert time_index.min() >= 0 and time_index.max() <= 999


def test_evo1_clamp_is_active_at_both_endpoints():
    # Beta(2, 2) draws land outside [0.02, 0.98] often enough that the clamp is load-bearing.
    torch.manual_seed(14)
    raw = sample_beta(2.0, 2.0, 20000, "cpu")
    assert (raw < 0.02).any() and (raw > 0.98).any()
    clamped = raw.clamp(0.02, 0.98)
    assert clamped.min() == pytest.approx(0.02)
    assert clamped.max() == pytest.approx(0.98)


def test_wall_x_recipe_keeps_its_device_side_rng_stream():
    alpha, beta, s, bsize = 1.5, 1.0, 0.999, 1024
    device = torch.device("cpu")

    torch.manual_seed(15)
    sample = sample_beta(alpha, beta, bsize, device, sampler=device_beta_sampler(device))
    time = (1 - sample) * s

    torch.manual_seed(15)
    expected = _historical_wall_x_sample_time(bsize, device, alpha, beta, s)

    assert time.dtype == torch.float32
    assert torch.equal(time, expected)


def test_wall_x_recipe_never_draws_from_the_cached_cpu_distribution():
    # Device-independent proof that wall_x still samples on its own device: the shared CPU
    # distribution is never even constructed, so the CPU generator is not advanced.
    device = torch.device("cpu")
    _beta_distribution.cache_clear()
    try:
        sample = sample_beta(1.5, 1.0, 64, device, sampler=device_beta_sampler(device))
        assert sample.shape == (64,)
        assert _beta_distribution.cache_info().currsize == 0
        assert _beta_distribution.cache_info().misses == 0
    finally:
        _beta_distribution.cache_clear()


def test_device_beta_sampler_builds_concentrations_on_the_requested_device():
    captured = {}

    def probe(alpha, beta, bsize):
        sampler = device_beta_sampler("cpu")
        out = sampler(alpha, beta, bsize)
        captured["device"] = out.device
        return out

    sample = sample_beta(1.5, 1.0, 8, "cpu", sampler=probe)
    assert captured["device"].type == "cpu"
    assert sample.shape == (8,)


def test_injected_sampler_bypasses_the_cached_cpu_distribution():
    draws = torch.linspace(0.0, 1.0, 8)
    sample = sample_beta(1.5, 1.0, 8, "cpu", sampler=lambda alpha, beta, bsize: draws)
    assert torch.equal(sample, draws)
    # Same injected draws, GR00T's cast-then-transform order, in bf16.
    sample_bf16 = sample_beta(
        1.5, 1.0, 8, "cpu", dtype=torch.bfloat16, sampler=lambda alpha, beta, bsize: draws
    )
    assert sample_bf16.dtype == torch.bfloat16
    assert torch.equal(sample_bf16, draws.to(torch.bfloat16))


def test_euler_integrate_constant_velocity_is_exact():
    # With v_t == c constant, x_0 = x_1 + sum(dt * c) = x_1 - c exactly (num_steps * dt = -1).
    noise = torch.randn(3, 5, 2)
    c = torch.randn(3, 5, 2)
    out = euler_integrate(lambda x_t, time: c, noise, num_steps=10)
    torch.testing.assert_close(out, noise - c, rtol=0, atol=1e-6)


def _reference_pi0_loop(denoise_fn, noise, num_steps, rtc_enabled, rtc_processor, kw):
    """Verbatim structure of the historical pi0/pi05/smolvla sample_actions loop."""
    bsize = noise.shape[0]
    device = noise.device
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return denoise_fn(input_x_t, current_timestep)

        if rtc_enabled:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=kw.get("prev_chunk_left_over"),
                inference_delay=kw.get("inference_delay"),
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=kw.get("execution_horizon"),
            )
        else:
            v_t = denoise_step_partial_call(x_t)
        x_t = x_t + dt * v_t
        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
    return x_t


class _StubRTCProcessor:
    def __init__(self, debug_enabled: bool):
        self._debug = debug_enabled
        self.tracked = []
        self.guidance_calls = []

    def is_debug_enabled(self):
        return self._debug

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon,
    ):
        self.guidance_calls.append(
            {
                "time": time,
                "inference_delay": inference_delay,
                "execution_horizon": execution_horizon,
                "x_t": x_t.clone(),
            }
        )
        return original_denoise_step_partial(x_t) * 0.5

    def track(self, time, x_t, v_t):
        self.tracked.append({"time": time, "x_t": x_t.clone(), "v_t": v_t.clone()})


def _make_denoise_fn():
    weight = torch.randn(4, 4) * 0.1

    def denoise_fn(x_t, time_tensor):
        return x_t @ weight + time_tensor[:, None, None]

    return denoise_fn


def test_euler_integrate_matches_historical_loop():
    torch.manual_seed(3)
    denoise_fn = _make_denoise_fn()
    noise = torch.randn(2, 6, 4)
    ref = _reference_pi0_loop(denoise_fn, noise, 10, rtc_enabled=False, rtc_processor=None, kw={})
    out = euler_integrate(denoise_fn, noise, 10)
    assert torch.equal(out, ref)


def test_euler_integrate_rtc_guidance_and_kwarg_forwarding():
    torch.manual_seed(4)
    denoise_fn = _make_denoise_fn()
    noise = torch.randn(2, 6, 4)
    leftover = torch.randn(2, 6, 4)
    kw = {"inference_delay": 3, "prev_chunk_left_over": leftover, "execution_horizon": 25}

    ref_proc, new_proc = _StubRTCProcessor(False), _StubRTCProcessor(False)
    ref = _reference_pi0_loop(denoise_fn, noise, 6, rtc_enabled=True, rtc_processor=ref_proc, kw=kw)
    out = euler_integrate(
        denoise_fn,
        noise,
        6,
        rtc_processor=new_proc,
        rtc_enabled=True,
        inference_delay=3,
        prev_chunk_left_over=leftover,
        execution_horizon=25,
    )
    assert torch.equal(out, ref)
    assert len(new_proc.guidance_calls) == 6
    for ref_call, new_call in zip(ref_proc.guidance_calls, new_proc.guidance_calls, strict=True):
        assert ref_call["time"] == new_call["time"]
        assert new_call["inference_delay"] == 3 and new_call["execution_horizon"] == 25
        # Guidance sees the PRE-update x_t.
        assert torch.equal(ref_call["x_t"], new_call["x_t"])


def test_euler_integrate_debug_tracking_fires_even_when_rtc_disabled():
    # Historical behavior: track() fires whenever the processor exists and has debugging
    # enabled, independent of whether RTC guidance is active.
    torch.manual_seed(5)
    denoise_fn = _make_denoise_fn()
    noise = torch.randn(2, 6, 4)
    proc = _StubRTCProcessor(True)
    out = euler_integrate(denoise_fn, noise, 4, rtc_processor=proc, rtc_enabled=False)
    assert len(proc.guidance_calls) == 0
    assert len(proc.tracked) == 4
    # track() receives the POST-update x_t; the last one is the returned sample.
    assert torch.equal(proc.tracked[-1]["x_t"], out)


def test_euler_integrate_clamps_trained_rtc_prefix_and_sets_clean_time():
    noise = torch.ones(1, 4, 1)
    hard_prefix = torch.tensor([[[2.0], [3.0], [0.0], [0.0]]])
    hard_prefix_mask = torch.tensor([[[True], [True], [False], [False]]])
    seen_times = []

    def denoise_fn(x_t, time_tensor):
        seen_times.append(time_tensor.clone())
        return torch.ones_like(x_t)

    out = euler_integrate(
        denoise_fn,
        noise,
        2,
        hard_prefix=hard_prefix,
        hard_prefix_mask=hard_prefix_mask,
    )

    torch.testing.assert_close(out[:, :2], hard_prefix[:, :2])
    assert all(torch.equal(time[:, :2], torch.zeros(1, 2)) for time in seen_times)
