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
"""

import torch

from lerobot.policies.common.flow_matching import (
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


def test_sample_noise_seeded():
    torch.manual_seed(2)
    n1 = sample_noise((2, 8, 4), "cpu")
    torch.manual_seed(2)
    n2 = sample_noise((2, 8, 4), "cpu")
    assert torch.equal(n1, n2)
    assert n1.dtype == torch.float32 and n1.shape == (2, 8, 4)


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
