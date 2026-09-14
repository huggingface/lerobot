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

``euler_integrate`` is compared against verbatim copies of the historical per-policy
sampling loops -- the pi0/pi05/smolvla one (``NOISE_AT_ONE``, including its RTC hook
semantics) and the evo1/groot/wall_x ones (``NOISE_AT_ZERO``): any divergence from those
references is a behavior change for released checkpoints.
"""

import pytest
import torch

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.common.flow_matching import (
    FlowConvention,
    euler_integrate,
    sample_beta,
    sample_noise,
    sample_time_beta,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


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


def test_euler_integrate_clamps_prefix_at_clean_end_of_forward_schedule():
    # Under NOISE_AT_ZERO the clean end of the schedule is t=1, so clean prefix tokens must be
    # handed model time 1.0 rather than the 0.0 used by the openpi convention.
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
        convention=FlowConvention.NOISE_AT_ZERO,
        hard_prefix=hard_prefix,
        hard_prefix_mask=hard_prefix_mask,
    )

    torch.testing.assert_close(out[:, :2], hard_prefix[:, :2])
    assert all(torch.equal(time[:, :2], torch.ones(1, 2)) for time in seen_times)
    # The non-prefix rows still follow the forward schedule 0, 1/2.
    assert [time[0, 2].item() for time in seen_times] == [0.0, 0.5]


def test_euler_integrate_requires_prefix_mask_in_both_conventions():
    noise = torch.ones(1, 4, 1)
    for convention in FlowConvention:
        with pytest.raises(ValueError, match="hard_prefix_mask is required"):
            euler_integrate(
                lambda x_t, time: torch.ones_like(x_t),
                noise,
                2,
                convention=convention,
                hard_prefix=torch.zeros(1, 4, 1),
            )


def test_euler_integrate_rejects_missing_and_conflicting_step_counts():
    noise = torch.ones(1, 4, 1)
    fn = lambda x_t, time: torch.ones_like(x_t)  # noqa: E731
    with pytest.raises(ValueError, match="requires either num_steps or time_grid"):
        euler_integrate(fn, noise)
    with pytest.raises(ValueError, match="conflicts with a time_grid"):
        euler_integrate(fn, noise, 5, time_grid=torch.linspace(0, 1, 4))
    with pytest.raises(ValueError, match="at least 2 entries"):
        euler_integrate(fn, noise, time_grid=torch.zeros(1))


# ---------------------------------------------------------------------------
# Cross-convention equivalence
# ---------------------------------------------------------------------------


def _make_mirrored_fields(num_steps):
    """A time-dependent velocity field and its NOISE_AT_ZERO mirror ``w(x, s) = -v(x, 1 - s)``.

    Reparametrising time this way traces the exact same trajectory, so the two conventions must
    agree bit-for-bit. The field is deliberately nonlinear in ``time`` so that an off-by-one time
    grid or a dropped sign changes the answer.
    """
    weight = torch.randn(4, 4) * 0.1

    def backward_field(x_t, time_tensor):
        t = time_tensor[:, None, None]
        return (x_t @ weight) * (1.0 + 3.0 * t * t) + torch.sin(4.0 * t)

    def forward_field(x_t, time_tensor):
        # Recover the integer step so the mirrored time is bit-identical to the backward run's.
        step = round(float(time_tensor[0]) * num_steps)
        mirrored = torch.full_like(time_tensor, 1.0 + step * (-1.0 / num_steps))
        return -backward_field(x_t, mirrored)

    return backward_field, forward_field


def test_forward_convention_mirrors_backward_convention():
    torch.manual_seed(6)
    num_steps = 7
    backward_field, forward_field = _make_mirrored_fields(num_steps)
    noise = torch.randn(2, 6, 4)

    backward_out = euler_integrate(backward_field, noise, num_steps)
    forward_out = euler_integrate(forward_field, noise, num_steps, convention=FlowConvention.NOISE_AT_ZERO)
    assert torch.equal(backward_out, forward_out)

    # Sanity: the convention really is doing something -- running the forward field under the
    # default backward convention integrates the wrong way and lands somewhere else.
    assert not torch.allclose(euler_integrate(forward_field, noise, num_steps), forward_out)


def test_forward_convention_walks_the_forward_time_grid():
    seen = []

    def denoise_fn(x_t, time_tensor):
        seen.append(time_tensor.clone())
        return torch.zeros_like(x_t)

    euler_integrate(denoise_fn, torch.zeros(1, 3, 4), 4, convention=FlowConvention.NOISE_AT_ZERO)
    assert [t.item() for t in seen] == [0.0, 0.25, 0.5, 0.75]


# ---------------------------------------------------------------------------
# RTC guidance under both conventions (against the real RTCProcessor)
# ---------------------------------------------------------------------------


def _make_rtc_processor(debug=False):
    return RTCProcessor(
        RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            max_guidance_weight=10.0,
            execution_horizon=4,
            debug=debug,
        )
    )


@pytest.mark.parametrize("leftover", [True, False])
def test_rtc_guidance_is_convention_invariant(leftover):
    """The NOISE_AT_ZERO time/velocity flips must reproduce the openpi-convention guided step.

    Covers both the guided path and the ``prev_chunk_left_over is None`` short-circuit, where
    ``RTCProcessor`` just returns the (flipped) base velocity.
    """
    torch.manual_seed(7)
    num_steps = 5
    backward_field, forward_field = _make_mirrored_fields(num_steps)
    noise = torch.randn(2, 6, 4)
    prev_chunk_left_over = torch.randn(2, 6, 4) if leftover else None

    kwargs = {
        "rtc_enabled": True,
        "inference_delay": 2,
        "prev_chunk_left_over": prev_chunk_left_over,
        "execution_horizon": 4,
    }
    backward_out = euler_integrate(
        backward_field, noise, num_steps, rtc_processor=_make_rtc_processor(), **kwargs
    )
    forward_out = euler_integrate(
        forward_field,
        noise,
        num_steps,
        convention=FlowConvention.NOISE_AT_ZERO,
        rtc_processor=_make_rtc_processor(),
        **kwargs,
    )
    assert torch.equal(backward_out, forward_out)

    unguided = euler_integrate(backward_field, noise, num_steps)
    if leftover:
        # Guidance must actually bite, otherwise the equality above would be vacuous.
        assert not torch.allclose(backward_out, unguided)
    else:
        # With no leftover prefix there is nothing to guide towards: plain integration.
        assert torch.equal(backward_out, unguided)


def test_rtc_debug_tracking_reports_noise_at_one_times_in_both_conventions():
    torch.manual_seed(8)
    num_steps = 4
    backward_field, forward_field = _make_mirrored_fields(num_steps)
    noise = torch.randn(2, 6, 4)

    backward_proc, forward_proc = _make_rtc_processor(debug=True), _make_rtc_processor(debug=True)
    euler_integrate(backward_field, noise, num_steps, rtc_processor=backward_proc)
    euler_integrate(
        forward_field,
        noise,
        num_steps,
        convention=FlowConvention.NOISE_AT_ZERO,
        rtc_processor=forward_proc,
    )

    backward_steps = backward_proc.get_all_debug_steps()
    forward_steps = forward_proc.get_all_debug_steps()
    assert [s.time for s in backward_steps] == [1.0, 0.75, 0.5, 0.25]
    assert [s.time for s in forward_steps] == [1.0, 0.75, 0.5, 0.25]
    for backward_step, forward_step in zip(backward_steps, forward_steps, strict=True):
        assert torch.equal(backward_step.v_t, forward_step.v_t)


# ---------------------------------------------------------------------------
# Equivalence with the replaced per-policy loops
# ---------------------------------------------------------------------------


def _evo1_velocity_model(seed):
    """Stand-in for EVO1's `predict_velocity`: depends on the looked-up timestep embedding."""
    torch.manual_seed(seed)
    weight = torch.randn(4, 4) * 0.1
    time_pos_enc_table = torch.randn(1, 1000, 4)

    def predict_velocity(seq, time_emb):
        return torch.tanh(seq @ weight) + time_emb[:, None, :]

    return predict_velocity, time_pos_enc_table


def _reference_evo1_loop(
    predict_velocity,
    time_pos_enc_table,
    action_seq,
    num_steps,
    *,
    use_rtc=False,
    rtc_processor=None,
    inference_delay=None,
    prev_chunk_left_over=None,
    execution_horizon=None,
):
    """Verbatim structure of the historical EVO1 `FlowmatchingActionHead.get_action` loop."""
    batch_size = action_seq.shape[0]
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i / num_steps
        time_index = min(int(t * 999), 999)
        time_emb = time_pos_enc_table[:, time_index, :].squeeze(0)
        time_emb = time_emb.unsqueeze(0).repeat(batch_size, 1)

        if use_rtc:
            guided = rtc_processor.denoise_step(
                x_t=action_seq,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=1.0 - t,
                original_denoise_step_partial=lambda seq, emb=time_emb: -predict_velocity(seq, emb),
                execution_horizon=execution_horizon,
            )
            velocity = -guided
        else:
            velocity = predict_velocity(action_seq, time_emb)

        action_seq = action_seq + dt * velocity
    return action_seq


def _migrated_evo1_loop(predict_velocity, time_pos_enc_table, action_seq, num_steps, **kwargs):
    """How `FlowmatchingActionHead.get_action` now calls the shared solver."""
    batch_size = action_seq.shape[0]

    def denoise_step(seq, time_tensor):
        i = round(float(time_tensor[0]) * num_steps)
        time_index = min(int((i / num_steps) * 999), 999)
        time_emb = time_pos_enc_table[:, time_index, :].squeeze(0)
        time_emb = time_emb.unsqueeze(0).repeat(batch_size, 1)
        return predict_velocity(seq, time_emb)

    return euler_integrate(
        denoise_step,
        action_seq,
        num_steps,
        convention=FlowConvention.NOISE_AT_ZERO,
        **kwargs,
    )


# 3 and 7 are the interesting counts: float32(i / n) * 999 truncates to a different bucket than
# the float64 expression, so a naive time -> index conversion would silently shift the embedding.
@pytest.mark.parametrize("num_steps", [1, 3, 4, 7, 20])
def test_evo1_loop_equivalence(num_steps):
    predict_velocity, table = _evo1_velocity_model(seed=9)
    action_seq = torch.randn(2, 6, 4)
    ref = _reference_evo1_loop(predict_velocity, table, action_seq, num_steps)
    out = _migrated_evo1_loop(predict_velocity, table, action_seq, num_steps)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("num_steps", [3, 7])
def test_evo1_loop_equivalence_with_rtc_guidance(num_steps):
    predict_velocity, table = _evo1_velocity_model(seed=10)
    action_seq = torch.randn(2, 6, 4)
    prev_chunk_left_over = torch.randn(2, 6, 4)

    ref = _reference_evo1_loop(
        predict_velocity,
        table,
        action_seq,
        num_steps,
        use_rtc=True,
        rtc_processor=_make_rtc_processor(),
        inference_delay=2,
        prev_chunk_left_over=prev_chunk_left_over,
        execution_horizon=4,
    )
    out = _migrated_evo1_loop(
        predict_velocity,
        table,
        action_seq,
        num_steps,
        rtc_processor=_make_rtc_processor(),
        rtc_enabled=True,
        inference_delay=2,
        prev_chunk_left_over=prev_chunk_left_over,
        execution_horizon=4,
    )
    assert torch.equal(out, ref)


def _groot_model(seed):
    """Stand-in for GR00T's action encoder / DiT / decoder stack, keyed on the timestep bucket."""
    torch.manual_seed(seed)
    weight = torch.randn(4, 4) * 0.1
    bucket_embedding = torch.randn(1000, 4)

    def model(actions, timesteps_tensor):
        assert timesteps_tensor.dtype == torch.long
        return torch.tanh(actions @ weight) + bucket_embedding[timesteps_tensor][:, None, :]

    return model


def _reference_groot_loop(model, actions, num_inference_timesteps, num_timestep_buckets, vel_strength):
    """Verbatim structure of the historical GR00T `get_action_with_features` loop."""
    batch_size = actions.shape[0]
    dt = 1.0 / num_inference_timesteps
    for t_step in range(num_inference_timesteps):
        t_cont = t_step / float(num_inference_timesteps)
        t_discretized = int(t_cont * num_timestep_buckets)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized)
        pred = model(actions, timesteps_tensor)
        actions = actions + dt * pred * vel_strength
    return actions


@pytest.mark.parametrize("num_inference_timesteps", [3, 4, 10])
def test_groot_loop_equivalence_with_frozen_prefix_weights(num_inference_timesteps):
    model = _groot_model(seed=11)
    torch.manual_seed(12)
    actions = torch.randn(2, 6, 4)
    num_timestep_buckets = 1000

    # Overlap initialization plus GR00T's frozen/ramped velocity weights: the first two steps are
    # frozen, the next two ramp in, the rest run free.
    vel_strength = torch.ones_like(actions)
    vel_strength[:, :2, :] = 0.0
    ramp = 1 - torch.exp(-torch.linspace(0.0, 1.0, 4) * 2.0)
    vel_strength[:, 2:4, :] = (ramp / ramp[-1].clamp_min(1e-8))[1:-1][None, :, None]

    ref = _reference_groot_loop(model, actions, num_inference_timesteps, num_timestep_buckets, vel_strength)
    out = euler_integrate(
        lambda a, time_tensor: model(
            a,
            torch.full(
                size=(a.shape[0],),
                fill_value=int(
                    (round(float(time_tensor[0]) * num_inference_timesteps) / float(num_inference_timesteps))
                    * num_timestep_buckets
                ),
            ),
        ),
        actions,
        num_inference_timesteps,
        convention=FlowConvention.NOISE_AT_ZERO,
        velocity_scale=vel_strength,
    )
    assert torch.equal(out, ref)
    # The frozen prefix must not have moved at all.
    assert torch.equal(out[:, :2], actions[:, :2])


def test_velocity_scale_preserves_multiplication_order():
    # `dt * v * scale` and `dt * (v * scale)` disagree in the last bit for non-dyadic dt, so the
    # solver has to keep GR00T's original left-to-right association.
    torch.manual_seed(13)
    v = torch.randn(2, 6, 4)
    scale = torch.rand(2, 6, 4)
    x0 = torch.zeros(2, 6, 4)
    dt = 1.0 / 10
    out = euler_integrate(
        lambda x_t, time: v,
        x0,
        10,
        convention=FlowConvention.NOISE_AT_ZERO,
        velocity_scale=scale,
    )
    expected = x0
    for _ in range(10):
        expected = expected + dt * v * scale
    assert torch.equal(out, expected)


def _wallx_model(seed):
    """Stand-in for Wall-X's `step`, sensitive to the continuous (non-bucketed) timestep."""
    torch.manual_seed(seed)
    weight = torch.randn(4, 4) * 0.1

    def model(noisy_action, timestep):
        return torch.tanh(noisy_action @ weight) * (1.0 + timestep[:, None, None])

    return model


def _reference_wallx_odeint_euler(step, y0, times):
    """`torchdiffeq.odeint(step, y0, times, method="euler")` unrolled.

    Fixed-grid Euler steps the supplied grid directly: ``dt = t[k+1] - t[k]`` as a tensor
    subtraction, ``y <- y + dt * f(t[k], y)``, and the returned trajectory's last entry is the
    final ``y``. Reproduced here so the equivalence check does not need torchdiffeq installed.
    """
    y = y0
    for t0, t1 in zip(times[:-1], times[1:], strict=True):
        y = y + (t1 - t0) * step(t0, y)
    return y


@pytest.mark.parametrize("num_inference_timesteps", [3, 7, 10])
def test_wallx_loop_equivalence(num_inference_timesteps):
    model = _wallx_model(seed=14)
    torch.manual_seed(15)
    noisy_action = torch.randn(2, 6, 4)
    times = torch.linspace(0, 1, num_inference_timesteps + 1, dtype=torch.float32)

    def reference_step(timestep, action):
        # The historical callback received a 0-dim time and broadcast it itself.
        return model(action, timestep.unsqueeze(0).repeat(action.shape[0]))

    ref = _reference_wallx_odeint_euler(reference_step, noisy_action, times)

    seen_times = []

    def migrated_step(action, timestep):
        seen_times.append(timestep[0].clone())
        return model(action, timestep)

    out = euler_integrate(
        migrated_step,
        noisy_action,
        convention=FlowConvention.NOISE_AT_ZERO,
        time_grid=times,
    )
    assert torch.equal(out, ref)
    # The explicit grid is honored bit-for-bit; `step / n` would differ here for n = 3 and 7.
    assert torch.equal(torch.stack(seen_times), times[:-1])


@pytest.mark.parametrize("num_inference_timesteps", [3, 7, 10])
def test_wallx_loop_equivalence_against_real_torchdiffeq(num_inference_timesteps):
    odeint = pytest.importorskip("torchdiffeq").odeint
    model = _wallx_model(seed=14)
    torch.manual_seed(15)
    noisy_action = torch.randn(2, 6, 4)
    times = torch.linspace(0, 1, num_inference_timesteps + 1, dtype=torch.float32)

    ref = odeint(
        lambda timestep, action: model(action, timestep.unsqueeze(0).repeat(action.shape[0])),
        noisy_action,
        times,
        method="euler",
    )[-1]
    out = euler_integrate(
        model,
        noisy_action,
        convention=FlowConvention.NOISE_AT_ZERO,
        time_grid=times,
    )
    assert torch.equal(out, ref)
