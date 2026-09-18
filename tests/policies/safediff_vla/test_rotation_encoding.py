"""Tests for `rotation_encoding.py`'s sin/cos Euler representation, and its integration into
`SafeDiffVLAPolicy`'s `temporal_decoder` architectures (the fix for raw-Euler branch-cut
discontinuities at +-pi -- see `examples/safediff_vla/euler_branch_audit.py`)."""

from __future__ import annotations

import math

import pytest
import torch

from lerobot.policies.safediff_vla import rotation_encoding as renc
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.safediff_vla.test_safediff_vla import make_batch, make_policy

ZERO_MEAN = torch.zeros(3)
UNIT_STD = torch.ones(3)


def _raw7(rx: float, ry: float, rz: float, *, xyz=(0.1, -0.2, 0.3), gripper: float = 0.5) -> torch.Tensor:
    """A single `[7]` raw (mean=0/std=1, i.e. "already raw") sample, for exercising `encode`/
    `decode` directly with `ZERO_MEAN`/`UNIT_STD` (an exact no-op un/re-normalize -- see
    `encode`/`decode`'s docstrings)."""
    return torch.tensor([*xyz, rx, ry, rz, gripper])


# ---- pure encode/decode properties -----------------------------------------------------------


def test_plus_pi_and_minus_pi_encode_to_nearly_identical_representations():
    """+pi and -pi are the same physical angle; the whole point of sin/cos encoding is that the
    representation no longer has a discontinuity there."""
    plus = _raw7(math.pi, 0.0, 0.0)
    minus = _raw7(-math.pi, 0.0, 0.0)
    encoded_plus = renc.encode(plus, ZERO_MEAN, UNIT_STD)
    encoded_minus = renc.encode(minus, ZERO_MEAN, UNIT_STD)
    torch.testing.assert_close(encoded_plus, encoded_minus, atol=1e-6, rtol=0)


def test_encode_decode_roundtrip_recovers_same_rotation():
    torch.manual_seed(0)
    for _ in range(20):
        raw_angles = (torch.rand(3) * 2 - 1) * math.pi  # sample across the full [-pi, pi) range
        raw = _raw7(*raw_angles.tolist())
        decoded = renc.decode(renc.encode(raw, ZERO_MEAN, UNIT_STD), ZERO_MEAN, UNIT_STD)
        # Compare via sin/cos (not the raw angle) so a recovered value of e.g. -pi vs +pi (the
        # same physical angle, opposite ends of atan2's branch) doesn't spuriously fail.
        torch.testing.assert_close(torch.sin(decoded[3:6]), torch.sin(raw[3:6]), atol=1e-5, rtol=0)
        torch.testing.assert_close(torch.cos(decoded[3:6]), torch.cos(raw[3:6]), atol=1e-5, rtol=0)
        torch.testing.assert_close(decoded[:3], raw[:3], atol=1e-6, rtol=0)
        torch.testing.assert_close(decoded[6], raw[6], atol=1e-6, rtol=0)


def test_trajectory_crossing_branch_cut_is_continuous_in_representation_space():
    """A trajectory sweeping through +-pi has a |raw delta| ~ 2*pi jump in the old representation
    -- the encoded (sin/cos) trajectory must have no such jump: consecutive-step deltas stay
    small and bounded throughout, including at the crossing."""
    n_steps = 200
    # rx ramps linearly from just under -pi to just over +pi, wrapped into [-pi, pi) exactly like
    # the dataset's stored actions -- so this reproduces a real branch-cut crossing.
    raw_rx = torch.linspace(-math.pi + 0.1, math.pi + 0.5, n_steps)
    wrapped_rx = torch.atan2(torch.sin(raw_rx), torch.cos(raw_rx))
    traj_raw = torch.stack([_raw7(float(rx), 0.0, 0.0) for rx in wrapped_rx])

    # The old representation genuinely has a large jump at the crossing (sanity-check the test
    # setup reproduces the bug being fixed).
    old_delta = (traj_raw[1:, 3] - traj_raw[:-1, 3]).abs()
    assert old_delta.max() > math.pi, "test setup should reproduce a real branch-cut jump"

    encoded = renc.encode(traj_raw, ZERO_MEAN, UNIT_STD)
    new_delta = (encoded[1:, 3:9] - encoded[:-1, 3:9]).abs().max(dim=-1).values
    # Max possible single-step change for a sin or cos component is 2.0 (from -1 to +1); a
    # smoothly wrapping trajectory's per-step change is far smaller than that everywhere,
    # including exactly at the old jump.
    assert new_delta.max() < 0.2, f"encoded trajectory has a discontinuity: max delta {new_delta.max()}"


def test_encode_output_dim_is_ten_and_decode_output_dim_is_seven():
    raw = _raw7(0.1, -0.2, 0.3)
    encoded = renc.encode(raw, ZERO_MEAN, UNIT_STD)
    assert encoded.shape == (renc.ENCODED_DIM,)
    decoded = renc.decode(encoded, ZERO_MEAN, UNIT_STD)
    assert decoded.shape == (renc.RAW_DIM,)


def test_decode_unit_normalizes_off_manifold_sincos_before_atan2():
    """The decoder has no constraint forcing sin**2+cos**2==1; `decode` must not silently produce
    a biased angle for an off-manifold (e.g. near-zero-magnitude) prediction."""
    x10 = torch.zeros(renc.ENCODED_DIM)
    x10[3], x10[4] = 3.0, 4.0  # sin=3, cos=4 (off-manifold, magnitude 5) -> normalized (0.6, 0.8)
    decoded = renc.decode(x10.unsqueeze(0), ZERO_MEAN, UNIT_STD)[0]
    assert torch.isfinite(decoded).all()
    expected_angle = math.atan2(0.6, 0.8)
    assert decoded[3].item() == pytest.approx(expected_angle, abs=1e-5)


# ---- policy-level integration ------------------------------------------------------------------


def test_policy_action_and_state_shapes_are_unaffected_externally():
    """`plan_action_chunk`'s external contract stays 7-D even though the decoder now works in the
    10-D encoded space internally."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch()
    actions, _ = policy.plan_action_chunk(batch)
    assert actions.shape == (2, policy.config.action_horizon, 7)
    encoded_state = policy._encode_state(policy._current_state(batch))
    assert encoded_state.shape == (2, renc.ENCODED_DIM)
    encoded_action = policy._encode_action_target(batch[ACTION])
    assert encoded_action.shape == (2, policy.config.action_horizon, renc.ENCODED_DIM)


def test_loss_is_finite_and_gradients_flow_through_decoder():
    policy = make_policy(architecture="temporal_decoder")
    loss, metrics = policy(make_batch())
    assert torch.isfinite(loss)
    for key in ("loss_pos", "loss_rot", "loss_grip"):
        assert math.isfinite(metrics[key])
    loss.backward()
    decoder_grads = [p.grad for p in policy.decoder.parameters() if p.requires_grad]
    assert decoder_grads, "decoder has no trainable parameters"
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in decoder_grads), (
        "no finite, nonzero gradient reached the decoder"
    )


def test_batch_at_the_branch_cut_produces_finite_loss_and_gradients():
    """The actual scenario this change targets: GT rotation actions sitting right at +-pi."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch(with_subgoal_label=False)
    batch[ACTION][..., 3] = math.pi - 1e-4  # rx pinned right at the branch cut, matching the
    batch[OBS_STATE][..., 3] = -math.pi + 1e-4  # audited dataset's most common rx region
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.decoder.parameters())


def test_rotation_stats_fallback_to_identity_without_dataset_stats_or_pretrained_path():
    """`make_policy()` never passes `dataset_stats`; the rotation-stat buffers must still exist
    and be finite (mean=0/std=1 fallback) rather than crashing construction."""
    policy = make_policy(architecture="temporal_decoder")
    for name in ("state_rot_mean", "state_rot_std", "action_rot_mean", "action_rot_std"):
        buf = getattr(policy, name)
        assert buf.shape == (3,)
        assert torch.isfinite(buf).all()
    torch.testing.assert_close(policy.state_rot_mean, torch.zeros(3))
    torch.testing.assert_close(policy.state_rot_std, torch.ones(3))
