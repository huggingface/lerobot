"""Tests for `architecture="temporal_decoder_grounded_grasp_v2_5"` (experimental) -- see
`configuration_safediff_vla.py`'s architecture docstring and `local_grasp_refiner.py`'s module
docstring. v2's global grounding/approach pipeline (`TargetQueryExtractor`,
`VisualGroundingCrossAttention`, `TargetPointHead`, `TemporalActionDecoder`) is reused completely
unmodified, in its own separate branch; the only new component is `LocalGraspRefiner` plus a
3-stage PRE_GRASP -> LOCAL_GRASP -> POST_GRASP runtime controller. Reuses `test_safediff_vla.py`'s
config/batch/policy builders and `test_grounded_grasp.py`'s `_build_modality_batch`/
`make_grounded_batch` helpers -- not a separate convention.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.safediff_vla.local_grasp_refiner import (
    LOCAL_GRASP,
    POST_GRASP,
    PRE_GRASP,
    LocalGraspRefiner,
)
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.safediff_vla.test_grounded_grasp import make_grounded_batch
from tests.policies.safediff_vla.test_safediff_vla import ACTION_DIM, STATE_DIM, make_config, make_policy

ARCH = "temporal_decoder_grounded_grasp_v2_5"


# ---- LocalGraspRefiner: output shape / horizon / bound / gradient ---------------------------


def test_local_grasp_refiner_output_shape_and_gradient() -> None:
    d, state_dim, horizon = 12, 10, 5
    refiner = LocalGraspRefiner(latent_dim=d, state_dim=state_dim, hidden_dim=16, horizon=horizon, max_delta_xyz=0.03, max_delta_rot=0.05)
    grounded = torch.randn(3, d, requires_grad=True)
    image_feature = torch.randn(3, d, requires_grad=True)
    current_state = torch.randn(3, state_dim, requires_grad=True)
    predicted_target_xyz = torch.randn(3, 3, requires_grad=True)
    relative_xyz = torch.randn(3, 3, requires_grad=True)

    delta_xyz, delta_rot, gripper = refiner(grounded, image_feature, current_state, predicted_target_xyz, relative_xyz)
    assert delta_xyz.shape == (3, horizon, 3)
    assert delta_rot.shape == (3, horizon, 3)
    assert gripper.shape == (3, horizon, 1)
    assert torch.isfinite(delta_xyz).all() and torch.isfinite(delta_rot).all() and torch.isfinite(gripper).all()

    (delta_xyz.sum() + delta_rot.sum() + gripper.sum()).backward()
    assert grounded.grad is not None and torch.any(grounded.grad != 0)
    assert image_feature.grad is not None and torch.any(image_feature.grad != 0)
    assert current_state.grad is not None and torch.any(current_state.grad != 0)
    assert predicted_target_xyz.grad is not None and torch.any(predicted_target_xyz.grad != 0)
    assert relative_xyz.grad is not None and torch.any(relative_xyz.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in refiner.parameters())


def test_local_grasp_refiner_residual_magnitude_bound() -> None:
    max_xyz, max_rot = 0.03, 0.05
    refiner = LocalGraspRefiner(latent_dim=8, state_dim=6, hidden_dim=16, horizon=5, max_delta_xyz=max_xyz, max_delta_rot=max_rot)
    # Push the last layer's weights/bias to saturate tanh, to exercise the bound at its edge.
    with torch.no_grad():
        for p in refiner.net[-1].parameters():
            p.mul_(0.0)
            p.add_(50.0)
    grounded = torch.randn(4, 8)
    image_feature = torch.randn(4, 8)
    current_state = torch.randn(4, 6)
    predicted_target_xyz = torch.randn(4, 3)
    relative_xyz = torch.randn(4, 3)
    delta_xyz, delta_rot, _ = refiner(grounded, image_feature, current_state, predicted_target_xyz, relative_xyz)
    assert torch.all(delta_xyz.abs() < max_xyz + 1e-6)
    assert torch.all(delta_rot.abs() < max_rot + 1e-6)
    # Saturated tanh(50) ~= 1, so the bound should actually be nearly achieved, not just respected.
    assert delta_xyz.abs().mean() > 0.9 * max_xyz
    assert delta_rot.abs().mean() > 0.9 * max_rot


def test_local_grasp_refiner_horizon_matches_config() -> None:
    for horizon in (1, 5, 8):
        refiner = LocalGraspRefiner(latent_dim=6, state_dim=6, hidden_dim=8, horizon=horizon, max_delta_xyz=0.03, max_delta_rot=0.05)
        delta_xyz, delta_rot, gripper = refiner(
            torch.randn(2, 6), torch.randn(2, 6), torch.randn(2, 6), torch.randn(2, 3), torch.randn(2, 3)
        )
        assert delta_xyz.shape[1] == horizon
        assert delta_rot.shape[1] == horizon
        assert gripper.shape[1] == horizon


def test_local_grasp_refiner_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="max_delta_xyz"):
        LocalGraspRefiner(latent_dim=8, state_dim=6, hidden_dim=8, horizon=5, max_delta_xyz=0.0, max_delta_rot=0.05)
    with pytest.raises(ValueError, match="max_delta_rot"):
        LocalGraspRefiner(latent_dim=8, state_dim=6, hidden_dim=8, horizon=5, max_delta_xyz=0.03, max_delta_rot=0.0)


def test_local_grasp_refiner_reacts_to_relative_xyz() -> None:
    """The explicit `relative_xyz` input must actually reach the output -- same
    inputs-fixed-except-one-changed pattern as every other module test on this branch."""
    refiner = LocalGraspRefiner(latent_dim=8, state_dim=6, hidden_dim=16, horizon=5, max_delta_xyz=0.03, max_delta_rot=0.05)
    grounded = torch.randn(1, 8)
    image_feature = torch.randn(1, 8)
    current_state = torch.randn(1, 6)
    predicted_target_xyz = torch.randn(1, 3)
    rel_a = torch.randn(1, 3)
    rel_b = torch.randn(1, 3)
    with torch.no_grad():
        out_a = refiner(grounded, image_feature, current_state, predicted_target_xyz, rel_a)
        out_b = refiner(grounded, image_feature, current_state, predicted_target_xyz, rel_b)
    assert not torch.allclose(out_a[0], out_b[0])


# ---- policy construction: modules built, v2 modules frozen -----------------------------------


def test_v2_5_builds_expected_modules_and_freezes_v2_by_default() -> None:
    policy = make_policy(architecture=ARCH)
    assert hasattr(policy, "target_query_extractor")
    assert hasattr(policy, "visual_grounding_cross_attn")
    assert hasattr(policy, "target_point_head")
    assert hasattr(policy, "local_grasp_refiner")
    assert not hasattr(policy, "sharpened_visual_grounding")  # v3's module, not v2_5's
    assert not hasattr(policy, "coarse_target_head")
    assert policy.decoder.use_grounded_grasp_v2 is True
    assert policy.decoder.use_target_xyz is True

    assert all(not p.requires_grad for p in policy.target_query_extractor.parameters())
    assert all(not p.requires_grad for p in policy.visual_grounding_cross_attn.parameters())
    assert all(not p.requires_grad for p in policy.target_point_head.parameters())
    assert all(not p.requires_grad for p in policy.decoder.parameters())
    assert all(not p.requires_grad for p in policy.backbone.parameters())
    assert any(p.requires_grad for p in policy.local_grasp_refiner.parameters())
    assert all(p.requires_grad for p in policy.local_grasp_refiner.parameters())


def test_v2_5_freeze_toggle_off_makes_v2_modules_trainable() -> None:
    policy = make_policy(architecture=ARCH, local_grasp_freeze_global_modules=False)
    assert any(p.requires_grad for p in policy.target_query_extractor.parameters())
    assert any(p.requires_grad for p in policy.decoder.parameters())


def test_get_optim_params_only_includes_trainable_when_frozen() -> None:
    policy = make_policy(architecture=ARCH)
    optim_params = list(policy.get_optim_params())
    refiner_params = list(policy.local_grasp_refiner.parameters())
    assert len(optim_params) == len(refiner_params)
    assert all(any(p is rp for rp in refiner_params) for p in optim_params)


# ---- training: loss_local, masking to the grasp window, gradient reaches only the refiner ----


def _batch_with_transition_at(step: int, batch_size: int = 2, horizon: int = 20) -> dict[str, torch.Tensor]:
    batch = make_grounded_batch(batch_size=batch_size, horizon=horizon)
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # pre-grasp (state convention: 0=open)
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0  # open throughout, except:
    batch[ACTION][:, step:, GRIPPER_INDEX_RAW] = -1.0  # closes at `step`
    return batch


def test_v2_5_trains_with_split_global_and_local_losses() -> None:
    policy = make_policy(architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10)
    batch = _batch_with_transition_at(step=3, horizon=20)  # within window_k=10
    loss, metrics = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {
        "loss", "loss_pos", "loss_rot", "loss_grip", "loss_target",
        "loss_local", "loss_local_pos", "loss_local_rot", "loss_local_grip", "local_eligible_frac",
    } <= metrics.keys()
    assert metrics["local_eligible_frac"] == pytest.approx(1.0)
    assert metrics["loss_local"] > 0
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.local_grasp_refiner.parameters())
    # v2 modules are frozen by default -- backward() must not populate their grads at all.
    assert all(p.grad is None for p in policy.target_query_extractor.parameters())
    assert all(p.grad is None for p in policy.decoder.parameters())
    assert all(p.grad is None for p in policy.backbone.parameters())


def test_v2_5_local_loss_zero_when_transition_outside_window_k() -> None:
    """A transition that exists but lands OUTSIDE `local_grasp_window_k` steps of the current
    sample must not contribute to `loss_local` -- proves masking is on the WINDOW, not just
    "does a valid target exist anywhere" (which `target_valid_mask`/`loss_target` alone would be)."""
    policy = make_policy(architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=5)
    batch = _batch_with_transition_at(step=15, horizon=20)  # valid transition, but far beyond window_k=5
    _, metrics = policy(batch)
    assert metrics["local_eligible_frac"] == pytest.approx(0.0)
    assert metrics["loss_local"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss_local_pos"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss_local_rot"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss_local_grip"] == pytest.approx(0.0, abs=1e-8)
    # v2's own loss_target must still be computed/trained normally -- only the NEW local loss is
    # scoped to the window.
    assert metrics["loss_target"] > 0


def test_v2_5_local_loss_zero_when_no_transition_at_all() -> None:
    policy = make_policy(architecture=ARCH)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0  # never closes
    _, metrics = policy(batch)
    assert metrics["local_eligible_frac"] == pytest.approx(0.0)
    assert metrics["loss_local"] == pytest.approx(0.0, abs=1e-8)


def test_v2_5_lambda_local_grasp_scales_loss() -> None:
    policy_default = make_policy(architecture=ARCH, action_horizon=20, execute_horizon=2, lambda_local_grasp=1.0)
    policy_zeroed = make_policy(architecture=ARCH, action_horizon=20, execute_horizon=2, lambda_local_grasp=0.0)
    policy_zeroed.load_state_dict(policy_default.state_dict())
    batch = _batch_with_transition_at(step=3, horizon=20)
    with torch.no_grad():
        loss_default, metrics = policy_default(batch)
        loss_zeroed, _ = policy_zeroed(batch)
    assert metrics["loss_local"] > 0
    assert loss_default.item() == pytest.approx(loss_zeroed.item() + metrics["loss_local"], abs=1e-6)


def test_v2_5_canonical_temporal_decoder_regression() -> None:
    policy = make_policy(architecture="temporal_decoder")
    loss, metrics = policy(make_grounded_batch())
    assert metrics["loss_local"] == 0.0
    assert metrics["loss_local_pos"] == 0.0
    assert metrics["loss_local_rot"] == 0.0
    assert metrics["loss_local_grip"] == 0.0
    assert "local_eligible_frac" not in metrics
    assert loss.item() == pytest.approx(metrics["loss"], abs=1e-6)


def test_v2_5_does_not_affect_v2_own_branch() -> None:
    """Regression guard: v2's own metrics/behavior are bit-for-bit what they were before v2_5
    existed -- v2 must never build/see any `local_grasp_*` machinery."""
    policy = make_policy(architecture="temporal_decoder_grounded_grasp_v2")
    assert not hasattr(policy, "local_grasp_refiner")
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0
    loss, metrics = policy(batch)
    assert metrics["loss_local"] == 0.0
    assert loss.item() == pytest.approx(metrics["loss"], abs=1e-6)


# ---- inference: 3-stage controller -------------------------------------------------------------


def test_v2_5_starts_in_pre_grasp() -> None:
    policy = make_policy(architecture=ARCH)
    assert policy._v25_stage == PRE_GRASP
    assert policy._v25_gripper_closed_latch is False
    assert policy._v25_last_target_xyz is None


def test_v2_5_local_mode_entry_threshold() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._v25_last_target_xyz = torch.zeros(1, 3)
    far_state = torch.zeros(1, STATE_DIM)
    far_state[0, :3] = 1.0  # 1m away, outside the 10cm default radius
    action = torch.zeros(1, ACTION_DIM)
    out = policy._maybe_enter_local_grasp(action, far_state)
    assert policy._v25_stage == PRE_GRASP
    assert torch.equal(out, action)

    near_state = torch.zeros(1, STATE_DIM)
    near_state[0, :3] = 0.01  # well within the 10cm default radius
    out2 = policy._maybe_enter_local_grasp(action, near_state)
    assert policy._v25_stage == LOCAL_GRASP
    assert torch.equal(out2, action)  # action itself unmodified -- only the stage flips


def test_v2_5_local_entry_clears_global_action_queue() -> None:
    policy = make_policy(architecture=ARCH, action_horizon=5, execute_horizon=5)
    policy.eval()
    policy.reset()
    policy._v25_last_target_xyz = torch.zeros(1, 3)
    policy._executor._action_queue.extend([torch.zeros(1, ACTION_DIM) for _ in range(5)])
    assert len(policy._executor._action_queue) == 5

    near_state = torch.zeros(1, STATE_DIM)
    near_state[0, :3] = 0.01
    policy._maybe_enter_local_grasp(torch.zeros(1, ACTION_DIM), near_state)
    assert len(policy._executor._action_queue) == 0


def test_v2_5_local_entry_is_one_shot_per_episode() -> None:
    """Once past PRE_GRASP, `_maybe_enter_local_grasp` must be a no-op forever (until `reset()`) --
    even if somehow called again while already in POST_GRASP."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._v25_last_target_xyz = torch.zeros(1, 3)
    near_state = torch.zeros(1, STATE_DIM)
    near_state[0, :3] = 0.01
    policy._maybe_enter_local_grasp(torch.zeros(1, ACTION_DIM), near_state)
    assert policy._v25_stage == LOCAL_GRASP

    policy._v25_stage = POST_GRASP  # simulate having already exited LOCAL_GRASP
    policy._maybe_enter_local_grasp(torch.zeros(1, ACTION_DIM), near_state)
    assert policy._v25_stage == POST_GRASP  # must NOT flip back to LOCAL_GRASP


def test_v2_5_plan_local_grasp_step_shape_and_relative_xyz_connection() -> None:
    """End-to-end (through the policy, not the bare module): zeroing `local_grasp_refiner`'s
    parameters must change its output relative to a randomly-initialized one, AND the step must
    actually consume `relative_xyz` derived from v2's own `predicted_target_xyz` -- proven by
    perturbing `target_point_head` (which changes `predicted_target_xyz`, hence `relative_xyz`)
    and confirming the local step's action changes too."""
    policy = make_policy(architecture=ARCH, local_grasp_freeze_global_modules=False)
    policy.eval()
    batch = make_grounded_batch(batch_size=1)
    with torch.no_grad():
        action_before, metrics_before = policy._plan_local_grasp_step(batch)
        assert action_before.shape == (1, ACTION_DIM)
        for p in policy.target_point_head.parameters():
            p.add_(1.0)
        action_after, metrics_after = policy._plan_local_grasp_step(batch)
    assert not torch.allclose(metrics_before["predicted_target_xyz"], metrics_after["predicted_target_xyz"])
    assert not torch.allclose(action_before, action_after)


def test_v2_5_execute_only_first_local_step_of_horizon() -> None:
    """`local_grasp_execute_horizon=1`: `_plan_local_grasp_step` must use index 0 of the refiner's
    predicted horizon, never any later step -- proven by making steps 1+ NOT match what's used, and
    steps 1+ differ from step 0 (deterministic per-position bias), then checking the emitted action
    tracks step 0 not the others."""
    policy = make_policy(architecture=ARCH, local_grasp_action_horizon=3)
    policy.eval()
    original_forward = policy.local_grasp_refiner.forward

    def spy_forward(*args, **kwargs):
        delta_xyz, delta_rot, gripper = original_forward(*args, **kwargs)
        delta_xyz = delta_xyz.clone()
        delta_xyz[:, 1:] += 999.0  # corrupt every step EXCEPT index 0
        return delta_xyz, delta_rot, gripper

    policy.local_grasp_refiner.forward = spy_forward
    batch = make_grounded_batch(batch_size=1)
    with torch.no_grad():
        action, metrics = policy._plan_local_grasp_step(batch)
    assert torch.all(metrics["delta_xyz_m"].abs() < 900)  # index-0 delta, NOT the corrupted 999-offset steps


def test_v2_5_replans_every_local_step_not_a_chunk() -> None:
    """Two consecutive `select_action` calls in LOCAL_GRASP with DIFFERENT observations must
    produce genuinely different actions computed fresh each time -- proves there is no queue/chunk
    being reused (unlike every global-stage architecture on this branch)."""
    policy = make_policy(architecture=ARCH, local_grasp_freeze_global_modules=False)
    policy.eval()
    policy.reset()
    policy._v25_stage = LOCAL_GRASP
    batch_a = make_grounded_batch(batch_size=1)
    batch_b = make_grounded_batch(batch_size=1)
    action_a = policy.select_action(batch_a)
    action_b = policy.select_action(batch_b)
    assert not torch.allclose(action_a, action_b)


def test_v2_5_select_action_local_grasp_bypasses_executor_queue() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._v25_stage = LOCAL_GRASP
    batch = make_grounded_batch(batch_size=1)
    policy.select_action(batch)
    assert len(policy._executor._action_queue) == 0  # never touched by the local path


# ---- gripper one-shot latch / LOCAL_GRASP -> POST_GRASP transition ---------------------------


def _make_v2_5_policy_with_forced_gripper(closed: bool) -> SafeDiffVLAPolicy:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    with torch.no_grad():
        for p in policy.local_grasp_refiner.net[-1].parameters():
            p.zero_()
        # gripper is channel index 6 of each 7-wide horizon block in the flattened output layer.
        bias = policy.local_grasp_refiner.net[-1].bias
        horizon = policy.local_grasp_refiner.horizon
        bias.view(horizon, 7)[:, 6] = -10.0 if closed else 10.0
    return policy


def test_v2_5_gripper_latch_fires_and_transitions_to_post_grasp() -> None:
    policy = _make_v2_5_policy_with_forced_gripper(closed=True)
    policy.reset()
    policy._v25_stage = LOCAL_GRASP
    batch = make_grounded_batch(batch_size=1)
    action = policy.select_action(batch)
    assert policy._v25_gripper_closed_latch is True
    assert policy._v25_stage == POST_GRASP
    closed_value = (0.0 - policy.action_grip_mean) / policy.action_grip_std
    assert action[0, GRIPPER_INDEX_RAW].item() == pytest.approx(closed_value.item())


def test_v2_5_gripper_does_not_latch_when_open() -> None:
    policy = _make_v2_5_policy_with_forced_gripper(closed=False)
    policy.reset()
    policy._v25_stage = LOCAL_GRASP
    batch = make_grounded_batch(batch_size=1)
    policy.select_action(batch)
    assert policy._v25_gripper_closed_latch is False
    assert policy._v25_stage == LOCAL_GRASP


def test_v2_5_gripper_latch_prevents_reopen_chatter() -> None:
    """Once latched closed, a LATER step where the refiner's raw output would read "open" again
    must NOT unlatch -- the whole point of the one-shot guard."""
    policy = _make_v2_5_policy_with_forced_gripper(closed=True)
    policy.reset()
    policy._v25_stage = LOCAL_GRASP
    batch = make_grounded_batch(batch_size=1)
    policy.select_action(batch)  # latches closed, transitions to POST_GRASP
    assert policy._v25_gripper_closed_latch is True
    assert policy._v25_stage == POST_GRASP

    # Force the stage back to LOCAL_GRASP and the refiner back to "open" -- simulating what would
    # be gripper chatter if the latch weren't sticky.
    policy._v25_stage = LOCAL_GRASP
    with torch.no_grad():
        bias = policy.local_grasp_refiner.net[-1].bias
        bias.view(policy.local_grasp_refiner.horizon, 7)[:, 6] = 10.0
    action2 = policy.select_action(batch)
    closed_value = (0.0 - policy.action_grip_mean) / policy.action_grip_std
    assert action2[0, GRIPPER_INDEX_RAW].item() == pytest.approx(closed_value.item())
    assert policy._v25_gripper_closed_latch is True


def test_v2_5_post_grasp_returns_to_v2_temporal_action_decoder() -> None:
    """After the LOCAL_GRASP -> POST_GRASP transition, `select_action` must route back through
    `self._executor`/`plan_action_chunk` (v2's own global decoder), conditioned on phase=POST_GRASP
    -- not stay on the local per-step path."""
    policy = make_policy(architecture=ARCH, action_horizon=4, execute_horizon=4)
    policy.eval()
    policy.reset()
    policy._v25_stage = POST_GRASP
    batch = make_grounded_batch(batch_size=1)
    action = policy.select_action(batch)
    assert action.shape == (1, ACTION_DIM)
    # The executor's queue must have been populated by a real chunk plan (execute_horizon-1 left
    # after popping one) -- proves `plan_action_chunk`/`_plan_temporal_decoder` actually ran.
    assert len(policy._executor._action_queue) == policy.config.execute_horizon - 1


def test_v2_5_post_grasp_conditions_decoder_on_phase_one() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    batch = make_grounded_batch(batch_size=1)
    policy._v25_stage = PRE_GRASP
    with torch.no_grad():
        actions_pre, _ = policy._plan_temporal_decoder(batch)
    policy._v25_stage = POST_GRASP
    with torch.no_grad():
        actions_post, _ = policy._plan_temporal_decoder(batch)
    assert not torch.allclose(actions_pre, actions_post)


def test_v2_5_reset_clears_stage_latch_and_cached_target() -> None:
    policy = make_policy(architecture=ARCH)
    policy._v25_stage = POST_GRASP
    policy._v25_gripper_closed_latch = True
    policy._v25_last_target_xyz = torch.zeros(1, 3)
    policy.reset()
    assert policy._v25_stage == PRE_GRASP
    assert policy._v25_gripper_closed_latch is False
    assert policy._v25_last_target_xyz is None


# ---- config validation -------------------------------------------------------------------------


def test_config_grounded_grasp_v2_5_is_a_registered_architecture() -> None:
    config = make_config(architecture=ARCH)
    assert config.architecture == ARCH


def test_config_rejects_invalid_local_grasp_fields() -> None:
    with pytest.raises(ValueError, match="local_grasp_radius_m"):
        make_config(architecture=ARCH, local_grasp_radius_m=0.0)
    with pytest.raises(ValueError, match="local_grasp_max_delta_xyz_m"):
        make_config(architecture=ARCH, local_grasp_max_delta_xyz_m=0.0)
    with pytest.raises(ValueError, match="local_grasp_max_delta_rot_rad"):
        make_config(architecture=ARCH, local_grasp_max_delta_rot_rad=0.0)
    with pytest.raises(ValueError, match="local_grasp_execute_horizon"):
        make_config(architecture=ARCH, local_grasp_execute_horizon=0)
    with pytest.raises(ValueError, match="local_grasp_execute_horizon"):
        make_config(architecture=ARCH, local_grasp_execute_horizon=10, local_grasp_action_horizon=5)
    with pytest.raises(ValueError, match="local_grasp_window_k"):
        make_config(architecture=ARCH, local_grasp_window_k=0)
    with pytest.raises(ValueError, match="lambda_local_grasp"):
        make_config(architecture=ARCH, lambda_local_grasp=-1.0)


# ---- no regression on other architectures ------------------------------------------------------


def test_other_architectures_never_build_v2_5_modules() -> None:
    for arch in (
        "temporal_decoder",
        "temporal_decoder_subgoal",
        "temporal_decoder_grounded_grasp",
        "temporal_decoder_instruction",
        "temporal_decoder_text_crossattn",
        "temporal_decoder_grounded_grasp_v2",
        "temporal_decoder_grounded_grasp_v3",
    ):
        policy = make_policy(architecture=arch)
        assert not hasattr(policy, "local_grasp_refiner")


# ---- LocalGraspCorrectionHead (opt-in, see `grasp_correction_head.py`) -------------------------


def test_correction_head_opt_in_default_off() -> None:
    """`local_grasp_use_correction_head` defaults to False -- existing v2_5 configs/checkpoints
    must be byte-for-byte unaffected unless explicitly turned on."""
    policy = make_policy(architecture=ARCH)
    assert not hasattr(policy, "local_grasp_correction_head")
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    _, metrics = policy(batch)
    assert "loss_grasp_correction" in metrics  # zero-valued placeholder, always present
    assert metrics["loss_grasp_correction"] == pytest.approx(0.0)
    assert "grasp_correction_eligible_frac" not in metrics


def test_correction_head_builds_and_freezes_refiner_by_default() -> None:
    policy = make_policy(architecture=ARCH, local_grasp_use_correction_head=True)
    assert hasattr(policy, "local_grasp_correction_head")
    # First-experiment default: refiner ALSO frozen, only the new head trainable.
    assert all(not p.requires_grad for p in policy.local_grasp_refiner.parameters())
    assert any(p.requires_grad for p in policy.local_grasp_correction_head.parameters())
    assert all(p.requires_grad for p in policy.local_grasp_correction_head.parameters())
    optim_params = list(policy.get_optim_params())
    head_params = list(policy.local_grasp_correction_head.parameters())
    assert len(optim_params) == len(head_params)
    assert all(any(p is hp for hp in head_params) for p in optim_params)


def test_correction_head_freeze_refiner_toggle_off() -> None:
    policy = make_policy(
        architecture=ARCH, local_grasp_use_correction_head=True, local_grasp_correction_freeze_refiner=False
    )
    assert any(p.requires_grad for p in policy.local_grasp_refiner.parameters())


def test_correction_head_trains_and_gradient_isolated_to_head_only() -> None:
    policy = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_use_correction_head=True,
        # This synthetic batch's xyz is unit-scale `torch.randn`, not real meters -- a huge clip
        # keeps the outlier-exclusion mechanism (tested separately, below) out of this test's way.
        local_grasp_correction_outlier_clip_m=1e6,
    )
    batch = _batch_with_transition_at(step=3, horizon=20)
    loss, metrics = policy(batch)
    assert {"loss_grasp_correction", "grasp_correction_eligible_frac"} <= metrics.keys()
    assert metrics["grasp_correction_eligible_frac"] == pytest.approx(1.0)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.local_grasp_correction_head.parameters())
    # First-experiment default: refiner frozen -- backward() must not populate its grads.
    assert all(p.grad is None for p in policy.local_grasp_refiner.parameters())
    assert all(p.grad is None for p in policy.target_query_extractor.parameters())


def test_correction_head_outlier_clip_excludes_large_residual_frames() -> None:
    """A frame whose own `gt_target_xyz - target_xyz` exceeds `local_grasp_correction_outlier_clip_m`
    must not contribute to `loss_grasp_correction` -- the B_WRONG_OBJECT mitigation."""
    policy = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_use_correction_head=True, local_grasp_correction_outlier_clip_m=1e-6,
    )
    batch = _batch_with_transition_at(step=3, horizon=20)
    _, metrics = policy(batch)
    # An essentially-zero clip excludes every real (non-degenerate) sample.
    assert metrics["grasp_correction_eligible_frac"] == pytest.approx(0.0)
    assert metrics["loss_grasp_correction"] == pytest.approx(0.0, abs=1e-8)


def test_correction_head_bounds_delta_and_shifts_corrected_target() -> None:
    from lerobot.policies.safediff_vla.grasp_correction_head import LocalGraspCorrectionHead

    max_corr = 0.16
    head = LocalGraspCorrectionHead(latent_dim=8, state_dim=6, hidden_dim=16, max_grasp_correction_m=max_corr)
    with torch.no_grad():
        for p in head.net[-1].parameters():
            p.mul_(0.0)
            p.add_(50.0)
    grounded = torch.randn(4, 8)
    image_feature = torch.randn(4, 8)
    current_state = torch.randn(4, 6)
    coarse_target_xyz = torch.randn(4, 3)
    relative_xyz = torch.randn(4, 3)
    delta = head(grounded, image_feature, current_state, coarse_target_xyz, relative_xyz)
    assert delta.shape == (4, 3)
    assert torch.all(delta.abs() < max_corr + 1e-6)
    assert delta.abs().mean() > 0.9 * max_corr


def test_correction_head_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="local_grasp_max_grasp_correction_m"):
        make_config(architecture=ARCH, local_grasp_max_grasp_correction_m=0.0)
    with pytest.raises(ValueError, match="local_grasp_correction_outlier_clip_m"):
        make_config(architecture=ARCH, local_grasp_correction_outlier_clip_m=0.0)
    with pytest.raises(ValueError, match="lambda_grasp_correction"):
        make_config(architecture=ARCH, lambda_grasp_correction=-1.0)


def test_correction_head_plan_local_grasp_step_logs_corrected_target() -> None:
    policy = make_policy(architecture=ARCH, local_grasp_use_correction_head=True, local_grasp_correction_freeze_refiner=False)
    policy.eval()
    batch = make_grounded_batch(batch_size=1)
    with torch.no_grad():
        action_before, metrics_before = policy._plan_local_grasp_step(batch)
        assert "delta_grasp_xyz_m" in metrics_before
        assert "corrected_target_xyz_m" in metrics_before
        assert metrics_before["delta_grasp_xyz_m"].shape == (1, 3)
        for p in policy.local_grasp_correction_head.parameters():
            p.add_(1.0)
        action_after, metrics_after = policy._plan_local_grasp_step(batch)
    # Perturbing ONLY the correction head must still change the emitted action -- proves it's
    # actually wired into what `local_grasp_refiner` is called with, not a dead computation.
    assert not torch.allclose(metrics_before["corrected_target_xyz_m"], metrics_after["corrected_target_xyz_m"])
    assert not torch.allclose(action_before, action_after)


# ---- grasp-phase-aware LocalGraspRefiner loss (loss-ONLY ablation, no new module -- the
# CorrectionHead direction's follow-up, see `local_grasp_use_phase_aware_loss`'s config docstring) ----


def test_phase_aware_loss_opt_in_default_off_matches_old_formula() -> None:
    policy_old = make_policy(architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10)
    policy_new = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_use_phase_aware_loss=False,
    )
    policy_new.load_state_dict(policy_old.state_dict())
    batch = _batch_with_transition_at(step=3, horizon=20)
    with torch.no_grad():
        _, metrics_old = policy_old(batch)
        _, metrics_new = policy_new(batch)
    assert metrics_new["loss_local"] == pytest.approx(metrics_old["loss_local"])
    assert metrics_new["loss_local_terminal_xyz"] == pytest.approx(0.0)
    assert metrics_new["loss_local_approach_xyz"] == pytest.approx(0.0)
    assert metrics_new["loss_local_grip_weighted"] == pytest.approx(0.0)


def test_phase_aware_loss_enabled_adds_new_terms_but_leaves_rot_and_old_grip_unchanged() -> None:
    policy_off = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_use_phase_aware_loss=False,
    )
    policy_on = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_use_phase_aware_loss=True,
    )
    policy_on.load_state_dict(policy_off.state_dict())
    batch = _batch_with_transition_at(step=3, horizon=20)
    with torch.no_grad():
        _, metrics_off = policy_off(batch)
        _, metrics_on = policy_on(batch)
    # rotation and the (still-always-computed) unweighted pos/grip terms are untouched by the flag.
    assert metrics_on["loss_local_rot"] == pytest.approx(metrics_off["loss_local_rot"])
    assert metrics_on["loss_local_pos"] == pytest.approx(metrics_off["loss_local_pos"])
    assert metrics_on["loss_local_grip"] == pytest.approx(metrics_off["loss_local_grip"])
    assert metrics_on["loss_local_terminal_xyz"] > 0
    assert metrics_on["loss_local_approach_xyz"] > 0
    assert metrics_on["loss_local_grip_weighted"] > 0
    assert metrics_on["loss_local"] != pytest.approx(metrics_off["loss_local"])


def test_phase_aware_loss_observability_metrics_present_and_raw_meters() -> None:
    """`terminal_grasp_distance_m`/`predicted_local_residual_magnitude_m` are pure observability
    (never affect any loss) -- present and positive when enabled/eligible, absent-as-zero when
    off, added for offline checkpoint-to-checkpoint comparison during the 100k-step long run."""
    policy_off = make_policy(architecture=ARCH, local_grasp_use_phase_aware_loss=False)
    policy_on = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_use_phase_aware_loss=True,
    )
    batch_off = make_grounded_batch()
    batch_off[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch_off[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    _, metrics_off = policy_off(batch_off)
    assert metrics_off["terminal_grasp_distance_m"] == pytest.approx(0.0)
    assert metrics_off["predicted_local_residual_magnitude_m"] == pytest.approx(0.0)

    batch_on = _batch_with_transition_at(step=3, horizon=20)
    _, metrics_on = policy_on(batch_on)
    assert metrics_on["terminal_grasp_distance_m"] > 0
    assert metrics_on["predicted_local_residual_magnitude_m"] > 0


def test_phase_aware_loss_gradient_isolated_to_refiner() -> None:
    policy = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_use_phase_aware_loss=True,
    )
    batch = _batch_with_transition_at(step=3, horizon=20)
    loss, metrics = policy(batch)
    assert metrics["loss_local_terminal_xyz"] > 0
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.local_grasp_refiner.parameters())
    assert all(p.grad is None for p in policy.target_query_extractor.parameters())
    assert all(p.grad is None for p in policy.decoder.parameters())
    assert all(p.grad is None for p in policy.backbone.parameters())


def test_phase_aware_loss_zero_when_no_transition() -> None:
    policy = make_policy(architecture=ARCH, local_grasp_use_phase_aware_loss=True)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0  # never closes
    _, metrics = policy(batch)
    assert metrics["loss_local_terminal_xyz"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss_local_approach_xyz"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss_local_grip_weighted"] == pytest.approx(0.0, abs=1e-8)


def test_phase_aware_terminal_loss_uses_clamped_close_index_beyond_horizon() -> None:
    """`close_idx_in_horizon` must clamp to `local_h - 1` when the real transition is farther out
    than the predicted horizon can see (but still within `local_grasp_window_k`) -- proven by
    forcing a transition beyond a short `local_grasp_action_horizon=3` and checking the terminal
    loss still contributes (not silently skipped just because the exact step isn't in-horizon)."""
    policy = make_policy(
        architecture=ARCH, action_horizon=20, execute_horizon=2, local_grasp_window_k=10,
        local_grasp_action_horizon=3, local_grasp_use_phase_aware_loss=True,
    )
    batch = _batch_with_transition_at(step=7, horizon=20)  # beyond local_h=3, within window_k=10
    _, metrics = policy(batch)
    assert metrics["local_eligible_frac"] == pytest.approx(1.0)
    assert metrics["loss_local_terminal_xyz"] > 0
    assert metrics["loss_local_approach_xyz"] > 0


def test_config_phase_aware_loss_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="lambda_local_terminal_xyz"):
        make_config(architecture=ARCH, lambda_local_terminal_xyz=-1.0)
    with pytest.raises(ValueError, match="lambda_local_approach_xyz"):
        make_config(architecture=ARCH, lambda_local_approach_xyz=-1.0)
    with pytest.raises(ValueError, match="lambda_local_grip_weighted"):
        make_config(architecture=ARCH, lambda_local_grip_weighted=-1.0)
    with pytest.raises(ValueError, match="local_grasp_grip_close_weight"):
        make_config(architecture=ARCH, local_grasp_grip_close_weight=0.0)
    with pytest.raises(ValueError, match="local_grasp_approach_weight_min"):
        make_config(architecture=ARCH, local_grasp_approach_weight_min=1.0, local_grasp_approach_weight_max=0.5)
    with pytest.raises(ValueError, match="local_grasp_huber_delta"):
        make_config(architecture=ARCH, local_grasp_huber_delta=0.0)
