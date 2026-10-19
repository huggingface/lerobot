"""Tests for `architecture="temporal_decoder_grounded_grasp_v3"` (experimental) -- see
`configuration_safediff_vla.py`'s architecture docstring for the v2 held-out eval numbers
motivating this design (v2 improved target-xyz localization error but never reached
grasp-relevant precision, and did not move task success). Reuses `test_safediff_vla.py`'s
config/batch/policy builders and `test_grounded_grasp.py`'s `_build_modality_batch`/
`make_grounded_batch` helpers -- not a separate convention.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.safediff_vla.grounded_grasp_v2 import TargetQueryExtractor
from lerobot.policies.safediff_vla.grounded_grasp_v3 import (
    CoarseTargetHead,
    FineResidualHead,
    SharpenedVisualGrounding,
)
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.policies.safediff_vla.temporal_decoder import TemporalActionDecoder
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.safediff_vla.test_grounded_grasp import _build_modality_batch, make_grounded_batch
from tests.policies.safediff_vla.test_safediff_vla import (
    ACTION_DIM,
    ENCODED_DIM,
    STATE_DIM,
    make_config,
    make_policy,
)

ARCH = "temporal_decoder_grounded_grasp_v3"


# ---- SharpenedVisualGrounding: token scores / normalization / sharpening --------------------


def test_sharpened_visual_grounding_shape_and_stats_shape() -> None:
    d = 12
    grounding = SharpenedVisualGrounding(d)
    target_query = torch.randn(3, d, requires_grad=True)
    image_tokens = torch.randn(3, 20, d, requires_grad=True)
    text_tokens = torch.randn(3, 4, d)
    state_token = torch.randn(3, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)

    feature, stats = grounding(target_query, tokens, pad_mask, modality_ids)
    assert feature.shape == (3, d)
    assert torch.isfinite(feature).all()
    assert stats["attention_weights"].shape == (3, tokens.shape[1])
    assert stats["entropy"].shape == (3,)
    assert stats["top1_mass"].shape == (3,)
    assert stats["topk_mass"].shape == (3,)

    feature.sum().backward()
    assert target_query.grad is not None and torch.any(target_query.grad != 0)
    assert image_tokens.grad is not None and torch.any(image_tokens.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in grounding.parameters())


def test_sharpened_visual_grounding_weights_finite_and_normalized() -> None:
    d = 12
    grounding = SharpenedVisualGrounding(d)
    target_query = torch.randn(4, d)
    image_tokens = torch.randn(4, 20, d)
    text_tokens = torch.randn(4, 4, d)
    state_token = torch.randn(4, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)

    _, stats = grounding(target_query, tokens, pad_mask, modality_ids)
    weights = stats["attention_weights"]
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    # Weight mass must sit entirely on IMAGE_MODALITY positions (text/state masked to -inf).
    image_only_mass = weights[:, : image_tokens.shape[1]].sum(dim=1)
    assert torch.allclose(image_only_mass, torch.ones(4), atol=1e-5)


def test_sharpened_visual_grounding_restricted_to_image_tokens_only() -> None:
    """Perturbing ONLY the text tokens must not move the attention weights over image positions --
    proves the similarity logits are computed against, and masked to, image tokens only."""
    d = 12
    grounding = SharpenedVisualGrounding(d)
    target_query = torch.randn(1, d)
    image_tokens = torch.randn(1, 20, d)
    state_token = torch.randn(1, 1, d)
    text_a = torch.randn(1, 4, d)
    text_b = torch.randn(1, 4, d)
    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_a, state_token)
    tokens_b, _, _ = _build_modality_batch(image_tokens, text_b, state_token)
    with torch.no_grad():
        _, stats_a = grounding(target_query, tokens_a, pad_mask, modality_ids)
        _, stats_b = grounding(target_query, tokens_b, pad_mask, modality_ids)
    assert torch.allclose(stats_a["attention_weights"], stats_b["attention_weights"])


def test_sharpened_visual_grounding_finite_when_every_image_token_padded() -> None:
    d = 12
    grounding = SharpenedVisualGrounding(d)
    target_query = torch.randn(2, d)
    image_tokens = torch.randn(2, 10, d)
    text_tokens = torch.randn(2, 4, d)
    state_token = torch.randn(2, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)
    pad_mask = pad_mask.clone()
    pad_mask[:, :10] = False  # mask out every image token
    feature, stats = grounding(target_query, tokens, pad_mask, modality_ids)
    assert torch.isfinite(feature).all()
    assert torch.isfinite(stats["attention_weights"]).all()


def test_sharpened_visual_grounding_top_k_zeroes_out_the_rest() -> None:
    """`top_k` must hard-restrict nonzero weight mass to exactly `top_k` image positions."""
    d = 12
    top_k = 3
    grounding = SharpenedVisualGrounding(d, top_k=top_k)
    target_query = torch.randn(2, d)
    image_tokens = torch.randn(2, 20, d)
    text_tokens = torch.randn(2, 4, d)
    state_token = torch.randn(2, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)
    _, stats = grounding(target_query, tokens, pad_mask, modality_ids)
    weights = stats["attention_weights"]
    nonzero_per_row = (weights > 0).sum(dim=1)
    assert torch.all(nonzero_per_row <= top_k)
    assert torch.allclose(stats["topk_mass"], torch.ones(2), atol=1e-5)  # reporting k == top_k here


def test_sharpened_visual_grounding_lower_temperature_is_sharper() -> None:
    """Same logits (fixed query/key projections, same inputs), lower temperature -> lower mean
    entropy (more concentrated attention) -- proves `temperature` actually controls sharpness."""
    d = 12
    torch.manual_seed(0)
    image_tokens = torch.randn(4, 20, d)
    text_tokens = torch.randn(4, 4, d)
    state_token = torch.randn(4, 1, d)
    target_query = torch.randn(4, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)

    sharp = SharpenedVisualGrounding(d, temperature=0.1)
    diffuse = SharpenedVisualGrounding(d, temperature=5.0)
    diffuse.load_state_dict(sharp.state_dict())  # identical projections, temperature is the only diff

    with torch.no_grad():
        _, stats_sharp = sharp(target_query, tokens, pad_mask, modality_ids)
        _, stats_diffuse = diffuse(target_query, tokens, pad_mask, modality_ids)
    assert stats_sharp["entropy"].mean() < stats_diffuse["entropy"].mean()


def test_sharpened_visual_grounding_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SharpenedVisualGrounding(12, temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        SharpenedVisualGrounding(12, top_k=0)


# ---- CoarseTargetHead / FineResidualHead -----------------------------------------------------


def test_coarse_target_head_shape_and_gradient() -> None:
    head = CoarseTargetHead(latent_dim=12, hidden_dim=16)
    grounded_feature = torch.randn(3, 12, requires_grad=True)
    coarse_xyz = head(grounded_feature)
    assert coarse_xyz.shape == (3, 3)
    assert torch.isfinite(coarse_xyz).all()
    coarse_xyz.sum().backward()
    assert grounded_feature.grad is not None
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in head.parameters())


def test_fine_residual_head_shape_gradient_and_bound() -> None:
    max_refine_offset = 0.2
    head = FineResidualHead(latent_dim=12, state_dim=5, hidden_dim=16, max_refine_offset=max_refine_offset)
    grounded_feature = torch.randn(4, 12, requires_grad=True)
    coarse_xyz = torch.randn(4, 3, requires_grad=True)
    current_state = torch.randn(4, 5, requires_grad=True)
    target_query = torch.randn(4, 12, requires_grad=True)

    delta = head(grounded_feature, coarse_xyz, current_state, target_query)
    assert delta.shape == (4, 3)
    assert torch.isfinite(delta).all()
    assert torch.all(delta.abs() < max_refine_offset)

    delta.sum().backward()
    assert grounded_feature.grad is not None and torch.any(grounded_feature.grad != 0)
    assert coarse_xyz.grad is not None and torch.any(coarse_xyz.grad != 0)
    assert current_state.grad is not None and torch.any(current_state.grad != 0)
    assert target_query.grad is not None and torch.any(target_query.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in head.parameters())


def test_fine_residual_head_rejects_non_positive_max_refine_offset() -> None:
    with pytest.raises(ValueError, match="max_refine_offset"):
        FineResidualHead(latent_dim=12, state_dim=5, hidden_dim=16, max_refine_offset=0.0)


def test_fine_residual_head_magnitude_scales_with_max_refine_offset() -> None:
    """A larger `max_refine_offset` must allow a strictly larger achievable residual magnitude for
    the same (saturating) raw MLP output -- proves the bound is actually load-bearing, not just
    present as an unused constructor argument."""
    small = FineResidualHead(latent_dim=8, state_dim=4, hidden_dim=8, max_refine_offset=0.1)
    large = FineResidualHead(latent_dim=8, state_dim=4, hidden_dim=8, max_refine_offset=2.0)
    large.load_state_dict(small.state_dict())
    with torch.no_grad():
        for p in large.net[-1].parameters():  # push the final linear layer's output to saturate tanh
            p.mul_(50.0)
        small.net.load_state_dict(large.net.state_dict())

    grounded_feature = torch.randn(2, 8)
    coarse_xyz = torch.randn(2, 3)
    current_state = torch.randn(2, 4)
    target_query = torch.randn(2, 8)
    with torch.no_grad():
        delta_small = small(grounded_feature, coarse_xyz, current_state, target_query)
        delta_large = large(grounded_feature, coarse_xyz, current_state, target_query)
    assert delta_large.abs().mean() > delta_small.abs().mean()


# ---- TemporalActionDecoder conditioning is reused unchanged (use_grounded_grasp_v2) ----------


def test_v3_reuses_use_grounded_grasp_v2_decoder_conditioning_unchanged() -> None:
    """v3's own module docstring's central claim: `TemporalActionDecoder` cannot tell v2's and
    v3's `grounded_target_feature`/`target_xyz` apart -- it is built with the SAME
    `use_grounded_grasp_v2=True` flag, no new decoder-side branch at all."""
    policy = make_policy(architecture=ARCH)
    assert policy.decoder.use_grounded_grasp_v2 is True
    assert policy.decoder.use_target_xyz is True
    assert hasattr(policy.decoder, "grounded_feature_encoder")
    assert hasattr(policy.decoder, "phase_encoder")
    assert hasattr(policy.decoder, "target_xyz_encoder")


def test_final_target_xyz_conditioning_changes_decoder_output() -> None:
    """Same sanity check as v2's `test_grounded_target_feature_conditioning_changes_decoder_output`
    -- proves the decoder's existing conditioning path is actually sensitive to what v3 feeds it."""
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM, state_dim=5, latent_dim=12, hidden_dim=16, horizon=4,
        num_layers=1, num_heads=2, ffn_dim=16, dropout=0.0,
        use_target_xyz=True, use_grounded_grasp_v2=True,
    )
    decoder.eval()
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    feature = torch.randn(2, 12)
    phase = torch.zeros(2, dtype=torch.long)
    final_a = torch.randn(2, 3)
    final_b = final_a + 5.0
    with torch.no_grad():
        out_a = decoder(latent_tokens, None, state, target_xyz=final_a, grounded_target_feature=feature, grasp_phase=phase)
        out_b = decoder(latent_tokens, None, state, target_xyz=final_b, grounded_target_feature=feature, grasp_phase=phase)
    assert not torch.allclose(out_a, out_b)


# ---- policy-level wiring: query reuse, coarse/final split, decoder connection ----------------


def test_grounded_grasp_v3_builds_expected_modules_only() -> None:
    policy = make_policy(architecture=ARCH)
    assert hasattr(policy, "target_query_extractor")
    assert isinstance(policy.target_query_extractor, TargetQueryExtractor)
    assert hasattr(policy, "sharpened_visual_grounding")
    assert hasattr(policy, "coarse_target_head")
    assert hasattr(policy, "fine_residual_head")
    assert not hasattr(policy, "visual_grounding_cross_attn")  # v2's module, not v3's
    assert not hasattr(policy, "target_point_head")  # v1/v2's module, not v3's
    assert not hasattr(policy, "language_grounded_target_pooling")
    assert not hasattr(policy, "modality_pool_projection")  # subgoal-only


def test_policy_final_target_equals_coarse_plus_residual() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    batch = make_grounded_batch()
    with torch.no_grad():
        _, metrics = policy.plan_action_chunk(batch)
    assert torch.allclose(
        metrics["predicted_coarse_target_xyz"] + metrics["predicted_residual_xyz"],
        metrics["predicted_target_xyz"],
        atol=1e-6,
    )


def test_policy_plan_action_chunk_actually_uses_coarse_and_fine_heads() -> None:
    """Zeroing `coarse_target_head`'s parameters must change the predicted target AND the action
    chunk -- proves the coarse head's prediction actually reaches the decoder (via `final_target_xyz`
    and `grounded_target_feature`), not just that the decoder can accept these arguments."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    batch = make_grounded_batch()
    with torch.no_grad():
        actions_before, metrics_before = policy.plan_action_chunk(batch)
        for p in policy.coarse_target_head.parameters():
            p.zero_()
        actions_after, metrics_after = policy.plan_action_chunk(batch)
    assert not torch.equal(metrics_before["predicted_target_xyz"], metrics_after["predicted_target_xyz"])
    assert not torch.allclose(actions_before, actions_after)


def test_policy_plan_action_chunk_actually_uses_fine_residual_head() -> None:
    """Zeroing `fine_residual_head`'s LAST layer (forcing its residual output to exactly zero)
    must make `final_target_xyz` equal `coarse_xyz` exactly, and perturbing it away from zero must
    change `final_target_xyz` again -- proves the residual head's output actually reaches
    `final_target_xyz`, not just that it's computed and discarded."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    batch = make_grounded_batch()
    with torch.no_grad():
        for p in policy.fine_residual_head.net[-1].parameters():
            p.zero_()
        _, metrics_zeroed = policy.plan_action_chunk(batch)
    assert torch.allclose(metrics_zeroed["predicted_residual_xyz"], torch.zeros_like(metrics_zeroed["predicted_residual_xyz"]))
    assert torch.allclose(metrics_zeroed["predicted_coarse_target_xyz"], metrics_zeroed["predicted_target_xyz"], atol=1e-6)

    with torch.no_grad():
        for p in policy.fine_residual_head.net[-1].parameters():
            p.add_(1.0)
        _, metrics_after = policy.plan_action_chunk(batch)
    assert not torch.allclose(metrics_after["predicted_residual_xyz"], torch.zeros_like(metrics_after["predicted_residual_xyz"]))


def test_policy_predicted_target_reacts_to_text_and_image_changes() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    d = policy._multimodal_latent_dim()

    def predict(image: torch.Tensor, text: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tokens, pad_mask, modality_ids = _build_modality_batch(image, text, state)
        with torch.no_grad():
            query = policy.target_query_extractor(tokens, pad_mask, modality_ids)
            grounded, _ = policy.sharpened_visual_grounding(query, tokens, pad_mask, modality_ids)
            coarse = policy.coarse_target_head(grounded)
            current_state = torch.zeros(image.shape[0], ENCODED_DIM)
            residual = policy.fine_residual_head(grounded, coarse, current_state, query)
            return coarse + residual

    image = torch.randn(1, 20, d)
    state = torch.randn(1, 1, d)
    text_a = torch.randn(1, 4, d)
    text_b = torch.randn(1, 4, d)
    assert not torch.allclose(predict(image, text_a, state), predict(image, text_b, state))

    image_b = torch.randn(1, 20, d)
    assert not torch.allclose(predict(image, text_a, state), predict(image_b, text_a, state))


# ---- training: coarse/final loss split, masking, lambda weighting ---------------------------


def test_grounded_grasp_v3_policy_trains_with_split_target_losses() -> None:
    policy = make_policy(architecture=ARCH)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # pre-grasp (state convention: 0=open)
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0  # guarantee a transition inside the horizon
    loss, metrics = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {
        "loss", "loss_pos", "loss_rot", "loss_grip", "loss_subgoal", "loss_smooth",
        "loss_target", "loss_target_coarse", "loss_target_final",
        "residual_xyz_magnitude_mean", "visual_selection_entropy_mean",
        "visual_selection_top1_mass_mean", "visual_selection_topk_mass_mean",
    } <= metrics.keys()
    assert metrics["loss_target_coarse"] > 0
    assert metrics["loss_target_final"] > 0
    assert metrics["loss_target"] == pytest.approx(metrics["loss_target_coarse"] + metrics["loss_target_final"], abs=1e-6)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.coarse_target_head.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.fine_residual_head.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.target_query_extractor.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.sharpened_visual_grounding.parameters())


def test_grounded_grasp_v3_loss_target_zero_when_no_valid_transition_in_horizon() -> None:
    policy = make_policy(architecture=ARCH)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0  # never closes
    _, metrics = policy(batch)
    assert metrics["loss_target_coarse"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss_target_final"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["loss_target"] == pytest.approx(0.0, abs=1e-8)


def test_grounded_grasp_v3_lambda_weights_scale_each_loss_independently() -> None:
    policy_default = make_policy(architecture=ARCH, lambda_target_coarse=1.0, lambda_target_refine=1.0)
    policy_coarse_only = make_policy(architecture=ARCH, lambda_target_coarse=1.0, lambda_target_refine=0.0)
    policy_coarse_only.load_state_dict(policy_default.state_dict())
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0
    with torch.no_grad():
        loss_default, metrics = policy_default(batch)
        loss_coarse_only, _ = policy_coarse_only(batch)
    expected_diff = metrics["loss_target_final"]  # lambda_target_refine going 1.0 -> 0.0 removes exactly this
    assert loss_default.item() == pytest.approx(loss_coarse_only.item() + expected_diff, abs=1e-6)


def test_config_rejects_negative_target_lambdas_and_invalid_v3_fields() -> None:
    with pytest.raises(ValueError, match="lambda_target_coarse"):
        make_config(architecture=ARCH, lambda_target_coarse=-1.0)
    with pytest.raises(ValueError, match="lambda_target_refine"):
        make_config(architecture=ARCH, lambda_target_refine=-1.0)
    with pytest.raises(ValueError, match="grounded_grasp_v3_temperature"):
        make_config(architecture=ARCH, grounded_grasp_v3_temperature=0.0)
    with pytest.raises(ValueError, match="grounded_grasp_v3_top_k"):
        make_config(architecture=ARCH, grounded_grasp_v3_top_k=0)
    with pytest.raises(ValueError, match="grounded_grasp_v3_max_refine_offset"):
        make_config(architecture=ARCH, grounded_grasp_v3_max_refine_offset=0.0)


# ---- reactive close: uses final_target_xyz, same threshold/latch/queue-clear behavior as v2 ---


def test_v3_proximity_close_trigger_fires_and_flips_phase() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    assert policy._v3_grasp_phase == 0
    policy._v3_last_target_xyz = torch.zeros(1, 3)
    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 0.01  # ~1.7cm away, within the 3cm default threshold
    action = torch.full((1, ACTION_DIM), 9.0)

    out = policy._grounded_grasp_v3_reactive_close(action, current_state)
    assert policy._v3_close_triggered is True
    assert policy._v3_grasp_phase == 1
    assert out[0, GRIPPER_INDEX_RAW].item() != pytest.approx(9.0)
    for i in range(ACTION_DIM):
        if i != GRIPPER_INDEX_RAW:
            assert out[0, i].item() == pytest.approx(9.0)


def test_v3_proximity_close_trigger_does_not_fire_outside_threshold() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._v3_last_target_xyz = torch.zeros(1, 3)
    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 1.0  # 1m away
    action = torch.full((1, ACTION_DIM), 9.0)

    out = policy._grounded_grasp_v3_reactive_close(action, current_state)
    assert policy._v3_close_triggered is False
    assert policy._v3_grasp_phase == 0
    assert torch.equal(out, action)


def test_v3_reactive_close_uses_final_target_not_coarse_only() -> None:
    """`_v3_last_target_xyz` (cached by `_plan_temporal_decoder`, read by
    `_grounded_grasp_v3_reactive_close`) must be the FINAL target (coarse + residual), matching
    `plan_action_chunk`'s own `predicted_target_xyz` metric exactly -- not the coarse-only
    estimate, which (with a randomly-initialized, near-certainly-nonzero residual head) differs
    from it."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    batch = make_grounded_batch(batch_size=1)
    with torch.no_grad():
        _, metrics = policy.plan_action_chunk(batch)
    assert policy._v3_last_target_xyz is not None
    assert torch.allclose(policy._v3_last_target_xyz, metrics["predicted_target_xyz"])
    assert not torch.allclose(policy._v3_last_target_xyz, metrics["predicted_coarse_target_xyz"])


def test_v3_one_shot_close_guard_does_not_reopen_or_flip_phase_back() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._v3_last_target_xyz = torch.zeros(1, 3)

    near_state = torch.zeros(1, STATE_DIM)
    near_state[0, :3] = 0.01
    action = torch.full((1, ACTION_DIM), 9.0)
    triggered_output = policy._grounded_grasp_v3_reactive_close(action.clone(), near_state)
    assert policy._v3_close_triggered is True
    assert policy._v3_grasp_phase == 1
    closed_value = triggered_output[0, GRIPPER_INDEX_RAW].item()

    far_state = torch.zeros(1, STATE_DIM)
    far_state[0, :3] = 5.0
    later_output = policy._grounded_grasp_v3_reactive_close(action.clone(), far_state)
    assert policy._v3_close_triggered is True
    assert policy._v3_grasp_phase == 1
    assert later_output[0, GRIPPER_INDEX_RAW].item() == pytest.approx(closed_value)


def test_v3_reset_clears_phase_trigger_and_cached_target() -> None:
    policy = make_policy(architecture=ARCH)
    policy._v3_grasp_phase = 1
    policy._v3_close_triggered = True
    policy._v3_last_target_xyz = torch.zeros(1, 3)
    policy.reset()
    assert policy._v3_grasp_phase == 0
    assert policy._v3_close_triggered is False
    assert policy._v3_last_target_xyz is None


def test_v3_select_action_applies_reactive_close_when_near_target() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    batch = make_grounded_batch(batch_size=1)
    batch[OBS_STATE][:, :3] = 0.0
    action = policy.select_action(batch)
    action = action.clone()
    action[0, GRIPPER_INDEX_RAW] = 5.0
    policy._v3_last_target_xyz = torch.zeros(1, 3)
    out = policy._grounded_grasp_v3_reactive_close(action, policy._current_state(batch))
    assert policy._v3_close_triggered is True
    assert out[0, GRIPPER_INDEX_RAW].item() != pytest.approx(action[0, GRIPPER_INDEX_RAW].item())


def test_v3_reactive_close_is_noop_before_any_chunk_planned() -> None:
    policy = make_policy(architecture=ARCH)
    policy.reset()
    assert policy._v3_last_target_xyz is None
    action = torch.full((1, ACTION_DIM), 3.0)
    out = policy._grounded_grasp_v3_reactive_close(action, torch.zeros(1, STATE_DIM))
    assert torch.equal(out, action)
    assert policy._v3_close_triggered is False
    assert policy._v3_grasp_phase == 0


# ---- no regression on v1/v2/canonical architectures ------------------------------------------


def test_other_architectures_never_build_grounded_grasp_v3_modules() -> None:
    for arch in (
        "temporal_decoder",
        "temporal_decoder_subgoal",
        "temporal_decoder_grounded_grasp",
        "temporal_decoder_instruction",
        "temporal_decoder_text_crossattn",
        "temporal_decoder_grounded_grasp_v2",
    ):
        policy = make_policy(architecture=arch)
        assert not hasattr(policy, "sharpened_visual_grounding")
        assert not hasattr(policy, "coarse_target_head")
        assert not hasattr(policy, "fine_residual_head")


def test_v2_still_builds_its_own_modules_unaffected_by_v3() -> None:
    """Regression guard: v2's own module set (and NOT v3's) is exactly as before."""
    policy = make_policy(architecture="temporal_decoder_grounded_grasp_v2")
    assert hasattr(policy, "target_query_extractor")
    assert hasattr(policy, "visual_grounding_cross_attn")
    assert hasattr(policy, "target_point_head")
    assert not hasattr(policy, "sharpened_visual_grounding")
    assert not hasattr(policy, "coarse_target_head")
    assert not hasattr(policy, "fine_residual_head")
    assert policy.decoder.use_grounded_grasp_v2 is True


def test_v2_policy_reactive_close_and_phase_state_unaffected_by_v3_additions() -> None:
    """v2's own reactive-close attributes/behavior must be untouched by adding v3 -- exercises the
    same scenario as `test_grounded_grasp_v2.py::test_v2_proximity_close_trigger_fires_and_flips_phase`."""
    policy = make_policy(architecture="temporal_decoder_grounded_grasp_v2")
    policy.eval()
    policy.reset()
    policy._v2_last_target_xyz = torch.zeros(1, 3)
    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 0.01
    action = torch.full((1, ACTION_DIM), 9.0)
    out = policy._grounded_grasp_v2_reactive_close(action, current_state)
    assert policy._v2_close_triggered is True
    assert policy._v2_grasp_phase == 1
    assert out[0, GRIPPER_INDEX_RAW].item() != pytest.approx(9.0)
    assert not hasattr(policy, "_v3_grasp_phase") or policy._v3_grasp_phase == 0


def test_canonical_temporal_decoder_unaffected_by_v3_additions() -> None:
    policy = make_policy(architecture="temporal_decoder")
    loss, metrics = policy(make_grounded_batch())
    assert metrics["loss_target"] == 0.0
    assert metrics["loss_target_coarse"] == 0.0
    assert metrics["loss_target_final"] == 0.0
    assert loss.item() == pytest.approx(metrics["loss"], abs=1e-6)
    assert not hasattr(policy, "sharpened_visual_grounding")
    assert policy.decoder.use_grounded_grasp_v2 is False


def test_config_grounded_grasp_v3_is_a_registered_architecture() -> None:
    config = make_config(architecture=ARCH)
    assert config.architecture == ARCH
