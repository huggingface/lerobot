from __future__ import annotations

from pathlib import Path

import draccus
import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.temporal_decoder import TemporalActionDecoder
from lerobot.policies.safediff_vla.utils import masked_mse, pad_or_crop_horizon, pad_or_crop_mask
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.safediff_vla.testing_utils import NominalCallForbiddenBackbone, TinyBackbone
from tests.utils import require_cuda

ACTION_DIM = 7  # SafeDiff-VLA's decomposed loss hard-codes 3 position + 3 orientation + 1 gripper.
STATE_DIM = 7  # temporal_decoder's sin/cos rotation encoding requires state to share action's layout.
ENCODED_DIM = 10  # xyz(3) + sin/cos(6) + gripper(1) -- see `rotation_encoding.py`.

# Real checkpoint from a completed temporal_decoder run, used by the checkpoint-load sanity test
# (section 9.E). Not committed to git (outputs/ is a local, gitignored directory) -- the test
# skips gracefully if it isn't present.
REAL_CHECKPOINT_DIR = Path(
    "outputs/train/safediff_vla_temporal_decoder_frac30_150k/checkpoints/150000/pretrained_model"
)


def make_config(**overrides) -> SafeDiffVLAConfig:
    values = {
        "architecture": "temporal_decoder",
        "device": "cpu",
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "action_horizon": 4,
        "execute_horizon": 2,
        "latent_dim": 8,
        "state_head_hidden_dim": 12,
        "decoder_hidden_dim": 16,
        "decoder_num_layers": 1,
        "decoder_num_heads": 2,
        "decoder_ffn_dim": 16,
        "decoder_dropout": 0.0,
    }
    values.update(overrides)
    return SafeDiffVLAConfig(**values)


def make_batch(batch_size: int = 2, with_subgoal_label: bool = True) -> dict[str, torch.Tensor]:
    batch = {OBS_STATE: torch.randn(batch_size, STATE_DIM), ACTION: torch.randn(batch_size, 4, ACTION_DIM)}
    if with_subgoal_label:
        batch["observation.subgoal_state"] = torch.randn(batch_size, STATE_DIM)
    return batch


def make_policy(backbone: torch.nn.Module | None = None, **overrides) -> SafeDiffVLAPolicy:
    config = make_config(**overrides)
    return SafeDiffVLAPolicy(config, backbone=backbone or TinyBackbone(4, ACTION_DIM, state_dim=STATE_DIM))


# ---- config -------------------------------------------------------------------------------


def test_config_serialization(tmp_path) -> None:
    config = make_config()
    config.save_pretrained(tmp_path)
    restored = SafeDiffVLAConfig.from_pretrained(tmp_path)
    assert isinstance(restored, SafeDiffVLAConfig)
    assert restored.action_horizon == config.action_horizon


def test_config_and_registration() -> None:
    config = make_config()
    assert config.type == "safediff_vla"
    assert isinstance(make_policy_config("safediff_vla", device="cpu"), SafeDiffVLAConfig)
    assert get_policy_class("safediff_vla") is SafeDiffVLAPolicy


def test_architecture_defaults_to_temporal_decoder() -> None:
    assert SafeDiffVLAConfig(device="cpu").architecture == "temporal_decoder"


def test_invalid_architecture_rejected() -> None:
    with pytest.raises(ValueError, match="architecture"):
        make_config(architecture="not_a_real_architecture")


def test_legacy_diffusion_is_no_longer_a_valid_main_path_architecture() -> None:
    """`legacy_diffusion` moved to its own policy type (`safediff_vla_legacy`, see
    `legacy/test_legacy_diffusion.py`) -- the main config must reject it."""
    with pytest.raises(ValueError, match="architecture"):
        make_config(architecture="legacy_diffusion")


def test_smolvla_finetune_requires_unfrozen_backbone() -> None:
    with pytest.raises(ValueError, match="freeze_backbone"):
        make_config(architecture="smolvla_finetune", freeze_backbone=True)


def test_decoder_hidden_dim_must_divide_num_heads() -> None:
    with pytest.raises(ValueError, match="decoder_hidden_dim"):
        make_config(decoder_hidden_dim=17, decoder_num_heads=2)


def test_action_dim_must_be_seven() -> None:
    """The decomposed loss (position/orientation/gripper) hard-codes the LIBERO 7-dim layout."""
    config = make_config(output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))})
    with pytest.raises(ValueError, match="7"):
        SafeDiffVLAPolicy(config, backbone=TinyBackbone(4, 6))


def test_completion_threshold_decodes_from_cli() -> None:
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--policy.type=safediff_vla",
            "--policy.architecture=temporal_decoder",
            "--policy.completion_threshold=0.1",
            "--dataset.repo_id=VLA/smolvla_libero",
        ],
    )
    assert isinstance(config.policy, SafeDiffVLAConfig)
    assert config.policy.completion_threshold == 0.1


def test_subgoal_labels_path_decodes_from_cli() -> None:
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--policy.type=safediff_vla",
            "--policy.architecture=temporal_decoder_subgoal",
            "--policy.subgoal_labels_path=outputs/data/vlabench_subgoal_labels/labels.parquet",
            "--dataset.repo_id=VLA/smolvla_libero",
        ],
    )
    assert isinstance(config.policy, SafeDiffVLAConfig)
    assert config.policy.subgoal_labels_path == "outputs/data/vlabench_subgoal_labels/labels.parquet"


# ---- temporal_decoder / temporal_decoder_subgoal -------------------------------------------


def test_temporal_action_decoder_shapes() -> None:
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM,
        state_dim=5,
        latent_dim=12,
        hidden_dim=16,
        horizon=4,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    actions = decoder(latent_tokens, None, state)
    assert actions.shape == (2, 4, ACTION_DIM)
    assert torch.isfinite(actions).all()


def test_temporal_action_decoder_with_subgoal() -> None:
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM,
        state_dim=5,
        latent_dim=12,
        hidden_dim=16,
        horizon=4,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        use_subgoal=True,
    )
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    subgoal_state = torch.randn(2, 5)  # [B, state_dim] -- a single target, not a [B, H, state_dim]
    actions = decoder(latent_tokens, None, state, subgoal_state)
    assert actions.shape == (2, 4, ACTION_DIM)
    with pytest.raises(ValueError, match="use_subgoal"):
        decoder(latent_tokens, None, state, None)


def test_temporal_decoder_policy_trains_and_freezes_backbone() -> None:
    policy = make_policy(architecture="temporal_decoder")
    loss, metrics = policy(make_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"loss", "loss_pos", "loss_rot", "loss_grip", "loss_subgoal", "loss_smooth"} <= metrics.keys()
    assert metrics["loss_subgoal"] == 0  # no subgoal head in this architecture
    loss.backward()
    assert all(parameter.grad is None for parameter in policy.backbone.parameters())
    assert any(parameter.grad is not None for parameter in policy.decoder.parameters())


def test_temporal_decoder_subgoal_regresses_subgoal_label() -> None:
    policy = make_policy(architecture="temporal_decoder_subgoal")
    loss, metrics = policy(make_batch(with_subgoal_label=True))
    assert torch.isfinite(loss)
    assert metrics["loss_subgoal"] > 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.subgoal_state_predictor.parameters())


def test_temporal_decoder_smoothness_regularizer_is_off_by_default() -> None:
    policy = make_policy(architecture="temporal_decoder")
    assert policy.config.lambda_smooth == 0.0
    _, metrics = policy(make_batch())
    assert metrics["loss_smooth"] == 0.0


def test_temporal_decoder_plan_action_chunk_shape() -> None:
    policy = make_policy(architecture="temporal_decoder")
    actions, metrics = policy.plan_action_chunk(make_batch())
    assert actions.shape == (2, 4, ACTION_DIM)
    assert "predicted_subgoal_state" not in metrics  # no subgoal signal for this architecture


def test_temporal_decoder_select_action_never_gates() -> None:
    """No subgoal signal exists for plain `temporal_decoder`, so every chunk boundary should
    commit a fresh full-length chunk regardless of `completion_threshold`."""
    policy = make_policy(architecture="temporal_decoder", completion_threshold=0.0, max_replan_retries=2)
    batch = make_batch()
    policy.select_action(batch)
    while policy._executor._action_queue:
        policy.select_action(batch)
    policy.select_action(batch)
    assert policy._executor._replan_retries == 0
    assert len(policy._executor._action_queue) == policy.config.execute_horizon - 1


def test_select_action_queue_and_reset() -> None:
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch()
    first = policy.select_action(batch)
    assert first.shape == (2, ACTION_DIM)
    assert len(policy._executor._action_queue) == 1
    policy.reset()
    assert len(policy._executor._action_queue) == 0
    assert policy._executor._pending_target_state is None
    assert policy._executor._last_gap is None


# ---- sanity tests (section 9) ---------------------------------------------------------------


def test_shape_contract_latent_state_actions() -> None:
    """A. shape test: latent tokens, current state, and the final action chunk all have the
    shapes the module docstring claims."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch()
    latent_tokens, _ = policy._encode_multimodal_latent(batch)
    current_state = policy._current_state(batch)
    assert latent_tokens.shape == (2, policy.backbone.num_latent_tokens, policy.backbone.multimodal_latent_dim)
    assert current_state.shape == (2, STATE_DIM)
    actions, _ = policy.plan_action_chunk(batch)
    assert actions.shape == (2, policy.config.action_horizon, ACTION_DIM)


def test_temporal_mixing_self_attention_couples_horizon_positions() -> None:
    """B. temporal mixing test. `TemporalActionDecoder` has no Conv1D mixer (that's
    `ConditionalDiffusionPlanner`, legacy-only) -- horizon-axis mixing here is
    `nn.TransformerDecoder`'s self-attention among the `horizon` query positions. Perturb only
    the learned query at horizon position 0 (latent tokens / current_state / subgoal are
    identical, shared inputs across every position) and confirm >=1 *other* position's output
    also changes -- the only way that can happen is self-attention across positions."""
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM,
        state_dim=5,
        latent_dim=12,
        hidden_dim=16,
        horizon=6,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    decoder.eval()
    latent_tokens = torch.randn(1, 6, 12)
    state = torch.randn(1, 5)
    with torch.no_grad():
        baseline = decoder(latent_tokens, None, state)
        decoder.action_queries.data[0] += 5.0
        perturbed = decoder(latent_tokens, None, state)
    changed = (baseline - perturbed).abs().sum(dim=-1).squeeze(0) > 1e-4
    assert changed[1:].any(), "perturbing one horizon position's query should propagate to others"


def test_current_state_conditioning_changes_output() -> None:
    """C. current-state conditioning test: same latent tokens, different current_state -> the
    action chunk changes."""
    policy = make_policy(architecture="temporal_decoder")
    policy.eval()
    batch = make_batch()
    latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(batch)
    state_a = policy._encode_state(policy._current_state(batch))
    state_b = policy._encode_state(policy._current_state(batch) + 1.0)
    with torch.no_grad():
        out_a = policy.decoder(latent_tokens, latent_pad_mask, state_a)
        out_b = policy.decoder(latent_tokens, latent_pad_mask, state_b)
    assert not torch.allclose(out_a, out_b)


def test_loss_decomposition_matches_weighted_sum() -> None:
    """D. loss decomposition test (part 1): total loss equals the dim-count-normalized weighted
    sum of its logged components exactly (see `configuration_safediff_vla.py`'s `lambda_pos`
    comment for why this isn't a plain unweighted sum of the three component MSEs)."""
    policy = make_policy(architecture="temporal_decoder")
    loss, metrics = policy(make_batch())
    position_dim, rotation_dim, gripper_dim = 3, 6, 1
    action_dim = position_dim + rotation_dim + gripper_dim
    expected_action = (
        policy.config.lambda_pos * position_dim * metrics["loss_pos"]
        + policy.config.lambda_rot * rotation_dim * metrics["loss_rot"]
        + policy.config.lambda_grip * gripper_dim * metrics["loss_grip"]
    ) / action_dim
    expected = (
        expected_action
        + policy.config.lambda_subgoal * metrics["loss_subgoal"]
        + policy.config.lambda_smooth * metrics["loss_smooth"]
    )
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_decomposed_loss_equals_old_pooled_mse_at_default_weights() -> None:
    """D. loss decomposition test (part 3), exact-equivalence: at the default weights
    (lambda_pos=lambda_rot=lambda_grip=1.0, lambda_smooth=0.0, and no subgoal contribution for
    plain `temporal_decoder`), the decomposed loss must be numerically identical to a single
    pooled `F.mse_loss(pred_actions, clean)` over all 10 *encoded* action dims -- this is the
    whole point of the dim-count normalization, not just "close"."""
    policy = make_policy(architecture="temporal_decoder")
    assert (policy.config.lambda_pos, policy.config.lambda_rot, policy.config.lambda_grip) == (1.0, 1.0, 1.0)
    assert policy.config.lambda_smooth == 0.0
    batch = make_batch(with_subgoal_label=False)

    latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(batch)
    current_state = policy._encode_state(policy._current_state(batch))
    with torch.no_grad():
        pred_actions = policy.decoder(latent_tokens, latent_pad_mask, current_state)
    clean_raw = pad_or_crop_horizon(batch[ACTION], policy.config.action_horizon)
    clean = policy._encode_action_target(clean_raw)
    expected_pooled_mse = F.mse_loss(pred_actions, clean).item()

    with torch.no_grad():
        loss, _ = policy(batch)
    assert loss.item() == pytest.approx(expected_pooled_mse, abs=1e-6)


# ---- action_is_pad masking (episode-end chunk padding must not leak into the loss) --------


def test_masked_mse_matches_f_mse_loss_when_all_valid() -> None:
    """`masked_mse` with an all-`True` mask must exactly reproduce plain `F.mse_loss` -- the
    no-padding case must be numerically unaffected by the masking fix."""
    pred = torch.randn(3, 5, 4)
    target = torch.randn(3, 5, 4)
    valid_mask = torch.ones(3, 5, dtype=torch.bool)
    assert masked_mse(pred, target, valid_mask).item() == pytest.approx(F.mse_loss(pred, target).item(), abs=1e-6)


def test_masked_mse_ignores_changes_at_padded_timesteps() -> None:
    """Changing `pred` only at timesteps marked padded (`valid_mask=False`) must not move the
    loss at all -- those steps are excluded, not just down-weighted."""
    pred = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    valid_mask = torch.tensor([[True, True, False, False], [True, False, False, True]])
    baseline = masked_mse(pred, target, valid_mask)

    pred_perturbed = pred.clone()
    pred_perturbed[0, 2:] += 1000.0  # both padded steps for row 0
    pred_perturbed[1, 1:3] += 1000.0  # both padded steps for row 1
    perturbed = masked_mse(pred_perturbed, target, valid_mask)
    assert perturbed.item() == pytest.approx(baseline.item(), abs=1e-6)


def test_masked_mse_reacts_to_changes_at_valid_timesteps() -> None:
    """Changing `pred` at a valid (unmasked) timestep must move the loss."""
    pred = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    valid_mask = torch.tensor([[True, True, False, False], [True, False, False, True]])
    baseline = masked_mse(pred, target, valid_mask)

    pred_perturbed = pred.clone()
    pred_perturbed[0, 0] += 1000.0  # a valid step for row 0
    perturbed = masked_mse(pred_perturbed, target, valid_mask)
    assert perturbed.item() != pytest.approx(baseline.item(), abs=1e-3)


def test_masked_mse_all_padded_is_safe_zero_not_nan() -> None:
    """An all-`False` mask (every timestep padded) must return a finite `0`, not `0/0` NaN or
    Inf -- this can happen for a very short episode where `action_horizon` exceeds its length."""
    pred = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    valid_mask = torch.zeros(2, 4, dtype=torch.bool)
    loss = masked_mse(pred, target, valid_mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_pad_or_crop_mask_crops_and_pads_like_pad_or_crop_horizon() -> None:
    mask = torch.tensor([[True, False, True]])
    assert pad_or_crop_mask(mask, 2).tolist() == [[True, False]]
    padded = pad_or_crop_mask(mask, 5)
    assert padded.tolist() == [[True, False, True, True, True]]  # new steps default to padded/excluded


def test_forward_temporal_decoder_no_padding_matches_old_pooled_mse() -> None:
    """Integration: a batch with `action_is_pad` present but entirely `False` (nothing padded)
    must give numerically the same loss as before this fix (the pre-existing exact-equivalence
    test above), proving the masking fix is a no-op for padding-free batches."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch(with_subgoal_label=False)
    batch["action_is_pad"] = torch.zeros(batch[ACTION].shape[0], batch[ACTION].shape[1], dtype=torch.bool)

    latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(batch)
    current_state = policy._encode_state(policy._current_state(batch))
    with torch.no_grad():
        pred_actions = policy.decoder(latent_tokens, latent_pad_mask, current_state)
    clean_raw = pad_or_crop_horizon(batch[ACTION], policy.config.action_horizon)
    clean = policy._encode_action_target(clean_raw)
    expected_pooled_mse = F.mse_loss(pred_actions, clean).item()

    with torch.no_grad():
        loss, _ = policy(batch)
    assert loss.item() == pytest.approx(expected_pooled_mse, abs=1e-6)


def test_forward_temporal_decoder_excludes_padded_timesteps_from_loss() -> None:
    """Integration: marking a relative timestep as padded and replacing its GT target with
    wildly different values must not move the loss -- proves `action_is_pad` actually reaches
    `_forward_temporal_decoder` and is applied, not just supported by the standalone util."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch(with_subgoal_label=False)
    action_is_pad = torch.zeros(batch[ACTION].shape[0], batch[ACTION].shape[1], dtype=torch.bool)
    action_is_pad[:, -1] = True  # last relative timestep is a repeated/padded "ghost" target
    batch["action_is_pad"] = action_is_pad

    with torch.no_grad():
        baseline_loss, _ = policy(batch)

    batch_perturbed = dict(batch)
    batch_perturbed[ACTION] = batch[ACTION].clone()
    batch_perturbed[ACTION][:, -1] += 1000.0  # garbage GT only at the padded step
    with torch.no_grad():
        perturbed_loss, _ = policy(batch_perturbed)

    assert perturbed_loss.item() == pytest.approx(baseline_loss.item(), abs=1e-5)


def test_forward_temporal_decoder_all_padded_batch_is_finite() -> None:
    """A batch where every timestep is padded (e.g. an episode shorter than `action_horizon`)
    must not blow up the training loop with NaN/Inf."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch(with_subgoal_label=False)
    batch["action_is_pad"] = torch.ones(batch[ACTION].shape[0], batch[ACTION].shape[1], dtype=torch.bool)
    with torch.no_grad():
        loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    for k in ("loss_pos", "loss_rot", "loss_grip"):
        assert metrics[k] == pytest.approx(0.0, abs=1e-8)


def test_loss_decomposition_slices_are_position_orientation_gripper() -> None:
    """D. loss decomposition test (part 2): loss_pos/loss_rot/loss_grip are computed from exactly
    the [:3] / [3:9] / [9:10] *encoded* action slices (xyz / sin-cos rotation / gripper)."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch()
    latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(batch)
    current_state = policy._encode_state(policy._current_state(batch))
    with torch.no_grad():
        pred_actions = policy.decoder(latent_tokens, latent_pad_mask, current_state)
    clean_raw = pad_or_crop_horizon(batch[ACTION], policy.config.action_horizon)
    clean = policy._encode_action_target(clean_raw)
    expected_pos = F.mse_loss(pred_actions[..., :3], clean[..., :3]).item()
    expected_rot = F.mse_loss(pred_actions[..., 3:9], clean[..., 3:9]).item()
    expected_grip = F.mse_loss(pred_actions[..., 9:10], clean[..., 9:10]).item()
    with torch.no_grad():
        _, metrics = policy(batch)
    assert metrics["loss_pos"] == pytest.approx(expected_pos, abs=1e-6)
    assert metrics["loss_rot"] == pytest.approx(expected_rot, abs=1e-6)
    assert metrics["loss_grip"] == pytest.approx(expected_grip, abs=1e-6)


def test_deterministic_inference_same_input_twice() -> None:
    """F. deterministic inference test: eval mode, same input called twice -> identical output."""
    policy = make_policy(architecture="temporal_decoder")
    policy.eval()
    batch = make_batch()
    actions_1, _ = policy.plan_action_chunk(batch)
    actions_2, _ = policy.plan_action_chunk(batch)
    assert torch.equal(actions_1, actions_2)


@pytest.mark.parametrize("architecture", ["temporal_decoder", "temporal_decoder_subgoal"])
def test_temporal_decoder_never_calls_nominal_action_head(architecture: str) -> None:
    """G. nominal independence test: neither `forward()` nor `plan_action_chunk()` ever call the
    backbone's own `predict_action_chunk` (SmolVLA's nominal action head)."""
    subgoal_kwargs = {"subgoal_labels_path": None} if architecture == "temporal_decoder_subgoal" else {}
    config = make_config(architecture=architecture, **subgoal_kwargs)
    policy = SafeDiffVLAPolicy(config, backbone=NominalCallForbiddenBackbone(4, ACTION_DIM, state_dim=STATE_DIM))
    policy(make_batch())
    policy.plan_action_chunk(make_batch())


@require_cuda
def test_checkpoint_load_state_dict_no_missing_or_unexpected_keys() -> None:
    """E. checkpoint load test, against a real completed `temporal_decoder` run. Skips if that
    local (gitignored) checkpoint directory isn't present."""
    if not REAL_CHECKPOINT_DIR.exists():
        pytest.skip(f"local checkpoint not found: {REAL_CHECKPOINT_DIR}")
    from safetensors.torch import load_file

    config = SafeDiffVLAConfig(
        architecture="temporal_decoder",
        backbone_name="lerobot/smolvla_vlabench",
        device="cuda",
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        action_horizon=50,
        execute_horizon=50,
        decoder_hidden_dim=512,
        decoder_num_layers=4,
        decoder_num_heads=8,
        decoder_ffn_dim=1024,
        decoder_dropout=0.1,
    )
    policy = SafeDiffVLAPolicy(config)
    state_dict = load_file(str(REAL_CHECKPOINT_DIR / "model.safetensors"))
    result = policy.load_state_dict(state_dict, strict=False)
    assert result.missing_keys == [], f"missing keys: {result.missing_keys}"
    assert result.unexpected_keys == [], f"unexpected keys: {result.unexpected_keys}"


# ---- smolvla_nominal / smolvla_finetune (eval-only ablation baselines) ----------------------


def test_smolvla_nominal_has_no_trainable_parameters() -> None:
    policy = make_policy(architecture="smolvla_nominal")
    assert not any(True for _ in policy.get_optim_params())


def test_smolvla_nominal_forward_raises() -> None:
    policy = make_policy(architecture="smolvla_nominal")
    with pytest.raises(NotImplementedError):
        policy(make_batch())


def test_smolvla_nominal_plan_action_chunk_returns_backbone_output() -> None:
    policy = make_policy(architecture="smolvla_nominal")
    batch = make_batch()
    actions, metrics = policy.plan_action_chunk(batch)
    assert actions.shape == (2, 4, ACTION_DIM)
    assert set(metrics) == {"mean_abs_delta_action", "mean_abs_delta2_action"}


def test_smolvla_finetune_has_trainable_parameters() -> None:
    # Unlike smolvla_nominal (frozen, no trainable params), smolvla_finetune requires
    # freeze_backbone=False, so the backbone's own params (all of TinyBackbone here) are trainable.
    policy = make_policy(architecture="smolvla_finetune", freeze_backbone=False)
    assert any(True for _ in policy.get_optim_params())


def test_smolvla_finetune_forward_returns_backbone_loss() -> None:
    policy = make_policy(architecture="smolvla_finetune", freeze_backbone=False)
    loss, metrics = policy(make_batch())
    assert loss.ndim == 0
    assert metrics["loss"] == pytest.approx(loss.item())
    assert "backbone_losses_after_forward" in metrics


def test_smolvla_finetune_forward_backprops_into_backbone() -> None:
    policy = make_policy(architecture="smolvla_finetune", freeze_backbone=False)
    loss, _ = policy(make_batch())
    loss.backward()
    grads = [p.grad for p in policy.backbone.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_smolvla_finetune_plan_action_chunk_matches_nominal_shape() -> None:
    policy = make_policy(architecture="smolvla_finetune", freeze_backbone=False)
    actions, metrics = policy.plan_action_chunk(make_batch())
    assert actions.shape == (2, 4, ACTION_DIM)
    assert set(metrics) == {"mean_abs_delta_action", "mean_abs_delta2_action"}
