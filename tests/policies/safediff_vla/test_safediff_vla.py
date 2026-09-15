from __future__ import annotations

import draccus
import pytest
import torch
from torch import Tensor, nn

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.safediff_vla.diffusion_planner import ConditionalDiffusionPlanner
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.state_predictor import StatePredictor, completion_gap
from lerobot.policies.safediff_vla.temporal_decoder import TemporalActionDecoder
from lerobot.utils.constants import ACTION, OBS_STATE


class TinyBackbone(nn.Module):
    """Serves both the legacy pooled-hook interface (`extract_safediff_features`, used by
    `architecture="legacy_diffusion"`) and the new token-sequence interface
    (`encode_multimodal_latent` / `multimodal_latent_dim`, used by `temporal_decoder*`), plus a
    bare `predict_action_chunk` for `architecture="smolvla_nominal"` — mirroring the three ways
    `SafeDiffVLAPolicy` can talk to a real SmolVLA backbone, without needing one."""

    def __init__(
        self, horizon: int, action_dim: int, feature_dim: int = 12, num_latent_tokens: int = 6
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.safediff_latent_dim = feature_dim
        self.multimodal_latent_dim = feature_dim
        self.num_latent_tokens = num_latent_tokens
        self.projection = nn.Linear(5, feature_dim)
        self.token_projection = nn.Linear(5, feature_dim)
        self.nominal_head = nn.Linear(5, action_dim)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def extract_safediff_features(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        # `SafeDiffVLAPolicy._backbone_outputs` always hands the backbone a plain 2D
        # [B, state_dim] current-state tensor.
        assert batch[OBS_STATE].ndim == 2
        latent = self.projection(batch[OBS_STATE])
        nominal = latent[:, None, : self.action_dim].expand(-1, self.horizon, -1).tanh()
        return nominal, latent

    def encode_multimodal_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        assert batch[OBS_STATE].ndim == 2
        tokens = self.token_projection(batch[OBS_STATE])[:, None, :].expand(-1, self.num_latent_tokens, -1)
        return tokens, None

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        assert batch[OBS_STATE].ndim == 2
        return self.nominal_head(batch[OBS_STATE])[:, None, :].expand(-1, self.horizon, -1).tanh()


def make_config(**overrides) -> SafeDiffVLAConfig:
    values = {
        "architecture": "legacy_diffusion",
        "device": "cpu",
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        "action_horizon": 4,
        "execute_horizon": 2,
        "num_diffusion_steps": 2,
        "latent_dim": 8,
        "planner_hidden_dim": 16,
        "state_head_hidden_dim": 12,
        "timestep_embedding_dim": 8,
        "decoder_hidden_dim": 16,
        "decoder_num_layers": 1,
        "decoder_num_heads": 2,
        "decoder_ffn_dim": 16,
        "decoder_dropout": 0.0,
    }
    values.update(overrides)
    return SafeDiffVLAConfig(**values)


def make_batch(batch_size: int = 2, with_subgoal_label: bool = True) -> dict[str, Tensor]:
    batch = {OBS_STATE: torch.randn(batch_size, 5), ACTION: torch.randn(batch_size, 4, 3)}
    if with_subgoal_label:
        batch["observation.subgoal_state"] = torch.randn(batch_size, 5)
    return batch


def make_policy(**overrides) -> SafeDiffVLAPolicy:
    config = make_config(**overrides)
    return SafeDiffVLAPolicy(config, backbone=TinyBackbone(4, 3))


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


def test_decoder_hidden_dim_must_divide_num_heads() -> None:
    with pytest.raises(ValueError, match="decoder_hidden_dim"):
        make_config(architecture="temporal_decoder", decoder_hidden_dim=17, decoder_num_heads=2)


def test_completion_threshold_decodes_from_cli() -> None:
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--policy.type=safediff_vla",
            "--policy.architecture=legacy_diffusion",
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
            "--policy.architecture=legacy_diffusion",
            "--policy.subgoal_labels_path=outputs/data/vlabench_subgoal_labels/labels.parquet",
            "--dataset.repo_id=VLA/smolvla_libero",
        ],
    )
    assert isinstance(config.policy, SafeDiffVLAConfig)
    assert config.policy.subgoal_labels_path == "outputs/data/vlabench_subgoal_labels/labels.parquet"


# ---- legacy_diffusion ---------------------------------------------------------------------


def test_planner_and_state_predictor_shapes() -> None:
    planner = ConditionalDiffusionPlanner(3, 5, 8, 16, 8)
    predictor = StatePredictor(3, 5, 8, 12)
    actions = torch.randn(2, 4, 3)
    latent = torch.randn(2, 8)
    state = torch.randn(2, 5)
    subgoal = torch.randn(2, 5)
    assert planner(actions, torch.tensor([0, 1]), latent, actions, state, subgoal).shape == actions.shape
    assert predictor(latent, actions, state).shape == (2, 5)


def test_backbone_and_state_predictor_are_trainable() -> None:
    policy = make_policy()
    assert all(not parameter.requires_grad for parameter in policy.backbone.parameters())
    assert any(parameter.requires_grad for parameter in policy.planner.parameters())
    assert any(parameter.requires_grad for parameter in policy.state_predictor.parameters())


def test_forward_regresses_against_subgoal_label_when_available() -> None:
    policy = make_policy()
    loss, metrics = policy(make_batch(with_subgoal_label=True))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"loss", "loss_diff", "loss_subgoal"} <= metrics.keys()
    assert metrics["loss_subgoal"] > 0


def test_forward_skips_subgoal_loss_without_label() -> None:
    policy = make_policy()
    loss, metrics = policy(make_batch(with_subgoal_label=False))
    assert torch.isfinite(loss)
    assert metrics["loss_subgoal"] == 0


def test_forward_state_predictor_uses_nominal_not_ground_truth_action() -> None:
    """The predicted subgoal must come from (latent, nominal, state) alone, so it's identical
    whether or not the ground-truth action chunk changes -- this is what keeps `forward()` (train)
    and `plan_action_chunk()` (inference) consistent (see `state_predictor.py`)."""
    policy = make_policy()
    batch = make_batch(with_subgoal_label=False)
    nominal, latent = policy._backbone_outputs(batch)
    state = policy._current_state(batch)
    expected = policy.state_predictor(latent, nominal, state)
    predicted_subgoal = policy.plan_action_chunk(batch)[1]["predicted_subgoal_state"]
    assert torch.allclose(predicted_subgoal, expected)


def test_sample_action_chunk_shape_and_no_nans() -> None:
    policy = make_policy()
    batch = make_batch()
    nominal, latent = policy._backbone_outputs(batch)
    state = policy._current_state(batch)
    subgoal = policy.state_predictor(latent, nominal, state)
    sample = policy._sample_action_chunk(latent, nominal, state, subgoal)
    assert sample.shape == nominal.shape
    assert torch.isfinite(sample).all()


def test_select_action_queue_and_reset() -> None:
    policy = make_policy(use_diffusion_refinement=False)
    batch = make_batch()
    first = policy.select_action(batch)
    assert first.shape == (2, 3)
    assert len(policy._action_queue) == 1
    policy.reset()
    assert len(policy._action_queue) == 0
    assert policy._pending_target_state is None
    assert policy._last_gap is None


def test_use_diffusion_refinement_false_returns_nominal() -> None:
    baseline = make_policy(use_diffusion_refinement=False)
    nominal, info = baseline.plan_action_chunk(make_batch())
    assert nominal.shape == (2, 4, 3)
    assert "predicted_subgoal_state" in info


def test_completion_gate_does_not_trigger_when_far_but_not_diverging() -> None:
    """Regression test for the original (timescale-mismatched) gate design: it flagged
    "not complete" whenever the gap merely still exceeded `completion_threshold`, which -- since
    a subgoal is often many `execute_horizon`s away (see `compute_subgoal_labels.py`) -- fired on
    almost every commit and collapsed execution into a near-permanent single-step replan loop.
    With a fixed (never-changing) batch, the measured gap can't grow between checks, so the gate
    must now commit a fresh full-length chunk even with `completion_threshold=0.0` (i.e. nowhere
    near "arrived")."""
    policy = make_policy(completion_threshold=0.0, max_replan_retries=2)
    batch = make_batch()
    policy.select_action(batch)
    while policy._action_queue:
        policy.select_action(batch)
    policy.select_action(batch)
    assert policy._replan_retries == 0
    assert len(policy._action_queue) == policy.config.execute_horizon - 1


def test_completion_gate_triggers_single_step_replan_when_gap_grows() -> None:
    policy = make_policy(completion_threshold=0.0, max_replan_retries=2)
    batch = make_batch()
    policy.select_action(batch)
    while policy._action_queue:
        policy.select_action(batch)
    # Force the next check to look like the gap grew since the last one.
    policy._last_gap = torch.zeros_like(policy._last_gap)

    policy.select_action(batch)
    assert policy._replan_retries == 1
    assert len(policy._action_queue) == 0

    policy._last_gap = torch.zeros_like(policy._last_gap)
    policy.select_action(batch)
    assert policy._replan_retries == 2
    assert len(policy._action_queue) == 0

    # Retry budget exhausted: this call must fall back to a fresh full-length commit regardless.
    policy._last_gap = torch.zeros_like(policy._last_gap)
    policy.select_action(batch)
    assert policy._replan_retries == 0
    assert len(policy._action_queue) == policy.config.execute_horizon - 1


def test_completion_gap() -> None:
    predicted = torch.zeros(2, 3)
    actual = torch.ones(2, 3)
    assert completion_gap(predicted, actual).allclose(torch.ones(2))


def test_use_completion_gate_false_disables_gating_even_with_subgoal() -> None:
    """With the gate off, `_pending_target_state` is never even populated (nothing to gate with),
    and every chunk boundary commits a fresh full-length chunk regardless."""
    policy = make_policy(completion_threshold=0.0, max_replan_retries=2, use_completion_gate=False)
    batch = make_batch()
    for _ in range(3):
        policy.select_action(batch)
        while policy._action_queue:
            policy.select_action(batch)
        assert policy._replan_retries == 0
        assert policy._pending_target_state is None


# ---- temporal_decoder / temporal_decoder_future_state -------------------------------------


def test_temporal_action_decoder_shapes() -> None:
    decoder = TemporalActionDecoder(
        action_dim=3,
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
    assert actions.shape == (2, 4, 3)
    assert torch.isfinite(actions).all()


def test_temporal_action_decoder_with_future_state() -> None:
    decoder = TemporalActionDecoder(
        action_dim=3,
        state_dim=5,
        latent_dim=12,
        hidden_dim=16,
        horizon=4,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        use_future_state=True,
    )
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    predicted_states = torch.randn(2, 4, 5)
    actions = decoder(latent_tokens, None, state, predicted_states)
    assert actions.shape == (2, 4, 3)
    with pytest.raises(ValueError, match="use_future_state"):
        decoder(latent_tokens, None, state, None)


def test_temporal_decoder_policy_trains_and_freezes_backbone() -> None:
    policy = make_policy(architecture="temporal_decoder")
    loss, metrics = policy(make_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"loss", "loss_action", "loss_subgoal", "loss_smooth"} <= metrics.keys()
    assert metrics["loss_subgoal"] == 0  # no future-state head in this architecture
    loss.backward()
    assert all(parameter.grad is None for parameter in policy.backbone.parameters())
    assert any(parameter.grad is not None for parameter in policy.decoder.parameters())


def test_temporal_decoder_future_state_regresses_subgoal_label() -> None:
    policy = make_policy(architecture="temporal_decoder_future_state")
    loss, metrics = policy(make_batch(with_subgoal_label=True))
    assert torch.isfinite(loss)
    assert metrics["loss_subgoal"] > 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.future_state_predictor.parameters())


def test_temporal_decoder_smoothness_regularizer_is_off_by_default() -> None:
    policy = make_policy(architecture="temporal_decoder")
    assert policy.config.lambda_smooth == 0.0
    _, metrics = policy(make_batch())
    assert metrics["loss_smooth"] == 0.0


def test_temporal_decoder_plan_action_chunk_shape() -> None:
    policy = make_policy(architecture="temporal_decoder")
    actions, metrics = policy.plan_action_chunk(make_batch())
    assert actions.shape == (2, 4, 3)
    assert "predicted_subgoal_state" not in metrics  # no subgoal signal for this architecture


def test_temporal_decoder_select_action_never_gates() -> None:
    """No subgoal signal exists for plain `temporal_decoder`, so every chunk boundary should
    commit a fresh full-length chunk regardless of `completion_threshold`."""
    policy = make_policy(architecture="temporal_decoder", completion_threshold=0.0, max_replan_retries=2)
    batch = make_batch()
    policy.select_action(batch)
    while policy._action_queue:
        policy.select_action(batch)
    policy.select_action(batch)
    assert policy._replan_retries == 0
    assert len(policy._action_queue) == policy.config.execute_horizon - 1


# ---- smolvla_nominal (eval-only ablation baseline) -----------------------------------------


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
    assert actions.shape == (2, 4, 3)
    assert metrics == {}
