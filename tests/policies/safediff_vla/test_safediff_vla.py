from __future__ import annotations

import draccus
import torch
from torch import Tensor, nn

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.safediff_vla.diffusion_planner import ConditionalDiffusionPlanner
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.state_predictor import StatePredictor, completion_gap, masked_mse_loss
from lerobot.utils.constants import ACTION, OBS_STATE


class TinyBackbone(nn.Module):
    def __init__(self, horizon: int, action_dim: int, feature_dim: int = 12) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.safediff_latent_dim = feature_dim
        self.projection = nn.Linear(5, feature_dim)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def extract_safediff_features(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        # `SafeDiffVLAPolicy._backbone_outputs` always hands the backbone a 2D [B, state_dim]
        # current-state tensor, even when the caller's own batch carries the extra
        # future-state slice `state_observation_delta_indices` adds.
        assert batch[OBS_STATE].ndim == 2
        latent = self.projection(batch[OBS_STATE])
        nominal = latent[:, None, : self.action_dim].expand(-1, self.horizon, -1).tanh()
        return nominal, latent


def make_config(**overrides) -> SafeDiffVLAConfig:
    values = {
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
    }
    values.update(overrides)
    return SafeDiffVLAConfig(**values)


def make_batch(batch_size: int = 2, with_future_state: bool = True) -> dict[str, Tensor]:
    state = torch.randn(batch_size, 2, 5) if with_future_state else torch.randn(batch_size, 5)
    return {OBS_STATE: state, ACTION: torch.randn(batch_size, 4, 3)}


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


def test_state_observation_delta_indices_span_execute_horizon() -> None:
    config = make_config(execute_horizon=3)
    assert config.state_observation_delta_indices == [0, 3]


def test_completion_threshold_decodes_from_cli() -> None:
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--policy.type=safediff_vla",
            "--policy.completion_threshold=0.1",
            "--dataset.repo_id=VLA/smolvla_libero",
        ],
    )
    assert isinstance(config.policy, SafeDiffVLAConfig)
    assert config.policy.completion_threshold == 0.1


def test_planner_and_state_predictor_shapes() -> None:
    planner = ConditionalDiffusionPlanner(3, 5, 8, 16, 8)
    predictor = StatePredictor(3, 5, 8, 12)
    actions = torch.randn(2, 4, 3)
    latent = torch.randn(2, 8)
    state = torch.randn(2, 5)
    assert planner(actions, torch.tensor([0, 1]), latent, actions, state).shape == actions.shape
    assert predictor(latent, actions).shape == (2, 5)


def test_backbone_and_state_predictor_are_trainable() -> None:
    policy = make_policy()
    assert all(not parameter.requires_grad for parameter in policy.backbone.parameters())
    assert any(parameter.requires_grad for parameter in policy.planner.parameters())
    assert any(parameter.requires_grad for parameter in policy.state_predictor.parameters())


def test_forward_regresses_against_future_state_when_available() -> None:
    policy = make_policy()
    loss, metrics = policy(make_batch(with_future_state=True))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"loss", "loss_diff", "loss_state_pred"} <= metrics.keys()
    assert metrics["loss_state_pred"] > 0


def test_forward_skips_state_pred_loss_without_future_state() -> None:
    policy = make_policy()
    loss, metrics = policy(make_batch(with_future_state=False))
    assert torch.isfinite(loss)
    assert metrics["loss_state_pred"] == 0


def test_sample_action_chunk_shape_and_no_nans() -> None:
    policy = make_policy()
    batch = make_batch()
    nominal, latent = policy._backbone_outputs(batch)
    state = policy._current_state(batch)
    sample = policy._sample_action_chunk(latent, nominal, state)
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


def test_use_diffusion_refinement_false_returns_nominal() -> None:
    baseline = make_policy(use_diffusion_refinement=False)
    nominal, info = baseline.plan_action_chunk(make_batch())
    assert nominal.shape == (2, 4, 3)
    assert "predicted_future_state" in info


def test_completion_gate_forces_single_step_replan_until_retry_budget_exhausted() -> None:
    policy = make_policy(completion_threshold=0.0, max_replan_retries=2)
    batch = make_batch()
    policy.select_action(batch)  # first chunk: no pending target yet, commits fully
    assert policy._replan_retries == 0
    while policy._action_queue:
        policy.select_action(batch)

    # A real gap of exactly 0.0 will essentially never be met by a random target, so each of the
    # next `max_replan_retries` chunk boundaries should fall back to a one-step replan instead of
    # a fresh full commit.
    policy.select_action(batch)
    assert policy._replan_retries == 1
    assert len(policy._action_queue) == 0
    policy.select_action(batch)
    assert policy._replan_retries == 2
    assert len(policy._action_queue) == 0

    # Retry budget exhausted: this call must fall back to a fresh full-length commit.
    policy.select_action(batch)
    assert policy._replan_retries == 0
    assert len(policy._action_queue) == policy.config.execute_horizon - 1


def test_completion_gap_and_masked_mse_loss() -> None:
    predicted = torch.zeros(2, 3)
    actual = torch.ones(2, 3)
    assert completion_gap(predicted, actual).allclose(torch.ones(2))
    loss = masked_mse_loss(predicted, actual, is_pad=torch.tensor([False, True]))
    # Only the first (non-padded) sample should contribute.
    assert loss.item() == 1.0
