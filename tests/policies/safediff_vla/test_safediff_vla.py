from __future__ import annotations

import draccus
import torch
from torch import Tensor, nn

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.safediff_vla.critics import TrajectoryCritic, score_candidates
from lerobot.policies.safediff_vla.diffusion_planner import ConditionalDiffusionPlanner
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
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
        "num_candidates": 3,
        "latent_dim": 8,
        "planner_hidden_dim": 16,
        "task_critic_hidden_dim": 12,
        "risk_critic_hidden_dim": 12,
        "timestep_embedding_dim": 8,
    }
    values.update(overrides)
    return SafeDiffVLAConfig(**values)


def make_batch(batch_size: int = 2) -> dict[str, Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, 5),
        ACTION: torch.randn(batch_size, 4, 3),
        "task_success": torch.tensor([1, 0][:batch_size]),
        "safety_violation": torch.tensor([0, 1][:batch_size]),
    }


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


def test_training_mode_decodes_from_cli() -> None:
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--policy.type=safediff_vla",
            "--policy.training_mode=diffusion",
            "--dataset.repo_id=VLA/smolvla_libero",
        ],
    )
    assert isinstance(config.policy, SafeDiffVLAConfig)
    assert config.policy.training_mode == "diffusion"


def test_planner_and_critic_shapes() -> None:
    planner = ConditionalDiffusionPlanner(3, 8, 16, 8)
    critic = TrajectoryCritic(3, 8, 12)
    actions = torch.randn(2, 4, 3)
    latent = torch.randn(2, 8)
    assert planner(actions, torch.tensor([0, 1]), latent, actions).shape == actions.shape
    assert critic(latent, actions).shape == (2,)
    assert critic(latent, actions[:, None].expand(-1, 3, -1, -1)).shape == (2, 3)


def test_backbone_freezing_and_training_modes() -> None:
    policy = make_policy(training_mode="diffusion")
    assert all(not parameter.requires_grad for parameter in policy.backbone.parameters())
    assert any(parameter.requires_grad for parameter in policy.planner.parameters())
    assert all(not parameter.requires_grad for parameter in policy.task_critic.parameters())


def test_joint_training_forward_with_and_without_labels() -> None:
    policy = make_policy(training_mode="joint")
    loss, metrics = policy(make_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"loss", "loss_diff", "loss_task", "loss_risk"} <= metrics.keys()
    unlabeled = make_batch()
    unlabeled.pop("task_success")
    unlabeled.pop("safety_violation")
    loss, metrics = policy(unlabeled)
    assert torch.isfinite(loss)
    assert metrics["loss_task"] == 0 and metrics["loss_risk"] == 0


def test_candidate_shape_and_no_nans() -> None:
    policy = make_policy()
    nominal, latent = policy._backbone_outputs(make_batch())
    candidates = policy.generate_candidates(latent, nominal)
    assert candidates.shape == (2, 3, 4, 3)
    assert torch.isfinite(candidates).all()


def test_candidate_scoring_argmax_is_batch_safe() -> None:
    nominal = torch.zeros(2, 4, 3)
    candidates = torch.stack((nominal, nominal + 1, nominal + 2), dim=1)
    task_logits = torch.tensor([[0.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
    risk_logits = torch.full_like(task_logits, -8.0)
    scores, distances = score_candidates(task_logits, risk_logits, candidates, nominal, 1.0, 0.0)
    assert scores.argmax(1).tolist() == [1, 2]
    assert distances.shape == (2, 3)


def test_select_action_queue_and_reset() -> None:
    policy = make_policy(use_diffusion_refinement=False)
    batch = make_batch()
    first = policy.select_action(batch)
    assert first.shape == (2, 3)
    assert len(policy._action_queue) == 1
    policy.reset()
    assert len(policy._action_queue) == 0


def test_ablation_and_adaptive_planning_paths() -> None:
    batch = make_batch()
    baseline = make_policy(use_diffusion_refinement=False)
    nominal, info = baseline.plan_action_chunk(batch)
    assert nominal.shape == (2, 4, 3) and info["planner_usage_rate"] == 0

    adaptive = make_policy(adaptive_planning=True, risk_threshold=2.0)
    _, info = adaptive.plan_action_chunk(batch)
    assert info["planner_usage_rate"] == 0

    guided = make_policy(use_critic_guidance=True, use_vla_prior_init=False)
    actions = guided.predict_action_chunk(batch)
    assert actions.shape == (2, 4, 3)
    assert torch.isfinite(actions).all()
