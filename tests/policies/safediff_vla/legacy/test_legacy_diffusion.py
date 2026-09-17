from __future__ import annotations

import draccus
import torch
from torch import Tensor

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.safediff_vla.legacy.configuration_legacy_diffusion import LegacySafeDiffVLAConfig
from lerobot.policies.safediff_vla.legacy.diffusion_planner import ConditionalDiffusionPlanner
from lerobot.policies.safediff_vla.legacy.modeling_legacy_diffusion import LegacySafeDiffVLAPolicy
from lerobot.policies.safediff_vla.legacy.state_predictor import StatePredictor
from lerobot.policies.safediff_vla.state_predictor import completion_gap
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.safediff_vla.testing_utils import TinyBackbone


def make_config(**overrides) -> LegacySafeDiffVLAConfig:
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
    return LegacySafeDiffVLAConfig(**values)


def make_batch(batch_size: int = 2, with_subgoal_label: bool = True) -> dict[str, Tensor]:
    batch = {OBS_STATE: torch.randn(batch_size, 5), ACTION: torch.randn(batch_size, 4, 3)}
    if with_subgoal_label:
        batch["observation.subgoal_state"] = torch.randn(batch_size, 5)
    return batch


def make_policy(**overrides) -> LegacySafeDiffVLAPolicy:
    config = make_config(**overrides)
    return LegacySafeDiffVLAPolicy(config, backbone=TinyBackbone(4, 3))


def test_config_serialization(tmp_path) -> None:
    config = make_config()
    config.save_pretrained(tmp_path)
    restored = LegacySafeDiffVLAConfig.from_pretrained(tmp_path)
    assert isinstance(restored, LegacySafeDiffVLAConfig)
    assert restored.action_horizon == config.action_horizon


def test_config_and_registration() -> None:
    config = make_config()
    assert config.type == "safediff_vla_legacy"
    assert isinstance(make_policy_config("safediff_vla_legacy", device="cpu"), LegacySafeDiffVLAConfig)
    assert get_policy_class("safediff_vla_legacy") is LegacySafeDiffVLAPolicy


def test_completion_threshold_decodes_from_cli() -> None:
    """Still reproducible end-to-end via `lerobot-train --policy.type=safediff_vla_legacy ...`."""
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--policy.type=safediff_vla_legacy",
            "--policy.completion_threshold=0.1",
            "--dataset.repo_id=VLA/smolvla_libero",
        ],
    )
    assert isinstance(config.policy, LegacySafeDiffVLAConfig)
    assert config.policy.completion_threshold == 0.1


def test_subgoal_labels_path_decodes_from_cli() -> None:
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--policy.type=safediff_vla_legacy",
            "--policy.subgoal_labels_path=outputs/data/vlabench_subgoal_labels/labels.parquet",
            "--dataset.repo_id=VLA/smolvla_libero",
        ],
    )
    assert isinstance(config.policy, LegacySafeDiffVLAConfig)
    assert config.policy.subgoal_labels_path == "outputs/data/vlabench_subgoal_labels/labels.parquet"


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
    assert len(policy._executor._action_queue) == 1
    policy.reset()
    assert len(policy._executor._action_queue) == 0
    assert policy._executor._pending_target_state is None
    assert policy._executor._last_gap is None


def test_use_diffusion_refinement_false_returns_nominal() -> None:
    baseline = make_policy(use_diffusion_refinement=False)
    nominal, info = baseline.plan_action_chunk(make_batch())
    assert nominal.shape == (2, 4, 3)
    assert "predicted_subgoal_state" in info


def test_completion_gate_does_not_trigger_when_far_but_not_diverging() -> None:
    """Regression test for the original (timescale-mismatched) gate design: it flagged
    "not complete" whenever the gap merely still exceeded `completion_threshold`, which -- since
    a subgoal is often many `execute_horizon`s away -- fired on almost every commit and collapsed
    execution into a near-permanent single-step replan loop. With a fixed (never-changing) batch,
    the measured gap can't grow between checks, so the gate must now commit a fresh full-length
    chunk even with `completion_threshold=0.0` (i.e. nowhere near "arrived")."""
    policy = make_policy(completion_threshold=0.0, max_replan_retries=2)
    batch = make_batch()
    policy.select_action(batch)
    while policy._executor._action_queue:
        policy.select_action(batch)
    policy.select_action(batch)
    assert policy._executor._replan_retries == 0
    assert len(policy._executor._action_queue) == policy.config.execute_horizon - 1


def test_completion_gate_triggers_single_step_replan_when_gap_grows() -> None:
    policy = make_policy(completion_threshold=0.0, max_replan_retries=2)
    batch = make_batch()
    policy.select_action(batch)
    while policy._executor._action_queue:
        policy.select_action(batch)
    # Force the next check to look like the gap grew since the last one.
    policy._executor._last_gap = torch.zeros_like(policy._executor._last_gap)

    policy.select_action(batch)
    assert policy._executor._replan_retries == 1
    assert len(policy._executor._action_queue) == 0

    policy._executor._last_gap = torch.zeros_like(policy._executor._last_gap)
    policy.select_action(batch)
    assert policy._executor._replan_retries == 2
    assert len(policy._executor._action_queue) == 0

    # Retry budget exhausted: this call must fall back to a fresh full-length commit regardless.
    policy._executor._last_gap = torch.zeros_like(policy._executor._last_gap)
    policy.select_action(batch)
    assert policy._executor._replan_retries == 0
    assert len(policy._executor._action_queue) == policy.config.execute_horizon - 1


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
        while policy._executor._action_queue:
            policy.select_action(batch)
        assert policy._executor._replan_retries == 0
        assert policy._executor._pending_target_state is None
