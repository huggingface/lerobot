#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the RL algorithm abstraction and SACAlgorithm implementation."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.algorithms import make_algorithm
from lerobot.rl.algorithms.base import RLAlgorithmConfig, TrainingStats
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.utils.random_utils import set_seed

# ---------------------------------------------------------------------------
# Helpers (reuse patterns from tests/policies/test_sac_policy.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(42)


def _make_sac_config(
    state_dim: int = 10,
    action_dim: int = 6,
    num_discrete_actions: int | None = None,
    utd_ratio: int = 1,
    policy_update_freq: int = 1,
    with_images: bool = False,
) -> SACConfig:
    config = SACConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
        utd_ratio=utd_ratio,
        policy_update_freq=policy_update_freq,
        num_discrete_actions=num_discrete_actions,
        use_torch_compile=False,
    )
    if with_images:
        config.input_features[OBS_IMAGE] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
        config.dataset_stats[OBS_IMAGE] = {
            "mean": torch.randn(3, 1, 1).tolist(),
            "std": torch.randn(3, 1, 1).abs().tolist(),
        }
        config.latent_dim = 32
        config.state_encoder_hidden_dim = 32
    config.validate_features()
    return config


def _make_algorithm(
    state_dim: int = 10,
    action_dim: int = 6,
    utd_ratio: int = 1,
    policy_update_freq: int = 1,
    num_discrete_actions: int | None = None,
    with_images: bool = False,
) -> tuple[SACAlgorithm, SACPolicy]:
    sac_cfg = _make_sac_config(
        state_dim=state_dim,
        action_dim=action_dim,
        utd_ratio=utd_ratio,
        policy_update_freq=policy_update_freq,
        num_discrete_actions=num_discrete_actions,
        with_images=with_images,
    )
    policy = SACPolicy(config=sac_cfg)
    policy.train()
    algo_config = SACAlgorithmConfig.from_policy_config(sac_cfg)
    algorithm = SACAlgorithm(policy=policy, config=algo_config)
    algorithm.make_optimizers()
    return algorithm, policy


def _make_batch(
    batch_size: int = 4,
    state_dim: int = 10,
    action_dim: int = 6,
    with_images: bool = False,
) -> dict:
    obs = {OBS_STATE: torch.randn(batch_size, state_dim)}
    next_obs = {OBS_STATE: torch.randn(batch_size, state_dim)}
    if with_images:
        obs[OBS_IMAGE] = torch.randn(batch_size, 3, 84, 84)
        next_obs[OBS_IMAGE] = torch.randn(batch_size, 3, 84, 84)
    return {
        ACTION: torch.randn(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": obs,
        "next_state": next_obs,
        "done": torch.zeros(batch_size),
        "complementary_info": {},
    }


def _batch_iterator(**batch_kwargs):
    """Infinite iterator that yields fresh batches (mirrors a real DataMixer iterator)."""
    while True:
        yield _make_batch(**batch_kwargs)


# ===========================================================================
# Registry / config tests
# ===========================================================================


def test_sac_algorithm_config_registered():
    """SACAlgorithmConfig should be discoverable through the registry."""
    assert "sac" in RLAlgorithmConfig.get_known_choices()
    cls = RLAlgorithmConfig.get_choice_class("sac")
    assert cls is SACAlgorithmConfig


def test_sac_algorithm_config_from_policy_config():
    """from_policy_config should copy relevant fields."""
    sac_cfg = _make_sac_config(utd_ratio=4, policy_update_freq=2)
    algo_cfg = SACAlgorithmConfig.from_policy_config(sac_cfg)
    assert algo_cfg.utd_ratio == 4
    assert algo_cfg.policy_update_freq == 2
    assert algo_cfg.clip_grad_norm == sac_cfg.grad_clip_norm


# ===========================================================================
# TrainingStats tests
# ===========================================================================


def test_training_stats_defaults():
    stats = TrainingStats()
    assert stats.loss_actor is None
    assert stats.loss_critic is None
    assert stats.loss_temperature is None
    assert stats.loss_discrete_critic is None
    assert stats.grad_norms == {}
    assert stats.extra == {}


# ===========================================================================
# get_weights
# ===========================================================================


def test_get_weights_returns_actor_state_dict():
    algorithm, policy = _make_algorithm()
    weights = algorithm.get_weights()
    assert "policy" in weights
    for key in policy.actor.state_dict():
        assert key in weights["policy"]
        assert torch.equal(weights["policy"][key].cpu(), policy.actor.state_dict()[key].cpu())


def test_get_weights_includes_discrete_critic_when_present():
    algorithm, policy = _make_algorithm(num_discrete_actions=3, action_dim=6)
    weights = algorithm.get_weights()
    assert "discrete_critic" in weights
    for key in policy.discrete_critic.state_dict():
        assert key in weights["discrete_critic"]


def test_get_weights_excludes_discrete_critic_when_absent():
    algorithm, _ = _make_algorithm()
    weights = algorithm.get_weights()
    assert "discrete_critic" not in weights


def test_get_weights_are_on_cpu():
    algorithm, _ = _make_algorithm()
    weights = algorithm.get_weights()
    for key, tensor in weights["policy"].items():
        assert tensor.device == torch.device("cpu"), f"{key} is not on CPU"


# ===========================================================================
# select_action
# ===========================================================================


def test_select_action_returns_correct_shape():
    action_dim = 6
    algorithm, _ = _make_algorithm(state_dim=10, action_dim=action_dim)
    obs = {OBS_STATE: torch.randn(10)}
    action = algorithm.select_action(obs)
    assert action.shape == (action_dim,)


def test_select_action_with_discrete_critic():
    continuous_dim = 5
    algorithm, _ = _make_algorithm(state_dim=10, action_dim=continuous_dim, num_discrete_actions=3)
    obs = {OBS_STATE: torch.randn(10)}
    action = algorithm.select_action(obs)
    assert action.shape == (continuous_dim + 1,)


# ===========================================================================
# update (single batch, utd_ratio=1)
# ===========================================================================


def test_update_returns_training_stats():
    algorithm, _ = _make_algorithm()
    stats = algorithm.update(_batch_iterator())
    assert isinstance(stats, TrainingStats)
    assert stats.loss_critic is not None
    assert isinstance(stats.loss_critic, float)


def test_update_populates_actor_and_temperature_losses():
    """With policy_update_freq=1 and step 0, actor/temperature should be updated."""
    algorithm, _ = _make_algorithm(policy_update_freq=1)
    stats = algorithm.update(_batch_iterator())
    assert stats.loss_actor is not None
    assert stats.loss_temperature is not None
    assert "temperature" in stats.extra


@pytest.mark.parametrize("policy_update_freq", [2, 3])
def test_update_skips_actor_at_non_update_steps(policy_update_freq):
    """Actor/temperature should only update when optimization_step % freq == 0."""
    algorithm, _ = _make_algorithm(policy_update_freq=policy_update_freq)
    it = _batch_iterator()

    # Step 0: should update actor
    stats_0 = algorithm.update(it)
    assert stats_0.loss_actor is not None

    # Step 1: should NOT update actor
    stats_1 = algorithm.update(it)
    assert stats_1.loss_actor is None


def test_update_increments_optimization_step():
    algorithm, _ = _make_algorithm()
    it = _batch_iterator()
    assert algorithm.optimization_step == 0
    algorithm.update(it)
    assert algorithm.optimization_step == 1
    algorithm.update(it)
    assert algorithm.optimization_step == 2


def test_update_with_discrete_critic():
    algorithm, _ = _make_algorithm(num_discrete_actions=3, action_dim=6)
    stats = algorithm.update(_batch_iterator(action_dim=7))  # continuous + 1 discrete
    assert stats.loss_discrete_critic is not None
    assert "discrete_critic" in stats.grad_norms


# ===========================================================================
# update with UTD ratio > 1
# ===========================================================================


@pytest.mark.parametrize("utd_ratio", [2, 4])
def test_update_with_utd_ratio(utd_ratio):
    algorithm, _ = _make_algorithm(utd_ratio=utd_ratio)
    stats = algorithm.update(_batch_iterator())
    assert isinstance(stats, TrainingStats)
    assert stats.loss_critic is not None
    assert algorithm.optimization_step == 1


def test_update_utd_ratio_pulls_utd_batches():
    """next(batch_iterator) should be called exactly utd_ratio times."""
    utd_ratio = 3
    algorithm, _ = _make_algorithm(utd_ratio=utd_ratio)

    call_count = 0

    def counting_iterator():
        nonlocal call_count
        while True:
            call_count += 1
            yield _make_batch()

    algorithm.update(counting_iterator())
    assert call_count == utd_ratio


def test_update_utd_ratio_3_critic_warmup_changes_weights():
    """With utd_ratio=3, critic weights should change after update (3 critic steps)."""
    algorithm, policy = _make_algorithm(utd_ratio=3)

    critic_params_before = {n: p.clone() for n, p in policy.critic_ensemble.named_parameters()}

    algorithm.update(_batch_iterator())

    changed = False
    for n, p in policy.critic_ensemble.named_parameters():
        if not torch.equal(p, critic_params_before[n]):
            changed = True
            break
    assert changed, "Critic weights should have changed after UTD update"


# ===========================================================================
# get_observation_features
# ===========================================================================


def test_get_observation_features_returns_none_without_frozen_encoder():
    algorithm, _ = _make_algorithm(with_images=False)
    obs = {OBS_STATE: torch.randn(4, 10)}
    next_obs = {OBS_STATE: torch.randn(4, 10)}
    feat, next_feat = algorithm.get_observation_features(obs, next_obs)
    assert feat is None
    assert next_feat is None


# ===========================================================================
# optimization_step setter
# ===========================================================================


def test_optimization_step_can_be_set_for_resume():
    algorithm, _ = _make_algorithm()
    algorithm.optimization_step = 100
    assert algorithm.optimization_step == 100


# ===========================================================================
# make_algorithm factory
# ===========================================================================


def test_make_algorithm_returns_sac_for_sac_policy():
    sac_cfg = _make_sac_config()
    policy = SACPolicy(config=sac_cfg)
    algorithm = make_algorithm(policy=policy, policy_cfg=sac_cfg)
    assert isinstance(algorithm, SACAlgorithm)
    assert algorithm.optimizers == {}


def test_make_optimizers_creates_expected_keys():
    """make_optimizers() should populate the algorithm with Adam optimizers."""
    sac_cfg = _make_sac_config()
    policy = SACPolicy(config=sac_cfg)
    algorithm = make_algorithm(policy=policy, policy_cfg=sac_cfg)
    optimizers = algorithm.make_optimizers()
    assert "actor" in optimizers
    assert "critic" in optimizers
    assert "temperature" in optimizers
    assert all(isinstance(v, torch.optim.Adam) for v in optimizers.values())
    assert algorithm.get_optimizers() is optimizers


def test_actor_side_no_optimizers():
    """Actor-side usage: no optimizers needed, make_optimizers is not called."""
    sac_cfg = _make_sac_config()
    policy = SACPolicy(config=sac_cfg)
    algorithm = make_algorithm(policy=policy, policy_cfg=sac_cfg)
    assert isinstance(algorithm, SACAlgorithm)
    assert algorithm.optimizers == {}


def test_make_algorithm_copies_config_fields():
    sac_cfg = _make_sac_config(utd_ratio=5, policy_update_freq=3)
    policy = SACPolicy(config=sac_cfg)
    algorithm = make_algorithm(policy=policy, policy_cfg=sac_cfg)
    assert algorithm.config.utd_ratio == 5
    assert algorithm.config.policy_update_freq == 3


def test_make_algorithm_raises_for_unknown_type():
    class FakeConfig:
        type = "unknown_algo"

    with pytest.raises(ValueError, match="No RLAlgorithmConfig"):
        make_algorithm(policy=None, policy_cfg=FakeConfig())


# ===========================================================================
# load_weights (round-trip with get_weights)
# ===========================================================================


def test_load_weights_round_trip():
    """get_weights -> load_weights should restore identical parameters on a fresh policy."""
    algo_src, _ = _make_algorithm(state_dim=10, action_dim=6)
    algo_src.update(_batch_iterator())

    sac_cfg = _make_sac_config(state_dim=10, action_dim=6)
    policy_dst = SACPolicy(config=sac_cfg)
    algo_dst = SACAlgorithm(policy=policy_dst, config=algo_src.config)

    weights = algo_src.get_weights()
    algo_dst.load_weights(weights, device="cpu")

    for key in weights["policy"]:
        assert torch.equal(
            algo_dst.policy.actor.state_dict()[key].cpu(),
            weights["policy"][key].cpu(),
        ), f"Actor param '{key}' mismatch after load_weights"


def test_load_weights_round_trip_with_discrete_critic():
    algo_src, _ = _make_algorithm(num_discrete_actions=3, action_dim=6)
    algo_src.update(_batch_iterator(action_dim=7))

    sac_cfg = _make_sac_config(num_discrete_actions=3, action_dim=6)
    policy_dst = SACPolicy(config=sac_cfg)
    algo_dst = SACAlgorithm(policy=policy_dst, config=algo_src.config)

    weights = algo_src.get_weights()
    algo_dst.load_weights(weights, device="cpu")

    for key in weights["discrete_critic"]:
        assert torch.equal(
            algo_dst.policy.discrete_critic.state_dict()[key].cpu(),
            weights["discrete_critic"][key].cpu(),
        ), f"Discrete critic param '{key}' mismatch after load_weights"


def test_load_weights_ignores_missing_discrete_critic():
    """load_weights should not fail when weights lack discrete_critic on a non-discrete policy."""
    algorithm, _ = _make_algorithm()
    weights = {"policy": algorithm.get_weights()["policy"]}
    algorithm.load_weights(weights, device="cpu")


# ===========================================================================
# TrainingStats generic losses dict
# ===========================================================================


def test_training_stats_generic_losses():
    stats = TrainingStats(
        losses={"loss_bc": 0.5, "loss_q": 1.2},
        extra={"temperature": 0.1},
    )
    assert stats.losses["loss_bc"] == 0.5
    assert stats.losses["loss_q"] == 1.2
    assert stats.extra["temperature"] == 0.1
    # backward-compat fields are still None
    assert stats.loss_actor is None


# ===========================================================================
# Registry-driven build_algorithm
# ===========================================================================


def test_build_algorithm_via_config():
    """SACAlgorithmConfig.build_algorithm should produce a working SACAlgorithm."""
    sac_cfg = _make_sac_config(utd_ratio=2)
    algo_config = SACAlgorithmConfig.from_policy_config(sac_cfg)
    policy = SACPolicy(config=sac_cfg)

    algorithm = algo_config.build_algorithm(policy)
    assert isinstance(algorithm, SACAlgorithm)
    assert algorithm.config.utd_ratio == 2


def test_make_algorithm_uses_build_algorithm():
    """make_algorithm should delegate to config.build_algorithm (no hardcoded if/else)."""
    sac_cfg = _make_sac_config()
    policy = SACPolicy(config=sac_cfg)
    algorithm = make_algorithm(policy=policy, policy_cfg=sac_cfg)
    assert isinstance(algorithm, SACAlgorithm)
