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

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig  # noqa: E402
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy  # noqa: E402
from lerobot.rl.algorithms.configs import RLAlgorithmConfig, TrainingStats  # noqa: E402
from lerobot.rl.algorithms.factory import make_algorithm  # noqa: E402
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (reuse patterns from tests/policies/test_gaussian_actor_policy.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(42)


def _make_sac_config(
    state_dim: int = 10,
    action_dim: int = 6,
    num_discrete_actions: int | None = None,
    with_images: bool = False,
) -> GaussianActorConfig:
    config = GaussianActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
        num_discrete_actions=num_discrete_actions,
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
) -> tuple[SACAlgorithm, GaussianActorPolicy]:
    sac_cfg = _make_sac_config(
        state_dim=state_dim,
        action_dim=action_dim,
        num_discrete_actions=num_discrete_actions,
        with_images=with_images,
    )
    policy = GaussianActorPolicy(config=sac_cfg)
    policy.train()
    algo_config = SACAlgorithmConfig.from_policy_config(sac_cfg)
    algo_config.utd_ratio = utd_ratio
    algo_config.policy_update_freq = policy_update_freq
    algorithm = SACAlgorithm(policy=policy, config=algo_config)
    algorithm.make_optimizers_and_scheduler()
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
    """from_policy_config embeds the policy config and uses SAC defaults."""
    sac_cfg = _make_sac_config()
    algo_cfg = SACAlgorithmConfig.from_policy_config(sac_cfg)
    assert algo_cfg.policy_config is sac_cfg
    assert algo_cfg.discrete_critic_network_kwargs is sac_cfg.discrete_critic_network_kwargs
    # Defaults come from SACAlgorithmConfig, not from the policy config.
    assert algo_cfg.utd_ratio == 1
    assert algo_cfg.policy_update_freq == 1
    assert algo_cfg.grad_clip_norm == 40.0
    assert algo_cfg.actor_lr == 3e-4


# ===========================================================================
# TrainingStats tests
# ===========================================================================


def test_training_stats_defaults():
    stats = TrainingStats()
    assert stats.losses == {}
    assert stats.grad_norms == {}
    assert stats.extra == {}


# ===========================================================================
# get_weights
# ===========================================================================


def test_get_weights_returns_policy_state_dict():
    algorithm, policy = _make_algorithm()
    weights = algorithm.get_weights()
    assert "policy" in weights
    actor_state_dict = policy.actor.state_dict()
    for key in actor_state_dict:
        assert key in weights["policy"]
        assert torch.equal(weights["policy"][key].cpu(), actor_state_dict[key].cpu())


def test_get_weights_includes_discrete_critic_when_present():
    algorithm, _ = _make_algorithm(num_discrete_actions=3, action_dim=6)
    weights = algorithm.get_weights()
    assert "discrete_critic" in weights
    assert len(weights["discrete_critic"]) > 0


def test_get_weights_excludes_discrete_critic_when_absent():
    algorithm, _ = _make_algorithm()
    weights = algorithm.get_weights()
    assert "discrete_critic" not in weights


def test_get_weights_are_on_cpu():
    algorithm, _ = _make_algorithm(num_discrete_actions=3, action_dim=6)
    weights = algorithm.get_weights()
    for group_name, state_dict in weights.items():
        for key, tensor in state_dict.items():
            assert tensor.device == torch.device("cpu"), f"{group_name}/{key} is not on CPU"


# ===========================================================================
# select_action (lives on the policy, not the algorithm)
# ===========================================================================


def test_select_action_returns_correct_shape():
    action_dim = 6
    _, policy = _make_algorithm(state_dim=10, action_dim=action_dim)
    policy.eval()
    obs = {OBS_STATE: torch.randn(10)}
    action = policy.select_action(obs)
    assert action.shape == (action_dim,)


def test_select_action_with_discrete_critic():
    continuous_dim = 5
    _, policy = _make_algorithm(state_dim=10, action_dim=continuous_dim, num_discrete_actions=3)
    policy.eval()
    obs = {OBS_STATE: torch.randn(10)}
    action = policy.select_action(obs)
    assert action.shape == (continuous_dim + 1,)


# ===========================================================================
# update (single batch, utd_ratio=1)
# ===========================================================================


def test_update_returns_training_stats():
    algorithm, _ = _make_algorithm()
    stats = algorithm.update(_batch_iterator())
    assert isinstance(stats, TrainingStats)
    assert "loss_critic" in stats.losses
    assert isinstance(stats.losses["loss_critic"], float)


def test_update_populates_actor_and_temperature_losses():
    """With policy_update_freq=1 and step 0, actor/temperature should be updated."""
    algorithm, _ = _make_algorithm(policy_update_freq=1)
    stats = algorithm.update(_batch_iterator())
    assert "loss_actor" in stats.losses
    assert "loss_temperature" in stats.losses
    assert "temperature" in stats.extra


@pytest.mark.parametrize("policy_update_freq", [2, 3])
def test_update_skips_actor_at_non_update_steps(policy_update_freq):
    """Actor/temperature should only update when optimization_step % freq == 0."""
    algorithm, _ = _make_algorithm(policy_update_freq=policy_update_freq)
    it = _batch_iterator()

    # Step 0: should update actor
    stats_0 = algorithm.update(it)
    assert "loss_actor" in stats_0.losses

    # Step 1: should NOT update actor
    stats_1 = algorithm.update(it)
    assert "loss_actor" not in stats_1.losses


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
    assert "loss_discrete_critic" in stats.losses
    assert "discrete_critic" in stats.grad_norms


# ===========================================================================
# update with UTD ratio > 1
# ===========================================================================


@pytest.mark.parametrize("utd_ratio", [2, 4])
def test_update_with_utd_ratio(utd_ratio):
    algorithm, _ = _make_algorithm(utd_ratio=utd_ratio)
    stats = algorithm.update(_batch_iterator())
    assert isinstance(stats, TrainingStats)
    assert "loss_critic" in stats.losses
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

    critic_params_before = {n: p.clone() for n, p in algorithm.critic_ensemble.named_parameters()}

    algorithm.update(_batch_iterator())

    changed = False
    for n, p in algorithm.critic_ensemble.named_parameters():
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
    policy = GaussianActorPolicy(config=sac_cfg)
    algorithm = make_algorithm(cfg=SACAlgorithmConfig.from_policy_config(sac_cfg), policy=policy)
    assert isinstance(algorithm, SACAlgorithm)
    assert algorithm.optimizers == {}


def test_make_optimizers_creates_expected_keys():
    """make_optimizers_and_scheduler() should populate the algorithm with Adam optimizers."""
    sac_cfg = _make_sac_config()
    policy = GaussianActorPolicy(config=sac_cfg)
    algorithm = make_algorithm(cfg=SACAlgorithmConfig.from_policy_config(sac_cfg), policy=policy)
    optimizers = algorithm.make_optimizers_and_scheduler()
    assert "actor" in optimizers
    assert "critic" in optimizers
    assert "temperature" in optimizers
    assert all(isinstance(v, torch.optim.Adam) for v in optimizers.values())
    assert algorithm.get_optimizers() is optimizers


def test_actor_side_no_optimizers():
    """Actor-side usage: no optimizers needed, make_optimizers_and_scheduler is not called."""
    sac_cfg = _make_sac_config()
    policy = GaussianActorPolicy(config=sac_cfg)
    algorithm = make_algorithm(cfg=SACAlgorithmConfig.from_policy_config(sac_cfg), policy=policy)
    assert isinstance(algorithm, SACAlgorithm)
    assert algorithm.optimizers == {}


def test_make_algorithm_uses_sac_algorithm_defaults():
    """make_algorithm populates SACAlgorithmConfig with its own defaults."""
    sac_cfg = _make_sac_config()
    policy = GaussianActorPolicy(config=sac_cfg)
    algorithm = make_algorithm(cfg=SACAlgorithmConfig.from_policy_config(sac_cfg), policy=policy)
    assert algorithm.config.utd_ratio == 1
    assert algorithm.config.policy_update_freq == 1
    assert algorithm.config.grad_clip_norm == 40.0


def test_unknown_algorithm_name_raises_in_registry():
    """The ChoiceRegistry is the source of truth for unknown algorithm names."""
    with pytest.raises(KeyError):
        RLAlgorithmConfig.get_choice_class("unknown_algo")


# ===========================================================================
# load_weights (round-trip with get_weights)
# ===========================================================================


def test_load_weights_round_trip():
    """get_weights -> load_weights should restore identical parameters on a fresh policy."""
    algo_src, _ = _make_algorithm(state_dim=10, action_dim=6)
    algo_src.update(_batch_iterator())

    sac_cfg = _make_sac_config(state_dim=10, action_dim=6)
    policy_dst = GaussianActorPolicy(config=sac_cfg)
    algo_dst = SACAlgorithm(policy=policy_dst, config=algo_src.config)

    weights = algo_src.get_weights()
    algo_dst.load_weights(weights, device="cpu")

    dst_actor_state_dict = algo_dst.policy.actor.state_dict()
    for key, tensor in weights["policy"].items():
        assert torch.equal(
            dst_actor_state_dict[key].cpu(),
            tensor.cpu(),
        ), f"Policy param '{key}' mismatch after load_weights"


def test_load_weights_round_trip_with_discrete_critic():
    algo_src, _ = _make_algorithm(num_discrete_actions=3, action_dim=6)
    algo_src.update(_batch_iterator(action_dim=7))

    sac_cfg = _make_sac_config(num_discrete_actions=3, action_dim=6)
    policy_dst = GaussianActorPolicy(config=sac_cfg)
    algo_dst = SACAlgorithm(policy=policy_dst, config=algo_src.config)

    weights = algo_src.get_weights()
    algo_dst.load_weights(weights, device="cpu")

    assert "discrete_critic" in weights
    assert len(weights["discrete_critic"]) > 0
    dst_discrete_critic_state_dict = algo_dst.policy.discrete_critic.state_dict()
    for key, tensor in weights["discrete_critic"].items():
        assert torch.equal(
            dst_discrete_critic_state_dict[key].cpu(),
            tensor.cpu(),
        ), f"Discrete critic param '{key}' mismatch after load_weights"


def test_load_weights_ignores_missing_discrete_critic():
    """load_weights should not fail when weights lack discrete_critic on a non-discrete policy."""
    algorithm, _ = _make_algorithm()
    weights = algorithm.get_weights()
    algorithm.load_weights(weights, device="cpu")


def test_actor_side_weight_sync_with_discrete_critic():
    """End-to-end: learner ``algorithm.get_weights()`` -> actor ``algorithm.load_weights()``."""
    # Learner side: train the source algorithm so its weights diverge from init.
    algo_src, _ = _make_algorithm(num_discrete_actions=3, action_dim=6)
    algo_src.update(_batch_iterator(action_dim=7))
    weights = algo_src.get_weights()

    # Actor side: fresh policy + fresh algorithm holding it.
    sac_cfg = _make_sac_config(num_discrete_actions=3, action_dim=6)
    policy_actor = GaussianActorPolicy(config=sac_cfg)
    algo_actor = SACAlgorithm(
        policy=policy_actor,
        config=SACAlgorithmConfig.from_policy_config(sac_cfg),
    )

    # Snapshot initial actor state for the "did it change?" assertion below.
    initial_discrete_critic_state_dict = {
        k: v.clone() for k, v in policy_actor.discrete_critic.state_dict().items()
    }

    algo_actor.load_weights(weights, device="cpu")

    # Actor weights match the learner's exported actor state dict.
    actor_state_dict = policy_actor.actor.state_dict()
    for key, tensor in weights["policy"].items():
        assert torch.equal(actor_state_dict[key].cpu(), tensor.cpu()), (
            f"Actor param '{key}' not synced by algorithm.load_weights"
        )

    # Discrete critic weights match the learner's exported discrete critic.
    discrete_critic_state_dict = policy_actor.discrete_critic.state_dict()
    for key, tensor in weights["discrete_critic"].items():
        assert torch.equal(discrete_critic_state_dict[key].cpu(), tensor.cpu()), (
            f"Discrete critic param '{key}' not synced by algorithm.load_weights"
        )

    # Sanity: the discrete critic actually changed (otherwise the sync is trivial).
    changed = any(
        not torch.equal(initial_discrete_critic_state_dict[key], discrete_critic_state_dict[key])
        for key in initial_discrete_critic_state_dict
        if key in discrete_critic_state_dict
    )
    assert changed, "Discrete critic weights did not change between init and after sync"


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


# ===========================================================================
# Registry-driven make_algorithm
# ===========================================================================


def test_make_algorithm_builds_sac():
    """make_algorithm should look up the SAC class from the registry and instantiate it."""
    sac_cfg = _make_sac_config()
    algo_config = SACAlgorithmConfig.from_policy_config(sac_cfg)
    algo_config.utd_ratio = 2
    policy = GaussianActorPolicy(config=sac_cfg)

    algorithm = make_algorithm(cfg=algo_config, policy=policy)
    assert isinstance(algorithm, SACAlgorithm)
    assert algorithm.config.utd_ratio == 2


# ===========================================================================
# state_dict / load_state_dict (algorithm-side resume)
# ===========================================================================


def test_state_dict_contains_algorithm_owned_tensors():
    """state_dict should pack critics, target networks, and log_alpha (no encoder bloat)."""
    algorithm, _ = _make_algorithm()
    sd = algorithm.state_dict()

    assert "log_alpha" in sd
    assert any(k.startswith("critic_ensemble.") for k in sd)
    assert any(k.startswith("critic_target.") for k in sd)
    # encoder weights live on the policy and must not be duplicated here.
    assert not any(".encoder." in k for k in sd)


def test_state_dict_includes_discrete_critic_target_when_present():
    algorithm, _ = _make_algorithm(num_discrete_actions=3, action_dim=6)
    sd = algorithm.state_dict()
    assert any(k.startswith("discrete_critic_target.") for k in sd)


def test_load_state_dict_round_trip_restores_critics_and_log_alpha():
    """state_dict -> load_state_dict on a fresh algorithm restores all bytes exactly."""
    sac_cfg = _make_sac_config(num_discrete_actions=3, action_dim=6)
    src_policy = GaussianActorPolicy(config=sac_cfg)
    src = SACAlgorithm(policy=src_policy, config=SACAlgorithmConfig.from_policy_config(sac_cfg))
    src.make_optimizers_and_scheduler()
    # Train a few steps so weights diverge from init (action_dim=7 = 6 continuous + 1 discrete).
    src.update(_batch_iterator(action_dim=7))
    src.update(_batch_iterator(action_dim=7))

    dst_policy = GaussianActorPolicy(config=sac_cfg)
    dst = SACAlgorithm(policy=dst_policy, config=SACAlgorithmConfig.from_policy_config(sac_cfg))
    dst.make_optimizers_and_scheduler()

    src_sd = src.state_dict()
    dst.load_state_dict(src_sd)
    dst_sd = dst.state_dict()

    assert set(dst_sd) == set(src_sd)
    for key in src_sd:
        assert torch.allclose(src_sd[key].cpu(), dst_sd[key].cpu()), f"{key} mismatch after round-trip"


def test_load_state_dict_preserves_log_alpha_parameter_identity():
    """The temperature optimizer holds a reference to log_alpha; identity must survive load."""
    algorithm, _ = _make_algorithm()
    log_alpha_id_before = id(algorithm.log_alpha)
    optimizer_param_id = id(algorithm.optimizers["temperature"].param_groups[0]["params"][0])
    assert log_alpha_id_before == optimizer_param_id

    new_state = algorithm.state_dict()
    new_state["log_alpha"] = torch.tensor([0.42])
    algorithm.load_state_dict(new_state)

    assert id(algorithm.log_alpha) == log_alpha_id_before
    assert id(algorithm.optimizers["temperature"].param_groups[0]["params"][0]) == log_alpha_id_before
    assert torch.allclose(algorithm.log_alpha.detach().cpu(), torch.tensor([0.42]))


def test_save_pretrained_round_trip_via_disk(tmp_path):
    """End-to-end: save_pretrained -> from_pretrained restores tensors and config."""
    sac_cfg = _make_sac_config()
    src_policy = GaussianActorPolicy(config=sac_cfg)
    src = SACAlgorithm(policy=src_policy, config=SACAlgorithmConfig.from_policy_config(sac_cfg))
    src.make_optimizers_and_scheduler()
    src.update(_batch_iterator())

    save_dir = tmp_path / "algorithm"
    src.save_pretrained(save_dir)
    assert (save_dir / "model.safetensors").is_file()
    assert (save_dir / "config.json").is_file()

    dst_policy = GaussianActorPolicy(config=sac_cfg)
    dst = SACAlgorithm.from_pretrained(save_dir, policy=dst_policy)

    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    assert set(src_sd) == set(dst_sd)
    for key in src_sd:
        assert torch.allclose(src_sd[key].cpu(), dst_sd[key].cpu()), f"{key} mismatch after disk round-trip"
