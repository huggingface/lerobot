#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""QF3 clip gradient, checkpoint roundtrip, and actor-learner wiring."""

import math
from dataclasses import dataclass
from pathlib import Path

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.flow_actor.configuration_flow_actor import FlowActorConfig  # noqa: E402
from lerobot.policies.flow_actor.modeling_flow_actor import FlowActorPolicy  # noqa: E402
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (  # noqa: E402
    ActorLearnerConfig,
    ConcurrencyConfig,
    GaussianActorConfig,
)
from lerobot.rl.algorithms.configs import RLAlgorithmConfig  # noqa: E402
from lerobot.rl.algorithms.factory import get_algorithm_class, make_algorithm  # noqa: E402
from lerobot.rl.algorithms.qf3.configuration_qf3 import QF3AlgorithmConfig  # noqa: E402
from lerobot.rl.algorithms.qf3.qf3_algorithm import (  # noqa: E402
    QF3Algorithm,
    qf3_actor_terms,
)
from lerobot.rl.train_rl import TrainRLServerPipelineConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


class _BiasVelocity(nn.Module):
    def __init__(self, value: Tensor) -> None:
        super().__init__()
        self.bias = nn.Parameter(value.clone())

    def forward(self, state: Tensor, action_sample: Tensor, flow_time: Tensor) -> Tensor:
        del action_sample, flow_time
        return self.bias.expand(state.shape[0], -1)


class _SumQ(nn.Module):
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        del state
        return action.sum(dim=-1, keepdim=True)


def _policy_config() -> FlowActorConfig:
    state_dim = 4
    action_dim = 3
    config = FlowActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        hidden_dims=[16, 16],
        time_dim=8,
        euler_steps=5,
        device="cpu",
    )
    config.validate_features()
    return config


def test_velocity_clip_gradient_matches_strict_mask():
    """Q reaches only coordinates with |delta| < alpha. Equality is masked."""
    velocity = torch.tensor([0.5, 2.0, 1.0])
    actor = _BiasVelocity(velocity)
    q1 = _SumQ()
    state = torch.zeros(1, 2)
    replay = torch.zeros(1, 3)
    noise = torch.zeros(1, 3)
    flow_time = torch.tensor([[0.25]])
    alpha = 1.0
    lambda_cfm = 0.2
    terms = qf3_actor_terms(
        actor,
        q1,
        state,
        replay,
        noise,
        flow_time,
        alpha=alpha,
        lambda_cfm=lambda_cfm,
        use_q=True,
        clip=True,
    )
    grad = torch.autograd.grad(terms.loss, actor.bias)[0]
    # d(loss)/dv = -(1-tau) * 1[|v|<alpha] + lambda_cfm * 2 * v, with u_buf = 0 and Q = sum(endpoint).
    mask = torch.tensor([1.0, 0.0, 0.0])
    expected = -(1.0 - 0.25) * mask + lambda_cfm * 2.0 * velocity
    assert torch.allclose(grad, expected, atol=1e-5)
    assert float(terms.identity_error) < 1e-6
    assert float(terms.bound_violation) < 1e-5

    direct = qf3_actor_terms(
        actor,
        q1,
        state,
        replay,
        noise,
        flow_time,
        alpha=alpha,
        lambda_cfm=lambda_cfm,
        use_q=True,
        clip=False,
    )
    direct_grad = torch.autograd.grad(direct.loss, actor.bias)[0]
    direct_expected = -(1.0 - 0.25) * torch.ones(3) + lambda_cfm * 2.0 * velocity
    assert torch.allclose(direct_grad, direct_expected, atol=1e-5)
    assert float(direct.bound_violation) > 0.5


def test_qf3_registers_updates_and_roundtrips(tmp_path: Path):
    torch.manual_seed(0)
    assert "flow_actor" in PreTrainedConfig.get_known_choices()
    assert get_policy_class("flow_actor") is FlowActorPolicy
    assert "qf3" in RLAlgorithmConfig.get_known_choices()
    assert get_algorithm_class("qf3") is QF3Algorithm

    config = _policy_config()
    policy = FlowActorPolicy(config)
    zeros = torch.zeros(1, 3)
    action = policy.select_action({OBS_STATE: torch.zeros(1, 4)}, noise=zeros)
    assert torch.allclose(action, zeros, atol=1e-6)

    preprocessor, postprocessor = make_pre_post_processors(config)
    processed = preprocessor({OBS_STATE: torch.zeros(4)})
    assert processed[OBS_STATE].shape[-1] == 4
    assert torch.allclose(postprocessor(action), action, atol=1e-6)

    algo_config = QF3AlgorithmConfig.from_policy_config(config)
    algo_config.policy_frequency = 1
    algo_config.hidden_dims = [16, 16]
    algorithm = make_algorithm(algo_config, policy)
    algorithm.make_optimizers_and_scheduler()
    before_actor = torch.nn.utils.parameters_to_vector(policy.velocity.parameters()).detach().clone()
    before_critic = torch.nn.utils.parameters_to_vector(algorithm.critic.parameters()).detach().clone()
    batch = {
        "state": {OBS_STATE: torch.randn(8, 4)},
        "action": torch.randn(8, 3),
        "reward": torch.randn(8),
        "next_state": {OBS_STATE: torch.randn(8, 4)},
        "done": torch.zeros(8),
    }
    stats = algorithm.update(iter([batch]))
    after_actor = torch.nn.utils.parameters_to_vector(policy.velocity.parameters()).detach()
    after_critic = torch.nn.utils.parameters_to_vector(algorithm.critic.parameters()).detach()
    assert torch.linalg.vector_norm(after_actor - before_actor) > 0
    assert torch.linalg.vector_norm(after_critic - before_critic) > 0
    assert math.isfinite(stats.losses["loss_critic"])
    assert math.isfinite(stats.losses["loss_actor"])
    assert stats.extra["bound_violation"] < 1e-4

    policy_dir = tmp_path / "policy"
    algo_dir = tmp_path / "algo"
    policy.save_pretrained(policy_dir)
    algorithm.save_pretrained(algo_dir)
    loaded_policy = FlowActorPolicy.from_pretrained(policy_dir)
    loaded = QF3Algorithm.from_pretrained(algo_dir, policy=loaded_policy)
    for key, value in algorithm.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
    for key, value in policy.velocity.state_dict().items():
        assert torch.equal(loaded_policy.velocity.state_dict()[key], value)


@dataclass
class _CLIConfig:
    policy: PreTrainedConfig


def _pipeline(policy: PreTrainedConfig, output_dir: Path, algorithm=None) -> TrainRLServerPipelineConfig:
    policy.push_to_hub = False
    cfg = TrainRLServerPipelineConfig(
        dataset=DatasetConfig(repo_id="local/qf3-service"),
        policy=policy,
        output_dir=output_dir,
        batch_size=4,
        save_checkpoint=False,
        algorithm=algorithm,
    )
    cfg.validate()
    return cfg


def test_flow_actor_service_fields_match_sac_and_parse():
    """The actor-learner fields are the SAC dataclasses, and the CLI can set them."""
    config = FlowActorConfig(device="cpu")
    gaussian = GaussianActorConfig(device="cpu")
    assert type(config.actor_learner_config) is ActorLearnerConfig
    assert type(config.concurrency) is ConcurrencyConfig
    assert config.online_steps == gaussian.online_steps
    assert config.online_buffer_capacity == gaussian.online_buffer_capacity
    assert config.offline_buffer_capacity == gaussian.offline_buffer_capacity
    assert config.online_step_before_learning == gaussian.online_step_before_learning
    assert config.storage_device == gaussian.storage_device
    assert config.num_discrete_actions is None
    assert config.dataset_stats is None

    parsed = draccus.parse(
        _CLIConfig,
        args=[
            "--policy.type=flow_actor",
            "--policy.device=cpu",
            "--policy.online_steps=12",
            "--policy.online_step_before_learning=3",
            "--policy.storage_device=cpu",
            "--policy.actor_learner_config.learner_port=50052",
            "--policy.concurrency.actor=threads",
        ],
    ).policy
    assert isinstance(parsed, FlowActorConfig)
    assert parsed.online_steps == 12
    assert parsed.online_step_before_learning == 3
    assert parsed.actor_learner_config.learner_port == 50052
    assert parsed.concurrency.actor == "threads"
    restored = draccus.decode(FlowActorConfig, draccus.encode(parsed))
    assert restored.online_steps == 12
    assert restored.actor_learner_config.learner_port == 50052
    assert type(restored.concurrency) is ConcurrencyConfig


def test_unset_algorithm_follows_the_policy(tmp_path: Path):
    """An omitted algorithm is QF3 for a flow actor and SAC for a Gaussian actor."""
    flow = FlowActorConfig(device="cpu", euler_steps=2)
    flow_cfg = _pipeline(flow, tmp_path / "flow")
    assert flow_cfg.algorithm.type == "qf3"
    assert flow_cfg.algorithm.policy_config is flow
    assert flow_cfg.algorithm.euler_steps == 2

    gaussian = GaussianActorConfig(device="cpu")
    sac_cfg = _pipeline(gaussian, tmp_path / "sac")
    assert sac_cfg.algorithm.type == "sac"
    assert sac_cfg.algorithm.policy_config is gaussian

    explicit = QF3AlgorithmConfig(euler_steps=4)
    kept = _pipeline(flow, tmp_path / "explicit", algorithm=explicit)
    assert kept.algorithm is explicit
    assert kept.algorithm.euler_steps == 4
    assert kept.algorithm.policy_config is flow


def test_learner_step_updates_qf3_and_actor_loads_weights(tmp_path: Path):
    """One learner step uses the service buffer, then the actor receives those weights."""
    from queue import Queue

    from lerobot.rl.actor import update_policy_parameters
    from lerobot.rl.data_sources import OnlineOfflineMixer
    from lerobot.rl.learner import initialize_replay_buffer, push_actor_policy_to_queue
    from lerobot.rl.trainer import RLTrainer

    torch.manual_seed(0)
    policy_cfg = FlowActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        hidden_dims=[8, 8],
        time_dim=4,
        euler_steps=2,
        device="cpu",
        online_buffer_capacity=16,
        online_step_before_learning=2,
        storage_device="cpu",
    )
    cfg = _pipeline(policy_cfg, tmp_path / "run")
    assert cfg.policy.online_buffer_capacity == 16
    policy = FlowActorPolicy(policy_cfg)
    algorithm = make_algorithm(cfg.algorithm, policy)
    buffer = initialize_replay_buffer(cfg, device="cpu", storage_device=cfg.policy.storage_device)
    for _ in range(4):
        buffer.add(
            state={OBS_STATE: torch.randn(1, 4)},
            action=torch.randn(1, 3),
            reward=0.0,
            next_state={OBS_STATE: torch.randn(1, 4)},
            done=False,
            truncated=False,
        )
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=cfg.policy.dataset_stats,
    )
    trainer = RLTrainer(
        algorithm=algorithm,
        data_mixer=OnlineOfflineMixer(online_buffer=buffer, offline_buffer=None),
        batch_size=cfg.batch_size,
        preprocessor=preprocessor,
    )
    before_critic = torch.nn.utils.parameters_to_vector(algorithm.critic.parameters()).detach().clone()
    stats = trainer.training_step()
    after_critic = torch.nn.utils.parameters_to_vector(algorithm.critic.parameters()).detach()
    assert torch.linalg.vector_norm(after_critic - before_critic) > 0
    assert math.isfinite(stats.losses["loss_critic"])
    assert algorithm.optimization_step == 1

    actor_policy = FlowActorPolicy(policy_cfg)
    actor_algorithm = make_algorithm(cfg.algorithm, actor_policy)
    before_actor = torch.nn.utils.parameters_to_vector(actor_policy.velocity.parameters()).detach().clone()
    trained = torch.nn.utils.parameters_to_vector(policy.velocity.parameters()).detach()
    assert torch.linalg.vector_norm(trained - before_actor) > 0

    # A thread queue makes put visible immediately. A process queue feeds in the background,
    # so a non-blocking read in this same call can observe it empty.
    parameters_queue: Queue = Queue()
    push_actor_policy_to_queue(parameters_queue, algorithm)
    update_policy_parameters(actor_algorithm, parameters_queue, device="cpu")
    loaded = torch.nn.utils.parameters_to_vector(actor_policy.velocity.parameters()).detach()
    assert torch.equal(loaded, trained)
