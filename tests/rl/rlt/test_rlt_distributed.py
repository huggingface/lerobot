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

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from lerobot.rl.algorithms.rlt import (
    AsyncRLTCollector,
    AsyncRLTLearner,
    RLTActorCriticConfig,
    RLTActorSnapshot,
    RLTCollectorDone,
    RLTCollectorProgress,
    RLTDualReplayBuffer,
    RLTOnlineConfig,
    RLTReplayBuffer,
    RLTTransition,
    RLTTransitionBatch,
    deserialize_rlt_message,
    run_async_rlt_learner,
    serialize_rlt_message,
    transition_has_intervention,
)
from lerobot.rl.algorithms.rlt.distributed import get_latest
from lerobot.rl.algorithms.rlt.online import ChunkOutcome


def _config(*, total_env_steps: int = 2, utd_ratio: int = 2) -> RLTOnlineConfig:
    return RLTOnlineConfig(
        actor_critic=RLTActorCriticConfig(
            rl_token_dim=1,
            proprio_dim=1,
            action_dim=1,
            chunk_length=2,
            hidden_dim=8,
            hidden_layers=1,
            fixed_std=0.0,
            reference_dropout=0.0,
        ),
        batch_size=2,
        utd_ratio=utd_ratio,
        critic_updates_per_actor=1,
        replay_capacity=16,
        expert_replay_capacity=16,
        online_sample_ratio=0.5,
        stride=1,
        warmup_env_steps=0,
        total_env_steps=total_env_steps,
        max_episode_steps=4,
        device="cpu",
    )


def _transition(
    value: float,
    *,
    intervention_mask: tuple[bool, bool] = (False, False),
    valid_horizon: int = 2,
) -> RLTTransition:
    return RLTTransition(
        state=torch.full((2,), value),
        reference=torch.full((2, 1), value),
        action=torch.full((2, 1), value),
        discounted_return=torch.tensor(value),
        next_state=torch.full((2,), value + 1),
        next_reference=torch.full((2, 1), value + 1),
        bootstrap_discount=torch.tensor(0.99),
        terminated=torch.tensor(False),
        truncated=torch.tensor(False),
        valid_horizon=torch.tensor(valid_horizon),
        intervention_mask=torch.tensor(intervention_mask),
    )


def _replay() -> RLTReplayBuffer:
    return RLTReplayBuffer(
        capacity=16,
        state_dim=2,
        chunk_length=2,
        action_dim=1,
        device="cpu",
    )


def test_transition_intervention_ignores_padded_actions() -> None:
    padded_only = _transition(1.0, intervention_mask=(False, True), valid_horizon=1)
    real_intervention = _transition(2.0, intervention_mask=(True, False), valid_horizon=1)

    assert not transition_has_intervention(padded_only)
    assert transition_has_intervention(real_intervention)


def test_dual_replay_duplicates_interventions_and_mixes_exact_ratio() -> None:
    buffers = RLTDualReplayBuffer(_replay(), _replay(), online_ratio=0.5)
    normal = _transition(1.0)
    padded_only = _transition(2.0, intervention_mask=(False, True), valid_horizon=1)
    intervention = _transition(3.0, intervention_mask=(True, False))

    assert buffers.add_online([normal, padded_only, intervention]) == 3
    assert len(buffers.online) == 3
    assert len(buffers.expert) == 1
    torch.testing.assert_close(buffers.expert.state[0], intervention.state)

    source_buffers = RLTDualReplayBuffer(_replay(), _replay(), online_ratio=0.5)
    source_buffers.add_online([_transition(10.0)])
    source_buffers.add_expert([_transition(20.0)])
    batch = source_buffers.sample(6)

    assert batch["state"].shape[0] == 6
    torch.testing.assert_close(batch["state"][:3], torch.full((3, 2), 10.0))
    torch.testing.assert_close(batch["state"][3:], torch.full((3, 2), 20.0))


def test_transition_cpu_returns_detached_independent_storage() -> None:
    transition = _transition(4.0)
    transition.state.requires_grad_(True)
    copied = transition.cpu()

    for field in fields(RLTTransition):
        original_tensor = getattr(transition, field.name)
        copied_tensor = getattr(copied, field.name)
        assert copied_tensor.device.type == "cpu"
        assert copied_tensor.data_ptr() != original_tensor.data_ptr()
        assert not copied_tensor.requires_grad

    with torch.no_grad():
        transition.state.fill_(99.0)
    torch.testing.assert_close(copied.state, torch.full((2,), 4.0))


def test_async_learner_routes_interventions_and_pays_exact_utd_budget(tmp_path: Path) -> None:
    config = _config(utd_ratio=2)
    transition_queue: Queue = Queue()
    parameter_queue: Queue = Queue()
    progress = RLTCollectorProgress(
        env_steps=2,
        episodes=1,
        successes=1,
        intervention_steps=1,
    )
    transition_queue.put(
        RLTTransitionBatch(
            transitions=(
                _transition(1.0),
                _transition(2.0, intervention_mask=(False, True)),
            ),
            progress=progress,
            episode_done=True,
        )
    )
    transition_queue.put(RLTCollectorDone(progress=progress))
    learner = AsyncRLTLearner(
        config,
        transition_queue,
        parameter_queue,
        output_dir=tmp_path,
        parameter_push_interval_s=0.0,
        max_updates_per_cycle=2,
        checkpoint_freq_episodes=0,
    )

    result = learner.run()

    assert result.error is None
    assert result.progress == progress
    assert result.online_buffer_size == 2
    assert result.expert_buffer_size == 1
    assert result.learner_state.transitions_received == 2
    assert result.learner_state.intervention_transitions == 1
    assert result.learner_state.gradient_updates == 4
    assert result.learner_state.update_budget == 0
    assert learner.agent.actor_updates == 4
    latest_snapshot = deserialize_rlt_message(get_latest(parameter_queue))
    assert isinstance(latest_snapshot, RLTActorSnapshot)
    assert latest_snapshot.ready
    assert (tmp_path / "rlt_training_state.pt").is_file()
    assert (tmp_path / "rlt_actor.pt").is_file()

    restored = AsyncRLTLearner(
        config,
        Queue(),
        Queue(),
        output_dir=tmp_path / "restored",
        checkpoint_freq_episodes=0,
        resume=tmp_path / "rlt_training_state.pt",
    )
    assert restored.state == result.learner_state
    assert restored.progress == progress
    assert len(restored.buffers.online) == 2
    assert len(restored.buffers.expert) == 1


class _FakeActorAgent:
    def __init__(self) -> None:
        self.loaded: list[dict[str, torch.Tensor]] = []

    def load_actor_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.loaded.append(state)


class _FakeCollector:
    replay = None

    def __init__(self, parameter_queue: Queue) -> None:
        self.parameter_queue = parameter_queue
        self.assembler = SimpleNamespace(pending=None)
        self.used_actor: list[bool] = []

    def reset(self) -> None:
        return None

    def run_chunk(self, *, use_actor: bool, max_steps: int) -> ChunkOutcome:
        self.used_actor.append(use_actor)
        if len(self.used_actor) == 1:
            self.parameter_queue.put(
                RLTActorSnapshot(
                    actor={"weight": torch.tensor([1.0])},
                    version=1,
                    critic_updates=1,
                    actor_updates=1,
                )
            )
        return ChunkOutcome(
            steps=min(1, max_steps),
            intervention_steps=int(len(self.used_actor) == 1),
            transitions_emitted=0,
            episode_done=False,
            terminated=False,
            truncated=False,
            success=False,
            reward=0.0,
            info={},
            transitions=(),
        )


def test_async_collector_switches_weights_only_between_chunks() -> None:
    config = _config(total_env_steps=2, utd_ratio=0)
    transition_queue: Queue = Queue()
    parameter_queue: Queue = Queue()
    fake_agent = _FakeActorAgent()
    fake_collector = _FakeCollector(parameter_queue)
    collector = AsyncRLTCollector(
        config,
        fake_agent,
        fake_collector,
        transition_queue,
        parameter_queue,
        learner_is_alive=lambda: True,
    )

    progress = collector.train()

    assert fake_collector.used_actor == [False, True]
    assert len(fake_agent.loaded) == 1
    assert progress.intervention_steps == 1
    assert isinstance(deserialize_rlt_message(transition_queue.get_nowait()), RLTCollectorDone)


def test_spawned_learner_process_smoke(tmp_path: Path) -> None:
    config = _config(utd_ratio=1)
    context = mp.get_context("spawn")
    transition_queue = context.Queue()
    parameter_queue = context.Queue()
    result_queue = context.Queue(maxsize=1)
    progress = RLTCollectorProgress(env_steps=2, episodes=1, intervention_steps=1)
    transition_queue.put(
        serialize_rlt_message(
            RLTTransitionBatch(
                transitions=(
                    _transition(1.0),
                    _transition(2.0, intervention_mask=(True, False)),
                ),
                progress=progress,
                episode_done=True,
            )
        )
    )
    transition_queue.put(serialize_rlt_message(RLTCollectorDone(progress=progress)))
    process = context.Process(
        target=run_async_rlt_learner,
        args=(
            config,
            transition_queue,
            parameter_queue,
            result_queue,
            tmp_path,
            0.0,
            2,
            0.01,
            0,
            None,
            None,
        ),
    )

    try:
        process.start()
        result = result_queue.get(timeout=30.0)
        process.join(timeout=30.0)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert result.error is None
        assert result.learner_state.gradient_updates == 2
        assert result.online_buffer_size == 2
        assert result.expert_buffer_size == 1
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        for queue in (transition_queue, parameter_queue, result_queue):
            queue.close()
            queue.cancel_join_thread()
