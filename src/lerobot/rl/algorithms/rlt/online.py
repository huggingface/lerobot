#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.rl_token.modeling_rl_token import RLTokenModel

from .configuration_rlt import RLTOnlineConfig
from .modeling_rlt import RLTAgent
from .replay import ChunkTransitionAssembler, ExecutedChunk, RLTReplayBuffer, RLTTransition


@dataclass(frozen=True)
class VLAInference:
    final_tokens: Tensor
    token_mask: Tensor
    reference_actions: Tensor
    proprio: Tensor


class VLAContextProvider(Protocol):
    def infer(self, observation: Any) -> VLAInference: ...


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    executed_action: Tensor | None = None
    intervened: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("a step cannot be both terminated and truncated")
        if self.intervened and self.executed_action is None:
            raise ValueError("an intervention must report the normalized executed_action")


class ChunkEnvironment(Protocol):
    def reset(self) -> Any: ...

    def step(self, action: Tensor) -> StepResult: ...


@dataclass(frozen=True)
class RLTContext:
    state: Tensor
    reference: Tensor


@dataclass(frozen=True)
class RLTPlan:
    state: Tensor
    reference: Tensor
    actions: Tensor


class RLTController:
    """Combine a frozen VLA, frozen RL token, and the online chunk actor."""

    def __init__(
        self,
        provider: VLAContextProvider,
        rl_token: RLTokenModel,
        agent: RLTAgent,
    ) -> None:
        self.provider = provider
        self.agent = agent
        self.rl_token_model = rl_token.to(agent.device).eval().requires_grad_(False)
        self.chunk_length = agent.config.actor_critic.chunk_length
        self.action_dim = agent.config.actor_critic.action_dim

    @torch.no_grad()
    def context(self, observation: Any) -> RLTContext:
        inference = self.provider.infer(observation)
        embeddings = inference.final_tokens.to(self.agent.device)
        token_mask = inference.token_mask.to(self.agent.device)
        if embeddings.shape[0] != 1:
            raise ValueError("online RLT collection currently expects a single environment")
        token = self.rl_token_model.rl_token(embeddings, token_mask)
        proprio = inference.proprio.to(device=token.device, dtype=token.dtype)
        expected_proprio = self.agent.config.actor_critic.proprio_dim
        if proprio.shape != (1, expected_proprio):
            raise ValueError(f"VLA proprio must have shape [1, {expected_proprio}]")

        reference = inference.reference_actions.to(device=token.device, dtype=token.dtype)
        if reference.ndim != 3 or reference.shape[1] < self.chunk_length:
            raise ValueError("VLA reference horizon is shorter than the RLT chunk")
        if reference.shape[2] < self.action_dim:
            raise ValueError("VLA reference action width is smaller than the RLT action width")
        state = torch.cat([token, proprio], dim=-1)
        return RLTContext(
            state=state,
            reference=reference[:, : self.chunk_length, : self.action_dim],
        )

    @torch.no_grad()
    def plan(self, observation: Any, *, use_actor: bool, deterministic: bool = False) -> RLTPlan:
        context = self.context(observation)
        actions = (
            self.agent.act(context.state, context.reference, deterministic=deterministic)
            if use_actor
            else context.reference
        )
        return RLTPlan(
            state=context.state[0],
            reference=context.reference[0],
            actions=actions[0],
        )


@dataclass(frozen=True)
class ChunkOutcome:
    steps: int
    intervention_steps: int
    transitions_emitted: int
    episode_done: bool
    terminated: bool
    truncated: bool
    success: bool
    reward: float
    info: dict[str, Any]
    transitions: tuple[RLTTransition, ...]


class RLTChunkCollector:
    """Execute chunks and emit stride-overlapping transitions after every chunk."""

    def __init__(
        self,
        env: ChunkEnvironment,
        controller: RLTController,
        assembler: ChunkTransitionAssembler,
        replay: RLTReplayBuffer | None,
        max_episode_steps: int,
    ) -> None:
        self.env = env
        self.controller = controller
        self.assembler = assembler
        self.replay = replay
        self.max_episode_steps = max_episode_steps
        self.observation: Any | None = None
        self.episode_steps = 0

    def reset(self) -> Any:
        self.assembler.start_episode()
        self.observation = self.env.reset()
        self.episode_steps = 0
        return self.observation

    def run_chunk(
        self,
        *,
        use_actor: bool,
        deterministic: bool = False,
        max_steps: int | None = None,
    ) -> ChunkOutcome:
        if self.observation is None:
            raise RuntimeError("call reset before run_chunk")

        c = self.controller.chunk_length
        stride = self.assembler.stride
        step_budget = c if max_steps is None else min(c, max_steps)
        plan = self.controller.plan(self.observation, use_actor=use_actor, deterministic=deterministic)
        states = [plan.state]
        references = [plan.reference]
        actions = torch.zeros_like(plan.actions)
        rewards = torch.zeros(c, dtype=torch.float32, device=plan.actions.device)
        intervention_mask = torch.zeros(c, dtype=torch.bool, device=plan.actions.device)
        last_result: StepResult | None = None

        for step_index in range(step_budget):
            if step_index > 0 and step_index % stride == 0:
                offset_context = self.controller.context(self.observation)
                states.append(offset_context.state[0])
                references.append(offset_context.reference[0])

            planned_action = plan.actions[step_index]
            result = self.env.step(planned_action)
            executed_action = result.executed_action if result.executed_action is not None else planned_action
            actions[step_index] = executed_action.to(actions)
            intervention_mask[step_index] = result.intervened
            rewards[step_index] = float(result.reward)
            self.observation = result.observation
            self.episode_steps += 1

            hit_trainer_limit = self.episode_steps >= self.max_episode_steps
            hit_budget_limit = step_index + 1 >= step_budget and step_budget < c
            if (hit_trainer_limit or hit_budget_limit) and not result.terminated and not result.truncated:
                result = StepResult(
                    observation=result.observation,
                    reward=result.reward,
                    truncated=True,
                    executed_action=executed_action,
                    intervened=result.intervened,
                    info={
                        **result.info,
                        "trainer_time_limit": hit_trainer_limit,
                        "training_budget_limit": hit_budget_limit,
                    },
                )
            last_result = result
            if result.terminated or result.truncated:
                break

        if last_result is None:
            raise RuntimeError("run_chunk requires a positive step budget")
        executed_steps = step_index + 1
        final_context = self.controller.context(self.observation)
        record = ExecutedChunk(
            states=torch.stack(states),
            references=torch.stack(references),
            actions=actions,
            rewards=rewards,
            intervention_mask=intervention_mask,
            executed_steps=executed_steps,
            final_state=final_context.state[0],
            final_reference=final_context.reference[0],
            terminated=last_result.terminated,
            truncated=last_result.truncated,
        )
        transitions = self.assembler.add_chunk(record)
        emitted = len(transitions)
        if self.replay is not None:
            self.replay.add_many(transitions)
        done = last_result.terminated or last_result.truncated
        return ChunkOutcome(
            steps=executed_steps,
            intervention_steps=int(intervention_mask[:executed_steps].sum().item()),
            transitions_emitted=emitted,
            episode_done=done,
            terminated=last_result.terminated,
            truncated=last_result.truncated,
            success=last_result.success,
            reward=float(rewards.sum()),
            info=last_result.info,
            transitions=tuple(transitions),
        )


@dataclass
class RLTTrainingState:
    env_steps: int = 0
    episodes: int = 0
    successes: int = 0
    intervention_steps: int = 0
    update_budget: int = 0
    gradient_updates: int = 0


class OnlineRLTTrainer:
    """Synchronous reference loop with UTD counted per newly emitted transition."""

    def __init__(
        self,
        config: RLTOnlineConfig,
        agent: RLTAgent,
        replay: RLTReplayBuffer,
        collector: RLTChunkCollector,
    ) -> None:
        self.config = config
        self.agent = agent
        self.replay = replay
        self.collector = collector
        self.state = RLTTrainingState()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    @property
    def actor_ready(self) -> bool:
        return (
            len(self.replay) >= self.config.batch_size
            and self.agent.actor_updates > 0
            and self.state.update_budget == 0
        )

    def train(self, log_fn=None) -> RLTTrainingState:
        self.collector.reset()
        metrics: dict[str, float] = {}
        while self.state.env_steps < self.config.total_env_steps:
            use_actor = self.state.env_steps >= self.config.warmup_env_steps and self.actor_ready
            remaining = self.config.total_env_steps - self.state.env_steps
            outcome = self.collector.run_chunk(use_actor=use_actor, max_steps=remaining)
            self.state.env_steps += outcome.steps
            self.state.intervention_steps += outcome.intervention_steps
            self.state.update_budget += outcome.transitions_emitted * self.config.utd_ratio
            metrics = self._pay_update_budget(metrics)

            if outcome.episode_done:
                self.state.episodes += 1
                self.state.successes += int(outcome.success)
            if log_fn is not None:
                log_fn(self._metrics(outcome, use_actor, metrics))
            if outcome.episode_done and self.state.env_steps < self.config.total_env_steps:
                self.collector.reset()

        if self.collector.assembler.pending is not None:
            pending = self.collector.assembler.pending
            transitions = self.collector.assembler.truncate_pending(
                pending.final_state, pending.final_reference
            )
            emitted = self.replay.add_many(transitions)
            self.state.update_budget += emitted * self.config.utd_ratio
            self._pay_update_budget(metrics)
        return self.state

    def _pay_update_budget(self, metrics: dict[str, float]) -> dict[str, float]:
        if len(self.replay) < self.config.batch_size:
            return metrics
        while self.state.update_budget > 0:
            metrics = self.agent.update(lambda: self.replay.sample(self.config.batch_size))
            self.state.update_budget -= 1
            self.state.gradient_updates += 1
        return metrics

    def _metrics(
        self,
        outcome: ChunkOutcome,
        used_actor: bool,
        update_metrics: dict[str, float],
    ) -> dict[str, float]:
        metrics = {
            "env_steps": float(self.state.env_steps),
            "episodes": float(self.state.episodes),
            "successes": float(self.state.successes),
            "intervention_steps": float(self.state.intervention_steps),
            "buffer_size": float(len(self.replay)),
            "gradient_updates": float(self.state.gradient_updates),
            "update_budget": float(self.state.update_budget),
            "used_actor": float(used_actor),
            "chunk_reward": outcome.reward,
        }
        metrics.update(update_metrics)
        return metrics

    def state_dict(self) -> dict[str, object]:
        if self.collector.assembler.pending is not None:
            raise RuntimeError("checkpoint online RLT only at an episode boundary")
        state: dict[str, object] = {
            "training_state": asdict(self.state),
            "agent": self.agent.training_state_dict(),
            "replay": self.replay.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.state = RLTTrainingState(**state["training_state"])
        self.agent.load_training_state_dict(state["agent"])
        self.replay.load_state_dict(state["replay"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if "cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])


@torch.no_grad()
def evaluate(
    env: ChunkEnvironment,
    controller: RLTController,
    *,
    episodes: int,
    max_episode_steps: int,
) -> list[dict[str, float | bool]]:
    results: list[dict[str, float | bool]] = []
    for _ in range(episodes):
        observation = env.reset()
        episode_return = 0.0
        steps = 0
        success = False
        done = False
        while not done and steps < max_episode_steps:
            plan = controller.plan(observation, use_actor=True, deterministic=True)
            for action in plan.actions:
                result = env.step(action)
                observation = result.observation
                episode_return += float(result.reward)
                steps += 1
                success = result.success
                done = result.terminated or result.truncated
                if done or steps >= max_episode_steps:
                    break
        results.append({"return": episode_return, "steps": float(steps), "success": success})
    return results
