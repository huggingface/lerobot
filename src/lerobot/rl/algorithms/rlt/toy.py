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

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lerobot.policies.rl_token import RLTokenConfig, RLTokenModel, RLTokenStage1Trainer

from .configuration_rlt import RLTActorCriticConfig, RLTOnlineConfig
from .modeling_rlt import RLTAgent
from .online import (
    OnlineRLTTrainer,
    RLTChunkCollector,
    RLTController,
    StepResult,
    VLAInference,
    evaluate,
)
from .replay import ChunkTransitionAssembler, RLTReplayBuffer


class TinyVLAContextProvider:
    """Small deterministic stand-in implementing the same contract as PI0ContextProvider."""

    def __init__(self, chunk_length: int = 4) -> None:
        self.chunk_length = chunk_length

    def infer(self, observation: torch.Tensor) -> VLAInference:
        proprio = observation.reshape(1, 2).float()
        position, target = proprio[0]
        features = torch.stack(
            [
                proprio[0],
                torch.stack([position.square(), target.square()]),
                torch.stack([target - position, position - target]),
            ]
        ).flatten()
        final_tokens = torch.stack(
            [features, features.roll(1), features.roll(2), features.roll(3)]
        ).unsqueeze(0)
        action = (target - position).clamp(-0.3, 0.3).reshape(1, 1, 1)
        return VLAInference(
            final_tokens=final_tokens,
            token_mask=torch.ones(1, 4, dtype=torch.bool),
            reference_actions=action.expand(1, self.chunk_length, 1).clone(),
            proprio=proprio,
        )


class TinyReachEnvironment:
    def __init__(self) -> None:
        self.position = 0.0
        self.target = 1.0

    def reset(self) -> torch.Tensor:
        self.position = 0.0
        return torch.tensor([self.position, self.target])

    def step(self, action: torch.Tensor) -> StepResult:
        executed = action.detach().cpu().clamp(-1.0, 1.0)
        self.position += float(executed[0])
        success = abs(self.target - self.position) < 0.12
        return StepResult(
            observation=torch.tensor([self.position, self.target]),
            reward=float(success),
            terminated=success,
            success=success,
            executed_action=executed,
        )


@dataclass(frozen=True)
class ToyWorkflowResult:
    stage1_steps: int
    env_steps: int
    episodes: int
    successes: int
    replay_size: int
    gradient_updates: int
    actor_updates: int
    evaluation: list[dict[str, float | bool]]


def run_toy_workflow(output_dir: str | Path, *, total_env_steps: int = 16) -> ToyWorkflowResult:
    """Run Stage 1, warmup, online RLT, evaluation, and checkpoint serialization on CPU."""
    torch.manual_seed(0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = TinyVLAContextProvider(chunk_length=4)

    token_model = RLTokenModel(
        RLTokenConfig(
            vla_dim=6,
            token_dim=4,
            max_tokens=4,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=2,
        )
    )
    stage1 = RLTokenStage1Trainer(token_model, lr=1e-3)
    for position in (0.0, 0.25, 0.5):
        inference = provider.infer(torch.tensor([position, 1.0]))
        stage1.step(inference.final_tokens, inference.token_mask)
    token_model.save_pretrained(output_dir / "rl_token")

    actor_config = RLTActorCriticConfig(
        rl_token_dim=4,
        proprio_dim=2,
        action_dim=1,
        chunk_length=4,
        hidden_dim=32,
        hidden_layers=2,
        fixed_std=0.02,
    )
    online_config = RLTOnlineConfig(
        actor_critic=actor_config,
        batch_size=2,
        utd_ratio=1,
        replay_capacity=64,
        stride=2,
        warmup_env_steps=3,
        total_env_steps=total_env_steps,
        max_episode_steps=8,
        device="cpu",
    )
    agent = RLTAgent(online_config)
    replay = RLTReplayBuffer(
        capacity=online_config.replay_capacity,
        state_dim=actor_config.state_dim,
        chunk_length=actor_config.chunk_length,
        action_dim=actor_config.action_dim,
    )
    assembler = ChunkTransitionAssembler(
        chunk_length=actor_config.chunk_length,
        action_dim=actor_config.action_dim,
        discount=online_config.discount,
        stride=online_config.stride,
    )
    controller = RLTController(provider, token_model, agent)
    collector = RLTChunkCollector(
        TinyReachEnvironment(),
        controller,
        assembler,
        replay,
        max_episode_steps=online_config.max_episode_steps,
    )
    trainer = OnlineRLTTrainer(online_config, agent, replay, collector)
    state = trainer.train()
    evaluation = evaluate(TinyReachEnvironment(), controller, episodes=2, max_episode_steps=8)

    torch.save(trainer.state_dict(), output_dir / "rlt_training_state.pt")
    torch.save(agent.actor.state_dict(), output_dir / "rlt_actor.pt")
    result = ToyWorkflowResult(
        stage1_steps=stage1.steps,
        env_steps=state.env_steps,
        episodes=state.episodes,
        successes=state.successes,
        replay_size=len(replay),
        gradient_updates=state.gradient_updates,
        actor_updates=agent.actor_updates,
        evaluation=evaluation,
    )
    with open(output_dir / "result.json", "w", encoding="utf-8") as result_file:
        json.dump(asdict(result), result_file, indent=2)
    return result
