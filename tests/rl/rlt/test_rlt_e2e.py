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

import unittest

import torch

from lerobot.policies.rl_token import RLTokenConfig, RLTokenModel, RLTokenStage1Trainer
from lerobot.rl.algorithms.rlt import (
    ChunkTransitionAssembler,
    RLTActorCriticConfig,
    RLTAgent,
    RLTOnlineConfig,
    RLTReplayBuffer,
)
from lerobot.rl.algorithms.rlt.online import (
    OnlineRLTTrainer,
    RLTChunkCollector,
    RLTController,
    StepResult,
    VLAInference,
    evaluate,
)


class _TinyVLAProvider:
    def __init__(self, chunk_length: int) -> None:
        self.chunk_length = chunk_length

    def infer(self, observation: torch.Tensor) -> VLAInference:
        proprio = observation.reshape(1, 2).float()
        position, target = proprio[0]
        base = torch.stack(
            [
                proprio[0],
                torch.tensor([position.square(), target.square()]),
                torch.tensor([target - position, position - target]),
            ]
        ).flatten()
        tokens = torch.stack([base, base.roll(1), base.roll(2), base.roll(3)]).unsqueeze(0)
        action = (target - position).clamp(-0.3, 0.3).reshape(1, 1, 1)
        reference = action.expand(1, self.chunk_length, 1).clone()
        return VLAInference(
            final_tokens=tokens,
            token_mask=torch.ones(1, 4, dtype=torch.bool),
            reference_actions=reference,
            proprio=proprio,
        )


class _TinyReachEnv:
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


class RLTEndToEndTest(unittest.TestCase):
    def test_stage1_collection_online_updates_and_evaluation(self) -> None:
        torch.manual_seed(0)
        provider = _TinyVLAProvider(chunk_length=4)
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
            metrics = stage1.step(inference.final_tokens, inference.token_mask)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

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
            total_env_steps=16,
            max_episode_steps=8,
            device="cpu",
        )
        agent = RLTAgent(online_config)
        replay = RLTReplayBuffer(
            capacity=online_config.replay_capacity,
            state_dim=actor_config.state_dim,
            chunk_length=actor_config.chunk_length,
            action_dim=actor_config.action_dim,
            device="cpu",
        )
        controller = RLTController(provider, token_model, agent)
        assembler = ChunkTransitionAssembler(
            chunk_length=actor_config.chunk_length,
            action_dim=actor_config.action_dim,
            discount=online_config.discount,
            stride=online_config.stride,
        )
        collector = RLTChunkCollector(
            _TinyReachEnv(),
            controller,
            assembler,
            replay,
            max_episode_steps=online_config.max_episode_steps,
        )
        trainer = OnlineRLTTrainer(online_config, agent, replay, collector)
        state = trainer.train()

        self.assertEqual(state.env_steps, online_config.total_env_steps)
        self.assertGreater(len(replay), 0)
        self.assertGreater(state.gradient_updates, 0)
        self.assertGreater(agent.actor_updates, 0)
        self.assertIsNone(assembler.pending)
        saved_state = trainer.state_dict()

        restored_agent = RLTAgent(online_config)
        restored_replay = RLTReplayBuffer(
            capacity=online_config.replay_capacity,
            state_dim=actor_config.state_dim,
            chunk_length=actor_config.chunk_length,
            action_dim=actor_config.action_dim,
            device="cpu",
        )
        restored_assembler = ChunkTransitionAssembler(
            chunk_length=actor_config.chunk_length,
            action_dim=actor_config.action_dim,
            discount=online_config.discount,
            stride=online_config.stride,
        )
        restored_controller = RLTController(provider, token_model, restored_agent)
        restored_collector = RLTChunkCollector(
            _TinyReachEnv(),
            restored_controller,
            restored_assembler,
            restored_replay,
            max_episode_steps=online_config.max_episode_steps,
        )
        restored_trainer = OnlineRLTTrainer(
            online_config, restored_agent, restored_replay, restored_collector
        )
        restored_trainer.load_state_dict(saved_state)
        self.assertEqual(restored_trainer.state, state)
        self.assertEqual(len(restored_replay), len(replay))
        self.assertEqual(restored_agent.actor_updates, agent.actor_updates)

        results = evaluate(_TinyReachEnv(), controller, episodes=2, max_episode_steps=8)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["steps"] > 0 for result in results))


if __name__ == "__main__":
    unittest.main()
