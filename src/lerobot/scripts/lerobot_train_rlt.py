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

"""Run PI0 + RL-token online chunk reinforcement learning on a LeRobot HIL environment."""

import logging
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty

import torch.multiprocessing as mp

from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0 import PI0Config, PI0Policy
from lerobot.policies.rl_token import RLTokenModel
from lerobot.rl.algorithms.rlt import (
    ChunkTransitionAssembler,
    RLTActorCriticConfig,
    RLTAgent,
    RLTAsyncLearnerResult,
    RLTCollectorDone,
    RLTCollectorProgress,
    RLTOnlineConfig,
    run_async_rlt_learner,
    serialize_rlt_message,
)
from lerobot.rl.algorithms.rlt.distributed import AsyncRLTCollector, load_collector_resume
from lerobot.rl.algorithms.rlt.hil_adapter import HILRLTEnvironment
from lerobot.rl.algorithms.rlt.online import RLTChunkCollector, RLTController
from lerobot.rl.algorithms.rlt.pi0_adapter import PI0ContextProvider, make_pi0_batch_builder
from lerobot.rl.gym_manipulator import make_processors, make_robot_env
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrainRLTConfig:
    env: HILSerlRobotEnvConfig
    pi0_path: str
    rl_token_path: Path
    task: str
    output_dir: Path
    device: str = "cuda"
    chunk_length: int = 10
    stride: int = 2
    discount: float = 0.99
    fixed_std: float = 0.05
    reference_dropout: float = 0.5
    reference_regularization: float = 1.0
    hidden_dim: int = 256
    hidden_layers: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    batch_size: int = 256
    utd_ratio: int = 5
    critic_updates_per_actor: int = 2
    replay_capacity: int = 200_000
    expert_replay_capacity: int = 100_000
    online_sample_ratio: float = 0.5
    warmup_env_steps: int = 2_000
    total_env_steps: int = 100_000
    max_episode_steps: int = 400
    checkpoint_freq_episodes: int = 10
    parameter_push_interval_s: float = 4.0
    learner_max_updates_per_cycle: int = 8
    learner_queue_get_timeout_s: float = 0.05
    learner_device: str | None = None
    expert_replay_path: Path | None = None
    seed: int = 0
    local_files_only: bool = False
    resume: Path | None = None


def _close_hardware(env, teleop_device) -> None:
    if hasattr(env, "close"):
        env.close()
    if teleop_device is not None and hasattr(teleop_device, "disconnect"):
        teleop_device.disconnect()


@parser.wrap()
def train(cfg: TrainRLTConfig) -> None:
    init_logging()
    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.device, log=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pi0_config = PI0Config.from_pretrained(
        cfg.pi0_path,
        local_files_only=cfg.local_files_only,
        device=str(device),
    )
    if pi0_config.use_relative_actions:
        raise NotImplementedError(
            "lerobot-train-rlt currently requires a PI0 checkpoint with use_relative_actions=false"
        )
    if pi0_config.chunk_size < cfg.chunk_length:
        raise ValueError("PI0 action horizon is shorter than the configured RLT chunk length")
    pi0 = PI0Policy.from_pretrained(
        cfg.pi0_path,
        config=pi0_config,
        local_files_only=cfg.local_files_only,
        strict=True,
    ).eval()
    if not pi0._pretrained_weights_loaded:  # noqa: SLF001
        raise RuntimeError(f"PI0 weights were not loaded from {cfg.pi0_path!r}")
    pi0.requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pi0_config,
        pretrained_path=cfg.pi0_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    rl_token = RLTokenModel.from_pretrained(cfg.rl_token_path, map_location=device)

    proprio_feature = pi0_config.input_features.get(OBS_STATE)
    action_feature = pi0_config.output_features.get(ACTION)
    if proprio_feature is None or action_feature is None:
        raise ValueError("PI0 checkpoint must define observation.state and action features")
    actor_critic_config = RLTActorCriticConfig(
        rl_token_dim=rl_token.config.token_dim,
        proprio_dim=proprio_feature.shape[0],
        action_dim=action_feature.shape[0],
        chunk_length=cfg.chunk_length,
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
        fixed_std=cfg.fixed_std,
        reference_dropout=cfg.reference_dropout,
    )
    online_config = RLTOnlineConfig(
        actor_critic=actor_critic_config,
        discount=cfg.discount,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        tau=cfg.tau,
        reference_regularization=cfg.reference_regularization,
        batch_size=cfg.batch_size,
        utd_ratio=cfg.utd_ratio,
        critic_updates_per_actor=cfg.critic_updates_per_actor,
        replay_capacity=cfg.replay_capacity,
        expert_replay_capacity=cfg.expert_replay_capacity,
        online_sample_ratio=cfg.online_sample_ratio,
        stride=cfg.stride,
        warmup_env_steps=cfg.warmup_env_steps,
        total_env_steps=cfg.total_env_steps,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed,
        device=str(device),
    )
    actor_agent = RLTAgent(online_config)
    initial_progress = RLTCollectorProgress()
    initial_actor_updates = 0
    if cfg.resume is not None:
        initial_progress, initial_actor_updates = load_collector_resume(cfg.resume, actor_agent)

    learner_device = cfg.learner_device or str(device)
    learner_config = replace(online_config, device=learner_device)
    assembler = ChunkTransitionAssembler(
        chunk_length=cfg.chunk_length,
        action_dim=actor_critic_config.action_dim,
        discount=cfg.discount,
        stride=cfg.stride,
    )

    raw_env, teleop_device = make_robot_env(cfg.env)
    learner_process = None
    transition_queue = None
    parameter_queue = None
    result_queue = None
    collector_finished = False
    learner_started = False
    async_collector = None
    collection_error: BaseException | None = None
    try:
        env_processor, action_processor = make_processors(raw_env, teleop_device, cfg.env, str(device))
        env = HILRLTEnvironment(
            raw_env,
            env_processor,
            action_processor,
            preprocessor,
            postprocessor,
            use_relative_actions=pi0_config.use_relative_actions,
            fps=cfg.env.fps,
        )
        batch_builder = make_pi0_batch_builder(task=cfg.task, input_features=set(pi0_config.input_features))
        provider = PI0ContextProvider(pi0, batch_builder, preprocessor=preprocessor)
        controller = RLTController(provider, rl_token, actor_agent)
        collector = RLTChunkCollector(
            env,
            controller,
            assembler,
            None,
            max_episode_steps=cfg.max_episode_steps,
        )

        mp_context = mp.get_context("spawn")
        transition_queue = mp_context.Queue()
        # The collector drains this queue and applies only the newest snapshot at chunk boundaries.
        parameter_queue = mp_context.Queue()
        result_queue = mp_context.Queue(maxsize=1)
        learner_process = mp_context.Process(
            target=run_async_rlt_learner,
            args=(
                learner_config,
                transition_queue,
                parameter_queue,
                result_queue,
                cfg.output_dir,
                cfg.parameter_push_interval_s,
                cfg.learner_max_updates_per_cycle,
                cfg.learner_queue_get_timeout_s,
                cfg.checkpoint_freq_episodes,
                cfg.resume,
                cfg.expert_replay_path,
            ),
            name="rlt-learner",
        )
        learner_process.start()
        learner_started = True

        async_collector = AsyncRLTCollector(
            online_config,
            actor_agent,
            collector,
            transition_queue,
            parameter_queue,
            initial_progress=initial_progress,
            initial_actor_updates=initial_actor_updates,
            learner_is_alive=learner_process.is_alive,
        )

        def log_metrics(metrics: dict[str, float]) -> None:
            logger.info(
                "env_steps=%d episodes=%d interventions=%d emitted=%d reward=%.1f actor=%d actor_version=%d",
                int(metrics["env_steps"]),
                int(metrics["episodes"]),
                int(metrics["intervention_steps"]),
                int(metrics["transitions_emitted"]),
                metrics["chunk_reward"],
                int(metrics["used_actor"]),
                int(metrics["actor_version"]),
            )

        async_collector.train(log_fn=log_metrics)
        collector_finished = True
    except BaseException as exc:
        collection_error = exc
    finally:
        _close_hardware(raw_env, teleop_device)

    if learner_process is not None and learner_started:
        if not collector_finished and learner_process.is_alive() and transition_queue is not None:
            progress = async_collector.progress if async_collector is not None else initial_progress
            transition_queue.put(serialize_rlt_message(RLTCollectorDone(progress=progress)))
        learner_process.join()

    learner_result = None
    if result_queue is not None:
        with suppress(Empty):
            learner_result = result_queue.get(timeout=1.0)

    for queue in (transition_queue, parameter_queue, result_queue):
        if queue is not None:
            queue.close()
            queue.cancel_join_thread()

    if collection_error is not None:
        raise collection_error
    if learner_process is None or not learner_started or learner_process.exitcode != 0:
        error = learner_result.error if isinstance(learner_result, RLTAsyncLearnerResult) else None
        raise RuntimeError(
            f"RLT learner failed with exit code "
            f"{None if learner_process is None else learner_process.exitcode}: {error or 'no result'}"
        )
    if not isinstance(learner_result, RLTAsyncLearnerResult):
        raise RuntimeError("RLT learner exited without returning a result")
    if learner_result.error is not None:
        raise RuntimeError(f"RLT learner failed: {learner_result.error}")

    logger.info(
        "RLT complete: env_steps=%d episodes=%d successes=%d interventions=%d updates=%d online=%d expert=%d",
        learner_result.progress.env_steps,
        learner_result.progress.episodes,
        learner_result.progress.successes,
        learner_result.progress.intervention_steps,
        learner_result.learner_state.gradient_updates,
        learner_result.online_buffer_size,
        learner_result.expert_buffer_size,
    )


def main() -> None:
    train()


if __name__ == "__main__":
    main()
