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

"""Asynchronous actor/learner runtime for online RL-Token training."""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from queue import Empty, Full
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .configuration_rlt import RLTOnlineConfig
from .modeling_rlt import RLTAgent
from .online import RLTChunkCollector
from .replay import (
    RLTDualReplayBuffer,
    RLTReplayBuffer,
    RLTTransition,
    transition_has_intervention,
)

logger = logging.getLogger(__name__)


def serialize_rlt_message(message: Any) -> bytes:
    """Serialize tensor-bearing IPC messages before they enter a multiprocessing queue."""
    if isinstance(message, RLTTransitionBatch):
        payload = {
            "type": "transition_batch",
            "transitions": [
                {field.name: getattr(transition, field.name) for field in fields(RLTTransition)}
                for transition in message.transitions
            ],
            "progress": asdict(message.progress),
            "episode_done": message.episode_done,
        }
    elif isinstance(message, RLTCollectorDone):
        payload = {
            "type": "collector_done",
            "progress": asdict(message.progress),
        }
    elif isinstance(message, RLTActorSnapshot):
        payload = {
            "type": "actor_snapshot",
            "actor": message.actor,
            "version": message.version,
            "critic_updates": message.critic_updates,
            "actor_updates": message.actor_updates,
        }
    else:
        raise TypeError(f"cannot serialize RLT message: {type(message)!r}")

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize_rlt_message(message: Any) -> Any:
    """Decode an RLT IPC payload while accepting direct objects for in-process use."""
    if not isinstance(message, bytes):
        return message

    payload = torch.load(io.BytesIO(message), map_location="cpu", weights_only=True)
    message_type = payload.get("type")
    if message_type == "transition_batch":
        return RLTTransitionBatch(
            transitions=tuple(RLTTransition(**transition) for transition in payload["transitions"]),
            progress=RLTCollectorProgress(**payload["progress"]),
            episode_done=bool(payload["episode_done"]),
        )
    if message_type == "collector_done":
        return RLTCollectorDone(progress=RLTCollectorProgress(**payload["progress"]))
    if message_type == "actor_snapshot":
        return RLTActorSnapshot(
            actor=payload["actor"],
            version=int(payload["version"]),
            critic_updates=int(payload["critic_updates"]),
            actor_updates=int(payload["actor_updates"]),
        )
    raise ValueError(f"unknown RLT message type: {message_type!r}")


@dataclass(frozen=True)
class RLTCollectorProgress:
    env_steps: int = 0
    episodes: int = 0
    successes: int = 0
    intervention_steps: int = 0


@dataclass(frozen=True)
class RLTTransitionBatch:
    transitions: tuple[RLTTransition, ...]
    progress: RLTCollectorProgress
    episode_done: bool = False


@dataclass(frozen=True)
class RLTCollectorDone:
    progress: RLTCollectorProgress


@dataclass(frozen=True)
class RLTActorSnapshot:
    actor: dict[str, Tensor]
    version: int
    critic_updates: int
    actor_updates: int

    @property
    def ready(self) -> bool:
        return self.actor_updates > 0


@dataclass
class RLTAsyncLearnerState:
    update_budget: int = 0
    gradient_updates: int = 0
    transitions_received: int = 0
    intervention_transitions: int = 0
    actor_version: int = 0


@dataclass(frozen=True)
class RLTAsyncLearnerResult:
    learner_state: RLTAsyncLearnerState
    progress: RLTCollectorProgress
    online_buffer_size: int
    expert_buffer_size: int
    checkpoint_path: str
    error: str | None = None


def put_latest(queue: Any, item: Any) -> None:
    """Publish a snapshot after discarding any snapshots already visible to this process."""
    while True:
        try:
            queue.get_nowait()
        except Empty:
            break
    queue.put_nowait(item)


def get_latest(queue: Any) -> Any | None:
    latest = None
    while True:
        try:
            latest = queue.get_nowait()
        except Empty:
            return latest


def make_dual_replay(config: RLTOnlineConfig) -> RLTDualReplayBuffer:
    replay_kwargs = {
        "state_dim": config.actor_critic.state_dim,
        "chunk_length": config.actor_critic.chunk_length,
        "action_dim": config.actor_critic.action_dim,
        "device": config.device,
    }
    return RLTDualReplayBuffer(
        online=RLTReplayBuffer(capacity=config.replay_capacity, **replay_kwargs),
        expert=RLTReplayBuffer(capacity=config.expert_replay_capacity, **replay_kwargs),
        online_ratio=config.online_sample_ratio,
    )


def _extract_expert_replay_state(payload: dict[str, object]) -> dict[str, object]:
    if "buffers" in payload:
        return payload["buffers"]["expert"]
    if "expert" in payload and isinstance(payload["expert"], dict):
        return payload["expert"]
    if "size" in payload:
        return payload
    raise ValueError("expert replay file does not contain an RLT replay state")


def load_collector_resume(
    checkpoint_path: Path,
    actor_agent: RLTAgent,
) -> tuple[RLTCollectorProgress, int]:
    """Restore the inference actor and collection counters at an episode boundary."""
    # Resume paths are trusted local RLT checkpoints because optimizer/RNG state needs pickle.
    checkpoint = torch.load(  # nosec B614
        checkpoint_path, map_location="cpu", weights_only=False
    )
    agent_state = checkpoint["agent"]
    actor_agent.load_actor_state_dict(agent_state["actor"])
    actor_updates = int(agent_state.get("actor_updates", 0))

    progress_state = checkpoint.get("collector_progress")
    if progress_state is None:
        training_state = checkpoint.get("training_state", {})
        progress_state = {
            "env_steps": training_state.get("env_steps", 0),
            "episodes": training_state.get("episodes", 0),
            "successes": training_state.get("successes", 0),
            "intervention_steps": training_state.get("intervention_steps", 0),
        }
    return RLTCollectorProgress(**progress_state), actor_updates


class AsyncRLTCollector:
    """Real-time-facing collection loop that never performs gradient updates."""

    def __init__(
        self,
        config: RLTOnlineConfig,
        actor_agent: RLTAgent,
        collector: RLTChunkCollector,
        transition_queue: Any,
        parameter_queue: Any,
        *,
        initial_progress: RLTCollectorProgress | None = None,
        initial_actor_updates: int = 0,
        learner_is_alive: Callable[[], bool] | None = None,
    ) -> None:
        if collector.replay is not None:
            raise ValueError("an asynchronous collector must not own a replay buffer")
        self.config = config
        self.actor_agent = actor_agent
        self.collector = collector
        self.transition_queue = transition_queue
        self.parameter_queue = parameter_queue
        self.progress = initial_progress or RLTCollectorProgress()
        self.latest_actor_updates = initial_actor_updates
        self.latest_actor_version = 0
        self.learner_is_alive = learner_is_alive

    def _refresh_actor(self) -> None:
        message = get_latest(self.parameter_queue)
        if message is None:
            return
        snapshot = deserialize_rlt_message(message)
        if not isinstance(snapshot, RLTActorSnapshot):
            raise TypeError(f"unexpected parameter message: {type(snapshot)!r}")
        if snapshot.version <= self.latest_actor_version:
            return
        self.actor_agent.load_actor_state_dict(snapshot.actor)
        self.latest_actor_updates = snapshot.actor_updates
        self.latest_actor_version = snapshot.version

    def _send_transitions(
        self,
        transitions: tuple[RLTTransition, ...],
        *,
        episode_done: bool,
    ) -> None:
        if not transitions and not episode_done:
            return
        message = RLTTransitionBatch(
            transitions=tuple(transition.cpu() for transition in transitions),
            progress=self.progress,
            episode_done=episode_done,
        )
        try:
            self.transition_queue.put_nowait(serialize_rlt_message(message))
        except Full as exc:
            raise RuntimeError(
                "RLT transition queue is full; refusing to stall or silently drop robot data"
            ) from exc

    def train(self, log_fn: Callable[[dict[str, float]], None] | None = None) -> RLTCollectorProgress:
        self.collector.reset()
        while self.progress.env_steps < self.config.total_env_steps:
            if self.learner_is_alive is not None and not self.learner_is_alive():
                raise RuntimeError("RLT learner exited while collection was still active")

            self._refresh_actor()
            use_actor = (
                self.progress.env_steps >= self.config.warmup_env_steps and self.latest_actor_updates > 0
            )
            remaining = self.config.total_env_steps - self.progress.env_steps
            outcome = self.collector.run_chunk(use_actor=use_actor, max_steps=remaining)
            self.progress = RLTCollectorProgress(
                env_steps=self.progress.env_steps + outcome.steps,
                episodes=self.progress.episodes + int(outcome.episode_done),
                successes=self.progress.successes + int(outcome.episode_done and outcome.success),
                intervention_steps=self.progress.intervention_steps + outcome.intervention_steps,
            )
            self._send_transitions(outcome.transitions, episode_done=outcome.episode_done)

            if log_fn is not None:
                log_fn(
                    {
                        "env_steps": float(self.progress.env_steps),
                        "episodes": float(self.progress.episodes),
                        "successes": float(self.progress.successes),
                        "intervention_steps": float(self.progress.intervention_steps),
                        "transitions_emitted": float(outcome.transitions_emitted),
                        "chunk_reward": outcome.reward,
                        "used_actor": float(use_actor),
                        "actor_version": float(self.latest_actor_version),
                    }
                )

            if outcome.episode_done and self.progress.env_steps < self.config.total_env_steps:
                self.collector.reset()

        if self.collector.assembler.pending is not None:
            pending = self.collector.assembler.pending
            transitions = self.collector.assembler.truncate_pending(
                pending.final_state, pending.final_reference
            )
            self._send_transitions(tuple(transitions), episode_done=False)

        self.transition_queue.put(serialize_rlt_message(RLTCollectorDone(progress=self.progress)))
        return self.progress


class AsyncRLTLearner:
    """Learner-side replay owner, update scheduler, and actor publisher."""

    def __init__(
        self,
        config: RLTOnlineConfig,
        transition_queue: Any,
        parameter_queue: Any,
        *,
        output_dir: Path,
        result_queue: Any | None = None,
        parameter_push_interval_s: float = 4.0,
        max_updates_per_cycle: int = 8,
        queue_get_timeout_s: float = 0.05,
        checkpoint_freq_episodes: int = 10,
        resume: Path | None = None,
        expert_replay_path: Path | None = None,
    ) -> None:
        if parameter_push_interval_s < 0.0:
            raise ValueError("parameter_push_interval_s must be non-negative")
        if max_updates_per_cycle <= 0:
            raise ValueError("max_updates_per_cycle must be positive")
        if queue_get_timeout_s <= 0.0:
            raise ValueError("queue_get_timeout_s must be positive")

        self.config = config
        self.transition_queue = transition_queue
        self.parameter_queue = parameter_queue
        self.result_queue = result_queue
        self.output_dir = output_dir
        self.parameter_push_interval_s = parameter_push_interval_s
        self.max_updates_per_cycle = max_updates_per_cycle
        self.queue_get_timeout_s = queue_get_timeout_s
        self.checkpoint_freq_episodes = checkpoint_freq_episodes
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.agent = RLTAgent(config)
        self.buffers = make_dual_replay(config)
        self.state = RLTAsyncLearnerState()
        self.progress = RLTCollectorProgress()
        self.collector_done = False
        self.last_parameter_push = 0.0
        self.last_checkpoint_episode = -1

        if resume is not None:
            self._load_checkpoint(resume)
        elif expert_replay_path is not None:
            # Expert replay paths are trusted local RLT files containing tensor state.
            payload = torch.load(  # nosec B614
                expert_replay_path, map_location="cpu", weights_only=False
            )
            self.buffers.expert.load_state_dict(_extract_expert_replay_state(payload))

    def _snapshot(self) -> RLTActorSnapshot:
        self.state.actor_version += 1
        return RLTActorSnapshot(
            actor=self.agent.actor_state_dict(),
            version=self.state.actor_version,
            critic_updates=self.agent.critic_updates,
            actor_updates=self.agent.actor_updates,
        )

    def _publish_actor(self) -> None:
        put_latest(self.parameter_queue, serialize_rlt_message(self._snapshot()))
        self.last_parameter_push = time.monotonic()

    def _ingest(self, message: RLTTransitionBatch) -> None:
        transitions = list(message.transitions)
        self.state.intervention_transitions += sum(
            transition_has_intervention(transition) for transition in transitions
        )
        inserted = self.buffers.add_online(transitions)
        self.state.transitions_received += inserted
        self.state.update_budget += inserted * self.config.utd_ratio
        self.progress = message.progress

        if (
            message.episode_done
            and self.checkpoint_freq_episodes > 0
            and self.progress.episodes > 0
            and self.progress.episodes % self.checkpoint_freq_episodes == 0
            and self.progress.episodes != self.last_checkpoint_episode
        ):
            checkpoint_path = self.output_dir / f"rlt_training_state_{self.progress.env_steps}.pt"
            self.save_checkpoint(checkpoint_path)
            self.last_checkpoint_episode = self.progress.episodes

    def _receive_messages(self, *, block: bool) -> None:
        try:
            first = (
                self.transition_queue.get(timeout=self.queue_get_timeout_s)
                if block
                else self.transition_queue.get_nowait()
            )
        except Empty:
            return

        messages = [first]
        for _ in range(63):
            try:
                messages.append(self.transition_queue.get_nowait())
            except Empty:
                break

        for encoded_message in messages:
            message = deserialize_rlt_message(encoded_message)
            if isinstance(message, RLTTransitionBatch):
                self._ingest(message)
            elif isinstance(message, RLTCollectorDone):
                self.progress = message.progress
                self.collector_done = True
            else:
                raise TypeError(f"unexpected RLT learner message: {type(message)!r}")

    def _train_available_budget(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if len(self.buffers.online) < self.config.batch_size:
            return metrics

        updates = min(self.state.update_budget, self.max_updates_per_cycle)
        actor_updates_before = self.agent.actor_updates
        for _ in range(updates):
            metrics = self.agent.update(lambda: self.buffers.sample(self.config.batch_size))
            self.state.update_budget -= 1
            self.state.gradient_updates += 1

        first_actor_became_ready = actor_updates_before == 0 and self.agent.actor_updates > 0
        push_due = time.monotonic() - self.last_parameter_push >= self.parameter_push_interval_s
        if first_actor_became_ready or (updates > 0 and push_due):
            self._publish_actor()
        return metrics

    def run(self) -> RLTAsyncLearnerResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._publish_actor()
        try:
            while True:
                needs_more_online_data = len(self.buffers.online) < self.config.batch_size
                self._receive_messages(
                    block=(self.state.update_budget == 0 or needs_more_online_data)
                    and not self.collector_done
                )
                metrics = self._train_available_budget()

                if metrics and self.state.gradient_updates % 100 == 0:
                    logger.info(
                        "RLT learner updates=%d budget=%d online=%d expert=%d actor_updates=%d",
                        self.state.gradient_updates,
                        self.state.update_budget,
                        len(self.buffers.online),
                        len(self.buffers.expert),
                        self.agent.actor_updates,
                    )

                cannot_train = len(self.buffers.online) < self.config.batch_size
                if self.collector_done and (self.state.update_budget == 0 or cannot_train):
                    if cannot_train and self.state.update_budget > 0:
                        logger.warning(
                            "RLT learner stopped with %d unpaid updates: online replay has %d/%d samples",
                            self.state.update_budget,
                            len(self.buffers.online),
                            self.config.batch_size,
                        )
                    break

            self._publish_actor()
            checkpoint_path = self.output_dir / "rlt_training_state.pt"
            self.save_checkpoint(checkpoint_path)
            torch.save(self.agent.actor_state_dict(), self.output_dir / "rlt_actor.pt")
            result = RLTAsyncLearnerResult(
                learner_state=self.state,
                progress=self.progress,
                online_buffer_size=len(self.buffers.online),
                expert_buffer_size=len(self.buffers.expert),
                checkpoint_path=str(checkpoint_path),
            )
            if self.result_queue is not None:
                self.result_queue.put(result)
            return result
        except Exception as exc:
            logger.exception("RLT learner failed")
            result = RLTAsyncLearnerResult(
                learner_state=self.state,
                progress=self.progress,
                online_buffer_size=len(self.buffers.online),
                expert_buffer_size=len(self.buffers.expert),
                checkpoint_path="",
                error=f"{type(exc).__name__}: {exc}",
            )
            if self.result_queue is not None:
                self.result_queue.put(result)
            raise

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = {
            "format_version": 2,
            "learner_state": asdict(self.state),
            "collector_progress": asdict(self.progress),
            "agent": self.agent.training_state_dict(),
            "buffers": self.buffers.state_dict(),
            "config": asdict(self.config),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        # Resume paths are trusted local RLT checkpoints because optimizer/RNG state needs pickle.
        checkpoint = torch.load(  # nosec B614
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.agent.load_training_state_dict(checkpoint["agent"])
        if "buffers" in checkpoint:
            self.buffers.load_state_dict(checkpoint["buffers"])
            self.state = RLTAsyncLearnerState(**checkpoint["learner_state"])
            self.progress = RLTCollectorProgress(**checkpoint["collector_progress"])
        else:
            self.buffers.online.load_state_dict(checkpoint["replay"])
            training_state = checkpoint.get("training_state", {})
            self.state.update_budget = int(training_state.get("update_budget", 0))
            self.state.gradient_updates = int(training_state.get("gradient_updates", 0))
            self.state.transitions_received = len(self.buffers.online)
            self.progress = RLTCollectorProgress(
                env_steps=int(training_state.get("env_steps", 0)),
                episodes=int(training_state.get("episodes", 0)),
                successes=int(training_state.get("successes", 0)),
                intervention_steps=int(training_state.get("intervention_steps", 0)),
            )

        if "torch_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng"])
        if "numpy_rng" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng"])
        if "cuda_rng" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def run_async_rlt_learner(
    config: RLTOnlineConfig,
    transition_queue: Any,
    parameter_queue: Any,
    result_queue: Any,
    output_dir: Path,
    parameter_push_interval_s: float,
    max_updates_per_cycle: int,
    queue_get_timeout_s: float,
    checkpoint_freq_episodes: int,
    resume: Path | None,
    expert_replay_path: Path | None,
) -> None:
    """Multiprocessing entry point; all arguments intentionally stay pickleable."""
    learner = AsyncRLTLearner(
        config,
        transition_queue,
        parameter_queue,
        output_dir=output_dir,
        result_queue=result_queue,
        parameter_push_interval_s=parameter_push_interval_s,
        max_updates_per_cycle=max_updates_per_cycle,
        queue_get_timeout_s=queue_get_timeout_s,
        checkpoint_freq_episodes=checkpoint_freq_episodes,
        resume=resume,
        expert_replay_path=expert_replay_path,
    )
    learner.run()
