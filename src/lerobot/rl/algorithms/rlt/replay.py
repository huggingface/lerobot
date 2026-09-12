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

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RLTTransition:
    state: Tensor
    reference: Tensor
    action: Tensor
    discounted_return: Tensor
    next_state: Tensor
    next_reference: Tensor
    bootstrap_discount: Tensor
    terminated: Tensor
    truncated: Tensor
    valid_horizon: Tensor
    intervention_mask: Tensor

    def cpu(self) -> RLTTransition:
        """Return an IPC-safe copy detached from accelerator storage."""
        return RLTTransition(
            state=self.state.detach().cpu().clone(),
            reference=self.reference.detach().cpu().clone(),
            action=self.action.detach().cpu().clone(),
            discounted_return=self.discounted_return.detach().cpu().clone(),
            next_state=self.next_state.detach().cpu().clone(),
            next_reference=self.next_reference.detach().cpu().clone(),
            bootstrap_discount=self.bootstrap_discount.detach().cpu().clone(),
            terminated=self.terminated.detach().cpu().clone(),
            truncated=self.truncated.detach().cpu().clone(),
            valid_horizon=self.valid_horizon.detach().cpu().clone(),
            intervention_mask=self.intervention_mask.detach().cpu().clone(),
        )


@dataclass
class ExecutedChunk:
    """One executed C-step plan plus contexts recomputed at every stride offset."""

    states: Tensor
    references: Tensor
    actions: Tensor
    rewards: Tensor
    intervention_mask: Tensor
    executed_steps: int
    final_state: Tensor
    final_reference: Tensor
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("chunk cannot be both terminated and truncated")
        if self.executed_steps <= 0 or self.executed_steps > self.actions.shape[0]:
            raise ValueError("executed_steps is outside the action chunk")


class ChunkTransitionAssembler:
    """Turn consecutive executed chunks into exact stride-overlapping transitions."""

    def __init__(
        self,
        *,
        chunk_length: int,
        action_dim: int,
        discount: float,
        stride: int = 2,
    ) -> None:
        if stride <= 0 or chunk_length % stride != 0:
            raise ValueError("stride must be a positive divisor of chunk_length")
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.discount = discount
        self.stride = stride
        self.pending: ExecutedChunk | None = None
        self._reward_weights = discount ** torch.arange(chunk_length, dtype=torch.float32)

    def start_episode(self) -> None:
        if self.pending is not None:
            raise RuntimeError("end the previous episode before starting another")

    def add_chunk(self, chunk: ExecutedChunk) -> list[RLTTransition]:
        self._validate_chunk(chunk)
        transitions: list[RLTTransition] = []
        if self.pending is not None:
            transitions.extend(self._emit_pair(self.pending, chunk))

        if chunk.terminated or chunk.truncated:
            transitions.extend(self._flush_final(chunk))
            self.pending = None
        else:
            if chunk.executed_steps != self.chunk_length:
                raise ValueError("a non-final chunk must execute all C actions")
            self.pending = chunk
        return transitions

    def truncate_pending(self, final_state: Tensor, final_reference: Tensor) -> list[RLTTransition]:
        if self.pending is None:
            return []
        self.pending.truncated = True
        self.pending.final_state = final_state
        self.pending.final_reference = final_reference
        transitions = self._flush_final(self.pending)
        self.pending = None
        return transitions

    def _emit_pair(self, current: ExecutedChunk, following: ExecutedChunk) -> list[RLTTransition]:
        transitions: list[RLTTransition] = []
        c = self.chunk_length
        for offset_index, offset in enumerate(range(0, c, self.stride)):
            following_steps = min(offset, following.executed_steps)
            actions = torch.cat([current.actions[offset:c], following.actions[:following_steps]], dim=0)
            rewards = torch.cat([current.rewards[offset:c], following.rewards[:following_steps]], dim=0)
            intervention = torch.cat(
                [current.intervention_mask[offset:c], following.intervention_mask[:following_steps]], dim=0
            )
            horizon = actions.shape[0]

            if offset < following.executed_steps:
                next_state = following.states[offset_index]
                next_reference = following.references[offset_index]
                terminated = False
                truncated = False
                bootstrap = self.discount**c
            else:
                next_state = following.final_state
                next_reference = following.final_reference
                terminated = following.terminated
                truncated = following.truncated
                bootstrap = 0.0 if terminated else self.discount**horizon

            transitions.append(
                self._make_transition(
                    state=current.states[offset_index],
                    reference=current.references[offset_index],
                    actions=actions,
                    rewards=rewards,
                    intervention_mask=intervention,
                    next_state=next_state,
                    next_reference=next_reference,
                    bootstrap_discount=bootstrap,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
        return transitions

    def _flush_final(self, chunk: ExecutedChunk) -> list[RLTTransition]:
        transitions: list[RLTTransition] = []
        for offset_index, offset in enumerate(range(0, self.chunk_length, self.stride)):
            if offset >= chunk.executed_steps:
                break
            actions = chunk.actions[offset : chunk.executed_steps]
            rewards = chunk.rewards[offset : chunk.executed_steps]
            intervention = chunk.intervention_mask[offset : chunk.executed_steps]
            horizon = chunk.executed_steps - offset
            transitions.append(
                self._make_transition(
                    state=chunk.states[offset_index],
                    reference=chunk.references[offset_index],
                    actions=actions,
                    rewards=rewards,
                    intervention_mask=intervention,
                    next_state=chunk.final_state,
                    next_reference=chunk.final_reference,
                    bootstrap_discount=0.0 if chunk.terminated else self.discount**horizon,
                    terminated=chunk.terminated,
                    truncated=chunk.truncated,
                )
            )
        return transitions

    def _make_transition(
        self,
        *,
        state: Tensor,
        reference: Tensor,
        actions: Tensor,
        rewards: Tensor,
        intervention_mask: Tensor,
        next_state: Tensor,
        next_reference: Tensor,
        bootstrap_discount: float,
        terminated: bool,
        truncated: bool,
    ) -> RLTTransition:
        horizon = actions.shape[0]
        padded_actions = self._pad(actions)
        padded_intervention = torch.zeros(self.chunk_length, dtype=torch.bool, device=actions.device)
        padded_intervention[:horizon] = intervention_mask.bool()

        training_reference = reference.clone()
        intervention_indices = padded_intervention[:horizon].nonzero(as_tuple=True)[0]
        training_reference[intervention_indices] = actions[intervention_indices]
        valid = torch.zeros(self.chunk_length, dtype=torch.bool, device=actions.device)
        valid[:horizon] = True
        training_reference = training_reference * valid.unsqueeze(-1)

        reward_weights = self._reward_weights[:horizon].to(rewards)
        discounted_return = (rewards * reward_weights).sum()
        return RLTTransition(
            state=state,
            reference=training_reference,
            action=padded_actions,
            discounted_return=discounted_return,
            next_state=next_state,
            next_reference=next_reference,
            bootstrap_discount=torch.as_tensor(bootstrap_discount, dtype=torch.float32),
            terminated=torch.as_tensor(terminated),
            truncated=torch.as_tensor(truncated),
            valid_horizon=torch.as_tensor(horizon, dtype=torch.long),
            intervention_mask=padded_intervention,
        )

    def _pad(self, actions: Tensor) -> Tensor:
        output = torch.zeros(self.chunk_length, self.action_dim, dtype=actions.dtype, device=actions.device)
        output[: actions.shape[0]] = actions
        return output

    def _validate_chunk(self, chunk: ExecutedChunk) -> None:
        c = self.chunk_length
        offsets = len(range(0, chunk.executed_steps, self.stride))
        if chunk.states.ndim != 2 or chunk.states.shape[0] != offsets:
            raise ValueError("states must contain every executed stride offset")
        if chunk.actions.shape != (c, self.action_dim):
            raise ValueError("actions have the wrong shape")
        if chunk.references.shape != (offsets, c, self.action_dim):
            raise ValueError("references must match every executed stride offset")
        if chunk.rewards.shape != (c,):
            raise ValueError("rewards have the wrong shape")
        if chunk.intervention_mask.shape != (c,):
            raise ValueError("intervention_mask has the wrong shape")
        if chunk.final_reference.shape != (c, self.action_dim):
            raise ValueError("final_reference has the wrong shape")


class RLTReplayBuffer:
    """Exact CPU replay storage; next states never rely on array adjacency."""

    _FLOAT_FIELDS = (
        "state",
        "reference",
        "action",
        "discounted_return",
        "next_state",
        "next_reference",
        "bootstrap_discount",
    )
    _BOOL_FIELDS = ("terminated", "truncated", "intervention_mask")

    def __init__(
        self,
        *,
        capacity: int,
        state_dim: int,
        chunk_length: int,
        action_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.state = torch.zeros(capacity, state_dim)
        self.reference = torch.zeros(capacity, chunk_length, action_dim)
        self.action = torch.zeros(capacity, chunk_length, action_dim)
        self.discounted_return = torch.zeros(capacity)
        self.next_state = torch.zeros(capacity, state_dim)
        self.next_reference = torch.zeros(capacity, chunk_length, action_dim)
        self.bootstrap_discount = torch.zeros(capacity)
        self.terminated = torch.zeros(capacity, dtype=torch.bool)
        self.truncated = torch.zeros(capacity, dtype=torch.bool)
        self.valid_horizon = torch.zeros(capacity, dtype=torch.long)
        self.intervention_mask = torch.zeros(capacity, chunk_length, dtype=torch.bool)
        self.size = 0
        self.pointer = 0

    def add(self, transition: RLTTransition) -> None:
        index = self.pointer
        for field in (*self._FLOAT_FIELDS, *self._BOOL_FIELDS, "valid_horizon"):
            getattr(self, field)[index].copy_(getattr(transition, field).detach().cpu())
        self.pointer = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_many(self, transitions: list[RLTTransition]) -> int:
        for transition in transitions:
            self.add(transition)
        return len(transitions)

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty replay buffer")
        indices = torch.randint(self.size, (batch_size,))
        fields = (*self._FLOAT_FIELDS, *self._BOOL_FIELDS, "valid_horizon")
        return {field: getattr(self, field)[indices].to(self.device) for field in fields}

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, object]:
        fields = (*self._FLOAT_FIELDS, *self._BOOL_FIELDS, "valid_horizon")
        return {
            "size": self.size,
            "pointer": self.pointer,
            **{field: getattr(self, field)[: self.size].clone() for field in fields},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError("saved replay is larger than current capacity")
        fields = (*self._FLOAT_FIELDS, *self._BOOL_FIELDS, "valid_horizon")
        for field in fields:
            getattr(self, field)[:size].copy_(state[field])
        self.size = size
        self.pointer = int(state["pointer"])


def transition_has_intervention(transition: RLTTransition) -> bool:
    """Whether a real, non-padding action in the transition came from a human."""
    horizon = int(transition.valid_horizon)
    if not 0 < horizon <= transition.intervention_mask.shape[0]:
        raise ValueError("valid_horizon is outside the intervention mask")
    return bool(transition.intervention_mask[:horizon].any())


def concatenate_rlt_batches(first: dict[str, Tensor], second: dict[str, Tensor]) -> dict[str, Tensor]:
    if first.keys() != second.keys():
        raise ValueError("RLT batches must contain identical fields")
    return {field: torch.cat([first[field], second[field]], dim=0) for field in first}


class RLTDualReplayBuffer:
    """HIL-SERL-style online/expert replay routing and mixed sampling.

    Every newly collected transition enters ``online``. Transitions containing
    at least one human-executed action additionally enter ``expert`` alongside
    any preloaded demonstrations. Duplicating interventions is intentional: it
    prevents scarce corrections from being diluted by autonomous experience.
    """

    def __init__(
        self,
        online: RLTReplayBuffer,
        expert: RLTReplayBuffer,
        *,
        online_ratio: float = 0.5,
    ) -> None:
        if not 0.0 <= online_ratio <= 1.0:
            raise ValueError("online_ratio must be in [0, 1]")
        self.online = online
        self.expert = expert
        self.online_ratio = online_ratio

    def add_online(self, transitions: list[RLTTransition] | tuple[RLTTransition, ...]) -> int:
        for transition in transitions:
            self.online.add(transition)
            if transition_has_intervention(transition):
                self.expert.add(transition)
        return len(transitions)

    def add_expert(self, transitions: list[RLTTransition] | tuple[RLTTransition, ...]) -> int:
        return self.expert.add_many(list(transitions))

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.online) == 0 and len(self.expert) == 0:
            raise RuntimeError("cannot sample empty online and expert buffers")
        if len(self.online) == 0:
            return self.expert.sample(batch_size)
        if len(self.expert) == 0:
            return self.online.sample(batch_size)
        if self.online_ratio >= 1.0:
            return self.online.sample(batch_size)
        if self.online_ratio <= 0.0:
            return self.expert.sample(batch_size)
        if batch_size == 1:
            source = self.online if self.online_ratio >= 0.5 else self.expert
            return source.sample(1)

        online_count = int(batch_size * self.online_ratio)
        online_count = min(max(online_count, 1), batch_size - 1)
        expert_count = batch_size - online_count
        return concatenate_rlt_batches(self.online.sample(online_count), self.expert.sample(expert_count))

    def state_dict(self) -> dict[str, object]:
        return {
            "online": self.online.state_dict(),
            "expert": self.expert.state_dict(),
            "online_ratio": self.online_ratio,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.online.load_state_dict(state["online"])
        self.expert.load_state_dict(state["expert"])
        saved_ratio = float(state.get("online_ratio", self.online_ratio))
        if saved_ratio != self.online_ratio:
            raise ValueError(
                f"saved online_ratio {saved_ratio} does not match configured {self.online_ratio}"
            )
