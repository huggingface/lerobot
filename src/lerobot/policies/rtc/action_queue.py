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

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Each queued action also carries the task (instruction) whose inference
    produced its chunk, kept in lockstep with the action queues under the same
    lock; ``get_with_task`` returns it so consumers can label dispatched
    actions with the instruction they were actually computed under.

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self._task_queue: list[str | None] | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        queued = self.get_with_task()
        return None if queued is None else queued[0]

    def get_with_task(self) -> tuple[Tensor, str | None] | None:
        """Get the next action together with the task that generated its chunk."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            if self._task_queue is not None and len(self._task_queue) != len(self.queue):
                # Every mutation keeps the two queues in lockstep under this
                # lock; a mismatch means a future edit broke that invariant.
                # Fail with a self-describing error instead of an opaque
                # IndexError on the control thread's hot path.
                raise RuntimeError(
                    f"ActionQueue task labels out of sync with actions "
                    f"({len(self._task_queue)} labels for {len(self.queue)} actions) — "
                    "a queue mutation broke the action/task lockstep invariant"
                )
            action = self.queue[self.last_index]
            task = None if self._task_queue is None else self._task_queue[self.last_index]
            self.last_index += 1
            return action.clone(), task

    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self._task_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            Tensor | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        """Get leftover processed actions (the actions currently executed by the robot).

        Returns:
            Tensor | None: Remaining processed actions (remaining_steps, action_dim),
                or None if no processed queue exists.
        """
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
        *,
        task: str | None = None,
    ):
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
            task: Instruction used to generate the incoming action chunk.
        """
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay, task)
                return

            self._append_actions_queue(original_actions, processed_actions, task)

    def _replace_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        task: str | None,
    ):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
            task: Instruction that generated the chunk; labels every queued action.
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].clone()
        self.queue = processed_actions[clamped_delay:].clone()
        self._task_queue = [task] * len(self.queue)

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}, clamped_delay: {clamped_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, task: str | None):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            task: Instruction that generated the appended chunk; actions already
                queued keep the label of the chunk that produced them.
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            self._task_queue = [task] * len(self.queue)
            return

        existing_tasks = self._task_queue or [None] * len(self.queue)
        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]
        self._task_queue = existing_tasks[self.last_index :] + [task] * len(processed_actions)

        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference.

        Args:
            real_delay: Delay computed from inference latency.
            action_index_before_inference: Action index when inference started.

        Returns:
            int: Delay to use.
        """
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
