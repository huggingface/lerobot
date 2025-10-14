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

from contextlib import suppress
from typing import TypedDict

import torch

from ..buffer import ReplayBuffer


class BatchTransitionNSteps(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    masks: torch.Tensor
    terminals: torch.Tensor
    valid: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


class ReplayBufferNSteps(ReplayBuffer):
    def sample_nstep_full(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
    ) -> BatchTransitionNSteps:
        """Sample a random batch of transitions and collate them into batched tensors.

        Args:
            batch_size (int): Size of batches to sample
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor

        Yields:
            BatchTransitionNSteps: Batched transitions
        """
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1.")

        batch_size = min(batch_size, self.size)
        high = (
            max(0, self.size - n_steps - 1)
            if self.optimize_memory and self.size < self.capacity
            else self.size
        )

        # Random indices for sampling - create on the same device as storage
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)
        steps = torch.arange(n_steps, device=self.storage_device).view(1, -1)

        # Build sequences
        indices = (idx[:, None] + steps) % self.capacity

        image_keys = [k for k in self.states if k.startswith("observation.image")] if self.use_drq else []
        batch_state_nsteps = {}
        batch_next_state_nsteps = {}

        for key in self.states:
            # Full sequence of observations
            batch_state_nsteps[key] = self.states[key][idx].to(self.device)
            if not self.optimize_memory:
                # Full sequence of next observations
                batch_next_state_nsteps[key] = self.next_states[key][idx].to(self.device)
            else:
                next_indices = (idx + n_steps) % self.capacity
                batch_next_state_nsteps[key] = self.states[key][next_indices].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state_nsteps[key])
                all_images.append(batch_next_state_nsteps[key])

            # Optimization: Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            for i, k in enumerate(image_keys):
                # Calculate offsets for the current image key:
                # For each key, we have 2*batch_size images (batch_size for states, batch_size for next_states)
                # States start at index i*2*batch_size and take up batch_size slots
                batch_state_nsteps[k] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next states start after the states at index (i*2+1)*batch_size and also take up batch_size slots
                batch_next_state_nsteps[k] = augmented_images[
                    (i * 2 + 1) * batch_size : (i * 2 + 2) * batch_size
                ]

        # Sample other tensors sequences
        action_seq = self.actions[indices].to(self.device)
        reward_seq = self.rewards[indices].to(self.device)

        # Get terminated and done flags
        terminated_seq = self.dones[indices].float().to(self.device)
        truncated_seq = self.truncateds[indices].float().to(self.device)
        done_seq = torch.logical_or(
            terminated_seq.bool(), truncated_seq.bool()
        ).float()  # done = terminated OR truncated

        # Calculate cumulative rewards, masks, terminals and valid
        rewards = torch.zeros((batch_size, n_steps), dtype=torch.float32, device=self.device)
        masks = torch.ones((batch_size, n_steps), dtype=torch.float32, device=self.device)
        terminals = torch.zeros((batch_size, n_steps), dtype=torch.float32, device=self.device)
        valid = torch.ones((batch_size, n_steps), dtype=torch.float32, device=self.device)

        discount_powers = gamma ** torch.arange(n_steps, device=self.device, dtype=torch.float32)

        # First step
        rewards[:, 0] = reward_seq[:, 0]
        masks[:, 0] = 1.0 - terminated_seq[:, 0]  # masks = 1.0 - terminated
        terminals[:, 0] = done_seq[:, 0]  # terminals = float(done)

        # Subsequent steps
        for i in range(1, n_steps):
            rewards[:, i] = rewards[:, i - 1] + reward_seq[:, i] * discount_powers[i]
            masks[:, i] = torch.minimum(masks[:, i - 1], 1.0 - terminated_seq[:, i])  # Cumulative masks
            terminals[:, i] = torch.maximum(terminals[:, i - 1], done_seq[:, i])  # Cumulative terminals
            valid[:, i] = 1.0 - terminals[:, i - 1]  # Valid mask

        # Sample complementary_info if available
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][idx].to(self.device)

        return BatchTransitionNSteps(
            state=batch_state_nsteps,
            action=action_seq,
            reward=rewards,
            next_state=batch_next_state_nsteps,
            masks=masks,
            terminals=terminals,
            valid=valid,
            complementary_info=batch_complementary_info,
        )

    def get_iterator_nstep(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)

        Yields:
            BatchTransitionNSteps: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator_nstep(batch_size, n_steps, gamma, queue_size)
            else:
                iterator = self._get_naive_iterator_nstep(batch_size, n_steps, gamma)

            # Yield all items from the iterator
            with suppress(StopIteration):
                yield from iterator

    def _get_naive_iterator_nstep(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
        queue_size: int = 2,
    ):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor
            queue_size (int): Number of initial batches to prefetch

        Yields:
            BatchTransitionNSteps: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample_nstep_full(batch_size, n_steps, gamma)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def _get_async_iterator_nstep(
        self,
        batch_size: int,
        n_steps: int,
        gamma: float,
        queue_size: int = 2,
    ):
        """
        Create an iterator that continuously yields prefetched batches in a
        background thread. The design is intentionally simple and avoids busy
        waiting / complex state management.

        Args:
            batch_size (int): Size of batches to sample.
            n_steps (int): Number of steps for n-step returns
            gamma (float): Discount factor
            queue_size (int): Maximum number of prefetched batches to keep in
                memory.

        Yields:
            BatchTransition: A batch sampled from the replay buffer.
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """Continuously put sampled batches into the queue until shutdown."""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample_nstep_full(batch_size, n_steps, gamma)
                    # The timeout ensures the thread unblocks if the queue is full
                    # and the shutdown event gets set meanwhile.
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # Queue is full – loop again (will re-check shutdown_event)
                    continue
                except Exception:
                    # Surface any unexpected error and terminate the producer.
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # If the producer already set the shutdown flag we exit.
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # Drain the queue quickly to help the thread exit if it's blocked on `put`.
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # Give the producer thread a bit of time to finish.
            producer_thread.join(timeout=1.0)
