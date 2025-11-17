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

import itertools
from collections.abc import Callable, Generator, Sequence
from contextlib import suppress
from typing import TypedDict

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, REWARD
from lerobot.utils.transition import Transition

from ..buffer import ReplayBuffer


class BatchTransitionNSteps(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    masks: torch.Tensor
    terminals: torch.Tensor
    truncateds: torch.Tensor
    valid: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


class ReplayBufferNSteps(ReplayBuffer):
    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # Pre-allocate tensors for storage
        self.states = {
            key: torch.zeros((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.zeros((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.zeros((self.capacity,), device=self.storage_device)
        self.mc_returns = torch.zeros((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
                key: torch.zeros((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.zeros((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.zeros((self.capacity,), dtype=torch.bool, device=self.storage_device)

        # Initialize storage for complementary_info
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # Pre-allocate tensors for each key in complementary_info
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.squeeze(0).shape
                    self.complementary_info[key] = torch.empty(
                        (self.capacity, *value_shape), device=self.storage_device
                    )
                elif isinstance(value, (int | float)):
                    # Handle scalar values similar to reward
                    self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

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
            else self.size - n_steps
        )

        # Random indices for sampling - create on the same device as storage
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)
        return self.sample_nstep_full_for_indices(idx, batch_size, n_steps, gamma)

    def sample_nstep_full_for_indices(
        self,
        idx: torch.Tensor,
        batch_size: int,
        n_steps: int,
        gamma: float,
    ) -> BatchTransitionNSteps:
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
                next_indices = (idx + n_steps - 1) % self.capacity
                batch_next_state_nsteps[key] = self.next_states[key][next_indices].to(self.device)
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
        truncateds = torch.zeros((batch_size, n_steps), dtype=torch.float32, device=self.device)

        discount_powers = gamma ** torch.arange(n_steps, device=self.device, dtype=torch.float32)

        # First step
        rewards[:, 0] = reward_seq[:, 0]
        masks[:, 0] = 1.0 - terminated_seq[:, 0]  # masks = 1.0 - terminated
        terminals[:, 0] = done_seq[:, 0]  # terminals = float(done)
        truncateds[:, 0] = truncated_seq[:, 0]  # truncateds = float(truncated)

        # Subsequent steps
        for i in range(1, n_steps):
            rewards[:, i] = rewards[:, i - 1] + reward_seq[:, i] * discount_powers[i]
            masks[:, i] = torch.minimum(masks[:, i - 1], 1.0 - terminated_seq[:, i])  # Cumulative masks
            terminals[:, i] = torch.maximum(terminals[:, i - 1], done_seq[:, i])  # Cumulative terminals
            truncateds[:, i] = torch.maximum(
                truncateds[:, i - 1], truncated_seq[:, i]
            )  # Cumulative truncateds
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
            truncateds=truncateds,
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
            BatchTransitionNSteps: A batch sampled from the replay buffer.
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

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        reward_neg: bool = False,
        is_sparse_reward: bool = True,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity. If None, uses dataset length.
            action_mask (Sequence[int] | None): Indices of action dimensions to keep.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        # Convert dataset to transitions generator
        list_transition = cls._lerobotdataset_to_transitions(
            dataset=lerobot_dataset,
            state_keys=state_keys,
            gamma=gamma,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            reward_neg=reward_neg,
            is_sparse_reward=is_sparse_reward,
        )

        # TODO: handle empty dataset case
        first_transition = next(list_transition, None)

        # Initialize the buffer with the first transition to set up storage tensors
        if first_transition is not None:
            first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
            first_action = first_transition["action"].to(device)

            # Get complementary info if available
            first_complementary_info = None
            if (
                "complementary_info" in first_transition
                and first_transition["complementary_info"] is not None
            ):
                first_complementary_info = {
                    k: v.to(device) for k, v in first_transition["complementary_info"].items()
                }

            replay_buffer._initialize_storage(
                state=first_state, action=first_action, complementary_info=first_complementary_info
            )

        # Merge first transition with remaining transitions using itertools.chain

        for data in itertools.chain([first_transition], list_transition):
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(storage_device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(storage_device)

            action = data["action"]

            replay_buffer.add(
                state=data["state"],
                action=action,
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=data["truncated"],  # NOTE: Truncation are not supported yet in lerobot dataset
                complementary_info=data.get("complementary_info", None),
            )

        return replay_buffer

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        reward_neg: bool = False,
        is_sparse_reward: bool = False,
    ) -> Generator[Transition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.

        Args:
            dataset (LeRobotDataset):
                The dataset to convert. Each item in the dataset is expected to have
                at least the following keys:
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                plus whatever your 'state_keys' specify.

            state_keys (Sequence[str] | None):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = DONE in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        current_transitions = []
        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample[ACTION].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = float(current_sample[REWARD].item())  # ensure float

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample[DONE].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            truncated = False
            if not done:
                #  This is important if the dataset has truncations, as it is likely that resuming training will have truncations.
                # If this is the last frame or if next frame is in a different episode, mark as truncated
                if i == num_frames - 1:
                    truncated = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        truncated = True

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # TODO: (azouitine) Check if it's necessary to convert to tensor
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )

            current_transitions.append(transition)

            if done or truncated:
                # Yield all transitions in the current episode
                current_transitions = add_mc_returns_to_trajectory(
                    current_transitions,
                    gamma=gamma,
                    reward_scale=reward_scale,
                    reward_bias=reward_bias,
                    reward_neg=reward_neg,
                    is_sparse_reward=is_sparse_reward,
                )
                yield from current_transitions
                current_transitions = []

        if current_transitions:
            # Yield any remaining transitions after the loop
            current_transitions = add_mc_returns_to_trajectory(
                current_transitions,
                gamma=gamma,
                reward_scale=reward_scale,
                reward_bias=reward_bias,
                reward_neg=reward_neg,
                is_sparse_reward=is_sparse_reward,
            )
            yield from current_transitions


def calc_return_to_go(rewards, terminals, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    A config dict for getting the default high/low reward values for each envs
    from https://github.com/s1lent4gnt/lerobot/blob/lilkm/port-conrft/src/lerobot/scripts/rl/data_util.py
    """
    if len(rewards) == 0:
        return np.array([])

    if is_sparse_reward:
        reward_neg = reward_neg * reward_scale + reward_bias
    else:
        # This assertion is from the JAX implementation, but in PyTorch we might not always have reward_neg
        # for dense rewards, so we can remove it or make it conditional.
        # For now, keeping it as a comment to reflect the original JAX logic.
        # assert not is_sparse_reward, "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."
        pass

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For example, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
            prev_return = return_to_go[-i - 1]

    return np.array(return_to_go, dtype=np.float32)


def add_mc_returns_to_trajectory(
    transitions: list[Transition],
    gamma: float,
    reward_scale: float,
    reward_bias: float,
    reward_neg: bool,
    is_sparse_reward: bool,
) -> list[Transition]:
    """
    Adds Monte Carlo returns to each transition in a trajectory.

    Args:
        transitions (list[Transition]): List of transitions representing a trajectory.
        gamma (float): Discount factor for future rewards.

    Returns:
        list[Transition]: The input list with updated transitions including MC returns.
    """
    rewards = [transition["reward"] for transition in transitions]
    terminals = [transition["done"] or transition["truncated"] for transition in transitions]

    mc_returns = calc_return_to_go(
        rewards=rewards,
        terminals=terminals,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg,
        is_sparse_reward=is_sparse_reward,
    )

    for i, transition in enumerate(transitions):
        # Ensure mc_returns is a float, not a numpy array
        transition["complementary_info"]["mc_returns"] = torch.tensor(
            float(mc_returns[i]), dtype=torch.float32
        )

    return transitions


def concatenate_batch_transitions_nstep(
    left_batch_transitions: BatchTransitionNSteps, right_batch_transition: BatchTransitionNSteps
) -> BatchTransitionNSteps:
    """
    Concatenates two BatchTransitionNSteps objects into one.

    This function merges the right BatchTransitionNSteps into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransitionNSteps): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransitionNSteps): The second batch to append to the first one.

    Returns:
        BatchTransitionNSteps: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # Concatenate basic fields
    left_batch_transitions[ACTION] = torch.cat(
        [left_batch_transitions[ACTION], right_batch_transition[ACTION]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["masks"] = torch.cat(
        [left_batch_transitions["masks"], right_batch_transition["masks"]], dim=0
    )
    left_batch_transitions["terminals"] = torch.cat(
        [left_batch_transitions["terminals"], right_batch_transition["terminals"]], dim=0
    )
    left_batch_transitions["truncateds"] = torch.cat(
        [left_batch_transitions["truncateds"], right_batch_transition["truncateds"]], dim=0
    )
    left_batch_transitions["valid"] = torch.cat(
        [left_batch_transitions["valid"], right_batch_transition["valid"]], dim=0
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # Only process if right_info exists
    if right_info is not None:
        # Initialize left complementary_info if needed
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # Concatenate each field
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions
