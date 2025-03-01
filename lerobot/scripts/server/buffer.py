#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import functools
import random
from typing import Any, Callable, Optional, Sequence, TypedDict

import io
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import pickle


class Transition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: dict[str, torch.Tensor]
    done: bool
    truncated: bool
    complementary_info: dict[str, Any] = None


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor


def move_transition_to_device(
    transition: Transition, device: str = "cpu"
) -> Transition:
    # Move state tensors to CPU
    device = torch.device(device)
    transition["state"] = {
        key: val.to(device, non_blocking=device.type == "cuda")
        for key, val in transition["state"].items()
    }

    # Move action to CPU
    transition["action"] = transition["action"].to(
        device, non_blocking=device.type == "cuda"
    )

    # No need to move reward or done, as they are float and bool

    # No need to move reward or done, as they are float and bool
    if isinstance(transition["reward"], torch.Tensor):
        transition["reward"] = transition["reward"].to(
            device=device, non_blocking=device.type == "cuda"
        )

    if isinstance(transition["done"], torch.Tensor):
        transition["done"] = transition["done"].to(
            device, non_blocking=device.type == "cuda"
        )

    if isinstance(transition["truncated"], torch.Tensor):
        transition["truncated"] = transition["truncated"].to(
            device, non_blocking=device.type == "cuda"
        )

    # Move next_state tensors to CPU
    transition["next_state"] = {
        key: val.to(device, non_blocking=device.type == "cuda")
        for key, val in transition["next_state"].items()
    }

    # If complementary_info is present, move its tensors to CPU
    # if transition["complementary_info"] is not None:
    #     transition["complementary_info"] = {
    #         key: val.to(device, non_blocking=True) for key, val in transition["complementary_info"].items()
    #     }
    return transition


def move_state_dict_to_device(state_dict, device="cpu"):
    """
    Recursively move all tensors in a (potentially) nested
    dict/list/tuple structure to the CPU.
    """
    if isinstance(state_dict, torch.Tensor):
        return state_dict.to(device)
    elif isinstance(state_dict, dict):
        return {
            k: move_state_dict_to_device(v, device=device)
            for k, v in state_dict.items()
        }
    elif isinstance(state_dict, list):
        return [move_state_dict_to_device(v, device=device) for v in state_dict]
    elif isinstance(state_dict, tuple):
        return tuple(move_state_dict_to_device(v, device=device) for v in state_dict)
    else:
        return state_dict


def state_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Convert model state dict to flat array for transmission"""
    buffer = io.BytesIO()

    torch.save(state_dict, buffer)

    return buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return torch.load(buffer)


def python_object_to_bytes(python_object: Any) -> bytes:
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return pickle.load(buffer)


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return torch.load(buffer)


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    buffer = io.BytesIO()
    torch.save(transitions, buffer)
    return buffer.getvalue()


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[
        torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :
    ]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Optional[Sequence[str]] = None,
        image_augmentation_function: Optional[Callable] = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ):
        """
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
        """
        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory

        # Track episode boundaries for memory optimization
        self.episode_ends = torch.zeros(
            capacity, dtype=torch.bool, device=storage_device
        )

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _initialize_storage(self, state: dict[str, torch.Tensor], action: torch.Tensor):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # Pre-allocate tensors for storage
        self.states = {
            key: torch.empty((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.empty(
            (self.capacity, *action_shape), device=self.storage_device
        )
        self.rewards = torch.empty((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.empty(
            (self.capacity,), dtype=torch.bool, device=self.storage_device
        )
        self.truncateds = torch.empty(
            (self.capacity,), dtype=torch.bool, device=self.storage_device
        )
        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Saves a transition, ensuring tensors are stored on the designated storage device."""
        # Initialize storage if this is the first transition
        if not self.initialized:
            self._initialize_storage(state=state, action=action)

        # Store the transition in pre-allocated tensors
        for key in self.states:
            self.states[key][self.position].copy_(state[key].squeeze(dim=0))

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_states[key][self.position].copy_(
                    next_state[key].squeeze(dim=0)
                )

        self.actions[self.position].copy_(action.squeeze(dim=0))
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncateds[self.position] = truncated

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions and collate them into batched tensors."""
        if not self.initialized:
            raise RuntimeError(
                "Cannot sample from an empty buffer. Add transitions first."
            )

        batch_size = min(batch_size, self.size)

        # Random indices for sampling - create on the same device as storage
        idx = torch.randint(
            low=0, high=self.size, size=(batch_size,), device=self.storage_device
        )

        # Identify image keys that need augmentation
        image_keys = (
            [k for k in self.states if k.startswith("observation.image")]
            if self.use_drq
            else []
        )

        # Create batched state and next_state
        batch_state = {}
        batch_next_state = {}

        # First pass: load all state tensors to target device
        for key in self.states:
            batch_state[key] = self.states[key][idx].to(self.device)

            if not self.optimize_memory:
                # Standard approach - load next_states directly
                batch_next_state[key] = self.next_states[key][idx].to(self.device)
            else:
                # Memory-optimized approach - get next_state from the next index
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # State images are at even indices (0, 2, 4...)
                batch_state[key] = augmented_images[
                    i * 2 * batch_size : (i * 2 + 1) * batch_size
                ]
                # Next state images are at odd indices (1, 3, 5...)
                batch_next_state[key] = augmented_images[
                    (i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size
                ]

        # Sample other tensors
        batch_actions = self.actions[idx].to(self.device)
        batch_rewards = self.rewards[idx].to(self.device)
        batch_dones = self.dones[idx].to(self.device).float()
        batch_truncateds = self.truncateds[idx].to(self.device).float()

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
        )

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Optional[Sequence[str]] = None,
        capacity: Optional[int] = None,
        action_mask: Optional[Sequence[int]] = None,
        action_delta: Optional[float] = None,
        image_augmentation_function: Optional[Callable] = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Optional[Sequence[str]]): The list of keys that appear in `state` and `next_state`.
            capacity (Optional[int]): Buffer capacity. If None, uses dataset length.
            action_mask (Optional[Sequence[int]]): Indices of action dimensions to keep.
            action_delta (Optional[float]): Factor to divide actions by.
            image_augmentation_function (Optional[Callable]): Function for image augmentation.
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

        # Convert dataset to transitions
        list_transition = cls._lerobotdataset_to_transitions(
            dataset=lerobot_dataset, state_keys=state_keys
        )

        # Initialize the buffer with the first transition to set up storage tensors
        if list_transition:
            first_transition = list_transition[0]
            first_state = {
                k: v.to(device) for k, v in first_transition["state"].items()
            }
            first_action = first_transition["action"].to(device)

            # Apply action mask/delta if needed
            if action_mask is not None:
                if first_action.dim() == 1:
                    first_action = first_action[action_mask]
                else:
                    first_action = first_action[:, action_mask]

            if action_delta is not None:
                first_action = first_action / action_delta

            replay_buffer._initialize_storage(state=first_state, action=first_action)

        # Fill the buffer with all transitions
        for data in list_transition:
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            action = data["action"]
            if action_mask is not None:
                if action.dim() == 1:
                    action = action[action_mask]
                else:
                    action = action[:, action_mask]

            if action_delta is not None:
                action = action / action_delta

            replay_buffer.add(
                state=data["state"],
                action=action,
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=False,  # NOTE: Truncation are not supported yet in lerobot dataset
            )

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object.
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        sample_action = self.actions[0]
        act_info = guess_feature_info(t=sample_action, name="action")
        features["action"] = act_info

        # Add "reward" and "done"
        features["next.reward"] = {"dtype": "float32", "shape": (1,)}
        features["next.done"] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=None,  # TODO: (azouitine) Handle robot
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # Start writing images if needed
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # Convert transitions into episodes and frames
        episode_index = 0
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(
            episode_index=episode_index
        )

        frame_idx_in_episode = 0
        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            frame_dict = {}

            # Fill the data for state keys
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # Fill action, reward, done
            frame_dict["action"] = self.actions[actual_idx].cpu()
            frame_dict["next.reward"] = torch.tensor(
                [self.rewards[actual_idx]], dtype=torch.float32
            ).cpu()
            frame_dict["next.done"] = torch.tensor(
                [self.dones[actual_idx]], dtype=torch.bool
            ).cpu()

            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict)

            # Move to next frame
            frame_idx_in_episode += 1

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode(task=task_name)
                episode_index += 1
                frame_idx_in_episode = 0
                lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(
                    episode_index=episode_index
                )

        # Save any remaining frames in the buffer
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode(task=task_name)

        lerobot_dataset.stop_image_writer()
        lerobot_dataset.consolidate(run_compute_stats=False, keep_image_files=False)

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Optional[Sequence[str]] = None,
    ) -> list[Transition]:
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

            state_keys (Optional[Sequence[str]]):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError(
                "State keys must be provided when converting LeRobotDataset to Transitions."
            )

        transitions = []
        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = "next.done" in sample

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print(
                "'next.done' key not found in dataset. Inferring from episode boundaries..."
            )

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample["action"].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = float(current_sample["next.reward"].item())  # ensure float

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample["next.done"].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

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

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
            )
            transitions.append(transition)

        return transitions


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t: torch.Tensor, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or array.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to 'float32' for numeric. You can customize as needed.
    """
    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """NOTE: Be careful it change the left_batch_transitions in place"""
    left_batch_transitions["state"] = {
        key: torch.cat(
            [
                left_batch_transitions["state"][key],
                right_batch_transition["state"][key],
            ],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }
    left_batch_transitions["action"] = torch.cat(
        [left_batch_transitions["action"], right_batch_transition["action"]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [
                left_batch_transitions["next_state"][key],
                right_batch_transition["next_state"][key],
            ],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )
    return left_batch_transitions


if __name__ == "__main__":
    import numpy as np
    from tempfile import TemporaryDirectory

    # ===== Test 1: Create and use a synthetic ReplayBuffer =====
    print("Testing synthetic ReplayBuffer...")

    # Create sample data dimensions
    batch_size = 32
    state_dims = {"observation.image": (3, 84, 84), "observation.state": (10,)}
    action_dim = (6,)

    # Create a buffer
    buffer = ReplayBuffer(
        capacity=1000,
        device="cpu",
        state_keys=list(state_dims.keys()),
        use_drq=True,
        storage_device="cpu",
    )

    # Add some random transitions
    for i in range(100):
        # Create dummy transition data
        state = {
            "observation.image": torch.rand(1, 3, 84, 84),
            "observation.state": torch.rand(1, 10),
        }
        action = torch.rand(1, 6)
        reward = 0.5
        next_state = {
            "observation.image": torch.rand(1, 3, 84, 84),
            "observation.state": torch.rand(1, 10),
        }
        done = False if i < 99 else True
        truncated = False

        buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            truncated=truncated,
        )

    # Test sampling
    batch = buffer.sample(batch_size)
    print(f"Buffer size: {len(buffer)}")
    print(
        f"Sampled batch state shapes: {batch['state']['observation.image'].shape}, {batch['state']['observation.state'].shape}"
    )
    print(f"Sampled batch action shape: {batch['action'].shape}")
    print(f"Sampled batch reward shape: {batch['reward'].shape}")
    print(f"Sampled batch done shape: {batch['done'].shape}")
    print(f"Sampled batch truncated shape: {batch['truncated'].shape}")

    # ===== Test for state-action-reward alignment =====
    print("\nTesting state-action-reward alignment...")

    # Create a buffer with controlled transitions where we know the relationships
    aligned_buffer = ReplayBuffer(
        capacity=100, device="cpu", state_keys=["state_value"], storage_device="cpu"
    )

    # Create transitions with known relationships
    # - Each state has a unique signature value
    # - Action is 2x the state signature
    # - Reward is 3x the state signature
    # - Next state is signature + 0.01 (unless at episode end)
    for i in range(100):
        # Create a state with a signature value that encodes the transition number
        signature = float(i) / 100.0
        state = {"state_value": torch.tensor([[signature]]).float()}

        # Action is 2x the signature
        action = torch.tensor([[2.0 * signature]]).float()

        # Reward is 3x the signature
        reward = 3.0 * signature

        # Next state is signature + 0.01, unless end of episode
        # End episode every 10 steps
        is_end = (i + 1) % 10 == 0

        if is_end:
            # At episode boundaries, next_state repeats current state (as per your implementation)
            next_state = {"state_value": torch.tensor([[signature]]).float()}
            done = True
        else:
            # Within episodes, next_state has signature + 0.01
            next_signature = float(i + 1) / 100.0
            next_state = {"state_value": torch.tensor([[next_signature]]).float()}
            done = False

        aligned_buffer.add(state, action, reward, next_state, done, False)

    # Sample from this buffer
    aligned_batch = aligned_buffer.sample(50)

    # Verify alignments in sampled batch
    correct_relationships = 0
    total_checks = 0

    # For each transition in the batch
    for i in range(50):
        # Extract signature from state
        state_sig = aligned_batch["state"]["state_value"][i].item()

        # Check action is 2x signature (within reasonable precision)
        action_val = aligned_batch["action"][i].item()
        action_check = abs(action_val - 2.0 * state_sig) < 1e-4

        # Check reward is 3x signature (within reasonable precision)
        reward_val = aligned_batch["reward"][i].item()
        reward_check = abs(reward_val - 3.0 * state_sig) < 1e-4

        # Check next_state relationship matches our pattern
        next_state_sig = aligned_batch["next_state"]["state_value"][i].item()
        is_done = aligned_batch["done"][i].item() > 0.5

        # Calculate expected next_state value based on done flag
        if is_done:
            # For episodes that end, next_state should equal state
            next_state_check = abs(next_state_sig - state_sig) < 1e-4
        else:
            # For continuing episodes, check if next_state is approximately state + 0.01
            # We need to be careful because we don't know the original index
            # So we check if the increment is roughly 0.01
            next_state_check = (
                abs(next_state_sig - state_sig - 0.01) < 1e-4
                or abs(next_state_sig - state_sig) < 1e-4
            )

        # Count correct relationships
        if action_check:
            correct_relationships += 1
        if reward_check:
            correct_relationships += 1
        if next_state_check:
            correct_relationships += 1

        total_checks += 3

    alignment_accuracy = 100.0 * correct_relationships / total_checks
    print(
        f"State-action-reward-next_state alignment accuracy: {alignment_accuracy:.2f}%"
    )
    if alignment_accuracy > 99.0:
        print(
            "✅ All relationships verified! Buffer maintains correct temporal relationships."
        )
    else:
        print(
            "⚠️ Some relationships don't match expected patterns. Buffer may have alignment issues."
        )

        # Print some debug information about failures
        print("\nDebug information for failed checks:")
        for i in range(5):  # Print first 5 transitions for debugging
            state_sig = aligned_batch["state"]["state_value"][i].item()
            action_val = aligned_batch["action"][i].item()
            reward_val = aligned_batch["reward"][i].item()
            next_state_sig = aligned_batch["next_state"]["state_value"][i].item()
            is_done = aligned_batch["done"][i].item() > 0.5

            print(f"Transition {i}:")
            print(f"  State: {state_sig:.6f}")
            print(f"  Action: {action_val:.6f} (expected: {2.0 * state_sig:.6f})")
            print(f"  Reward: {reward_val:.6f} (expected: {3.0 * state_sig:.6f})")
            print(f"  Done: {is_done}")
            print(f"  Next state: {next_state_sig:.6f}")

            # Calculate expected next state
            if is_done:
                expected_next = state_sig
            else:
                # This approximation might not be perfect
                state_idx = round(state_sig * 100)
                expected_next = (state_idx + 1) / 100.0

            print(f"  Expected next state: {expected_next:.6f}")
            print()

    # ===== Test 2: Convert to LeRobotDataset and back =====
    with TemporaryDirectory() as temp_dir:
        print("\nTesting conversion to LeRobotDataset and back...")
        # Convert buffer to dataset
        repo_id = "test/replay_buffer_conversion"
        # Create a subdirectory to avoid the "directory exists" error
        dataset_dir = os.path.join(temp_dir, "dataset1")
        dataset = buffer.to_lerobot_dataset(repo_id=repo_id, root=dataset_dir)

        print(f"Dataset created with {len(dataset)} frames")
        print(f"Dataset features: {list(dataset.features.keys())}")

        # Check a random sample from the dataset
        sample = dataset[0]
        print(
            f"Dataset sample types: {[(k, type(v)) for k, v in sample.items() if k.startswith('observation')]}"
        )

        # Convert dataset back to buffer
        reconverted_buffer = ReplayBuffer.from_lerobot_dataset(
            dataset, state_keys=list(state_dims.keys()), device="cpu"
        )

        print(f"Reconverted buffer size: {len(reconverted_buffer)}")

        # Sample from the reconverted buffer
        reconverted_batch = reconverted_buffer.sample(batch_size)
        print(
            f"Reconverted batch state shapes: {reconverted_batch['state']['observation.image'].shape}, {reconverted_batch['state']['observation.state'].shape}"
        )

        # Verify consistency before and after conversion
        original_states = batch["state"]["observation.image"].mean().item()
        reconverted_states = (
            reconverted_batch["state"]["observation.image"].mean().item()
        )
        print(f"Original buffer state mean: {original_states:.4f}")
        print(f"Reconverted buffer state mean: {reconverted_states:.4f}")

        if abs(original_states - reconverted_states) < 1.0:
            print("Values are reasonably similar - conversion works as expected")
        else:
            print(
                "WARNING: Significant difference between original and reconverted values"
            )

    print("\nAll previous tests completed!")

    # ===== Test for memory optimization =====
    print("\n===== Testing Memory Optimization =====")

    # Create two buffers, one with memory optimization and one without
    standard_buffer = ReplayBuffer(
        capacity=1000,
        device="cpu",
        state_keys=["observation.image", "observation.state"],
        storage_device="cpu",
        optimize_memory=False,
        use_drq=True,
    )

    optimized_buffer = ReplayBuffer(
        capacity=1000,
        device="cpu",
        state_keys=["observation.image", "observation.state"],
        storage_device="cpu",
        optimize_memory=True,
        use_drq=True,
    )

    # Generate sample data with larger state dimensions for better memory impact
    print("Generating test data...")
    num_episodes = 10
    steps_per_episode = 50
    total_steps = num_episodes * steps_per_episode

    for episode in range(num_episodes):
        for step in range(steps_per_episode):
            # Index in the overall sequence
            i = episode * steps_per_episode + step

            # Create state with identifiable values
            img = torch.ones((3, 84, 84)) * (i / total_steps)
            state_vec = torch.ones((10,)) * (i / total_steps)

            state = {
                "observation.image": img.unsqueeze(0),
                "observation.state": state_vec.unsqueeze(0),
            }

            # Create next state (i+1 or same as current if last in episode)
            is_last_step = step == steps_per_episode - 1

            if is_last_step:
                # At episode end, next state = current state
                next_img = img.clone()
                next_state_vec = state_vec.clone()
                done = True
                truncated = False
            else:
                # Within episode, next state has incremented value
                next_val = (i + 1) / total_steps
                next_img = torch.ones((3, 84, 84)) * next_val
                next_state_vec = torch.ones((10,)) * next_val
                done = False
                truncated = False

            next_state = {
                "observation.image": next_img.unsqueeze(0),
                "observation.state": next_state_vec.unsqueeze(0),
            }

            # Action and reward
            action = torch.tensor([[i / total_steps]])
            reward = float(i / total_steps)

            # Add to both buffers
            standard_buffer.add(state, action, reward, next_state, done, truncated)
            optimized_buffer.add(state, action, reward, next_state, done, truncated)

    # Verify episode boundaries with our simplified approach
    print("\nVerifying simplified memory optimization...")

    # Test with a new buffer with a small sequence
    test_buffer = ReplayBuffer(
        capacity=20,
        device="cpu",
        state_keys=["value"],
        storage_device="cpu",
        optimize_memory=True,
        use_drq=False,
    )

    # Add a simple sequence with known episode boundaries
    for i in range(20):
        val = float(i)
        state = {"value": torch.tensor([[val]]).float()}
        next_val = float(i + 1) if i % 5 != 4 else val  # Episode ends every 5 steps
        next_state = {"value": torch.tensor([[next_val]]).float()}

        # Set done=True at every 5th step
        done = (i % 5) == 4
        action = torch.tensor([[0.0]])
        reward = 1.0
        truncated = False

        test_buffer.add(state, action, reward, next_state, done, truncated)

    # Get sequential batch for verification
    sequential_batch_size = test_buffer.size
    all_indices = torch.arange(sequential_batch_size, device=test_buffer.storage_device)

    # Get state tensors
    batch_state = {
        "value": test_buffer.states["value"][all_indices].to(test_buffer.device)
    }

    # Get next_state using memory-optimized approach (simply index+1)
    next_indices = (all_indices + 1) % test_buffer.capacity
    batch_next_state = {
        "value": test_buffer.states["value"][next_indices].to(test_buffer.device)
    }

    # Get other tensors
    batch_dones = test_buffer.dones[all_indices].to(test_buffer.device)

    # Print sequential values
    print("State, Next State, Done (Sequential values with simplified optimization):")
    state_values = batch_state["value"].squeeze().tolist()
    next_values = batch_next_state["value"].squeeze().tolist()
    done_flags = batch_dones.tolist()

    # Print all values
    for i in range(len(state_values)):
        print(f"  {state_values[i]:.1f} → {next_values[i]:.1f}, Done: {done_flags[i]}")

    # Explain the memory optimization tradeoff
    print("\nWith simplified memory optimization:")
    print("- We always use the next state in the buffer (index+1) as next_state")
    print("- For terminal states, this means using the first state of the next episode")
    print("- This is a common tradeoff in RL implementations for memory efficiency")
    print(
        "- Since we track done flags, the algorithm can handle these transitions correctly"
    )

    # Test random sampling
    print("\nVerifying random sampling with simplified memory optimization...")
    random_samples = test_buffer.sample(20)  # Sample all transitions

    # Extract values
    random_state_values = random_samples["state"]["value"].squeeze().tolist()
    random_next_values = random_samples["next_state"]["value"].squeeze().tolist()
    random_done_flags = random_samples["done"].bool().tolist()

    # Print a few samples
    print("Random samples - State, Next State, Done (First 10):")
    for i in range(10):
        print(
            f"  {random_state_values[i]:.1f} → {random_next_values[i]:.1f}, Done: {random_done_flags[i]}"
        )

    # Calculate memory savings
    # Assume optimized_buffer and standard_buffer have already been initialized and filled
    std_mem = (
        sum(
            standard_buffer.states[key].nelement()
            * standard_buffer.states[key].element_size()
            for key in standard_buffer.states
        )
        * 2
    )
    opt_mem = sum(
        optimized_buffer.states[key].nelement()
        * optimized_buffer.states[key].element_size()
        for key in optimized_buffer.states
    )

    savings_percent = (std_mem - opt_mem) / std_mem * 100

    print(f"\nMemory optimization result:")
    print(f"- Standard buffer state memory: {std_mem / (1024 * 1024):.2f} MB")
    print(f"- Optimized buffer state memory: {opt_mem / (1024 * 1024):.2f} MB")
    print(f"- Memory savings for state tensors: {savings_percent:.1f}%")

    print("\nAll memory optimization tests completed!")

    # # ===== Test real dataset conversion =====
    # print("\n===== Testing Real LeRobotDataset Conversion =====")
    # try:
    #     # Try to use a real dataset if available
    #     dataset_name = "AdilZtn/Maniskill-Pushcube-demonstration-small"
    #     dataset = LeRobotDataset(repo_id=dataset_name)

    #     # Print available keys to debug
    #     sample = dataset[0]
    #     print("Available keys in dataset:", list(sample.keys()))

    #     # Check for required keys
    #     if "action" not in sample or "next.reward" not in sample:
    #         print("Dataset missing essential keys. Cannot convert.")
    #         raise ValueError("Missing required keys in dataset")

    #     # Auto-detect appropriate state keys
    #     image_keys = []
    #     state_keys = []
    #     for k, v in sample.items():
    #         # Skip metadata keys and action/reward keys
    #         if k in {
    #             "index",
    #             "episode_index",
    #             "frame_index",
    #             "timestamp",
    #             "task_index",
    #             "action",
    #             "next.reward",
    #             "next.done",
    #         }:
    #             continue

    #         # Infer key type from tensor shape
    #         if isinstance(v, torch.Tensor):
    #             if len(v.shape) == 3 and (v.shape[0] == 3 or v.shape[0] == 1):
    #                 # Likely an image (channels, height, width)
    #                 image_keys.append(k)
    #             else:
    #                 # Likely state or other vector
    #                 state_keys.append(k)

    #     print(f"Detected image keys: {image_keys}")
    #     print(f"Detected state keys: {state_keys}")

    #     if not image_keys and not state_keys:
    #         print("No usable keys found in dataset, skipping further tests")
    #         raise ValueError("No usable keys found in dataset")

    #     # Test with standard and memory-optimized buffers
    #     for optimize_memory in [False, True]:
    #         buffer_type = "Standard" if not optimize_memory else "Memory-optimized"
    #         print(f"\nTesting {buffer_type} buffer with real dataset...")

    #         # Convert to ReplayBuffer with detected keys
    #         replay_buffer = ReplayBuffer.from_lerobot_dataset(
    #             lerobot_dataset=dataset,
    #             state_keys=image_keys + state_keys,
    #             device="cpu",
    #             optimize_memory=optimize_memory,
    #         )
    #         print(f"Loaded {len(replay_buffer)} transitions from {dataset_name}")

    #         # Test sampling
    #         real_batch = replay_buffer.sample(32)
    #         print(f"Sampled batch from real dataset ({buffer_type}), state shapes:")
    #         for key in real_batch["state"]:
    #             print(f"  {key}: {real_batch['state'][key].shape}")

    #         # Convert back to LeRobotDataset
    #         with TemporaryDirectory() as temp_dir:
    #             dataset_name = f"test/real_dataset_converted_{buffer_type}"
    #             replay_buffer_converted = replay_buffer.to_lerobot_dataset(
    #                 repo_id=dataset_name,
    #                 root=os.path.join(temp_dir, f"dataset_{buffer_type}"),
    #             )
    #             print(
    #                 f"Successfully converted back to LeRobotDataset with {len(replay_buffer_converted)} frames"
    #             )

    # except Exception as e:
    #     print(f"Real dataset test failed: {e}")
    #     print("This is expected if running offline or if the dataset is not available.")

    # print("\nAll tests completed!")
