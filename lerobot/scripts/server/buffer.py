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

import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class Transition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: dict[str, torch.Tensor]
    done: bool
    complementary_info: dict[str, Any] = None


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor


def move_transition_to_device(transition: Transition, device: str = "cpu") -> Transition:
    # Move state tensors to CPU
    transition["state"] = {key: val.to(device, non_blocking=True) for key, val in transition["state"].items()}

    # Move action to CPU
    transition["action"] = transition["action"].to(device, non_blocking=True)

    # No need to move reward or done, as they are float and bool

    # Move next_state tensors to CPU
    transition["next_state"] = {
        key: val.to(device, non_blocking=True) for key, val in transition["next_state"].items()
    }

    # If complementary_info is present, move its tensors to CPU
    if transition["complementary_info"] is not None:
        transition["complementary_info"] = {
            key: val.to(device, non_blocking=True) for key, val in transition["complementary_info"].items()
        }
    return transition


def move_state_dict_to_device(state_dict, device):
    """
    Recursively move all tensors in a (potentially) nested
    dict/list/tuple structure to the CPU.
    """
    if isinstance(state_dict, torch.Tensor):
        return state_dict.to(device)
    elif isinstance(state_dict, dict):
        return {k: move_state_dict_to_device(v, device=device) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [move_state_dict_to_device(v, device=device) for v in state_dict]
    elif isinstance(state_dict, tuple):
        return tuple(move_state_dict_to_device(v, device=device) for v in state_dict)
    else:
        return state_dict


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
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
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
    ):
        """
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
        """
        self.capacity = capacity
        self.device = device
        self.memory: list[Transition] = []
        self.position = 0

        # If no state_keys provided, default to an empty list
        # (you can handle this differently if needed)
        self.state_keys = state_keys if state_keys is not None else []
        if image_augmentation_function is None:
            self.image_augmentation_function = functools.partial(random_shift, pad=4)
        self.use_drq = use_drq

    def __len__(self):
        return len(self.memory)

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        complementary_info: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # Create and store the Transition
        self.memory[self.position] = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            complementary_info=complementary_info,
        )
        self.position: int = (self.position + 1) % self.capacity

    # TODO: ADD image_augmentation and use_drq arguments in this function in order to instantiate the class with them
    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Optional[Sequence[str]] = None,
        capacity: Optional[int] = None,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device . Defaults to "cuda:0".
            state_keys (Optional[Sequence[str]], optional): The list of keys that appear in `state` and `next_state`.
            Defaults to None.

        Returns:
            ReplayBuffer: The replay buffer with offline dataset transitions.
        """
        # We convert the LeRobotDataset into a replay buffer, because it is more efficient to sample from
        # a replay buffer than from a lerobot dataset.
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        replay_buffer = cls(capacity=capacity, device=device, state_keys=state_keys)
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)
        # Fill the replay buffer with the lerobot dataset transitions
        for data in list_transition:
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            replay_buffer.add(
                state=data["state"],
                action=data["action"],
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
            )
        return replay_buffer

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

        # If not provided, you can either raise an error or define a default:
        if state_keys is None:
            raise ValueError("You must provide a list of keys in `state_keys` that define your 'state'.")

        transitions: list[Transition] = []
        num_frames = len(dataset)

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
            done = bool(current_sample["next.done"].item())  # ensure bool

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
            )
            transitions.append(transition)

        return transitions

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions and collate them into batched tensors."""
        list_of_transitions = random.sample(self.memory, batch_size)

        # -- Build batched states --
        batch_state = {}
        for key in self.state_keys:
            batch_state[key] = torch.cat([t["state"][key] for t in list_of_transitions], dim=0).to(
                self.device
            )
            if key.startswith("observation.image") and self.use_drq:
                batch_state[key] = self.image_augmentation_function(batch_state[key])

        # -- Build batched actions --
        batch_actions = torch.cat([t["action"] for t in list_of_transitions]).to(self.device)

        # -- Build batched rewards --
        batch_rewards = torch.tensor([t["reward"] for t in list_of_transitions], dtype=torch.float32).to(
            self.device
        )

        # -- Build batched next states --
        batch_next_state = {}
        for key in self.state_keys:
            batch_next_state[key] = torch.cat([t["next_state"][key] for t in list_of_transitions], dim=0).to(
                self.device
            )
            if key.startswith("observation.image") and self.use_drq:
                batch_next_state[key] = self.image_augmentation_function(batch_next_state[key])

        # -- Build batched dones --
        batch_dones = torch.tensor([t["done"] for t in list_of_transitions], dtype=torch.float32).to(
            self.device
        )
        batch_dones = torch.tensor([t["done"] for t in list_of_transitions], dtype=torch.float32).to(
            self.device
        )

        # Return a BatchTransition typed dict
        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
        )

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,  # If you have real timestamps, adjust this
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object,
        splitting episodes by transitions where 'done=True'.

        Returns:
            LeRobotDataset: The resulting offline dataset.
        """
        if len(self.memory) == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Infer the shapes and dtypes of your features
        #    We'll create a features dict that is suitable for LeRobotDataset
        # --------------------------------------------------------------------------------------------
        # First, grab one transition to inspect shapes
        first_transition = self.memory[0]

        # We'll store default metadata for every episode: indexes, timestamps, etc.
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        act_info = guess_feature_info(
            first_transition["action"].squeeze(dim=0), "action"
        )  # Remove batch dimension
        features["action"] = act_info

        # Add "reward" (scalars)
        features["next.reward"] = {"dtype": "float32", "shape": (1,)}

        # Add "done" (boolean scalars)
        features["next.done"] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.state_keys:
            sample_val = first_transition["state"][key].squeeze(dim=0)  # Remove batch dimension
            if not isinstance(sample_val, torch.Tensor):
                raise ValueError(
                    f"State key '{key}' is not a torch.Tensor. Please ensure your states are stored as torch.Tensors."
                )
            f_info = guess_feature_info(sample_val, key)
            features[key] = f_info

        # --------------------------------------------------------------------------------------------
        # Create an empty LeRobotDataset
        #    We'll store all frames as separate images only if we detect shape = (3, H, W) or (1, H, W).
        #    By default we won't do videos, but feel free to adapt if you have them.
        # --------------------------------------------------------------------------------------------
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,  # If you have real timestamps, adjust this
            root=root,  # Or some local path where you'd like the dataset files to go
            robot=None,
            robot_type=None,
            features=features,
            use_videos=True,  # We won't do actual video encoding for a replay buffer
        )

        # Start writing images if needed. If you have no image features, this is harmless.
        # Set num_processes or num_threads if you want concurrency.
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=2)

        # --------------------------------------------------------------------------------------------
        # Convert transitions into episodes and frames
        #    We detect episode boundaries by `done == True`.
        # --------------------------------------------------------------------------------------------
        episode_index = 0
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(episode_index)

        frame_idx_in_episode = 0
        for global_frame_idx, transition in enumerate(self.memory):
            frame_dict = {}

            # Fill the data for state keys
            for key in self.state_keys:
                # Expand dimension to match what the dataset expects (the dataset wants the raw shape)
                # We assume your buffer has shape [C, H, W] (if image) or [D] if vector
                # This is typically already correct, but if needed you can reshape below.
                frame_dict[key] = transition["state"][key].cpu().squeeze(dim=0)  # Remove batch dimension

            # Fill action, reward, done
            # Make sure they are shape (X,) or (X,Y,...) as needed.
            frame_dict["action"] = transition["action"].cpu().squeeze(dim=0)  # Remove batch dimension
            frame_dict["next.reward"] = (
                torch.tensor([transition["reward"]], dtype=torch.float32).cpu().squeeze(dim=0)
            )
            frame_dict["next.done"] = (
                torch.tensor([transition["done"]], dtype=torch.bool).cpu().squeeze(dim=0)
            )
            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict)

            # Move to next frame
            frame_idx_in_episode += 1

            # If we reached an episode boundary, call save_episode, reset counters
            if transition["done"]:
                # Use some placeholder name for the task
                lerobot_dataset.save_episode(task="from_replay_buffer")
                episode_index += 1
                frame_idx_in_episode = 0
                # Start a new buffer for the next episode
                lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(episode_index)

        # We are done adding frames
        # If the last transition wasn't done=True, we still have an open buffer with frames.
        # We'll consider that an incomplete episode and still save it:
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode(task=task_name)

        lerobot_dataset.stop_image_writer()

        lerobot_dataset.consolidate(run_compute_stats=False, keep_image_files=False)

        return lerobot_dataset


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
        key: torch.cat([left_batch_transitions["state"][key], right_batch_transition["state"][key]], dim=0)
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
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]], dim=0
        )
        for key in left_batch_transitions["next_state"]
    }
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    return left_batch_transitions


# if __name__ == "__main__":
#     dataset_name = "lerobot/pusht_image"
#     dataset = LeRobotDataset(repo_id=dataset_name, episodes=range(1, 3))
#     replay_buffer = ReplayBuffer.from_lerobot_dataset(
#         lerobot_dataset=dataset, state_keys=["observation.image", "observation.state"]
#     )
#     replay_buffer_converted = replay_buffer.to_lerobot_dataset(repo_id="AdilZtn/pusht_image_converted")
#     for i in range(len(replay_buffer_converted)):
#         replay_convert = replay_buffer_converted[i]
#         dataset_convert = dataset[i]
#         for key in replay_convert.keys():
#             if key in {"index", "episode_index", "frame_index", "timestamp", "task_index"}:
#                 continue
#             if key in dataset_convert.keys():
#                 assert torch.equal(replay_convert[key], dataset_convert[key])
#                 print(f"Key {key} is equal : {replay_convert[key].size()}, {dataset_convert[key].size()}")
#     re_reconverted_dataset = ReplayBuffer.from_lerobot_dataset(
#         replay_buffer_converted, state_keys=["observation.image", "observation.state"], device="cpu"
#     )
#     for _ in range(20):
#         batch = re_reconverted_dataset.sample(32)

#         for key in batch.keys():
#             if key in {"state", "next_state"}:
#                 for key_state in batch[key].keys():
#                     print(key_state, batch[key][key_state].size())
#                 continue
#             print(key, batch[key].size())
