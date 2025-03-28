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
import logging
import random
from pprint import pformat
from typing import Callable, Optional, Sequence, TypedDict

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

# TODO: Remove the import of maniskill
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.envs.factory import make_maniskill_env
from lerobot.common.envs.utils import preprocess_maniskill_observation
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
    set_global_seed,
)


def make_optimizers_and_scheduler(cfg, policy):
    optimizer_actor = torch.optim.Adam(
        # NOTE: Handle the case of shared encoder where the encoder weights are not optimized with the gradient of the actor
        params=policy.actor.parameters_to_optimize,
        lr=policy.config.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(
        params=policy.critic_ensemble.parameters(), lr=policy.config.critic_lr
    )
    # We wrap policy log temperature in list because this is a torch tensor and not a nn.Module
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=policy.config.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    return optimizers, lr_scheduler


class Transition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: dict[str, torch.Tensor]
    done: bool
    complementary_info: dict[str, torch.Tensor] = None


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape
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
        replay_buffer = cls(capacity=len(lerobot_dataset), device=device, state_keys=state_keys)
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)
        # Fill the replay buffer with the lerobot dataset transitions
        for data in list_transition:
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
    return left_batch_transitions


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    # Create an env dedicated to online episodes collection from policy rollout.
    # online_env = make_env(cfg, n_envs=cfg.training.online_rollout_batch_size)
    # NOTE: Off policy algorithm are efficient enought to use a single environment
    logging.info("make_env online")
    # online_env = make_env(cfg, n_envs=1)
    # TODO: Remove the import of maniskill and unifiy with make env
    online_env = make_maniskill_env(cfg, n_envs=1)
    if cfg.training.eval_freq > 0:
        logging.info("make_env eval")
        # eval_env = make_env(cfg, n_envs=1)
        # TODO: Remove the import of maniskill and unifiy with make env
        eval_env = make_maniskill_env(cfg, n_envs=1)

    # TODO: Add a way to resume training

    # log metrics to terminal and wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")
    # TODO: At some point we should just need make sac policy
    policy: SACPolicy = make_policy(
        hydra_cfg=cfg,
        # dataset_stats=offline_dataset.meta.stats if not cfg.resume else None,
        # Hack: But if we do online traning, we do not need dataset_stats
        dataset_stats=None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir) if cfg.resume else None,
        device=device,
    )
    assert isinstance(policy, nn.Module)

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg, policy)

    # TODO: Handle resume

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    obs, info = online_env.reset()

    # HACK for maniskill
    # obs = preprocess_observation(obs)
    obs = preprocess_maniskill_observation(obs)
    obs = {key: obs[key].to(device, non_blocking=True) for key in obs}

    replay_buffer = ReplayBuffer(
        capacity=cfg.training.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_shapes.keys(),
    )

    batch_size = cfg.training.batch_size

    if cfg.dataset_repo_id is not None:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
        logging.info("Convertion to a offline replay buffer")
        offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
            offline_dataset, device=device, state_keys=cfg.policy.input_shapes.keys()
        )
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0

    for interaction_step in range(cfg.training.online_steps):
        # NOTE: At some point we should use a  wrapper to handle the observation

        if interaction_step >= cfg.training.online_step_before_learning:
            action = policy.select_action(batch=obs)
            next_obs, reward, done, truncated, info = online_env.step(action.cpu().numpy())
        else:
            action = online_env.action_space.sample()
            next_obs, reward, done, truncated, info = online_env.step(action)
            # HACK
            action = torch.tensor(action, dtype=torch.float32).to(device, non_blocking=True)

        # HACK: For maniskill
        # next_obs = preprocess_observation(next_obs)
        next_obs = preprocess_maniskill_observation(next_obs)
        next_obs = {key: next_obs[key].to(device, non_blocking=True) for key in obs}
        sum_reward_episode += float(reward[0])
        # Because we are using a single environment
        # we can safely assume that the episode is done
        if done[0] or truncated[0]:
            logging.info(f"Global step {interaction_step}: Episode reward: {sum_reward_episode}")
            logger.log_dict({"Sum episode reward": sum_reward_episode}, interaction_step)
            sum_reward_episode = 0
            # HACK: This is for maniskill
            logging.info(
                f"global step {interaction_step}: episode success: {info['success'].float().item()} \n"
            )
            logger.log_dict({"Episode success": info["success"].float().item()}, interaction_step)

        replay_buffer.add(
            state=obs,
            action=action,
            reward=float(reward[0]),
            next_state=next_obs,
            done=done[0],
        )
        obs = next_obs

        if interaction_step < cfg.training.online_step_before_learning:
            continue
        for _ in range(cfg.policy.utd_ratio - 1):
            batch = replay_buffer.sample(batch_size)
            if cfg.dataset_repo_id is not None:
                batch_offline = offline_replay_buffer.sample(batch_size)
                batch = concatenate_batch_transitions(batch, batch_offline)

            actions = batch["action"]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]

            loss_critic = policy.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
            )
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            optimizers["critic"].step()

        batch = replay_buffer.sample(batch_size)
        if cfg.dataset_repo_id is not None:
            batch_offline = offline_replay_buffer.sample(batch_size)
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        loss_critic = policy.compute_loss_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
        )
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        optimizers["critic"].step()

        training_infos = {}
        training_infos["loss_critic"] = loss_critic.item()

        if interaction_step % cfg.training.policy_update_freq == 0:
            # TD3 Trick
            for _ in range(cfg.training.policy_update_freq):
                loss_actor = policy.compute_loss_actor(observations=observations)

                optimizers["actor"].zero_grad()
                loss_actor.backward()
                optimizers["actor"].step()

                training_infos["loss_actor"] = loss_actor.item()

                loss_temperature = policy.compute_loss_temperature(observations=observations)
                optimizers["temperature"].zero_grad()
                loss_temperature.backward()
                optimizers["temperature"].step()

                training_infos["loss_temperature"] = loss_temperature.item()

        if interaction_step % cfg.training.log_freq == 0:
            logger.log_dict(training_infos, interaction_step, mode="train")

        policy.update_target_networks()


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train_notebook(out_dir=None, job_name=None, config_name="default", config_path="../configs"):
    from hydra import compose, initialize

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    train(cfg, out_dir=out_dir, job_name=job_name)


if __name__ == "__main__":
    train_cli()
