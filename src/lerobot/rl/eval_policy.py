# !/usr/bin/env python

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
import logging
import time
from pprint import pformat

import gymnasium as gym
from termcolor import colored
from torch import nn

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.processor import (
    DataProcessorPipeline,
    EnvTransition,
    TransitionKey,
    create_transition,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    TimerManager,
    init_logging,
)

from .actor import get_frequency_stats
from .gym_manipulator import (
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

logging.basicConfig(level=logging.INFO)


def eval_policy(
    env: gym.Env,
    policy,
    n_episodes,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: TrainRLServerPipelineConfig,
    fps=None,
):
    sum_reward_episode = []
    episode_timer = TimerManager("Episode duration", log=True, logger=logging)
    for _ in range(n_episodes):
        episode_timer.reset()
        policy.reset()
        obs, info = env.reset()
        complementary_data = {}
        env_processor.reset()
        action_processor.reset()
        if hasattr(teleop_device, "reset"):
            teleop_device.reset()

        # Process initial observation
        transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
        transition = env_processor(data=transition)

        # Determine if gripper is used
        # use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

        episode_reward = 0.0

        start_time_for_episode = time.perf_counter()
        episode_steps = 0

        while True:
            episode_timer.start()
            start_time = time.perf_counter()
            observation = {
                k: v
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }
            start_time_policy = time.perf_counter()
            action = policy.select_action(observation)
            end_time_policy = time.perf_counter()

            # obs, reward, terminated, truncated, _ = env.step(action)
            # Use the new step function

            start_time_env_step = time.perf_counter()
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            end_time_env_step = time.perf_counter()
            logging.info(
                f"Times: policy {end_time_policy - start_time_policy:.4f}s, env step {end_time_env_step - start_time_env_step:.4f}s"
            )
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)
            reward = transition.get(TransitionKey.REWARD, 0.0)

            episode_reward += reward
            episode_steps += 1
            episode_timer.stop()
            if terminated or truncated:
                break

            if fps is not None:
                dt_time = time.perf_counter() - start_time
                busy_wait(1 / fps - dt_time)

        sum_reward_episode.append(episode_reward)
        episode_action_fps = episode_steps / (time.perf_counter() - start_time_for_episode)
        logging.info(f"Episode action fps: {episode_action_fps:.2f}")
        stats = get_frequency_stats(episode_timer)
        logging.info(f"Policy frequency stats: {stats}")

    logging.info(f"Success after 20 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode)}")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    cfg.output_dir = None
    cfg.validate()

    init_logging()
    logging.info(pformat(cfg.to_dict()))

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    env_cfg = cfg.env
    env, teleop_device = make_robot_env(env_cfg)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    # set_seed(cfg.seed)
    # device = get_safe_torch_device(cfg.policy.device, log=True)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")
    # dataset_cfg = cfg.dataset
    # dataset = LeRobotDataset(repo_id=dataset_cfg.repo_id)
    # dataset_meta = dataset.meta

    # if env_cfg.pretrained_policy_name_or_path is not None:
    #     cfg.policy.pretrained_path = env_cfg.pretrained_policy_name_or_path
    # else:
    # Construct path to the last checkpoint directory
    # checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    # logging.info(f"Loading training state from {checkpoint_dir}")

    # pretrained_policy_name_or_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR)
    # cfg.policy.pretrained_path = pretrained_policy_name_or_path

    logging.info(f"Using pretrained policy from {cfg.policy.pretrained_path}")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        # ds_meta=dataset_meta,
    )
    # policy.from_pretrained(env_cfg.pretrained_policy_name_or_path)
    policy.eval()
    assert isinstance(policy, nn.Module)

    assert env_cfg.fps is not None, "env_cfg.fps must be set for eval_policy"
    eval_policy(
        env,
        policy=policy,
        n_episodes=10,
        env_processor=env_processor,
        action_processor=action_processor,
        teleop_device=teleop_device,
        cfg=cfg,
        fps=env_cfg.fps,
    )


if __name__ == "__main__":
    register_third_party_devices()
    main()
