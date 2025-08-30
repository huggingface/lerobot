# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.scripts.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.acfql.modeling_acfql import ACFQLPolicy
from lerobot.policies.factory import make_policy
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.scripts.rl import learner_service
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

LOG_PREFIX = "[LEARNER]"


#################################################
# MAIN ENTRY POINTS AND CORE ALGORITHM FUNCTIONS #
#################################################


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.utils.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


#################################################
# Core algorithm functions #
#################################################


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    policy: ACFQLPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    if cfg.policy.pretrain_steps == 0:
        push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # Initialize iterators
    online_iterator = None
    offline_iterator = None

    pretrain_steps = cfg.policy.pretrain_steps

    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    while True:
        # Exit the training loop if shutdown is requested
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        if optimization_step < pretrain_steps:
            if offline_replay_buffer is not None and offline_iterator is None:
                logging.info(
                    f"[LEARNER] Pretraining step {optimization_step}/{pretrain_steps}, "
                    "sampling from offline replay buffer"
                )
                offline_iterator = offline_replay_buffer.get_iterator(
                    batch_size=batch_size * 2,  # Use larger batch size for pretraining
                    async_prefetch=async_prefetch,
                    queue_size=2,
                )
        else:
            # Process all available transitions to the replay buffer, send by the actor server
            process_transitions(
                transition_queue=transition_queue,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                device=device,
                dataset_repo_id=dataset_repo_id,
                shutdown_event=shutdown_event,
                # chunk_size=cfg.policy.chunk_size,
            )

            # Process all available interaction messages sent by the actor server
            interaction_message = process_interaction_messages(
                interaction_message_queue=interaction_message_queue,
                interaction_step_shift=interaction_step_shift,
                wandb_logger=wandb_logger,
                shutdown_event=shutdown_event,
            )

            # Wait until the replay buffer has enough samples to start training
            if len(replay_buffer) < online_step_before_learning and not cfg.offline_learning_only:
                continue

            if optimization_step == pretrain_steps:
                logging.info(
                    f"[LEARNER] Pretraining finished, starting online training with {len(replay_buffer)} transitions"
                )
                offline_iterator = None  # Reset offline iterator after pretraining
                if cfg.policy.reset_critics_after_pretraining:
                    logging.info("[LEARNER] Resetting critics after pretraining")
                    policy._init_encoders()
                    policy._init_critics(cfg.policy.output_features["action"].shape[0])
                    policy.to(cfg.policy.device)

                    optimizers["critic"] = torch.optim.Adam(
                        params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr
                    )

            if online_iterator is None and not cfg.offline_learning_only:
                online_iterator = replay_buffer.get_iterator(
                    batch_size=batch_size if not cfg.online_learning_only else batch_size * 2,
                    async_prefetch=async_prefetch,
                    queue_size=2,
                )

            if (
                offline_replay_buffer is not None
                and offline_iterator is None
                and not cfg.online_learning_only
            ):
                offline_iterator = offline_replay_buffer.get_iterator(
                    batch_size=batch_size,
                    async_prefetch=async_prefetch,
                    queue_size=2,
                    n_steps=cfg.policy.chunk_size,
                    gamma=cfg.policy.discount,
                )

        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            # Sample from the iterators
            if not cfg.offline_learning_only and optimization_step >= pretrain_steps:
                batch = next(online_iterator)

                if dataset_repo_id is not None and not cfg.online_learning_only:
                    batch_offline = next(offline_iterator)
                    # batch_offline["action_is_pad"]
                    # batch = batch_offline
                    batch = concatenate_batch_transitions(
                        left_batch_transitions=batch, right_batch_transition=batch_offline
                    )
                    # batch["action_is_pad"]
            else:
                batch = next(offline_iterator)

            actions = batch["action"]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]
            actions_is_pad = batch["action_is_pad"]
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            reward_nsteps = batch["reward_nsteps"]
            next_observation_nsteps = batch["next_state_nsteps"]
            done_nsteps = batch["done_nsteps"]
            truncated_nsteps = batch["truncated_nsteps"]
            discount_nsteps = batch["discount_nsteps"]
            mc_returns = batch["mc_returns"]

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = {
                "action": actions,
                "actions_is_pad": actions_is_pad,
                # "actions_is_pad": torch.zeros_like(actions, dtype=torch.bool),
                # "actions_is_pad": torch.zeros(*actions.shape[:-1], dtype=torch.bool, device=actions.device),
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "complementary_info": batch["complementary_info"],
                "reward_nsteps": reward_nsteps,
                "next_state_nsteps": next_observation_nsteps,
                "done_nsteps": done_nsteps,
                "truncated_nsteps": truncated_nsteps,
                "discount_nsteps": discount_nsteps,
                "mc_returns": mc_returns,
            }

            # Use the forward method for critic loss
            critic_output = policy.forward(forward_batch, model="critic")

            # Main critic optimization
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            # optimizers["discrete_critic"].zero_grad()  # Reset discrete critic optimizer if available
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            # discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            #     parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            # )
            optimizers["critic"].step()
            # optimizers["discrete_critic"].step()  # Step discrete critic optimizer if available

            # Discrete critic optimization (if available)
            # if policy.config.num_discrete_actions is not None:
            #     discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            #     loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            #     optimizers["discrete_critic"].zero_grad()
            #     loss_discrete_critic.backward()
            #     discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            #         parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            #     )
            #     optimizers["discrete_critic"].step()

            # Update target networks (main and discrete)
            policy.update_target_networks()

        if not cfg.offline_learning_only and optimization_step >= pretrain_steps:
            # Sample for the last update in the UTD ratio
            batch = next(online_iterator)

            if dataset_repo_id is not None and not cfg.online_learning_only:
                batch_offline = next(offline_iterator)
                # batch_offline["action"]
                # batch = batch_offline
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )
        else:
            batch = next(offline_iterator)

        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]
        actions_is_pad = batch["action_is_pad"]

        reward_nsteps = batch["reward_nsteps"]
        next_observation_nsteps = batch["next_state_nsteps"]
        done_nsteps = batch["done_nsteps"]
        truncated_nsteps = batch["truncated_nsteps"]
        discount_nsteps = batch["discount_nsteps"]
        mc_returns = batch["mc_returns"]

        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            "action": actions,
            "actions_is_pad": actions_is_pad,
            # "actions_is_pad": torch.zeros_like(actions, dtype=torch.bool),
            # "actions_is_pad": torch.zeros(*actions.shape[:-1], dtype=torch.bool, device=actions.device),
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "reward_nsteps": reward_nsteps,
            "next_state_nsteps": next_observation_nsteps,
            "done_nsteps": done_nsteps,
            "truncated_nsteps": truncated_nsteps,
            "discount_nsteps": discount_nsteps,
            "mc_returns": mc_returns,
        }

        critic_output = policy.forward(forward_batch, model="critic")

        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        # optimizers["discrete_critic"].zero_grad()  # Reset discrete critic optimizer if available
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        # discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        #         parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
        #     ).item()
        optimizers["critic"].step()
        # optimizers["discrete_critic"].step()  # Step discrete critic optimizer if available

        # Initialize training info dictionary
        training_infos = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
            # "discrete_critic_grad_norm": discrete_critic_grad_norm,
        }

        if "info" in critic_output:
            for k, v in critic_output["info"].items():
                training_infos[f"critic_{k}"] = v.item()

        # Discrete critic optimization (if available)
        # if policy.config.num_discrete_actions is not None:
        #     discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
        #     loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
        #     optimizers["discrete_critic"].zero_grad()
        #     loss_discrete_critic.backward()
        #     discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        #         parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
        #     ).item()
        #     optimizers["discrete_critic"].step()

        #     # Add discrete critic info to training info
        #     training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
        #     training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

        #     if "info" in discrete_critic_output:
        #         for k, v in discrete_critic_output["info"].items():
        #             training_infos[f"discrete_critic_{k}"] = v.item()

        # Actor and temperature optimization (at specified frequency)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor BC flow optimization
                actor_bc_flow_output = policy.forward(forward_batch, model="actor_bc_flow")
                loss_actor_bc_flow = actor_bc_flow_output["loss_actor_bc_flow"]
                optimizers["actor_bc_flow"].zero_grad()
                loss_actor_bc_flow.backward()
                actor_bc_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_bc_flow.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor_bc_flow"].step()

                # Add actor info to training info
                training_infos["loss_actor_bc_flow"] = loss_actor_bc_flow.item()
                training_infos["actor_bc_flow_grad_norm"] = actor_bc_flow_grad_norm

                if "info" in actor_bc_flow_output:
                    for k, v in actor_bc_flow_output["info"].items():
                        training_infos[f"actor_bc_flow_{k}"] = v.item()

                # Actor optimization
                # discrete_actor_output = policy.forward(forward_batch, model="discrete_actor")
                # loss_discrete_actor = discrete_actor_output["loss_discrete_actor"]
                # optimizers["discrete_actor"].zero_grad()
                # loss_discrete_actor.backward()
                # discrete_actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                #     parameters=policy.discrete_actor.parameters(), max_norm=clip_grad_norm_value
                # ).item()
                # optimizers["discrete_actor"].step()

                # # Add actor info to training info
                # training_infos["loss_discrete_actor"] = loss_discrete_actor.item()
                # training_infos["discrete_actor_grad_norm"] = discrete_actor_grad_norm

                # if "info" in discrete_actor_output:
                #     for k, v in discrete_actor_output["info"].items():
                #         training_infos[f"discrete_actor_{k}"] = v.item()

                # Actor onestep flow optimization
                actor_onestep_flow_output = policy.forward(forward_batch, model="actor_onestep_flow")
                loss_actor_onestep_flow = actor_onestep_flow_output["loss_actor_onestep_flow"]
                optimizers["actor_onestep_flow"].zero_grad()
                loss_actor_onestep_flow.backward()
                actor_onestep_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_onestep_flow.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor_onestep_flow"].step()

                # Add actor info to training info
                training_infos["loss_actor_onestep_flow"] = loss_actor_onestep_flow.item()
                training_infos["actor_onestep_flow_grad_norm"] = actor_onestep_flow_grad_norm

                if "info" in actor_onestep_flow_output:
                    for k, v in actor_onestep_flow_output["info"].items():
                        training_infos[f"actor_onestep_flow_{k}"] = v.item()

                # Temperature optimization
                # temperature_output = policy.forward(forward_batch, model="temperature")
                # loss_temperature = temperature_output["loss_temperature"]
                # optimizers["temperature"].zero_grad()
                # loss_temperature.backward()
                # temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                #     parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                # ).item()
                # optimizers["temperature"].step()

                # # Add temperature info to training info
                # training_infos["loss_temperature"] = loss_temperature.item()
                # training_infos["temperature_grad_norm"] = temp_grad_norm
                # training_infos["temperature"] = policy.temperature

                # # Update temperature
                # policy.update_temperature()

        # Push policy to actors if needed
        if (
            time.time() - last_time_policy_pushed > policy_parameters_push_frequency
            and optimization_step + 1 >= pretrain_steps
        ):
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        # Update target networks (main and discrete)
        policy.update_target_networks()

        # Log training metrics at specified intervals
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            # Log training metrics
            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        # Calculate and log optimization frequency
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        # Log optimization frequency
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # Save checkpoint at specified intervals
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer if optimization_step > pretrain_steps else None,
                offline_replay_buffer=offline_replay_buffer if optimization_step > pretrain_steps else None,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    service = learner_service.LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(learner_service.SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer | None = None,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    """
    Save training checkpoint and associated data.

    This function performs the following steps:
    1. Creates a checkpoint directory with the current optimization step
    2. Saves the policy model, configuration, and optimizer states
    3. Saves the current interaction step for resuming training
    4. Updates the "last" checkpoint symlink to point to this checkpoint
    5. Saves the replay buffer as a dataset for later use
    6. If an offline replay buffer exists, saves it as a separate dataset

    Args:
        cfg: Training configuration
        optimization_step: Current optimization step
        online_steps: Total number of online steps
        interaction_message: Dictionary containing interaction information
        policy: Policy model to save
        optimizers: Dictionary of optimizers
        replay_buffer: Replay buffer to save as dataset
        offline_replay_buffer: Optional offline replay buffer to save
        dataset_repo_id: Repository ID for dataset
        fps: Frames per second for dataset
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # Create checkpoint directory
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # Save checkpoint
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save interaction step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)

    # TODO : temporary save replay buffer here, remove later when on the robot
    # We want to control this with the keyboard inputs
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Save dataset
    # NOTE: Handle the case where the dataset repo id is not specified in the config
    # eg. RL training without demonstrations data

    if replay_buffer is not None:
        repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
        replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,
            fps=fps,
            root=dataset_offline_dir,
        )

    logging.info("Resume training")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    params_to_skip = [
        "encoder.vla.model.vlm_with_expert.vlm.",
        # "encoder.vla.model.state_proj.",
    ]
    optimizer_actor_bc_flow = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor_bc_flow.named_parameters()
            # if not policy.config.shared_encoder or not n.startswith("encoder")
            if not any(n.startswith(p) for p in params_to_skip)
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_actor_onestep_flow = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor_onestep_flow.named_parameters()
            # if not policy.config.shared_encoder or not n.startswith("encoder")
            if not any(n.startswith(p) for p in params_to_skip)
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    # if cfg.policy.num_discrete_actions is not None:
    #     optimizer_discrete_critic = torch.optim.Adam(
    #         params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
    #     )
    #     optimizer_discrete_actor = torch.optim.Adam(
    #         params=[
    #             p
    #             for n, p in policy.discrete_actor.named_parameters()
    #             if not policy.config.shared_encoder or not n.startswith("encoder")
    #         ],
    #         lr=cfg.policy.actor_lr,
    #     )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor_bc_flow": optimizer_actor_bc_flow,
        "actor_onestep_flow": optimizer_actor_onestep_flow,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    # if cfg.policy.num_discrete_actions is not None:
    #     optimizers["discrete_critic"] = optimizer_discrete_critic
    #     optimizers["discrete_actor"] = optimizer_discrete_actor
    return optimizers, lr_scheduler


#################################################
# Training setup functions #
#################################################


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration

    Returns:
        TrainRLServerPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    """
    if not cfg.resume:
        return None, None

    # Construct path to the last checkpoint directory
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # Use the utility function from train_utils which loads the optimizer state
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # Load interaction step separately from training_state.pt
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
            n_steps=cfg.policy.chunk_size,
            gamma=cfg.policy.discount,
            force_full_n_steps=cfg.policy.force_full_n_steps,
            use_terminal_for_next_state=cfg.policy.use_terminal_for_next_state,
        )

    logging.info("Resume training load the online dataset")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # NOTE: In RL is possible to not have a dataset.
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
        n_steps=cfg.policy.chunk_size,
        gamma=cfg.policy.discount,
        force_full_n_steps=cfg.policy.force_full_n_steps,
        use_terminal_for_next_state=cfg.policy.use_terminal_for_next_state,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    logging.info("Convert to a offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        n_steps=cfg.policy.chunk_size,
        gamma=cfg.policy.discount,
        force_full_n_steps=cfg.policy.force_full_n_steps,
        use_terminal_for_next_state=cfg.policy.use_terminal_for_next_state,
    )
    return offline_replay_buffer


#################################################
# Utilities/Helpers functions #
#################################################


def get_observation_features(
    policy: ACFQLPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    return None, None

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(
            observations, normalize=True
        )
        next_observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(
            next_observations, normalize=True
        )

    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {
        "policy": move_state_dict_to_device(
            {
                k: v
                for k, v in policy.actor_onestep_flow.state_dict().items()
                if not any(k.startswith(p) for p in ("encoder.vla.model.vlm_with_expert.vlm.",))
            },
            device="cpu",
        )
    }

    # # Add discrete critic if it exists
    # if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
    #     state_dicts["discrete_critic"] = move_state_dict_to_device(
    #         policy.discrete_critic.state_dict(), device="cpu"
    #     )
    #     logging.debug("[LEARNER] Including discrete critic in state dict push")

    # Add actor_bc_flow if it exists
    # if hasattr(policy, "actor_bc_flow") and policy.actor_bc_flow is not None:
    #     state_dicts["actor_bc_flow"] = move_state_dict_to_device(
    #         {
    #             k: v
    #             for k, v in policy.actor_bc_flow.state_dict().items()
    #             if not any(
    #                 k.startswith(p) for p in ("encoder.vla.model.vlm_with_expert.vlm.",)
    #             )
    #         },
    #         device="cpu",
    #     )
    #     logging.debug("[LEARNER] Including actor_bc_flow in state dict push")

    # Add discrete actor if it exists
    if hasattr(policy, "discrete_actor") and policy.discrete_actor is not None:
        state_dicts["discrete_actor"] = move_state_dict_to_device(
            policy.discrete_actor.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete actor in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
    # chunk_size: int,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition["action"],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            # # pad to [1, 50, 4]
            # action = transition["action"] # [1, 4]
            # action = einops.repeat(action, "b a -> b e a", e=chunk_size)
            # transition["action"] = action

            # transition["action_is_pad"] = torch.cat([
            #     torch.zeros(action.shape[0], 1, dtype=torch.bool,device=action.device),
            #     torch.ones(action.shape[0], chunk_size-1, dtype=torch.bool, device=action.device)
            # ], dim=1)

            # reward = transition["reward"]
            # reward = einops.repeat(reward, "b -> b e", e=chunk_size)
            # transition["reward"] = reward

            # done = transition["done"]
            # done = einops.repeat(done, "b -> b e", e=chunk_size)
            # transition["done"] = done

            # state = transition["state"]
            # # ['observation.images.front', 'observation.images.wrist', 'observation.state']
            # # Actual state and next chunk size is chunk_size+1
            # for k in state.keys():
            #     if state[k].dim() == 2:
            #         # If the state is 2D, we need to repeat it to match the chunk size
            #         state[k] = einops.repeat(state[k], "b a -> b e a", e=chunk_size+1)
            #     elif state[k].dim() == 3:
            #         # If the state is 3D, we need to repeat it to match the chunk size
            #         state[k] = einops.repeat(state[k], "b c h -> b e c h", e=chunk_size+1)
            #     elif state[k].dim() == 4:
            #         # If the state is 4D, we need to repeat it to match the chunk size
            #         state[k] = einops.repeat(state[k], "b c h w -> b e c h w", e=chunk_size+1)
            #     else:
            #         raise ValueError(
            #             f"Unsupported state dimension {state[k].dim()} for key {k}. Expected 2D or 3D tensor."
            #         )
            # transition["state"] = state

            # import pdb; pdb.set_trace()

            replay_buffer.add(**transition)

            # Add to offline buffer if it's an intervention
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                "is_intervention"
            ):
                offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
