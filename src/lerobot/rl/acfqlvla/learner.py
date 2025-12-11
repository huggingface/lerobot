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
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
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
from pprint import pformat

import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.acfqlvla.modeling_acfqlvla import ACFQLVLAPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.rl.learner import (
    check_nan_in_transition,
    load_training_state,
    log_training_info,
    process_interaction_messages,
    start_learner,
    use_threads,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so100_follower, so101_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport.utils import (
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)

from .buffer import (
    ReplayBufferNSteps as ReplayBuffer,
    concatenate_batch_transitions_nstep as concatenate_batch_transitions,
)
from .configs import ACFQLTrainRLServerPipelineConfig as TrainRLServerPipelineConfig


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
        from lerobot.rl.wandb_utils import WandBLogger

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


# Core algorithm functions


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
    is divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Extract all configuration variables at the beginning
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device_offline_replay_buffer = get_safe_torch_device(
        try_device=cfg.policy.storage_device_offline_replay_buffer
    )
    storage_device_replay_buffer = get_safe_torch_device(try_device=cfg.policy.storage_device_replay_buffer)
    critic_grad_clip_norm_value = cfg.policy.critic_grad_clip_norm
    actor_bc_grad_clip_norm_value = cfg.policy.actor_bc_grad_clip_norm
    actor_onestep_grad_clip_norm_value = cfg.policy.actor_onestep_grad_clip_norm
    value_grad_clip_norm_value = cfg.policy.critic_grad_clip_norm  # Use same as critic
    online_step_before_learning = cfg.policy.online_step_before_learning
    only_successful_online_step_before_learning = cfg.policy.only_successful_online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    offline_steps = cfg.policy.offline_steps
    async_prefetch = cfg.policy.async_prefetch

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    policy: ACFQLVLAPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = cfg.policy.dataset_stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": cfg.policy.dataset_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": cfg.policy.dataset_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # This was commented because the policy will be sent to the actor when the online phase starts
    # push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    batch_size = cfg.batch_size
    replay_buffer = None
    offline_replay_buffer = None

    if online_steps > 0:
        replay_buffer = initialize_replay_buffer(cfg, device, storage_device_replay_buffer)

    if cfg.dataset is not None and offline_steps > 0:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device_offline_replay_buffer,
        )
        # batch_size: int = batch_size // 2  # We will sample from both replay buffer

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    offline_step = min(offline_steps, optimization_step) if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # =============================================================================
    # PHASE 1: OFFLINE PRETRAINING
    # =============================================================================

    offline_iterator = None

    if offline_steps > 0 and offline_replay_buffer is not None and optimization_step < offline_steps:
        logging.info(f"[LEARNER] Starting offline pretraining for {offline_steps} steps")

        offline_iterator = offline_replay_buffer.get_iterator_nstep(
            batch_size=batch_size,
            n_steps=cfg.policy.chunk_size,
            gamma=cfg.policy.discount,
            async_prefetch=async_prefetch,
            queue_size=2,
        )

        for _ in range(offline_step, offline_steps):
            if shutdown_event is not None and shutdown_event.is_set():
                logging.info("[LEARNER] Shutdown signal received during offline training. Exiting...")
                return

            time_for_one_optimization_step = time.time()
            for _ in range(utd_ratio - 1):
                batch = next(offline_iterator)

                # Extract n-step batch components
                actions = batch[ACTION]  # [B, h, action_dim]
                observations = batch["state"]
                next_observations = batch["next_state"]

                # TODO(jpizarrom): encapsulate this, find a better way to avoid permute many times
                observations = preprocessor(
                    {
                        **{"observation.state": observations["observation.state"]},
                        # [B, C, H, W] -> [B, H, W, C]
                        **{
                            k: v.permute(0, 2, 3, 1)
                            for k, v in observations.items()
                            if "observation.images" in k
                        },
                        **{"action": actions},
                        **{"task": ["pick up the pink cube"] * batch_size},
                    }
                )

                actions = observations.pop("action")

                # The preprocessor may add extra keys, filter them out
                observations = {
                    k: v
                    for k, v in observations.items()
                    if k in cfg.policy.input_features
                    or k in ["observation.language.tokens", "observation.language.attention_mask"]
                }

                observations = {
                    **{k: observations[k] for k in observations if "observation.images" not in k},
                    # [B, H, W, C] -> [B, C, H, W]
                    **{
                        k: v.permute(0, 3, 1, 2) for k, v in observations.items() if "observation.images" in k
                    },
                    **{f"{k}.raw": batch["state"][k] for k in batch["state"] if "observation.images" in k},
                }

                next_observations = preprocessor(
                    {
                        **{"observation.state": next_observations["observation.state"]},
                        # [B, C, H, W] -> [B, H, W, C]
                        **{
                            k: v.permute(0, 2, 3, 1)
                            for k, v in next_observations.items()
                            if "observation.images" in k
                        },
                        **{"task": ["pick up the pink cube"] * batch_size},
                    }
                )
                # The preprocessor may add extra keys, filter them out
                next_observations = {
                    k: v
                    for k, v in next_observations.items()
                    if k in cfg.policy.input_features
                    or k in ["observation.language.tokens", "observation.language.attention_mask"]
                }
                next_observations = {
                    **{k: next_observations[k] for k in next_observations if "observation.images" not in k},
                    # [B, H, W, C] -> [B, C, H, W]
                    **{
                        k: v.permute(0, 3, 1, 2)
                        for k, v in next_observations.items()
                        if "observation.images" in k
                    },
                    **{
                        f"{k}.raw": batch["next_state"][k]
                        for k in batch["next_state"]
                        if "observation.images" in k
                    },
                }

                check_nan_in_transition(
                    observations=observations,
                    actions=actions.reshape(actions.shape[0], -1),
                    next_state=next_observations,
                )

                observation_features, next_observation_features = get_observation_features(
                    policy=policy,
                    observations=observations,
                    next_observations=next_observations,
                )

                observation_features_vla, next_observation_features_vla = get_observation_features_vla(
                    policy=policy,
                    observations=observations,
                    next_observations=next_observations,
                )

                # Create a batch dictionary with all required elements for the forward method
                forward_batch = {
                    "state": observations,
                    ACTION: actions,
                    "reward": batch["reward"],
                    "terminal": batch.get("terminals"),
                    "mask": batch.get("masks"),
                    "truncated": batch.get("truncateds"),
                    "valid": batch.get("valid"),
                    "next_state": next_observations,
                    "observation_feature": observation_features,
                    "next_observation_feature": next_observation_features,
                    "observation_feature_vla": observation_features_vla,
                    "next_observation_feature_vla": next_observation_features_vla,
                    "complementary_info": batch.get("complementary_info"),
                }

                # Use the forward method for critic loss
                critic_output = policy.forward(forward_batch, model="critic")

                # Main critic optimization
                loss_critic = critic_output["loss_critic"]
                optimizers["critic"].zero_grad()
                loss_critic.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.critic_ensemble.parameters(), max_norm=critic_grad_clip_norm_value
                )
                optimizers["critic"].step()

                # Update target networks
                policy.update_target_networks()  # keep EMA on critic target as before

            # Sample from the iterators
            batch = next(offline_iterator)

            # Extract n-step batch components
            actions = batch[ACTION]  # [B, h, action_dim]
            observations = batch["state"]
            next_observations = batch["next_state"]

            observations = preprocessor(
                {
                    **{"observation.state": observations["observation.state"]},
                    # [B, C, H, W] -> [B, H, W, C]
                    **{
                        k: v.permute(0, 2, 3, 1) for k, v in observations.items() if "observation.images" in k
                    },
                    **{"action": actions},
                    **{"task": ["pick up the pink cube"] * batch_size},
                }
            )

            actions = observations.pop("action")

            # The preprocessor may add extra keys, filter them out
            observations = {
                k: v
                for k, v in observations.items()
                if k in cfg.policy.input_features
                or k in ["observation.language.tokens", "observation.language.attention_mask"]
            }

            observations = {
                **{k: observations[k] for k in observations if "observation.images" not in k},
                # [B, H, W, C] -> [B, C, H, W]
                **{k: v.permute(0, 3, 1, 2) for k, v in observations.items() if "observation.images" in k},
                **{f"{k}.raw": batch["state"][k] for k in batch["state"] if "observation.images" in k},
            }

            next_observations = preprocessor(
                {
                    **{"observation.state": next_observations["observation.state"]},
                    # [B, C, H, W] -> [B, H, W, C]
                    **{
                        k: v.permute(0, 2, 3, 1)
                        for k, v in next_observations.items()
                        if "observation.images" in k
                    },
                    **{"task": ["pick up the pink cube"] * batch_size},
                }
            )
            # The preprocessor may add extra keys, filter them out
            next_observations = {
                k: v
                for k, v in next_observations.items()
                if k in cfg.policy.input_features
                or k in ["observation.language.tokens", "observation.language.attention_mask"]
            }
            next_observations = {
                **{k: next_observations[k] for k in next_observations if "observation.images" not in k},
                # [B, H, W, C] -> [B, C, H, W]
                **{
                    k: v.permute(0, 3, 1, 2)
                    for k, v in next_observations.items()
                    if "observation.images" in k
                },
                **{
                    f"{k}.raw": batch["next_state"][k]
                    for k in batch["next_state"]
                    if "observation.images" in k
                },
            }

            check_nan_in_transition(
                observations=observations,
                actions=actions.reshape(actions.shape[0], -1),
                next_state=next_observations,
            )

            observation_features, next_observation_features = get_observation_features(
                policy=policy,
                observations=observations,
                next_observations=next_observations,
            )

            observation_features_vla, next_observation_features_vla = get_observation_features_vla(
                policy=policy,
                observations=observations,
                next_observations=next_observations,
            )

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = {
                "state": observations,
                ACTION: actions,
                "reward": batch["reward"],
                "terminal": batch.get("terminals"),
                "truncated": batch.get("truncateds"),
                "mask": batch.get("masks"),
                "valid": batch.get("valid"),
                "next_state": next_observations,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "observation_feature_vla": observation_features_vla,
                "next_observation_feature_vla": next_observation_features_vla,
                "complementary_info": batch.get("complementary_info"),
            }

            critic_output = policy.forward(forward_batch, model="critic")

            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=critic_grad_clip_norm_value
            ).item()
            optimizers["critic"].step()

            training_infos = {
                f"critic/{k}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in critic_output["info"].items()
            }
            training_infos["critic/grad_norm"] = critic_grad_norm

            # Value network update (if enabled)
            if (
                cfg.policy.recap_style_advantages
                and "value" in optimizers
                and batch.get("complementary_info")
                and "mc_returns" in batch.get("complementary_info", {})
            ):
                value_output = policy.forward(forward_batch, model="value")
                loss_value = value_output["loss_value"]
                optimizers["value"].zero_grad()
                loss_value.backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.value_net.parameters(), max_norm=value_grad_clip_norm_value
                ).item()
                optimizers["value"].step()

                training_infos["value/grad_norm"] = value_grad_norm
                training_infos.update(
                    {
                        f"value/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in value_output["info"].items()
                    }
                )

            if optimization_step % policy_update_freq == 0:
                for _ in range(policy_update_freq):
                    # Actor BC flow optimization
                    actor_bc_flow_output = policy.forward(forward_batch, model="actor_bc_flow")
                    loss_actor_bc_flow = actor_bc_flow_output["loss_actor_bc_flow"]
                    optimizers["actor_bc_flow"].zero_grad()
                    loss_actor_bc_flow.backward()
                    actor_bc_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.actor_bc_flow.parameters(), max_norm=actor_bc_grad_clip_norm_value
                    ).item()
                    optimizers["actor_bc_flow"].step()

                    # Add actor info to training info
                    # training_infos["actor_bc/loss"] = loss_actor_bc_flow.item()
                    training_infos["actor_bc/grad_norm"] = actor_bc_flow_grad_norm

                    training_infos.update(
                        {
                            f"actor_bc/{k}": v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in actor_bc_flow_output["info"].items()
                        }
                    )

                    # Actor onestep flow optimization
                    actor_onestep_flow_output = policy.forward(forward_batch, model="actor_onestep_flow")
                    loss_actor_onestep_flow = actor_onestep_flow_output["loss_actor_onestep_flow"]
                    optimizers["actor_onestep_flow"].zero_grad()
                    loss_actor_onestep_flow.backward()
                    actor_onestep_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.actor_onestep_flow.parameters(),
                        max_norm=actor_onestep_grad_clip_norm_value,
                    ).item()
                    optimizers["actor_onestep_flow"].step()

                    # Add actor info to training info
                    # training_infos["actor_one/loss"] = loss_actor_onestep_flow.item()
                    training_infos["actor_one/grad_norm"] = actor_onestep_flow_grad_norm

                    training_infos.update(
                        {
                            f"actor_one/{k}": v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in actor_onestep_flow_output["info"].items()
                        }
                    )

            # Logging
            if optimization_step % log_freq == 0:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
                training_infos["Optimization step"] = optimization_step
                training_infos["phase"] = "offline"

                # Calculate and log optimization frequency
                time_for_one_optimization_step = time.time() - time_for_one_optimization_step
                frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)
                training_infos["Optimization frequency loop [Hz]"] = frequency_for_one_optimization_step
                logging.info(
                    f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}"
                )

                if wandb_logger:
                    wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

                logging.info(
                    f"[LEARNER] Offline step {offline_step}/{offline_steps}, optimization step {optimization_step}"
                )

            optimization_step += 1
            offline_step += 1

            # Save checkpoint
            if saving_checkpoint and (
                optimization_step % save_freq == 0 or optimization_step == offline_steps
            ):
                save_training_checkpoint(
                    cfg=cfg,
                    optimization_step=optimization_step,
                    online_steps=online_steps,
                    interaction_message=interaction_message,
                    policy=policy,
                    optimizers=optimizers,
                    replay_buffer=replay_buffer if cfg.save_replay_buffer_on_checkpoint else None,
                    offline_replay_buffer=offline_replay_buffer
                    if cfg.save_offline_replay_buffer_on_checkpoint
                    else None,
                    dataset_repo_id=cfg.dataset.repo_id if cfg.dataset else None,
                    fps=fps,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )

        logging.info(f"[LEARNER] Completed offline pretraining after {offline_steps} steps")

    if online_steps == 0:
        logging.info("[LEARNER] No online steps specified, training complete.")
        return

    # =============================================================================
    # PHASE 2: ONLINE FINE-TUNING
    # =============================================================================
    logging.info("[LEARNER] Starting online fine-tuning phase")
    logging.info(f"[LEARNER] Online step before learning steps: {online_step_before_learning}")
    online_iterator = None
    has_replay_buffer_enough_samples = False

    if cfg.dataset is not None and offline_replay_buffer is None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device_offline_replay_buffer,
        )

    if offline_iterator is not None:
        # TODO(jpizarrom): clean memory used by offline iterator
        offline_iterator = None

    if dataset_repo_id is not None:
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    # Push policy to actors to start collecting transitions
    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
    last_time_policy_pushed = time.time()

    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    while True:
        # Exit the training loop if shutdown is requested
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # Process all available transitions to the replay buffer, send by the actor server
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
            process_successful_only=not has_replay_buffer_enough_samples
            and only_successful_online_step_before_learning,
        )

        # Process all available interaction messages sent by the actor server
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # Wait until the replay buffer has enough samples to start training
        if len(replay_buffer) < online_step_before_learning:
            continue
        has_replay_buffer_enough_samples = True

        if offline_iterator is None and offline_replay_buffer is not None:
            offline_iterator = offline_replay_buffer.get_iterator_nstep(
                batch_size=batch_size,
                n_steps=cfg.policy.chunk_size,
                gamma=cfg.policy.discount,
                async_prefetch=async_prefetch,
                queue_size=2,
            )

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator_nstep(
                batch_size=batch_size,
                n_steps=cfg.policy.chunk_size,
                gamma=cfg.policy.discount,
                async_prefetch=async_prefetch,
                queue_size=2,
            )

        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            # Sample from the iterators
            batch = next(online_iterator)
            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator)
                # Merge both batches
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            # Extract n-step batch components
            actions = batch[ACTION]  # [B, h, action_dim]
            observations = batch["state"]
            next_observations = batch["next_state"]
            # done = batch["done"]

            observations = preprocessor(
                {
                    **{"observation.state": observations["observation.state"]},
                    # [B, C, H, W] -> [B, H, W, C]
                    **{
                        k: v.permute(0, 2, 3, 1) for k, v in observations.items() if "observation.images" in k
                    },
                    **{"action": actions},
                    **{"task": ["pick up the pink cube"] * batch_size},
                }
            )

            actions = observations.pop("action")

            # The preprocessor may add extra keys, filter them out
            observations = {
                k: v
                for k, v in observations.items()
                if k in cfg.policy.input_features
                or k in ["observation.language.tokens", "observation.language.attention_mask"]
            }

            observations = {
                **{k: observations[k] for k in observations if "observation.images" not in k},
                # [B, H, W, C] -> [B, C, H, W]
                **{k: v.permute(0, 3, 1, 2) for k, v in observations.items() if "observation.images" in k},
                **{f"{k}.raw": batch["state"][k] for k in batch["state"] if "observation.images" in k},
            }

            next_observations = preprocessor(
                {
                    **{"observation.state": next_observations["observation.state"]},
                    # [B, C, H, W] -> [B, H, W, C]
                    **{
                        k: v.permute(0, 2, 3, 1)
                        for k, v in next_observations.items()
                        if "observation.images" in k
                    },
                    **{"task": ["pick up the pink cube"] * batch_size},
                }
            )
            # The preprocessor may add extra keys, filter them out
            next_observations = {
                k: v
                for k, v in next_observations.items()
                if k in cfg.policy.input_features
                or k in ["observation.language.tokens", "observation.language.attention_mask"]
            }
            next_observations = {
                **{k: next_observations[k] for k in next_observations if "observation.images" not in k},
                # [B, H, W, C] -> [B, C, H, W]
                **{
                    k: v.permute(0, 3, 1, 2)
                    for k, v in next_observations.items()
                    if "observation.images" in k
                },
                **{
                    f"{k}.raw": batch["next_state"][k]
                    for k in batch["next_state"]
                    if "observation.images" in k
                },
            }

            check_nan_in_transition(
                observations=observations,
                actions=actions.reshape(actions.shape[0], -1),
                next_state=next_observations,
            )

            observation_features, next_observation_features = get_observation_features(
                policy=policy,
                observations=observations,
                next_observations=next_observations,
            )

            observation_features_vla, next_observation_features_vla = get_observation_features_vla(
                policy=policy,
                observations=observations,
                next_observations=next_observations,
            )

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = {
                "state": observations,
                ACTION: actions,
                "reward": batch["reward"],
                "terminal": batch.get("terminals"),
                "mask": batch.get("masks"),
                "truncated": batch.get("truncateds"),
                "valid": batch.get("valid"),
                "next_state": next_observations,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "observation_feature_vla": observation_features_vla,
                "next_observation_feature_vla": next_observation_features_vla,
                "complementary_info": batch.get("complementary_info"),
            }

            # Use the forward method for critic loss
            critic_output = policy.forward(forward_batch, model="critic")

            # Main critic optimization
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=critic_grad_clip_norm_value
            )
            optimizers["critic"].step()

            # Update target networks (main and discrete)
            policy.update_target_networks()

        # Sample for the last update in the UTD ratio
        batch = next(online_iterator)
        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            # Merge both batches
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        # Extract n-step batch components
        actions = batch[ACTION]  # [B, h, action_dim]
        observations = batch["state"]
        next_observations = batch["next_state"]

        observations = preprocessor(
            {
                **{"observation.state": observations["observation.state"]},
                # [B, C, H, W] -> [B, H, W, C]
                **{k: v.permute(0, 2, 3, 1) for k, v in observations.items() if "observation.images" in k},
                **{"action": actions},
                **{"task": ["pick up the pink cube"] * batch_size},
            }
        )
        actions = observations.pop("action")

        # The preprocessor may add extra keys, filter them out
        observations = {
            k: v
            for k, v in observations.items()
            if k in cfg.policy.input_features
            or k in ["observation.language.tokens", "observation.language.attention_mask"]
        }

        observations = {
            **{k: observations[k] for k in observations if "observation.images" not in k},
            # [B, H, W, C] -> [B, C, H, W]
            **{k: v.permute(0, 3, 1, 2) for k, v in observations.items() if "observation.images" in k},
            **{f"{k}.raw": batch["state"][k] for k in batch["state"] if "observation.images" in k},
        }

        next_observations = preprocessor(
            {
                **{"observation.state": next_observations["observation.state"]},
                # [B, C, H, W] -> [B, H, W, C]
                **{
                    k: v.permute(0, 2, 3, 1)
                    for k, v in next_observations.items()
                    if "observation.images" in k
                },
                **{"task": ["pick up the pink cube"] * batch_size},
            }
        )
        # The preprocessor may add extra keys, filter them out
        next_observations = {
            k: v
            for k, v in next_observations.items()
            if k in cfg.policy.input_features
            or k in ["observation.language.tokens", "observation.language.attention_mask"]
        }
        next_observations = {
            **{k: next_observations[k] for k in next_observations if "observation.images" not in k},
            # [B, H, W, C] -> [B, C, H, W]
            **{k: v.permute(0, 3, 1, 2) for k, v in next_observations.items() if "observation.images" in k},
            **{f"{k}.raw": batch["next_state"][k] for k in batch["next_state"] if "observation.images" in k},
        }

        check_nan_in_transition(
            observations=observations,
            actions=actions.reshape(actions.shape[0], -1),
            next_state=next_observations,
        )

        observation_features, next_observation_features = get_observation_features(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
        )

        observation_features_vla, next_observation_features_vla = get_observation_features_vla(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            "state": observations,
            ACTION: actions,
            "reward": batch["reward"],
            "terminal": batch.get("terminals"),
            "mask": batch.get("masks"),
            "truncated": batch.get("truncateds"),
            "valid": batch.get("valid"),
            "next_state": next_observations,
            # "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "observation_feature_vla": observation_features_vla,
            "next_observation_feature_vla": next_observation_features_vla,
        }

        critic_output = policy.forward(forward_batch, model="critic")

        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=critic_grad_clip_norm_value
        ).item()
        optimizers["critic"].step()

        training_infos = {
            f"critic/{k}": v.item() if isinstance(v, torch.Tensor) else v
            for k, v in critic_output["info"].items()
        }
        training_infos["critic/grad_norm"] = critic_grad_norm

        # Value network update (if enabled)
        if (
            cfg.policy.recap_style_advantages
            and "value" in optimizers
            and batch.get("complementary_info")
            and "mc_returns" in batch.get("complementary_info", {})
        ):
            value_output = policy.forward(forward_batch, model="value")
            loss_value = value_output["loss_value"]
            optimizers["value"].zero_grad()
            loss_value.backward()
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.value_net.parameters(), max_norm=value_grad_clip_norm_value
            ).item()
            optimizers["value"].step()

            training_infos["value/grad_norm"] = value_grad_norm
            training_infos.update(
                {
                    f"value/{k}": v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in value_output["info"].items()
                }
            )

        # Actor optimization (at specified frequency)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor BC flow optimization
                actor_bc_flow_output = policy.forward(forward_batch, model="actor_bc_flow")
                loss_actor_bc_flow = actor_bc_flow_output["loss_actor_bc_flow"]
                optimizers["actor_bc_flow"].zero_grad()
                loss_actor_bc_flow.backward()
                actor_bc_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_bc_flow.parameters(), max_norm=actor_bc_grad_clip_norm_value
                ).item()
                optimizers["actor_bc_flow"].step()

                # Add actor info to training info
                # training_infos["actor_bc/loss"] = loss_actor_bc_flow.item()
                training_infos["actor_bc/grad_norm"] = actor_bc_flow_grad_norm

                training_infos.update(
                    {
                        f"actor_bc/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in actor_bc_flow_output["info"].items()
                    }
                )

                # Actor onestep flow optimization
                actor_onestep_flow_output = policy.forward(forward_batch, model="actor_onestep_flow")
                loss_actor_onestep_flow = actor_onestep_flow_output["loss_actor_onestep_flow"]
                optimizers["actor_onestep_flow"].zero_grad()
                loss_actor_onestep_flow.backward()
                actor_onestep_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_onestep_flow.parameters(),
                    max_norm=actor_onestep_grad_clip_norm_value,
                ).item()
                optimizers["actor_onestep_flow"].step()

                # Add actor info to training info
                # training_infos["actor_one/loss"] = loss_actor_onestep_flow.item()
                training_infos["actor_one/grad_norm"] = actor_onestep_flow_grad_norm

                training_infos.update(
                    {
                        f"actor_one/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in actor_onestep_flow_output["info"].items()
                    }
                )

        # Push policy to actors if needed
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
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
            training_infos["phase"] = "online"

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
                replay_buffer=replay_buffer if cfg.save_replay_buffer_on_checkpoint else None,
                offline_replay_buffer=offline_replay_buffer
                if cfg.save_offline_replay_buffer_on_checkpoint
                else None,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
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
        preprocessor: Optional preprocessor for dataset
        postprocessor: Optional postprocessor for dataset
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
        preprocessor=preprocessor,
        postprocessor=postprocessor,
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
    if replay_buffer is not None and len(replay_buffer) > 0:
        repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
        logging.info(f"Saving replay buffer to {dataset_dir} with repo id {repo_id_buffer_save}")
        replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        logging.info(
            f"Saving offline replay buffer to {dataset_offline_dir} with repo id {cfg.dataset.repo_id}"
        )
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
    Creates and returns optimizers for the actor, critic, and value components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor BC network**, Behavior Cloning actor.
    - The **actor onestep network**, One-step actor.
    - The **critic ensemble**, which evaluates the value function.
    - The **value network** (optional), for Recap-style advantage estimation.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.


    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and value components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    optimizer_params = policy.get_optim_params()

    optimizer_actor_bc_flow = torch.optim.Adam(
        params=optimizer_params["actor_bc_flow"], lr=cfg.policy.actor_lr
    )
    optimizer_actor_onestep_flow = torch.optim.Adam(
        params=optimizer_params["actor_onestep_flow"], lr=cfg.policy.actor_lr
    )
    optimizer_critic = torch.optim.Adam(params=optimizer_params["critic"], lr=cfg.policy.critic_lr)

    optimizers = {
        "actor_bc_flow": optimizer_actor_bc_flow,
        "actor_onestep_flow": optimizer_actor_onestep_flow,
        "critic": optimizer_critic,
    }

    # Add value network optimizer if Recap-style advantages are enabled
    if cfg.policy.recap_style_advantages and "value" in optimizer_params:
        optimizer_value = torch.optim.Adam(params=optimizer_params["value"], lr=cfg.policy.critic_lr)
        optimizers["value"] = optimizer_value
        logging.info("Added value network optimizer for Recap-style advantage estimation")

    lr_scheduler = None

    return optimizers, lr_scheduler


# Training setup functions


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

    # This is needed to populate pretrained_path and checkpoint_path
    checkpoint_cfg.validate()

    return checkpoint_cfg


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

    dataset_path = os.path.join(cfg.output_dir, "dataset")

    if cfg.resume and os.path.exists(dataset_path):
        logging.info("Resume training load the online dataset")

        # NOTE: In RL is possible to not have a dataset.
        repo_id = None
        if cfg.dataset is not None:
            repo_id = cfg.dataset.repo_id
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
        )
    elif cfg.online_dataset is not None:
        logging.info(f"Load the online dataset from the repo {cfg.online_dataset.repo_id}")
        dataset = LeRobotDataset(
            repo_id=cfg.online_dataset.repo_id,
            # root=cfg.online_dataset.path,
        )
    else:
        logging.info(f"Make an empty online replay buffer with capacity {cfg.policy.online_buffer_capacity}")
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info(
        f"Convert to a online replay buffer with {len(dataset)} samples and capacity {cfg.policy.online_buffer_capacity}"
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
        gamma=cfg.policy.discount,
        reward_scale=1.0,
        reward_bias=0.0,
        reward_neg=0.0,
        is_sparse_reward=True,
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
    dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")

    if not cfg.resume or not os.path.exists(dataset_offline_path):
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("load offline dataset")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    logging.info(
        f"Convert to a offline replay buffer with {len(offline_dataset)} samples and capacity {cfg.policy.offline_buffer_capacity}"
    )
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        gamma=cfg.policy.discount,
        reward_scale=1.0,
        reward_bias=0.0,
        reward_neg=0.0,
        is_sparse_reward=True,
    )
    return offline_replay_buffer


# Utilities/Helpers functions


def get_observation_features(
    policy: ACFQLVLAPolicy, observations: torch.Tensor, next_observations: torch.Tensor
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

    if (
        policy.config.vision_encoder_name is None
        or not policy.config.freeze_vision_encoder
        or not policy.config.cache_observation_features
    ):
        return None, None

    with torch.no_grad():
        observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(observations)
        next_observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(
            next_observations
        )

    return observation_features, next_observation_features


def get_observation_features_vla(
    policy: ACFQLVLAPolicy, observations: torch.Tensor, next_observations: torch.Tensor
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

    if (
        # policy.config.vision_encoder_name is None
        # or not policy.config.freeze_vision_encoder
        not policy.config.cache_observation_features_vla
    ):
        return None, None

    with torch.no_grad():
        observation_features = policy.actor_bc_flow.encoder.get_cached_features(observations)
        next_observation_features = None
        # next_observation_features = policy.actor_bc_flow.encoder.get_cached_features(
        #     next_observations
        # )

    return observation_features, next_observation_features


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {"policy": move_state_dict_to_device(policy.actor_onestep_flow.state_dict(), device="cpu")}

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
    process_successful_only: bool = False,
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

        if process_successful_only and len(transition_list) > 0:
            last_transition = transition_list[-1]
            last_reward = last_transition.get("reward", None)

            if last_reward is not None and last_reward <= 0:
                logging.info("[LEARNER] Skipping unsuccessful episode transitions")
                continue

        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition)

            # Add to offline buffer if it's an intervention
            # TODO(jpizarrom): single intervention should not be added to offline buffer when using action chunks, but a chunk where there are intervention make sense
            # TODO(jpizarrom): Review if the enum or the str value is available in the complementary info
            # if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
            #     TeleopEvents.IS_INTERVENTION
            # ):
            #     offline_replay_buffer.add(**transition)


if __name__ == "__main__":
    register_third_party_plugins()
    train_cli()
    logging.info("[LEARNER] main finished")
