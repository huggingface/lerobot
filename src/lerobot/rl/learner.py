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
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
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

from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService


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

    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    # QC (action-chunking) uses a plugin-local buffer that emits chunk-aware
    # batches (state, action chunk, reward_chunk, state_at_h, valid_h, mask_h).
    # SAC path (default) uses the standard step-level ReplayBuffer.
    _is_qc = getattr(cfg.policy, "type", None) == "qc_ext"
    if _is_qc:
        from lerobot_policy_qc.qc_utils import QCReplayBuffer

        replay_buffer = QCReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            storage_device=storage_device,
        )
    else:
        replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    # QC dual-buffer mode: offline buffer is a chunk-aware buffer that
    # holds demonstrations + intervention transitions. Critic samples 50/50
    # (online, offline). Actor bc_flow loss samples 100% from offline so the
    # behavior-cloning target distribution stays demo-pure (mirrors paper
    # QC-FQL D-distribution that's demo-dominated during online phase).
    # When cfg.policy.offline_preload_n_episodes > 0, the offline buffer is
    # pre-filled with the first N episodes of cfg.dataset.repo_id at start.
    if _is_qc:
        offline_replay_buffer = QCReplayBuffer(
            capacity=int(cfg.policy.offline_buffer_capacity),
            device=device,
            storage_device=storage_device,
        )
        _n_preload = int(getattr(cfg.policy, "offline_preload_n_episodes", 0) or 0)
        if _n_preload > 0 and cfg.dataset is not None and cfg.dataset.repo_id:
            try:
                from lerobot_policy_qc.qc_utils import preload_qc_buffer_from_dataset
                _state_keys = list(cfg.policy.input_features.keys())
                _img_size = (
                    tuple(cfg.env.image_size)
                    if hasattr(cfg.env, "image_size") and cfg.env.image_size is not None
                    else None
                )
                n_loaded = preload_qc_buffer_from_dataset(
                    offline_replay_buffer,
                    dataset_repo_id=cfg.dataset.repo_id,
                    state_keys=_state_keys,
                    n_episodes=_n_preload,
                    target_image_size=_img_size,
                )
                logging.info(
                    f"[QC] Preloaded {n_loaded} demo transitions from "
                    f"{cfg.dataset.repo_id} ({_n_preload} eps) into offline buffer"
                )
            except Exception as e:
                logging.error(f"[QC] offline preload failed: {e}; continuing with empty buffer")
    elif cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )
        batch_size: int = batch_size // 2  # We will sample from both replay buffer
        # Pre-encode all offline images once with the (frozen) image encoder so
        # subsequent offline samples can skip the dominant per-opt-step cost.
        if (
            policy.config.vision_encoder_name is not None
            and policy.config.freeze_vision_encoder
        ):
            if hasattr(policy, "actor") and getattr(policy, "actor", None) is not None:
                _img_enc = policy.actor.encoder.image_encoder
            elif hasattr(policy, "encoder") and getattr(policy, "encoder", None) is not None:
                _img_enc = policy.encoder.image_encoder
            else:
                _img_enc = None
            if _img_enc is not None:
                logging.info(
                    "Precomputing image features for offline buffer (%d frames)…",
                    len(offline_replay_buffer),
                )
                t0 = time.time()
                offline_replay_buffer.precompute_image_features(
                    image_encoder=_img_enc,
                    encode_batch_size=64,
                    encode_device=device,
                )
                logging.info(
                    "Offline image-feature cache built in %.1fs", time.time() - t0
                )

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # ---- BC + RABC auxiliary loss setup ----
    bc_loss_weight_init = float(getattr(policy.config, "bc_loss_weight", 0.0))
    bc_loss_weight_final = float(getattr(policy.config, "bc_loss_weight_final", 0.0))
    bc_anneal_steps = int(getattr(policy.config, "bc_anneal_steps", 0))
    bc_use_rabc = bool(getattr(policy.config, "bc_use_rabc", False))
    rabc_provider = None
    if bc_loss_weight_init > 0.0 and bc_use_rabc:
        rabc_path = getattr(policy.config, "bc_rabc_progress_path", None)
        if rabc_path:
            from lerobot.utils.rabc import RABCWeights

            rabc_provider = RABCWeights(
                progress_path=rabc_path,
                chunk_size=int(policy.config.bc_rabc_chunk_size),
                head_mode=str(policy.config.bc_rabc_head_mode),
                kappa=float(policy.config.bc_rabc_kappa),
                device=device,
            )
            logging.info("BC+RABC enabled: %s", rabc_provider.get_stats())
        else:
            logging.warning(
                "bc_use_rabc=True but bc_rabc_progress_path not set; BC will run uniform-weighted."
            )

    def _current_bc_weight(step: int) -> float:
        if bc_loss_weight_init <= 0.0:
            return 0.0
        if bc_anneal_steps <= 0 or step >= bc_anneal_steps:
            return bc_loss_weight_final if bc_anneal_steps > 0 else bc_loss_weight_init
        frac = step / float(bc_anneal_steps)
        return bc_loss_weight_init + frac * (bc_loss_weight_final - bc_loss_weight_init)

    # Initialize iterators
    online_iterator = None
    offline_iterator = None

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

        # DEBUG: confirm learning actually started + steady-state throughput.
        # Uses raw print(..., flush=True) so it survives Python stdout buffering
        # even when running under tee/redirect with threaded gRPC.
        _warm_n_dbg = int(getattr(cfg.policy, "critic_warmup_steps", 0) or 0)
        _warm_active_dbg = (
            getattr(cfg.policy, "residual_mode", False)
            and _warm_n_dbg > 0
            and optimization_step < _warm_n_dbg
        )
        if not hasattr(add_actor_information_and_train, "_dbg_started"):
            add_actor_information_and_train._dbg_started = True
            add_actor_information_and_train._dbg_t0 = time.time()
            add_actor_information_and_train._dbg_step0 = optimization_step
            print(
                f"[LEARNER-DBG] training STARTED: buffer={len(replay_buffer)} "
                f"threshold={online_step_before_learning} step0={optimization_step} "
                f"critic_warmup_steps={_warm_n_dbg}"
                + (" (critic-only on Q^pi_base)" if _warm_active_dbg else ""),
                flush=True,
            )
            add_actor_information_and_train._dbg_warm_end_logged = (not _warm_active_dbg)
        if getattr(add_actor_information_and_train, "_dbg_started", False) \
                and not getattr(add_actor_information_and_train, "_dbg_warm_end_logged", True) \
                and not _warm_active_dbg:
            print(
                f"[LEARNER-DBG] critic warmup DONE at opt_step={optimization_step} "
                "— actor + temperature updates ENABLED",
                flush=True,
            )
            add_actor_information_and_train._dbg_warm_end_logged = True
        elif (optimization_step - add_actor_information_and_train._dbg_step0) > 0 \
                and optimization_step % 50 == 0:
            _dt = time.time() - add_actor_information_and_train._dbg_t0
            _dn = optimization_step - add_actor_information_and_train._dbg_step0
            _rate = _dn / max(_dt, 1e-6)
            print(
                f"[LEARNER-DBG] opt_step={optimization_step} "
                f"rate={_rate:.2f} steps/s buffer={len(replay_buffer)}",
                flush=True,
            )

        # ----- QC (action-chunking) branch ----------------------------------
        # Type-gated on cfg.policy.type == "qc_ext". SAC path below is bit-
        # identical when this branch is skipped. The QC opt step is
        # self-contained: it samples chunk batches via
        # QCReplayBuffer.sample_chunk_sequence, calls compute_loss_critic /
        # compute_loss_actor directly (bypassing the SAC-shaped forward()
        # dispatcher), and runs its own logging + push + checkpoint
        # bookkeeping before `continue`-ing past the SAC opt body.
        if getattr(cfg.policy, "type", None) == "qc_ext":
            t0_qc = time.time()
            qc_h = int(cfg.policy.horizon_length)
            _utd = max(1, int(getattr(cfg.policy, "utd_ratio", 1) or 1))

            # Critic batch: 50/50 (online ∪ offline) when offline has data,
            # else online only. Critic learns value of all observed chunks.
            def _sample_critic_batch() -> dict:
                _off = offline_replay_buffer
                if (
                    _off is not None
                    and getattr(_off, "size", 0) >= qc_h
                ):
                    _half = batch_size // 2
                    _b_on = replay_buffer.sample_chunk_sequence(
                        batch_size=batch_size - _half, horizon=qc_h
                    )
                    _b_off = _off.sample_chunk_sequence(
                        batch_size=_half, horizon=qc_h
                    )
                    _merged = {}
                    for _k in _b_on:
                        if isinstance(_b_on[_k], dict):
                            _merged[_k] = {
                                _kk: torch.cat([_b_on[_k][_kk], _b_off[_k][_kk]], dim=0)
                                for _kk in _b_on[_k]
                            }
                        else:
                            _merged[_k] = torch.cat([_b_on[_k], _b_off[_k]], dim=0)
                    return _merged
                return replay_buffer.sample_chunk_sequence(
                    batch_size=batch_size, horizon=qc_h
                )

            # Actor batch: 100% offline (demos + interventions) so bc_flow
            # target distribution stays demo-pure. Falls back to online when
            # offline buffer hasn't filled yet.
            def _sample_actor_batch() -> dict:
                _off = offline_replay_buffer
                if _off is not None and getattr(_off, "size", 0) >= qc_h:
                    return _off.sample_chunk_sequence(
                        batch_size=batch_size, horizon=qc_h
                    )
                return replay_buffer.sample_chunk_sequence(
                    batch_size=batch_size, horizon=qc_h
                )

            # Warmup flag: computed before critic update so CQL can be gated.
            _qc_warmup_n = int(getattr(cfg.policy, "critic_warmup_steps", 0) or 0)
            _in_qc_warmup = (_qc_warmup_n > 0) and (optimization_step < _qc_warmup_n)

            # --- critic updates (UTD inner loop) ---
            for _utd_i in range(_utd):
                batch_qc = _sample_critic_batch()
                loss_critic, critic_info = policy.compute_loss_critic(
                    batch_qc, return_components=True,
                    in_critic_warmup=_in_qc_warmup,
                )
                optimizers["critic"].zero_grad()
                loss_critic.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["critic"].step()

            # --- actor update (flow + one_step) ---
            # critic_warmup_steps: keep actor FROZEN at the pretrained init
            # for the first N opt-steps so the value head can learn the new
            # online reward signal before the policy gradient drags the
            # actor off-manifold.
            if not _in_qc_warmup:
                # One-time actor param snapshot at unfreeze for drift tracking.
                if not hasattr(add_actor_information_and_train, "_actor_init_snapshot"):
                    add_actor_information_and_train._actor_init_snapshot = [
                        p.detach().clone()
                        for p in list(policy.flow_actor.parameters())
                        + list(policy.one_step_actor.parameters())
                    ]
                # Actor uses offline-only batch (demos + interventions) for
                # bc_flow + distill target distribution purity. Q-loss inside
                # actor still evaluates Q(s_offline, μ(s_offline, z)).
                batch_actor = _sample_actor_batch()
                loss_actor, actor_info = policy.compute_loss_actor(batch_actor, return_components=True)
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(policy.flow_actor.parameters())
                    + list(policy.one_step_actor.parameters()),
                    max_norm=clip_grad_norm_value,
                ).item()
                optimizers["actor"].step()
                # Drift: ||θ_t - θ_unfreeze||_2 over flow + one_step params.
                with torch.no_grad():
                    _drift_sq = 0.0
                    for p_now, p_init in zip(
                        list(policy.flow_actor.parameters())
                        + list(policy.one_step_actor.parameters()),
                        add_actor_information_and_train._actor_init_snapshot,
                        strict=True,
                    ):
                        _drift_sq += (p_now - p_init).pow(2).sum().item()
                    actor_param_drift = float(_drift_sq ** 0.5)
            else:
                # Still compute the loss for monitoring (no gradient step).
                with torch.no_grad():
                    batch_actor = _sample_actor_batch()
                    loss_actor, actor_info = policy.compute_loss_actor(batch_actor, return_components=True)
                actor_grad_norm = 0.0
                actor_param_drift = 0.0
                if optimization_step == 0 or optimization_step % 200 == 0:
                    logging.info(
                        "[LEARNER] critic-warmup active: opt_step=%d / %d — "
                        "actor frozen, loss_actor=%.4f (monitor only)",
                        optimization_step, _qc_warmup_n, loss_actor.item(),
                    )

            # --- target Polyak EMA ---
            policy.update_target_networks()

            # --- bookkeeping (mirrors the SAC tail of the loop) ---
            if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
                push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
                last_time_policy_pushed = time.time()

            training_infos = {
                "loss_critic": loss_critic.item(),
                "loss_actor": loss_actor.item(),
                "critic_grad_norm": critic_grad_norm,
                "actor_grad_norm": actor_grad_norm,
                "actor_param_drift": actor_param_drift,
                "replay_buffer_size": len(replay_buffer),
                "offline_replay_buffer_size": (
                    len(offline_replay_buffer) if offline_replay_buffer is not None else 0
                ),
                "Optimization step": optimization_step,
            }
            for _k, _v in critic_info.items():
                training_infos[f"critic/{_k}"] = float(_v.item()) if hasattr(_v, "item") else float(_v)
            for _k, _v in actor_info.items():
                training_infos[f"actor/{_k}"] = float(_v.item()) if hasattr(_v, "item") else float(_v)
            if wandb_logger and optimization_step % log_freq == 0:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

            dt_qc = time.time() - t0_qc
            hz_qc = 1.0 / max(dt_qc, 1e-9)
            logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {hz_qc}")
            if wandb_logger:
                wandb_logger.log_dict(
                    {"Optimization frequency loop [Hz]": hz_qc, "Optimization step": optimization_step},
                    mode="train",
                    custom_step_key="Optimization step",
                )

            optimization_step += 1
            if optimization_step % log_freq == 0:
                logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

            if saving_checkpoint and (
                optimization_step % save_freq == 0 or optimization_step == online_steps
            ):
                save_training_checkpoint(
                    cfg=cfg,
                    optimization_step=optimization_step,
                    online_steps=online_steps,
                    interaction_message=interaction_message,
                    policy=policy,
                    optimizers=optimizers,
                    replay_buffer=replay_buffer,
                    offline_replay_buffer=offline_replay_buffer,
                    dataset_repo_id=dataset_repo_id,
                    fps=fps,
                )

            continue
        # ----- /QC branch ---------------------------------------------------

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            # Sample from the iterators
            batch = next(online_iterator)
            online_size_inner = batch[ACTION].shape[0]
            offline_img_feat_inner = None
            offline_next_img_feat_inner = None

            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator)
                offline_img_feat_inner = batch_offline.get("image_features")
                offline_next_img_feat_inner = batch_offline.get("next_image_features")
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            actions = batch[ACTION]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            observation_features, next_observation_features = get_observation_features(
                policy=policy,
                observations=observations,
                next_observations=next_observations,
                online_size=online_size_inner if offline_img_feat_inner is not None else None,
                offline_image_features=offline_img_feat_inner,
                offline_next_image_features=offline_next_img_feat_inner,
            )

            # Create a batch dictionary with all required elements for the forward method
            _critic_warmup_steps = int(getattr(policy.config, "critic_warmup_steps", 0) or 0)
            _in_critic_warmup = (
                getattr(policy.config, "residual_mode", False)
                and _critic_warmup_steps > 0
                and optimization_step < _critic_warmup_steps
            )
            forward_batch = {
                ACTION: actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "complementary_info": batch["complementary_info"],
                "critic_warmup": _in_critic_warmup,
            }

            # Use the forward method for critic loss
            critic_output = policy.forward(forward_batch, model="critic")

            # Main critic optimization
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()

            # Discrete critic optimization (if available)
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["discrete_critic"].step()

            # Update target networks (main and discrete)
            policy.update_target_networks()

        # Sample for the last update in the UTD ratio
        batch = next(online_iterator)
        online_size = batch[ACTION].shape[0]
        batch_offline_for_bc = None
        offline_img_feat = None
        offline_next_img_feat = None

        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            batch_offline_for_bc = batch_offline
            offline_img_feat = batch_offline.get("image_features")
            offline_next_img_feat = batch_offline.get("next_image_features")
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        actions = batch[ACTION]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
            online_size=online_size if offline_img_feat is not None else None,
            offline_image_features=offline_img_feat,
            offline_next_image_features=offline_next_img_feat,
        )

        _critic_warmup_steps = int(getattr(policy.config, "critic_warmup_steps", 0) or 0)
        _in_critic_warmup = (
            getattr(policy.config, "residual_mode", False)
            and _critic_warmup_steps > 0
            and optimization_step < _critic_warmup_steps
        )
        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            ACTION: actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "critic_warmup": _in_critic_warmup,
        }

        critic_output = policy.forward(forward_batch, model="critic")

        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        # Initialize training info dictionary
        training_infos = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
            "critic_warmup": int(_in_critic_warmup),
        }

        # Build BC inputs from offline batch (demo actions) for actor opt step.
        bc_w_now = _current_bc_weight(optimization_step)
        if bc_w_now > 0.0 and batch_offline_for_bc is not None:
            bc_state = batch_offline_for_bc["state"]
            bc_action = batch_offline_for_bc[ACTION]
            # Reuse the offline tail of the just-encoded features rather than
            # running the image encoder a third time on bc_state.
            if observation_features is not None:
                bc_obs_feat = {k: v[online_size:] for k, v in observation_features.items()}
            else:
                bc_obs_feat = None
            bc_weights_tensor = None
            if rabc_provider is not None:
                # RABCWeights expects a flat `index` field; thread it through from the
                # offline buffer's complementary_info (preserved at ingestion time).
                rabc_batch = dict(batch_offline_for_bc)
                comp = batch_offline_for_bc.get("complementary_info") or {}
                if "dataset_index" in comp and "index" not in rabc_batch:
                    rabc_batch["index"] = comp["dataset_index"]
                bc_weights_tensor, bc_w_stats = rabc_provider.compute_batch_weights(rabc_batch)
                training_infos["bc_rabc_mean_weight"] = float(
                    bc_w_stats.get("raw_mean_weight", 0.0)
                )
                training_infos["bc_rabc_num_zero"] = int(bc_w_stats.get("num_zero_weight", 0))
                training_infos["bc_rabc_num_full"] = int(bc_w_stats.get("num_full_weight", 0))
            forward_batch["bc_state"] = bc_state
            forward_batch["bc_action"] = bc_action
            forward_batch["bc_observation_feature"] = bc_obs_feat
            forward_batch["bc_weights"] = bc_weights_tensor
            forward_batch["bc_loss_weight"] = bc_w_now
            training_infos["bc_loss_weight"] = bc_w_now

        # Discrete critic optimization (if available)
        if policy.config.num_discrete_actions is not None:
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()

            # Add discrete critic info to training info
            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

        # Actor and temperature optimization (at specified frequency).
        # Skip during critic warmup so the critic stabilizes on Q^{π_base}
        # before the actor starts moving residuals.
        if optimization_step % policy_update_freq == 0 and not _in_critic_warmup:
            for _ in range(policy_update_freq):
                # Actor optimization
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # Add actor info to training info
                training_infos["loss_actor"] = loss_actor.item()
                training_infos["actor_grad_norm"] = actor_grad_norm
                if "loss_actor_sac" in actor_output:
                    training_infos["loss_actor_sac"] = float(actor_output["loss_actor_sac"].item())
                if "loss_actor_bc" in actor_output:
                    training_infos["loss_actor_bc"] = float(actor_output["loss_actor_bc"].item())

                # Temperature optimization
                temperature_output = policy.forward(forward_batch, model="temperature")
                loss_temperature = temperature_output["loss_temperature"]
                optimizers["temperature"].zero_grad()
                loss_temperature.backward()
                temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                ).item()
                optimizers["temperature"].step()

                # Add temperature info to training info
                training_infos["loss_temperature"] = loss_temperature.item()
                training_infos["temperature_grad_norm"] = temp_grad_norm
                training_infos["temperature"] = policy.temperature

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
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
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

    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
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
    server.stop(SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


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
    # Plugins (e.g. QCPolicy) that have no monolithic `.actor`/`.log_alpha`
    # surface opt into this path by defining `get_optim_params()` returning
    # `{"actor": [...params...], "critic": [...params...]}`. No SAC-style
    # entropy temperature for QC — `flow_actor + one_step_actor` is the
    # actor group and we keep them under a single Adam. SAC stays on the
    # untouched code path below.
    if hasattr(policy, "get_optim_params"):
        groups = policy.get_optim_params()
        optimizer_actor = torch.optim.Adam(params=list(groups["actor"]), lr=cfg.policy.actor_lr)
        optimizer_critic = torch.optim.Adam(params=list(groups["critic"]), lr=cfg.policy.critic_lr)
        return {"actor": optimizer_actor, "critic": optimizer_critic}, None

    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
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

    stride = int(getattr(cfg, "offline_dataset_stride", 1))
    drop_idle = float(getattr(cfg, "offline_drop_idle_threshold", 0.0))
    logging.info(
        "Convert to a offline replay buffer (stride=%d, drop_idle=%s)",
        stride,
        drop_idle,
    )
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        stride=stride,
        drop_idle_threshold=drop_idle,
        # Frozen vision encoder + precomputed feature cache make image
        # augmentation a wasted CPU/GPU pass on offline samples.
        use_drq=False,
    )
    return offline_replay_buffer


# Utilities/Helpers functions


def get_observation_features(
    policy: SACPolicy,
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    online_size: int | None = None,
    offline_image_features: dict[str, torch.Tensor] | None = None,
    offline_next_image_features: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Encode current/next observation images, optionally splicing pre-computed
    features for the offline portion of the batch.

    When `offline_image_features` is provided, it must already correspond to the
    offline tail of `observations` (i.e. observations[online_size:]). Only the
    online prefix is run through the (frozen) image encoder; the cached offline
    features are concatenated on the channel-batch dim.
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        if offline_image_features is None:
            observation_features = policy.actor.encoder.get_cached_image_features(observations)
            next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations)
        else:
            assert online_size is not None, (
                "online_size must be provided when offline_image_features is supplied"
            )
            online_obs = {k: observations[k][:online_size] for k in observations}
            online_next = {k: next_observations[k][:online_size] for k in next_observations}
            online_feat = policy.actor.encoder.get_cached_image_features(online_obs)
            online_next_feat = policy.actor.encoder.get_cached_image_features(online_next)
            observation_features = {
                k: torch.cat([online_feat[k], offline_image_features[k]], dim=0)
                for k in online_feat
            }
            if offline_next_image_features is None:
                # Offline buffer is built with optimize_memory=True so next state
                # is just states[(i+1) % cap]. Without an explicit cache, we have
                # to encode the offline next images. Fall back to single-pass
                # encode of the full next batch.
                next_observation_features = policy.actor.encoder.get_cached_image_features(
                    next_observations
                )
            else:
                next_observation_features = {
                    k: torch.cat([online_next_feat[k], offline_next_image_features[k]], dim=0)
                    for k in online_next_feat
                }

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

    # Plugins (e.g. QCPolicy) with no monolithic `.actor` submodule opt into
    # this path by defining `get_actor_state_dict()` returning a {name:
    # state_dict} dict already moved to CPU. SAC keeps its untouched
    # `policy.actor.state_dict()` form below — back-compat preserved.
    if hasattr(policy, "get_actor_state_dict"):
        actor_state = policy.get_actor_state_dict()
    else:
        actor_state = move_state_dict_to_device(policy.actor.state_dict(), device="cpu")
    state_dicts = {"policy": actor_state}

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

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
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition)

            # Add to offline buffer if it's an intervention. QC dual-buffer
            # mode creates an EMPTY offline buffer at init (no dataset
            # preload), so gate solely on offline_replay_buffer existence.
            # Key is the enum's string value (torch.save w/ weights_only=True
            # strips Enum objects, so the actor serializes the string).
            _ci = transition.get("complementary_info", {}) or {}
            _interv_flag = _ci.get(TeleopEvents.IS_INTERVENTION.value)
            if hasattr(_interv_flag, "item"):
                _interv_flag = bool(_interv_flag.item())
            if offline_replay_buffer is not None and bool(_interv_flag):
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
