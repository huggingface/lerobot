#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
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
import io
import logging
import pickle
import queue
import shutil
import time
from pprint import pformat
from threading import Lock, Thread

import grpc

# Import generated stubs
import hilserl_pb2  # type: ignore
import hilserl_pb2_grpc  # type: ignore
import hydra
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from torch import nn
from torch.optim.optimizer import Optimizer

# TODO: Remove the import of maniskill
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.utils.utils import (
    format_big_number,
    get_global_random_state,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_random_state,
    set_global_seed,
)
from lerobot.scripts.server.buffer import (
    ReplayBuffer,
    move_state_dict_to_device,
    move_transition_to_device,
)

logging.basicConfig(level=logging.INFO)

transition_queue = queue.Queue()
interaction_message_queue = queue.Queue()


def handle_resume_logic(cfg: DictConfig, out_dir: str) -> DictConfig:
    if not cfg.resume:
        if Logger.get_last_checkpoint_dir(out_dir).exists():
            raise RuntimeError(
                f"Output directory {Logger.get_last_checkpoint_dir(out_dir)} already exists. "
                "Use `resume=true` to resume training."
            )
        return cfg

    # if resume == True
    checkpoint_dir = Logger.get_last_checkpoint_dir(out_dir)
    if not checkpoint_dir.exists():
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    checkpoint_cfg_path = str(Logger.get_last_pretrained_model_dir(out_dir) / "config.yaml")
    logging.info(
        colored(
            "Resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)
    diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))

    if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
        del diff["values_changed"]["root['resume']"]

    if len(diff) > 0:
        logging.warning(
            f"Differences between the checkpoint config and the provided config detected: \n{pformat(diff)}\n"
            "Checkpoint configuration takes precedence."
        )

    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: DictConfig,
    logger: Logger,
    optimizers: Optimizer | dict,
):
    if not cfg.resume:
        return None, None

    training_state = torch.load(logger.last_checkpoint_dir / logger.training_state_file_name)

    if isinstance(training_state["optimizer"], dict):
        assert set(training_state["optimizer"].keys()) == set(optimizers.keys())
        for k, v in training_state["optimizer"].items():
            optimizers[k].load_state_dict(v)
    else:
        optimizers.load_state_dict(training_state["optimizer"])

    set_global_random_state({k: training_state[k] for k in get_global_random_state()})
    return training_state["step"], training_state["interaction_step"]


def log_training_info(cfg: DictConfig, out_dir: str, policy: nn.Module) -> None:
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(cfg: DictConfig, logger: Logger, device: str) -> ReplayBuffer:
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.training.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_shapes.keys(),
        )

    dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id, local_files_only=True, root=logger.log_dir / "dataset"
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.training.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_shapes.keys(),
    )


def start_learner_threads(
    cfg: DictConfig,
    device: str,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    batch_size: int,
    optimizers: dict,
    policy: SACPolicy,
    policy_lock: Lock,
    logger: Logger,
    resume_optimization_step: int | None = None,
    resume_interaction_step: int | None = None,
) -> None:
    actor_ip = cfg.actor_learner_config.actor_ip
    port = cfg.actor_learner_config.port

    server_thread = Thread(
        target=stream_transitions_from_actor,
        args=(
            actor_ip,
            port,
        ),
        daemon=True,
    )

    transition_thread = Thread(
        target=add_actor_information_and_train,
        daemon=True,
        args=(
            cfg,
            device,
            replay_buffer,
            offline_replay_buffer,
            batch_size,
            optimizers,
            policy,
            policy_lock,
            logger,
            resume_optimization_step,
            resume_interaction_step,
        ),
    )

    param_push_thread = Thread(
        target=learner_push_parameters,
        args=(policy, policy_lock, actor_ip, port, 15),
        daemon=True,
    )

    server_thread.start()
    transition_thread.start()
    param_push_thread.start()

    param_push_thread.join()
    transition_thread.join()
    server_thread.join()


def stream_transitions_from_actor(host="127.0.0.1", port=50051):
    """
    Runs a gRPC client that listens for transition and interaction messages from an Actor service.

    This function establishes a gRPC connection with the given `host` and `port`, then continuously
    streams transition data from the `ActorServiceStub`. The received transition data is deserialized
    and stored in a queue (`transition_queue`). Similarly, interaction messages are also deserialized
    and stored in a separate queue (`interaction_message_queue`).

    Args:
        host (str, optional): The IP address or hostname of the gRPC server. Defaults to `"127.0.0.1"`.
        port (int, optional): The port number on which the gRPC server is running. Defaults to `50051`.

    """
    # NOTE: This is waiting for the handshake to be done
    # In the future we will do it in a canonical way with a proper handshake
    time.sleep(10)
    channel = grpc.insecure_channel(
        f"{host}:{port}",
        options=[("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)],
    )
    stub = hilserl_pb2_grpc.ActorServiceStub(channel)
    for response in stub.StreamTransition(hilserl_pb2.Empty()):
        if response.HasField("transition"):
            buffer = io.BytesIO(response.transition.transition_bytes)
            transition = torch.load(buffer)
            transition_queue.put(transition)
        if response.HasField("interaction_message"):
            content = pickle.loads(response.interaction_message.interaction_message_bytes)
            interaction_message_queue.put(content)


def learner_push_parameters(
    policy: nn.Module, policy_lock: Lock, actor_host="127.0.0.1", actor_port=50052, seconds_between_pushes=5
):
    """
    As a client, connect to the Actor's gRPC server (ActorService)
    and periodically push new parameters.
    """
    time.sleep(10)
    channel = grpc.insecure_channel(
        f"{actor_host}:{actor_port}",
        options=[("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)],
    )
    actor_stub = hilserl_pb2_grpc.ActorServiceStub(channel)

    while True:
        with policy_lock:
            params_dict = policy.actor.state_dict()
            if policy.config.vision_encoder_name is not None and policy.config.freeze_vision_encoder:
                params_dict = {k: v for k, v in params_dict if not k.startswith("encoder.")}

        params_dict = move_state_dict_to_device(params_dict, device="cpu")
        # Serialize
        buf = io.BytesIO()
        torch.save(params_dict, buf)
        params_bytes = buf.getvalue()

        # Push them to the Actor’s "SendParameters" method
        logging.info("[LEARNER] Publishing parameters to the Actor")
        response = actor_stub.SendParameters(hilserl_pb2.Parameters(parameter_bytes=params_bytes))  # noqa: F841
        time.sleep(seconds_between_pushes)


def add_actor_information_and_train(
    cfg,
    device: str,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    batch_size: int,
    optimizers: dict[str, torch.optim.Optimizer],
    policy: nn.Module,
    policy_lock: Lock,
    logger: Logger,
    resume_optimization_step: int | None = None,
    resume_interaction_step: int | None = None,
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

    **NOTE:**
    - This function performs multiple responsibilities (data transfer, training, and logging).
      It should ideally be split into smaller functions in the future.
    - Due to Python's **Global Interpreter Lock (GIL)**, running separate threads for different tasks
      significantly reduces performance. Instead, this function executes all operations in a single thread.

    Args:
        cfg: Configuration object containing hyperparameters.
        device (str): The computing device (`"cpu"` or `"cuda"`).
        replay_buffer (ReplayBuffer): The primary replay buffer storing online transitions.
        offline_replay_buffer (ReplayBuffer): An additional buffer for offline transitions.
        batch_size (int): The number of transitions to sample per training step.
        optimizers (Dict[str, torch.optim.Optimizer]): A dictionary of optimizers (`"actor"`, `"critic"`, `"temperature"`).
        policy (nn.Module): The reinforcement learning policy with critic, actor, and temperature parameters.
        policy_lock (Lock): A threading lock to ensure safe policy updates.
        logger (Logger): Logger instance for tracking training progress.
        resume_optimization_step (int | None): In the case of resume training, start from the last optimization step reached.
        resume_interaction_step (int | None): In the case of resume training, shift the interaction step with the last saved step in order to not break logging.
    """
    # NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    # in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    # are divided by 200. So we need to have a single thread that does all the work.
    time.time()
    interaction_message, transition = None, None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0
    while True:
        while not transition_queue.empty():
            transition_list = transition_queue.get()
            for transition in transition_list:
                transition = move_transition_to_device(transition, device=device)
                replay_buffer.add(**transition)

                if transition.get("complementary_info", {}).get("is_interaction"):
                    offline_replay_buffer.add(**transition)

        while not interaction_message_queue.empty():
            interaction_message = interaction_message_queue.get()
            # If cfg.resume, shift the interaction step with the last checkpointed step in order to not break the logging
            interaction_message["Interaction step"] += interaction_step_shift
            logger.log_dict(interaction_message, mode="train", custom_step_key="Interaction step")

        if len(replay_buffer) < cfg.training.online_step_before_learning:
            continue
        time_for_one_optimization_step = time.time()
        for _ in range(cfg.policy.utd_ratio - 1):
            batch = replay_buffer.sample(batch_size)

            # if cfg.offline_dataset_repo_id is not None:
            #     batch_offline = offline_replay_buffer.sample(batch_size)
            #     batch = concatenate_batch_transitions(batch, batch_offline)

            actions = batch["action"]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]

            with policy_lock:
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

        # if cfg.offline_dataset_repo_id is not None:
        #     batch_offline = offline_replay_buffer.sample(batch_size)
        #     batch = concatenate_batch_transitions(
        #         left_batch_transitions=batch, right_batch_transition=batch_offline
        #     )

        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        with policy_lock:
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

        if optimization_step % cfg.training.policy_update_freq == 0:
            for _ in range(cfg.training.policy_update_freq):
                with policy_lock:
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

        policy.update_target_networks()
        if optimization_step % cfg.training.log_freq == 0:
            logger.log_dict(training_infos, step=optimization_step, mode="train")

        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        logging.debug(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        logger.log_dict(
            {"Optimization frequency loop [Hz]": frequency_for_one_optimization_step},
            step=optimization_step,
            mode="train",
        )

        optimization_step += 1
        if optimization_step % cfg.training.log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        if cfg.training.save_checkpoint and (
            optimization_step % cfg.training.save_freq == 0 or optimization_step == cfg.training.online_steps
        ):
            logging.info(f"Checkpoint policy after step {optimization_step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            _num_digits = max(6, len(str(cfg.training.online_steps)))
            step_identifier = f"{optimization_step:0{_num_digits}d}"
            interaction_step = (
                interaction_message["Interaction step"] if interaction_message is not None else 0
            )
            logger.save_checkpoint(
                optimization_step,
                policy,
                optimizers,
                scheduler=None,
                identifier=step_identifier,
                interaction_step=interaction_step,
            )

            # TODO : temporarly save replay buffer here, remove later when on the robot
            # We want to control this with the keyboard inputs
            dataset_dir = logger.log_dir / "dataset"
            if dataset_dir.exists() and dataset_dir.is_dir():
                shutil.rmtree(
                    dataset_dir,
                )
            replay_buffer.to_lerobot_dataset(
                cfg.dataset_repo_id, fps=cfg.fps, root=logger.log_dir / "dataset"
            )

            logging.info("Resume training")


def make_optimizers_and_scheduler(cfg, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    **NOTE:**
    - If the encoder is shared, its parameters are excluded from the actor’s optimization process.
    - The policy’s log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
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


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    cfg = handle_resume_logic(cfg, out_dir)

    set_global_seed(cfg.seed)

    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy intance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    # TODO: At some point we should just need make sac policy
    policy_lock = Lock()
    policy: SACPolicy = make_policy(
        hydra_cfg=cfg,
        # dataset_stats=offline_dataset.meta.stats if not cfg.resume else None,
        # Hack: But if we do online traning, we do not need dataset_stats
        dataset_stats=None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir) if cfg.resume else None,
    )
    # compile policy
    policy = torch.compile(policy)
    assert isinstance(policy, nn.Module)

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg, policy)
    resume_optimization_step, resume_interaction_step = load_training_state(cfg, logger, optimizers)

    log_training_info(cfg, out_dir, policy)

    replay_buffer = initialize_replay_buffer(cfg, logger, device)
    batch_size = cfg.training.batch_size
    offline_replay_buffer = None

    # if cfg.dataset_repo_id is not None:
    #     logging.info("make_dataset offline buffer")
    #     offline_dataset = make_dataset(cfg)
    #     logging.info("Convertion to a offline replay buffer")
    #     offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
    #         offline_dataset, device=device, state_keys=cfg.policy.input_shapes.keys()
    #     )
    #     batch_size: int = batch_size // 2  # We will sample from both replay buffer

    start_learner_threads(
        cfg,
        device,
        replay_buffer,
        offline_replay_buffer,
        batch_size,
        optimizers,
        policy,
        policy_lock,
        logger,
        resume_optimization_step,
        resume_interaction_step,
    )


@hydra.main(version_base="1.2", config_name="default", config_path="../../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    train_cli()
