import grpc
from concurrent import futures
import functools
import logging
import queue
import pickle
import torch
import torch.nn.functional as F
import io
import time

from pprint import pformat
import random
from typing import Optional, Sequence, TypedDict, Callable

import hydra
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from threading import Thread, Lock

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# TODO: Remove the import of maniskill
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.server.buffer import ReplayBuffer, move_transition_to_device, concatenate_batch_transitions, move_state_dict_to_device, Transition

# Import generated stubs
import hilserl_pb2
import hilserl_pb2_grpc

logging.basicConfig(level=logging.INFO)



# TODO: Implement it in cleaner way maybe
transition_queue = queue.Queue()
interaction_message_queue = queue.Queue()


# 1) Implement the LearnerService so the Actor can send transitions here.
class LearnerServiceServicer(hilserl_pb2_grpc.LearnerServiceServicer):
    # def SendTransition(self, request, context):
    #     """
    #     Actor calls this method to push a Transition -> Learner.
    #     """
    #     buffer = io.BytesIO(request.transition_bytes)
    #     transition = torch.load(buffer)
    #     transition_queue.put(transition)
    #     return hilserl_pb2.Empty()
    def SendInteractionMessage(self, request, context):
        """
        Actor calls this method to push a Transition -> Learner.
        """
        content = pickle.loads(request.interaction_message_bytes)
        interaction_message_queue.put(content)
        return hilserl_pb2.Empty()



def stream_transitions_from_actor(port=50051):
    """
    Runs a gRPC server listening for transitions from the Actor.
    """
    time.sleep(10)
    channel = grpc.insecure_channel(f'127.0.0.1:{port}',
                             options=[('grpc.max_send_message_length', -1),
                                      ('grpc.max_receive_message_length', -1)])
    stub = hilserl_pb2_grpc.ActorServiceStub(channel)
    for response in stub.StreamTransition(hilserl_pb2.Empty()):
        if response.HasField('transition'):
            buffer = io.BytesIO(response.transition.transition_bytes)
            transition = torch.load(buffer)
            transition_queue.put(transition)
        if response.HasField('interaction_message'):
            content = pickle.loads(response.interaction_message.interaction_message_bytes)
            interaction_message_queue.put(content)
        # NOTE: Cool down the CPU, if you comment this line you will make a huge bottleneck
        time.sleep(0.001)

def learner_push_parameters(
    policy: nn.Module, policy_lock: Lock, actor_host="127.0.0.1", actor_port=50052, seconds_between_pushes=5
):
    """
    As a client, connect to the Actor's gRPC server (ActorService)
    and periodically push new parameters.
    """
    time.sleep(10)
    # The Actor's server is presumably listening on a different port, e.g. 50052
    channel = grpc.insecure_channel(f"{actor_host}:{actor_port}",
                             options=[('grpc.max_send_message_length', -1),
                                      ('grpc.max_receive_message_length', -1)])
    actor_stub = hilserl_pb2_grpc.ActorServiceStub(channel)

    while True:
        with policy_lock:
            params_dict = policy.actor.state_dict()
        params_dict = move_state_dict_to_device(params_dict, device="cpu")
        # Serialize
        buf = io.BytesIO()
        torch.save(params_dict, buf)
        params_bytes = buf.getvalue()

        # Push them to the Actor’s "SendParameters" method
        logging.info(f"[LEARNER] Pushing parameters to the Actor")
        response = actor_stub.SendParameters(hilserl_pb2.Parameters(parameter_bytes=params_bytes))
        time.sleep(seconds_between_pushes)


# Checked 
def add_actor_information(
    cfg,
    device,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    batch_size: int,
    optimizers,
    policy, 
    policy_lock: Lock,
    buffer_lock: Lock,
    offline_buffer_lock: Lock,
    logger_lock: Lock,
    logger: Logger,
):
    """
    In a real application, you might run your training loop here,
    reading from the transition queue and doing gradient updates.
    """
    # NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    # in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    # are divided by 200. So we need to have a single thread that does all the work.
    start = time.time()
    optimization_step = 0
    timeout_for_adding_transitions = 1
    while True:
        time_for_adding_transitions = time.time()
        while not transition_queue.empty():

            transition_list = transition_queue.get()
            for transition in transition_list:
                transition = move_transition_to_device(transition, device=device)
                replay_buffer.add(**transition)
                # logging.info(f"[LEARNER] size of replay buffer: {len(replay_buffer)}")
                # logging.info(f"[LEARNER] size of transition queues: {transition_queue.qsize()}")
                # logging.info(f"[LEARNER] size of replay buffer: {len(replay_buffer)}")
                # logging.info(f"[LEARNER] size of transition queues: {transition }")
            if len(replay_buffer) > cfg.training.online_step_before_learning:
                logging.info(f"[LEARNER] size of replay buffer: {len(replay_buffer)}")

        while not interaction_message_queue.empty():
            interaction_message = interaction_message_queue.get()
            logger.log_dict(interaction_message,mode="train",custom_step_key="interaction_step")
            # logging.info(f"[LEARNER] size of interaction message queue: {interaction_message_queue.qsize()}")

        if len(replay_buffer) < cfg.training.online_step_before_learning:
            continue
        time_for_one_optimization_step = time.time()
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

        if optimization_step % cfg.training.log_freq == 0:
            logger.log_dict(training_infos, step=optimization_step, mode="train")

        policy.update_target_networks()
        optimization_step += 1
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step

        logging.info(f"[LEARNER] Time for one optimization step: {time_for_one_optimization_step}")
        logger.log_dict({"Time optimization step":time_for_one_optimization_step}, step=optimization_step, mode="train")


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




def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    logger_lock = Lock()

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
    with logger_lock:
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

    buffer_lock = Lock()
    replay_buffer = ReplayBuffer(
        capacity=cfg.training.online_buffer_capacity, device=device, state_keys=cfg.policy.input_shapes.keys()
    )

    batch_size = cfg.training.batch_size
    offline_buffer_lock = None
    offline_replay_buffer = None
    if cfg.dataset_repo_id is not None:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
        logging.info("Convertion to a offline replay buffer")
        offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
            offline_dataset, device=device, state_keys=cfg.policy.input_shapes.keys()
        )
        offline_buffer_lock = Lock()
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    server_thread = Thread(target=stream_transitions_from_actor, args=(50051,), daemon=True)
    server_thread.start()


    # Start a background thread to process transitions from the queue
    transition_thread = Thread(
        target=add_actor_information,
        daemon=True,
        args=(cfg,
              device,
              replay_buffer,
              offline_replay_buffer,
              batch_size,
              optimizers,
              policy,
              policy_lock,
              buffer_lock,
              offline_buffer_lock,
              logger_lock,
              logger),
    )
    transition_thread.start()

    param_push_thread = Thread(
        target=learner_push_parameters,
        args=(policy, policy_lock, "127.0.0.1", 50051, 15),
        # args=("127.0.0.1", 50052),
        daemon=True,
    )
    param_push_thread.start()

        # interaction_thread = Thread(
    #     target=add_message_interaction_to_wandb,
    #     daemon=True,
    #     args=(cfg, logger, logger_lock),
    # )
    # interaction_thread.start()

    transition_thread.join()
    # param_push_thread.join()
    server_thread.join()
    # interaction_thread.join()


@hydra.main(version_base="1.2", config_name="default", config_path="../../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    train_cli()
