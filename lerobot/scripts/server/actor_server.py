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
import logging
import functools
from pprint import pformat
import random
from typing import Optional, Sequence, TypedDict, Callable
import pickle

import hydra
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# TODO: Remove the import of maniskill
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.factory import make_env, make_maniskill_env
from lerobot.common.envs.utils import preprocess_observation, preprocess_maniskill_observation
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
# from lerobot.scripts.eval import eval_policy
from threading import Thread
import queue

import grpc
from lerobot.scripts.server import hilserl_pb2, hilserl_pb2_grpc
import io
import time
import logging
from concurrent import futures
from threading import Thread
from lerobot.scripts.server.buffer import move_state_dict_to_device, move_transition_to_device, Transition

import faulthandler
import signal

logging.basicConfig(level=logging.INFO)

parameters_queue = queue.Queue(maxsize=1)
message_queue = queue.Queue(maxsize=1_000_000)

class ActorInformation:
    def __init__(self, transition=None, interaction_message=None):
        self.transition = transition
        self.interaction_message = interaction_message


# 1) Implement ActorService so the Learner can send parameters to this Actor.
class ActorServiceServicer(hilserl_pb2_grpc.ActorServiceServicer):
    def StreamTransition(self, request, context):
        while True:
            # logging.info(f"[ACTOR] before message.empty()")
            # logging.info(f"[ACTOR] size transition queue {message_queue.qsize()}")
            # time.sleep(0.01)
            # if message_queue.empty():
            #     continue
            # logging.info(f"[ACTOR] after message.empty()")
            start = time.time()
            message = message_queue.get(block=True)
            # logging.info(f"[ACTOR] Message queue get time {time.time() - start}")

            if message.transition is not None:
                # transition_to_send_to_learner = move_transition_to_device(message.transition, device="cpu")
                transition_to_send_to_learner = [move_transition_to_device(T, device="cpu") for T in message.transition]
                # logging.info(f"[ACTOR] Message queue get time {time.time() - start}")

                # Serialize it
                buf = io.BytesIO()
                torch.save(transition_to_send_to_learner, buf)
                transition_bytes = buf.getvalue()
                
                transition_message = hilserl_pb2.Transition(
                    transition_bytes=transition_bytes
                )

                response = hilserl_pb2.ActorInformation(
                    transition=transition_message
                )
                logging.info(f"[ACTOR] time to yield transition response {time.time() - start}")
                logging.info(f"[ACTOR] size transition queue {message_queue.qsize()}")
                
            elif message.interaction_message is not None:
                # Serialize it and send it to the Learner's server
                content = hilserl_pb2.InteractionMessage(
                    interaction_message_bytes=pickle.dumps(message.interaction_message)
                    )
                response = hilserl_pb2.ActorInformation(
                    interaction_message=content
                )

            # logging.info(f"[ACTOR] yield response before")
            yield response
            # logging.info(f"[ACTOR] response yielded after")

    def SendParameters(self, request, context):
        """
        Learner calls this with updated Parameters -> Actor
        """
        # logging.info("[ACTOR] Received parameters from Learner.")
        buffer = io.BytesIO(request.parameter_bytes)
        params = torch.load(buffer)
        parameters_queue.put(params)
        return hilserl_pb2.Empty()


def serve_actor_service(port=50052):
    """
    Runs a gRPC server so that the Learner can push parameters to the Actor.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20),
                            options=[('grpc.max_send_message_length', -1),
                                     ('grpc.max_receive_message_length', -1)])
    hilserl_pb2_grpc.add_ActorServiceServicer_to_server(
        ActorServiceServicer(), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f"[ACTOR] gRPC server listening on port {port}")
    server.wait_for_termination()

def act_with_policy(cfg: DictConfig, 
                   out_dir: str | None = None, 
                   job_name: str | None = None):

    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    logging.info("make_env online")

    # online_env = make_env(cfg, n_envs=1)
    # TODO: Remove the import of maniskill and unifiy with make env
    online_env = make_maniskill_env(cfg, n_envs=1)
    if cfg.training.eval_freq > 0:
        logging.info("make_env eval")
        # eval_env = make_env(cfg, n_envs=1)
        # TODO: Remove the import of maniskill and unifiy with make env
        eval_env = make_maniskill_env(cfg, n_envs=1)

    set_global_seed(cfg.seed)
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")
    
    
    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy intance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    # TODO: At some point we should just need make sac policy
    policy: SACPolicy = make_policy(
        hydra_cfg=cfg,
        # dataset_stats=offline_dataset.meta.stats if not cfg.resume else None,
        # Hack: But if we do online traning, we do not need dataset_stats
        dataset_stats=None,
        # TODO: Handle resume training
        pretrained_policy_name_or_path=None,
        device=device,
    )
    assert isinstance(policy, nn.Module)

    # HACK for maniskill
    obs, info = online_env.reset()

    # obs = preprocess_observation(obs)
    obs = preprocess_maniskill_observation(obs)
    obs = {key: obs[key].to(device, non_blocking=True) for key in obs}
    ### ACTOR ==================
    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []

    for interaction_step in range(cfg.training.online_steps):
        # NOTE: At some point we should use a  wrapper to handle the observation

        # start = time.time()
        if interaction_step >= cfg.training.online_step_before_learning:
            action = policy.select_action(batch=obs)
            next_obs, reward, done, truncated, info = online_env.step(action.cpu().numpy())
        else:
            action = online_env.action_space.sample()
            next_obs, reward, done, truncated, info = online_env.step(action)
            # HACK
            action = torch.tensor(action, dtype=torch.float32).to(device, non_blocking=True)

        # logging.info(f"[ACTOR] Time for env step {time.time() - start}")

        # HACK: For maniskill
        # next_obs = preprocess_observation(next_obs)
        next_obs = preprocess_maniskill_observation(next_obs)
        next_obs = {key: next_obs[key].to(device, non_blocking=True) for key in obs}
        sum_reward_episode += float(reward[0])
        # Because we are using a single environment
        # we can safely assume that the episode is done
        if done[0].item() or truncated[0].item():
            # TODO: Handle logging for episode information
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            if not parameters_queue.empty():
                logging.info("[ACTOR] Load new parameters from Learner.")
                # Load new parameters from Learner
                state_dict = parameters_queue.get()
                state_dict = move_state_dict_to_device(state_dict, device=device)
                policy.actor.load_state_dict(state_dict)
            
            if len(list_transition_to_send_to_learner) > 0:
                logging.info(f"[ACTOR] Sending {len(list_transition_to_send_to_learner)} transitions to Learner.")
                message_queue.put(ActorInformation(transition=list_transition_to_send_to_learner))
                list_transition_to_send_to_learner = []

            # Send episodic reward to the learner
            message_queue.put(ActorInformation(interaction_message={"episodic_reward": sum_reward_episode,"interaction_step": interaction_step}))
            sum_reward_episode = 0.0

        # ============================
        # Prepare transition to send
        # ============================
        # Label the reward
        # if config.label_reward_on_actor:
        #     reward = reward_classifier(obs)

        list_transition_to_send_to_learner.append(Transition(
        # transition_to_send_to_learner = Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                    complementary_info=None,
                )
        )
        # message_queue.put(ActorInformation(transition=transition_to_send_to_learner))

        # assign obs to the next obs and continue the rollout
        obs = next_obs

@hydra.main(version_base="1.2", config_name="default", config_path="../../configs")
def actor_cli(cfg: dict):
        server_thread = Thread(target=serve_actor_service, args=(50051,), daemon=True)
        server_thread.start()
        policy_thread = Thread(target=act_with_policy, 
                               daemon=True, 
                               args=(cfg,hydra.core.hydra_config.HydraConfig.get().run.dir, hydra.core.hydra_config.HydraConfig.get().job.name))
        policy_thread.start()
        policy_thread.join()
        server_thread.join()

if __name__ == "__main__":
    with open("traceback.log", "w") as f:
        faulthandler.register(signal.SIGUSR1, file=f)

    actor_cli()