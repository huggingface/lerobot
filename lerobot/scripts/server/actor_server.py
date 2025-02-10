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
import io
import logging
import pickle
import queue
import time
from concurrent import futures
from statistics import mean, quantiles

# from lerobot.scripts.eval import eval_policy
from threading import Thread

import grpc
import hydra
import torch
from omegaconf import DictConfig
from torch import nn

# TODO: Remove the import of maniskill
from lerobot.common.envs.factory import make_maniskill_env
from lerobot.common.envs.utils import preprocess_maniskill_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    set_global_seed,
)
from lerobot.scripts.server import hilserl_pb2, hilserl_pb2_grpc
from lerobot.scripts.server.buffer import Transition, move_state_dict_to_device, move_transition_to_device

logging.basicConfig(level=logging.INFO)

parameters_queue = queue.Queue(maxsize=1)
message_queue = queue.Queue(maxsize=1_000_000)


class ActorInformation:
    """
    This helper class is used to differentiate between two types of messages that are placed in the same queue during streaming:

    - **Transition Data:** Contains experience tuples (observation, action, reward, next observation) collected during interaction.
    - **Interaction Messages:** Encapsulates statistics related to the interaction process.

    Attributes:
        transition (Optional): Transition data to be sent to the learner.
        interaction_message (Optional): Iteraction message providing additional statistics for logging.
    """

    def __init__(self, transition=None, interaction_message=None):
        self.transition = transition
        self.interaction_message = interaction_message


class ActorServiceServicer(hilserl_pb2_grpc.ActorServiceServicer):
    """
    gRPC service for actor-learner communication in reinforcement learning.

    This service is responsible for:
    1. Streaming batches of transition data and statistical metrics from the actor to the learner.
    2. Receiving updated network parameters from the learner.
    """

    def StreamTransition(self, request, context):  # noqa: N802
        """
        Streams data from the actor to the learner.

        This function continuously retrieves messages from the queue and processes them based on their type:

        - **Transition Data:**
          - A batch of transitions (observation, action, reward, next observation) is collected.
          - Transitions are moved to the CPU and serialized using PyTorch.
          - The serialized data is wrapped in a `hilserl_pb2.Transition` message and sent to the learner.

        - **Interaction Messages:**
          - Contains useful statistics about episodic rewards and policy timings.
          - The message is serialized using `pickle` and sent to the learner.

        Yields:
            hilserl_pb2.ActorInformation: The response message containing either transition data or an interaction message.
        """
        while True:
            message = message_queue.get(block=True)

            if message.transition is not None:
                transition_to_send_to_learner = [
                    move_transition_to_device(T, device="cpu") for T in message.transition
                ]

                buf = io.BytesIO()
                torch.save(transition_to_send_to_learner, buf)
                transition_bytes = buf.getvalue()

                transition_message = hilserl_pb2.Transition(transition_bytes=transition_bytes)

                response = hilserl_pb2.ActorInformation(transition=transition_message)

            elif message.interaction_message is not None:
                content = hilserl_pb2.InteractionMessage(
                    interaction_message_bytes=pickle.dumps(message.interaction_message)
                )
                response = hilserl_pb2.ActorInformation(interaction_message=content)

            yield response

    def SendParameters(self, request, context):  # noqa: N802
        """
        Receives updated parameters from the learner and updates the actor.

        The learner calls this method to send new model parameters. The received parameters are deserialized
        and placed in a queue to be consumed by the actor.

        Args:
            request (hilserl_pb2.ParameterUpdate): The request containing serialized network parameters.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            hilserl_pb2.Empty: An empty response to acknowledge receipt.
        """
        buffer = io.BytesIO(request.parameter_bytes)
        params = torch.load(buffer)
        parameters_queue.put(params)
        return hilserl_pb2.Empty()


def serve_actor_service(port=50052):
    """
    Runs a gRPC server to start streaming the data from the actor to the learner.
     Throught this server the learner can push parameters to the Actor as well.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=20),
        options=[("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)],
    )
    hilserl_pb2_grpc.add_ActorServiceServicer_to_server(ActorServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logging.info(f"[ACTOR] gRPC server listening on port {port}")
    server.wait_for_termination()


def act_with_policy(cfg: DictConfig):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg (DictConfig): Configuration settings for the interaction process.
    """

    logging.info("make_env online")

    # online_env = make_env(cfg, n_envs=1)
    # TODO: Remove the import of maniskill and unifiy with make env
    online_env = make_maniskill_env(cfg, n_envs=1)

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
        # Hack: But if we do online training, we do not need dataset_stats
        dataset_stats=None,
        # TODO: Handle resume training
    )
    #     pretrained_policy_name_or_path=None,
    #     device=device,
    # )
    policy = torch.compile(policy)
    assert isinstance(policy, nn.Module)

    # HACK for maniskill
    obs, info = online_env.reset()

    # obs = preprocess_observation(obs)
    obs = preprocess_maniskill_observation(obs)
    obs = {key: obs[key].to(device, non_blocking=True) for key in obs}

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    list_policy_fps = []

    for interaction_step in range(cfg.training.online_steps):
        if interaction_step >= cfg.training.online_step_before_learning:
            start = time.perf_counter()
            action = policy.select_action(batch=obs)
            list_policy_fps.append(1.0 / (time.perf_counter() - start + 1e-9))
            if list_policy_fps[-1] < cfg.fps:
                logging.warning(
                    f"[ACTOR] policy frame rate {list_policy_fps[-1]} during interaction step {interaction_step} is below the required control frame rate {cfg.fps}"
                )

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

        # Because we are using a single environment we can index at zero
        if done[0].item() or truncated[0].item():
            # TODO: Handle logging for episode information
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            if not parameters_queue.empty():
                logging.debug("[ACTOR] Load new parameters from Learner.")
                state_dict = parameters_queue.get()
                state_dict = move_state_dict_to_device(state_dict, device=device)
                # strict=False for the case when the image encoder is frozen and not sent through
                # the network. Becareful might cause issues if the wrong keys are passed
                policy.actor.load_state_dict(state_dict, strict=False)

            if len(list_transition_to_send_to_learner) > 0:
                logging.debug(
                    f"[ACTOR] Sending {len(list_transition_to_send_to_learner)} transitions to Learner."
                )
                message_queue.put(ActorInformation(transition=list_transition_to_send_to_learner))
                list_transition_to_send_to_learner = []

            stats = {}
            if len(list_policy_fps) > 0:
                policy_fps = mean(list_policy_fps)
                quantiles_90 = quantiles(list_policy_fps, n=10)[-1]
                logging.debug(f"[ACTOR] Average policy frame rate: {policy_fps}")
                logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {quantiles_90}")
                stats = {"Policy frequency [Hz]": policy_fps, "Policy frequency 90th-p [Hz]": quantiles_90}
                list_policy_fps = []

            # Send episodic reward to the learner
            message_queue.put(
                ActorInformation(
                    interaction_message={
                        "Episodic reward": sum_reward_episode,
                        "Interaction step": interaction_step,
                        **stats,
                    }
                )
            )
            sum_reward_episode = 0.0

        # TODO (michel-aractingi): Label the reward
        # if config.label_reward_on_actor:
        #     reward = reward_classifier(obs)
        if info["is_intervention"]:
            # TODO: Check the shape
            action = info["action_intervention"]

        list_transition_to_send_to_learner.append(
            Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                complementary_info=info,  # TODO Handle information for the transition, is_demonstraction: bool
            )
        )

        # assign obs to the next obs and continue the rollout
        obs = next_obs


@hydra.main(version_base="1.2", config_name="default", config_path="../../configs")
def actor_cli(cfg: dict):
    port = cfg.actor_learner_config.port
    server_thread = Thread(target=serve_actor_service, args=(port,), daemon=True)
    server_thread.start()
    policy_thread = Thread(
        target=act_with_policy,
        daemon=True,
        args=(cfg,),
    )
    policy_thread.start()
    policy_thread.join()
    server_thread.join()


if __name__ == "__main__":
    actor_cli()
