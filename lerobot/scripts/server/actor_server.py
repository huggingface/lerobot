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
# from lerobot.common.envs.factory import make_maniskill_env
# from lerobot.common.envs.utils import preprocess_maniskill_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    set_global_seed,
)
from lerobot.scripts.server import hilserl_pb2, hilserl_pb2_grpc
from lerobot.scripts.server.buffer import Transition, move_state_dict_to_device, move_transition_to_device
from lerobot.scripts.server.gym_manipulator import get_classifier, make_robot_env

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


def update_policy_parameters(policy: SACPolicy, parameters_queue: queue.Queue, device):
    if not parameters_queue.empty():
        logging.debug("[ACTOR] Load new parameters from Learner.")
        state_dict = parameters_queue.get()
        state_dict = move_state_dict_to_device(state_dict, device=device)
        policy.load_state_dict(state_dict)


def act_with_policy(cfg: DictConfig, robot: Robot, reward_classifier: nn.Module):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg (DictConfig): Configuration settings for the interaction process.
    """

    logging.info("make_env online")

    online_env = make_robot_env(robot=robot, reward_classifier=reward_classifier, cfg=cfg.env)

    set_global_seed(cfg.seed)
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    # HACK: This is an ugly hack to pass the normalization parameters to the policy
    # Because the action space is dynamic so we override the output normalization parameters
    # it's ugly, we know ... and we will fix it
    min_action_space: list = online_env.action_space.spaces[0].low.tolist()
    max_action_space: list = online_env.action_space.spaces[0].high.tolist()
    output_normalization_params: dict[dict[str, list]] = {
        "action": {"min": min_action_space, "max": max_action_space}
    }
    cfg.policy.output_normalization_params = output_normalization_params

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
        device=device,
    )
    policy = torch.compile(policy)
    assert isinstance(policy, nn.Module)

    obs, info = online_env.reset()

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    list_policy_time = []

    for interaction_step in range(cfg.training.online_steps):
        if interaction_step >= cfg.training.online_step_before_learning:
            # Time policy inference and check if it meets FPS requirement
            with TimerManager(
                elapsed_time_list=list_policy_time, label="Policy inference time", log=False
            ) as timer:  # noqa: F841
                action = policy.select_action(batch=obs) * 0.0
            policy_fps = 1.0 / (list_policy_time[-1] + 1e-9)

            log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

            next_obs, reward, done, truncated, info = online_env.step(action.squeeze(dim=0).cpu().numpy())
        else:
            # TODO (azouitine): Make a custom space for torch tensor
            action = online_env.action_space.sample()
            next_obs, reward, done, truncated, info = online_env.step(action)

            # HACK: We have only one env but we want to batch it, it will be resolved with the torch box
            action = torch.from_numpy(action[0]).to(device, non_blocking=True).unsqueeze(dim=0)

        sum_reward_episode += float(reward)

        # NOTE: We overide the action if the intervention is True, because the action applied is the intervention action
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

        # HACK: We have only one env but we want to batch it, it will be resolved with the torch box
        # Because we are using a single environment we can index at zero
        if done or truncated:
            # TODO: Handle logging for episode information
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            # update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

            if len(list_transition_to_send_to_learner) > 0:
                send_transitions_in_chunks(
                    transitions=list_transition_to_send_to_learner, message_queue=message_queue, chunk_size=4
                )
                list_transition_to_send_to_learner = []

            stats = get_frequency_stats(list_policy_time)
            list_policy_time.clear()

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
            obs, info = online_env.reset()


def send_transitions_in_chunks(transitions: list, message_queue, chunk_size: int = 100):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    for i in range(0, len(transitions), chunk_size):
        chunk = transitions[i : i + chunk_size]
        logging.debug(f"[ACTOR] Sending chunk of {len(chunk)} transitions to Learner.")
        message_queue.put(ActorInformation(transition=chunk))


def get_frequency_stats(list_policy_time: list[float]) -> dict[str, float]:
    stats = {}
    list_policy_fps = [1.0 / t for t in list_policy_time]
    if len(list_policy_fps) > 0:
        policy_fps = mean(list_policy_fps)
        quantiles_90 = quantiles(list_policy_fps, n=10)[-1]
        logging.debug(f"[ACTOR] Average policy frame rate: {policy_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {quantiles_90}")
        stats = {"Policy frequency [Hz]": policy_fps, "Policy frequency 90th-p [Hz]": quantiles_90}
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: DictConfig, interaction_step: int):
    if policy_fps < cfg.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.fps} at step {interaction_step}"
        )


@hydra.main(version_base="1.2", config_name="default", config_path="../../configs")
def actor_cli(cfg: dict):
    robot = make_robot(cfg=cfg.robot)

    server_thread = Thread(target=serve_actor_service, args=(cfg.actor_learner_config.port,), daemon=True)
    reward_classifier = get_classifier(
        pretrained_path=cfg.env.reward_classifier.pretrained_path,
        config_path=cfg.env.reward_classifier.config_path,
    )
    policy_thread = Thread(
        target=act_with_policy,
        daemon=True,
        args=(cfg, robot, reward_classifier),
    )
    server_thread.start()
    policy_thread.start()
    policy_thread.join()
    server_thread.join()


if __name__ == "__main__":
    actor_cli()
