#!/usr/bin/env python

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
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `lerobot-find-joint-limits`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time
from pprint import pformat

import torch
from torch import nn
from torch.multiprocessing import Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.policies.acfql.modeling_acfql import ACFQLPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.rl.actor import (
    establish_learner_connection,
    learner_service_client,
    log_policy_frequency_issue,
    push_transitions_to_transport_queue,
    receive_policy,
    send_interactions,
    send_transitions,
    use_threads,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport.utils import (
    bytes_to_state_dict,
    python_object_to_bytes,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .configs import ACFQLTrainRLServerPipelineConfig as TrainRLServerPipelineConfig
from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from .utils import get_frequency_stats

# Main entry point


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# Core algorithm functions


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info(pformat(cfg.to_dict()))
    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a ACFQLPolicy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    policy: ACFQLPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    policy = policy.eval()
    assert isinstance(policy, nn.Module)

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

    # get the initial policy parameters from the learner
    if cfg.policy.offline_steps > 0:
        logging.info("[ACTOR] Waiting for initial policy parameters from learner")
        update_policy_parameters(
            policy=policy, parameters_queue=parameters_queue, device=device, wait_for_update=True
        )

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0

    policy_timer = TimerManager("Policy inference", log=False)
    fps_tracker = TimerManager("Episode FPS", log=False)
    episode_started = True
    episode_start_time = time.perf_counter()

    for interaction_step in range(cfg.policy.online_steps):
        fps_tracker.start()
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        observation_for_inference = preprocessor(
            {
                **{"observation.state": observation["observation.state"]},
                # [B, C, H, W] -> [B, H, W, C]
                **{k: v.permute(0, 2, 3, 1) for k, v in observation.items() if "observation.images" in k},
            }
        )

        # The preprocessor may add extra keys, filter them out
        observation_for_inference = {
            k: v for k, v in observation_for_inference.items() if k in cfg.policy.input_features
        }

        observation_for_inference = {
            **{"observation.state": observation_for_inference["observation.state"]},
            # [B, H, W, C] -> [B, C, H, W]
            **{
                k: v.permute(0, 3, 1, 2)
                for k, v in observation_for_inference.items()
                if "observation.images" in k
            },
        }

        # Time policy inference and check if it meets FPS requirement
        with policy_timer:
            # Extract observation from transition for policy
            action = policy.select_action(batch=observation_for_inference)
        policy_fps = policy_timer.fps_last

        action = postprocessor(action)

        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

        # Use the new step function
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # Extract values from processed transition
        next_observation = {
            k: v
            for k, v in new_transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }

        # Teleop action is the action that was executed in the environment
        # It is either the action from the teleop device or the action from the policy
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        sum_reward_episode += float(reward)
        episode_total_steps += 1

        # Check for intervention from transition info
        intervention_info = new_transition[TransitionKey.INFO]
        if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
            episode_intervention = True
            episode_intervention_steps += 1
            # TODO: The policy state should be reset when an intervention occurs when chunking is enabled, to clear the action queue
            policy.reset()  # Reset policy state if needed

        complementary_info = {
            "discrete_penalty": torch.tensor(
                [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
            ),
        }
        # Create transition for learner (convert to old format)
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )

        # Update transition for next iteration
        transition = new_transition

        if done or truncated:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}, Episode time: {episode_time:.2f}s"
            )

            stats = get_frequency_stats(policy_timer)

            stats = get_frequency_stats(fps_tracker)
            logging.info(", ".join([f"{k} : {v:.2f}" for k, v in stats.items()]))

            if len(list_transition_to_send_to_learner) > 0:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                list_transition_to_send_to_learner = []

            # Calculate intervention rate
            intervention_rate = 0.0
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps

            # Send episodic reward to the learner
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Episodic reward": sum_reward_episode,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(episode_intervention),
                        "Intervention rate": intervention_rate,
                        **stats,
                    }
                )
            )

            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

            policy_timer.reset()
            fps_tracker.reset()
            episode_started = False

            # Reset intervention counters and environment
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_intervention_steps = 0
            episode_total_steps = 0

            # Reset environment and processors
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()

            policy.reset()  # Reset policy state if needed

            episode_start_time = time.perf_counter()

            # Process initial observation
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            busy_wait(1 / cfg.env.fps - dt_time)

        if not episode_started:
            # This is needed to track the fps correctly after reset
            episode_started = True
        else:
            fps_tracker.stop()


#  Policy functions


def update_policy_parameters(
    policy: ACFQLPolicy, parameters_queue: Queue, device, wait_for_update: bool = False
):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)

    while bytes_state_dict is None and wait_for_update:
        # logging.info("[ACTOR] Waiting for updated policy parameters from Learner...")
        bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
        time.sleep(2)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)

        # Load actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        policy.actor_onestep_flow.load_state_dict(actor_state_dict, strict=True)

        # # Load discrete critic if present
        # if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
        #     discrete_critic_state_dict = move_state_dict_to_device(
        #         state_dicts["discrete_critic"], device=device
        #     )
        #     policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
        #     logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")


if __name__ == "__main__":
    register_third_party_devices()
    actor_cli()
