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
"""Evaluate a policy by running rollouts on the real robot and computing metrics.

Usage examples: evaluate a checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval_on_robot.py \
    -p outputs/train/model/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

**NOTE** (michel-aractingi): This script is incomplete and it is being prepared
for running training on the real robot.
"""

import argparse
import logging
import time
from copy import deepcopy
from queue import Queue
from threading import Thread

import torch

from lerobot.common.envs.factory import (
    make_env,
)
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.robot_devices.control_utils import busy_wait, is_headless
from lerobot.common.robot_devices.robots.factory import Robot, make_robot
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    log_say,
    set_global_seed,
)

REPLAY_BUFFER_START_SIZE = 1000


def rollout(robot: Robot, policy: Policy, fps: int, control_time_s: float = 20, use_amp: bool = True) -> dict:
    """Run a batched policy rollout on the real robot.

    The return dictionary contains:
        "robot": A a dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE the that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        robot: The robot class that defines the interface with the real robot.
        policy: The policy. Must be a PyTorch nn module.

    Returns:
        The dictionary described above.
    """
    # assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    # device = get_device_from_parameters(policy)

    # define keyboard listener
    listener, events = init_keyboard_listener()

    # Reset the policy. TODO (michel-aractingi) add real policy evaluation once the code is ready.
    # policy.reset()

    # Get observation from real robot
    observation = robot.capture_observation()

    # Calculate reward. TODO (michel-aractingi)
    # in HIL-SERL it will be with a reward classifier
    reward = calculate_reward(observation)
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []

    start_episode_t = time.perf_counter()
    timestamp = 0.0
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        all_observations.append(deepcopy(observation))
        # observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

        # Apply the next action.
        while events["pause_policy"] and not events["human_intervention_step"]:
            busy_wait(0.5)

        if events["human_intervention_step"]:
            # take over the robot's actions
            observation, action = robot.teleop_step(record_data=True)
            action = action["action"]  # teleop step returns torch tensors but in a dict
        else:
            # explore with policy
            with torch.inference_mode():
                action = robot.follower_arms["main"].read("Present_Position")
                action = torch.from_numpy(action)
                robot.send_action(action)
                # action = predict_action(observation, policy, device, use_amp)

        observation = robot.capture_observation()
        # Calculate reward
        # in HIL-SERL it will be with a reward classifier
        reward = calculate_reward(observation)

        all_actions.append(action)
        all_rewards.append(torch.from_numpy(reward))
        all_successes.append(torch.tensor([False]))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            events["human_intervention_step"] = False
            events["pause_policy"] = False
            break
    all_observations.append(deepcopy(observation))

    dones = torch.tensor([False] * len(all_actions))
    dones[-1] = True
    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "next.reward": torch.stack(all_rewards, dim=1),
        "next.success": torch.stack(all_successes, dim=1),
        "done": dones,
    }
    stacked_observations = {}
    for key in all_observations[0]:
        stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
    ret["observation"] = stacked_observations

    listener.stop()

    return ret


def start_learner(replay_buffer, stop_event):
    """Thread that handles policy learning and updates"""

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=REPLAY_BUFFER_START_SIZE,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )

    while len(replay_buffer) < REPLAY_BUFFER_START_SIZE and not stop_event.is_set():
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    train_critic_networks_to_update = frozenset({"critic"})
    train_networks_to_update = frozenset({"critic", "actor", "temperature"})

    for step in tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


def process_actor():
    """Thread that handles interaction with robot and data collection"""
    # while not stop_event.is_set():
    # Collect data from robot interactions
    # Add your data collection logic here
    # data = {"observations": [], "actions": [], "rewards": []}  # Your collected data
    # queue.put(data)
    logging.info("Actor: Collecting data")

    # if FLAGS.eval_checkpoint_step:
    #     success_counter = 0
    #     time_list = []

    #     ckpt = checkpoints.restore_checkpoint(
    #         os.path.abspath(FLAGS.checkpoint_path),
    #         agent.state,
    #         step=FLAGS.eval_checkpoint_step,
    #     )
    #     agent = agent.replace(state=ckpt)

    #     for episode in range(FLAGS.eval_n_trajs):
    #         obs, _ = env.reset()
    #         done = False
    #         start_time = time.time()
    #         while not done:
    #             sampling_rng, key = jax.random.split(sampling_rng)
    #             actions = agent.sample_actions(
    #                 observations=jax.device_put(obs),
    #                 argmax=False,
    #                 seed=key
    #             )
    #             actions = np.asarray(jax.device_get(actions))

    #             next_obs, reward, done, truncated, info = env.step(actions)
    #             obs = next_obs

    #             if done:
    #                 if reward:
    #                     dt = time.time() - start_time
    #                     time_list.append(dt)
    #                     print(dt)

    #                 success_counter += reward
    #                 print(reward)
    #                 print(f"{success_counter}/{episode + 1}")

    #     print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    #     print(f"average time: {np.mean(time_list)}")
    #     return  # after done eval, return and exit

    # start_step = (
    #     int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
    #     if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
    #     else 0
    # )

    # datastore_dict = {
    #     "actor_env": data_store,
    #     "actor_env_intvn": intvn_data_store,
    # }

    # client = TrainerClient(
    #     "actor_env",
    #     FLAGS.ip,
    #     make_trainer_config(),
    #     data_stores=datastore_dict,
    #     wait_for_server=True,
    #     timeout_ms=3000,
    # )

    # # Function to update the agent with new params
    # def update_params(params):
    #     nonlocal agent
    #     agent = agent.replace(state=agent.state.replace(params=params))

    # client.recv_network_callback(update_params)

    # transitions = []
    # demo_transitions = []

    # obs, _ = env.reset()
    # done = False

    # # training loop
    # timer = Timer()
    # running_return = 0.0
    # already_intervened = False
    # intervention_count = 0
    # intervention_steps = 0

    # pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    # for step in pbar:
    #     timer.tick("total")

    #     with timer.context("sample_actions"):
    #         if step < config.random_steps:
    #             actions = env.action_space.sample()
    #         else:
    #             sampling_rng, key = jax.random.split(sampling_rng)
    #             actions = agent.sample_actions(
    #                 observations=jax.device_put(obs),
    #                 seed=key,
    #                 argmax=False,
    #             )
    #             actions = np.asarray(jax.device_get(actions))

    #     # Step environment
    #     with timer.context("step_env"):

    #         next_obs, reward, done, truncated, info = env.step(actions)
    #         if "left" in info:
    #             info.pop("left")
    #         if "right" in info:
    #             info.pop("right")

    #         # override the action with the intervention action
    #         if "intervene_action" in info:
    #             actions = info.pop("intervene_action")
    #             intervention_steps += 1
    #             if not already_intervened:
    #                 intervention_count += 1
    #             already_intervened = True
    #         else:
    #             already_intervened = False

    #         running_return += reward
    #         transition = dict(
    #             observations=obs,
    #             actions=actions,
    #             next_observations=next_obs,
    #             rewards=reward,
    #             masks=1.0 - done,
    #             dones=done,
    #         )
    #         if 'grasp_penalty' in info:
    #             transition['grasp_penalty']= info['grasp_penalty']
    #         data_store.insert(transition)
    #         transitions.append(copy.deepcopy(transition))
    #         if already_intervened:
    #             intvn_data_store.insert(transition)
    #             demo_transitions.append(copy.deepcopy(transition))

    #         obs = next_obs
    #         if done or truncated:
    #             info["episode"]["intervention_count"] = intervention_count
    #             info["episode"]["intervention_steps"] = intervention_steps
    #             stats = {"environment": info}  # send stats to the learner to log
    #             client.request("send-stats", stats)
    #             pbar.set_description(f"last return: {running_return}")
    #             running_return = 0.0
    #             intervention_count = 0
    #             intervention_steps = 0
    #             already_intervened = False
    #             client.update()
    #             obs, _ = env.reset()

    #     if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
    #         # dump to pickle file
    #         buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
    #         demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
    #         if not os.path.exists(buffer_path):
    #             os.makedirs(buffer_path)
    #         if not os.path.exists(demo_buffer_path):
    #             os.makedirs(demo_buffer_path)
    #         with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
    #             pkl.dump(transitions, f)
    #             transitions = []
    #         with open(
    #             os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
    #         ) as f:
    #             pkl.dump(demo_transitions, f)
    #             demo_transitions = []

    #     timer.tock("total")

    #     if step % config.log_period == 0:
    #         stats = {"timer": timer.get_average_times()}
    #         client.request("send-stats", stats)


def eval_policy(
    robot: Robot,
    policy: torch.nn.Module,
    reward_classifier: torch.nn.Module,
    fps: float,
    n_episodes: int,
    control_time_s: int = 20,
    use_amp: bool = True,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    # TODO (michel-aractingi) comment this out for testing with a fixed policy
    # assert isinstance(policy, Policy)
    # policy.eval()

    learner_thread = Thread(target=start_learner, args=(queue, stop_event))
    actor_thread = Thread(target=start_actor, args=(queue, stop_event))

    from threading import Event

    # Create communication queue
    queue = Queue()
    stop_event = Event()

    # Create threads
    learner_thread = Thread(target=start_learner, args=(queue, stop_event))

    # Start threads
    learner_thread.start()

    try:
        # Let threads run until interrupted
        while True:
            time.sleep(0.1)
            process_actor()
    except KeyboardInterrupt:
        print("\nStopping threads...")
        stop_event.set()

    # Wait for threads to complete
    learner_thread.join()
    print("Threads stopped successfully")

    # sum_rewards = []
    # max_rewards = []
    # successes = []
    # rollouts = []

    # start_eval = time.perf_counter()
    # progbar = trange(n_episodes, desc="Evaluating policy on real robot")
    # for _batch_idx in progbar:
    #     rollout_data = rollout(robot, policy, fps, control_time_s, use_amp)

    #     rollouts.append(rollout_data)
    #     sum_rewards.append(sum(rollout_data["next.reward"]))
    #     max_rewards.append(max(rollout_data["next.reward"]))
    #     successes.append(rollout_data["next.success"][-1])

    # info = {
    #     "per_episode": [
    #         {
    #             "episode_ix": i,
    #             "sum_reward": sum_reward,
    #             "max_reward": max_reward,
    #             "pc_success": success * 100,
    #         }
    #         for i, (sum_reward, max_reward, success) in enumerate(
    #             zip(
    #                 sum_rewards[:n_episodes],
    #                 max_rewards[:n_episodes],
    #                 successes[:n_episodes],
    #                 strict=False,
    #             )
    #         )
    #     ],
    #     "aggregated": {
    #         "avg_sum_reward": float(np.nanmean(torch.cat(sum_rewards[:n_episodes]))),
    #         "avg_max_reward": float(np.nanmean(torch.cat(max_rewards[:n_episodes]))),
    #         "pc_success": float(np.nanmean(torch.cat(successes[:n_episodes])) * 100),
    #         "eval_s": time.time() - start_eval,
    #         "eval_ep_s": (time.time() - start_eval) / n_episodes,
    #     },
    # }

    if robot.is_connected:
        robot.disconnect()

    return info


def calculate_reward(observation):
    """
    Method to calculate reward function in some way.
    In HIL-SERL this is done through defining a reward classifier
    """
    # reward = reward_classifier(observation)
    return np.array([0.0])


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["pause_policy"] = False
    events["human_intervention_step"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                # check if first space press then pause the policy for the user to get ready
                # if second space press then the user is ready to start intervention
                if not events["pause_policy"]:
                    print(
                        "Space key pressed. Human intervention required.\n"
                        "Place the leader in similar pose to the follower and press space again."
                    )
                    events["pause_policy"] = True
                    log_say("Human intervention stage. Get ready to take over.", play_sounds=True)
                else:
                    events["human_intervention_step"] = True
                    print("Space key pressed. Human intervention starting.")
                    log_say("Starting human intervention.", play_sounds=True)

        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    group.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        default="lerobot/configs/hil-serl.yaml",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )

    group.add_argument(
        "--pretrained-reward-classifier-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )

    args = parser.parse_args()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)
    # if not robot.is_connected:
    #     robot.connect()

    cfg = init_hydra_config(args.config) if args.config else None

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(cfg.seed)

    # log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(cfg)

    logging.info("Making policy.")

    # Create main policy - SAC or Dagger
    policy = make_policy(cfg, env)

    reward_classifier = load_reward_classifier_model(cfg, args.pretrained_reward_classifier_or_path)

    demo_buffer = []
    replay_buffer = []

    eval_policy(robot, policy, reward_classifier, fps=40, n_episodes=2, control_time_s=100)
