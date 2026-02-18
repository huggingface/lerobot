import multiprocessing as mp
import signal
from pathlib import Path
from queue import Empty, Full

import torch
import torch.optim as optim

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.robots.so_follower import SO100FollowerConfig
from lerobot.teleoperators.so_leader import SO100LeaderConfig
from lerobot.teleoperators.utils import TeleopEvents

LOG_EVERY = 10
SEND_EVERY = 10
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20


def run_learner(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_learner: SACPolicy,
    online_buffer: ReplayBuffer,
    offline_buffer: ReplayBuffer,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: torch.device = "mps",
):
    """The learner process - trains SAC policy on transitions streamed from the actor, updating parameters
    for the actor to adopt."""
    policy_learner.train()
    policy_learner.to(device)

    # Create Adam optimizer from scratch - simple and clean
    optimizer = optim.Adam(policy_learner.parameters(), lr=lr)

    print(f"[LEARNER] Online buffer capacity: {online_buffer.capacity}")
    print(f"[LEARNER] Offline buffer capacity: {offline_buffer.capacity}")

    training_step = 0

    while not shutdown_event.is_set():
        # retrieve incoming transitions from the actor process
        try:
            transitions = transitions_queue.get(timeout=0.1)
            for transition in transitions:
                # HIL-SERL: Add ALL transitions to online buffer
                online_buffer.add(**transition)

                # HIL-SERL: Add ONLY human intervention transitions to offline buffer
                is_intervention = transition.get("complementary_info", {}).get("is_intervention", False)
                if is_intervention:
                    offline_buffer.add(**transition)
                    print(
                        f"[LEARNER] Human intervention detected! Added to offline buffer (now {len(offline_buffer)} transitions)"
                    )

        except Empty:
            pass  # No transitions available, continue

        # Train if we have enough data
        if len(online_buffer) >= policy_learner.config.online_step_before_learning:
            # Sample from online buffer (autonomous + human data)
            online_batch = online_buffer.sample(batch_size // 2)

            # Sample from offline buffer (human demonstrations only, either precollected or at runtime)
            offline_batch = offline_buffer.sample(batch_size // 2)

            # Combine batches - this is the key HIL-SERL mechanism!
            batch = {}
            for key in online_batch:
                if key in offline_batch:
                    batch[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
                else:
                    batch[key] = online_batch[key]

            loss, _ = policy_learner.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_step += 1

            if training_step % LOG_EVERY == 0:
                print(
                    f"[LEARNER] Training step {training_step}, Loss: {loss.item():.4f}, "
                    f"Buffers: Online={len(online_buffer)}, Offline={len(offline_buffer)}"
                )

            # Send updated parameters to actor every 10 training steps
            if training_step % SEND_EVERY == 0:
                try:
                    state_dict = {k: v.cpu() for k, v in policy_learner.state_dict().items()}
                    parameters_queue.put_nowait(state_dict)
                    print("[LEARNER] Sent updated parameters to actor")
                except Full:
                    # Missing write due to queue not being consumed (should happen rarely)
                    pass

    print("[LEARNER] Learner process finished")


def run_actor(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    policy_actor: SACPolicy,
    reward_classifier: Classifier,
    env_cfg: HILSerlRobotEnvConfig,
    device: torch.device = "mps",
    output_directory: Path | None = None,
):
    """The actor process - interacts with environment and collects data.
    The policy is frozen and only the parameters are updated, popping the most recent ones from a queue."""
    policy_actor.eval()
    policy_actor.to(device)

    reward_classifier.eval()
    reward_classifier.to(device)

    # Create robot environment inside the actor process
    env, teleop_device = make_robot_env(env_cfg)

    try:
        for episode in range(MAX_EPISODES):
            if shutdown_event.is_set():
                break

            obs, _info = env.reset()
            episode_reward = 0.0
            step = 0
            episode_transitions = []

            print(f"[ACTOR] Starting episode {episode + 1}")

            while step < MAX_STEPS_PER_EPISODE and not shutdown_event.is_set():
                try:
                    new_params = parameters_queue.get_nowait()
                    policy_actor.load_state_dict(new_params)
                    print("[ACTOR] Updated policy parameters from learner")
                except Empty:  # No new updated parameters available from learner, waiting
                    pass

                # Get action from policy
                policy_obs = make_policy_obs(obs, device=device)
                action_tensor = policy_actor.select_action(policy_obs)  # predicts a single action
                action = action_tensor.squeeze(0).cpu().numpy()

                # Step environment
                next_obs, _env_reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated

                # Predict reward
                policy_next_obs = make_policy_obs(next_obs, device=device)
                reward = reward_classifier.predict_reward(policy_next_obs)

                if reward >= 1.0 and not done:  # success detected! halt episode
                    terminated = True
                    done = True

                # In HIL-SERL, human interventions come from the teleop device
                is_intervention = False
                if hasattr(teleop_device, "get_teleop_events"):
                    # Real intervention detection from teleop device
                    teleop_events = teleop_device.get_teleop_events()
                    is_intervention = teleop_events.get(TeleopEvents.IS_INTERVENTION, False)

                # Store transition with intervention metadata
                transition = {
                    "state": policy_obs,
                    "action": action,
                    "reward": float(reward) if hasattr(reward, "item") else reward,
                    "next_state": policy_next_obs,
                    "done": done,
                    "truncated": truncated,
                    "complementary_info": {
                        "is_intervention": is_intervention,
                    },
                }

                episode_transitions.append(transition)

                episode_reward += reward
                step += 1

                obs = next_obs

                if done:
                    break

            # Send episode transitions to learner
            transitions_queue.put_nowait(episode_transitions)

    except KeyboardInterrupt:
        print("[ACTOR] Interrupted by user")
    finally:
        # Clean up
        if hasattr(env, "robot") and env.robot.is_connected:
            env.robot.disconnect()
        if teleop_device and hasattr(teleop_device, "disconnect"):
            teleop_device.disconnect()
        if output_directory is not None:
            policy_actor.save_pretrained(output_directory)
            print(f"[ACTOR] Latest actor policy saved at: {output_directory}")

        print("[ACTOR] Actor process finished")


def make_policy_obs(obs, device: torch.device = "cpu"):
    return {
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device),
        **{
            f"observation.image.{k}": torch.from_numpy(obs["pixels"][k]).float().unsqueeze(0).to(device)
            for k in obs["pixels"]
        },
    }


def main():
    """Main function - coordinates actor and learner processes."""

    device = "mps"  # or "cuda" or "cpu"
    output_directory = Path("outputs/robot_learning_tutorial/hil_serl")
    output_directory.mkdir(parents=True, exist_ok=True)

    # find ports using lerobot-find-port
    follower_port = ...
    leader_port = ...

    # the robot ids are used the load the right calibration files
    follower_id = ...
    leader_id = ...

    # A pretrained model (to be used in-distribution!)
    reward_classifier_id = "<user>/reward_classifier_hil_serl_example"
    reward_classifier = Classifier.from_pretrained(reward_classifier_id)

    reward_classifier.to(device)
    reward_classifier.eval()

    # Robot and environment configuration
    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id)
    teleop_cfg = SO100LeaderConfig(port=leader_port, id=leader_id)
    processor_cfg = HILSerlProcessorConfig(control_mode="leader")

    env_cfg = HILSerlRobotEnvConfig(robot=robot_cfg, teleop=teleop_cfg, processor=processor_cfg)

    # Create robot environment
    env, teleop_device = make_robot_env(env_cfg)

    obs_features = hw_to_dataset_features(env.robot.observation_features, "observation")
    action_features = hw_to_dataset_features(env.robot.action_features, "action")

    # Create SAC policy for action selection
    policy_cfg = SACConfig(
        device=device,
        input_features=obs_features,
        output_features=action_features,
    )

    policy_actor = SACPolicy(policy_cfg)
    policy_learner = SACPolicy(policy_cfg)

    demonstrations_repo_id = "lerobot/example_hil_serl_dataset"
    offline_dataset = LeRobotDataset(repo_id=demonstrations_repo_id)

    # Online buffer: initialized from scratch
    online_replay_buffer = ReplayBuffer(device=device, state_keys=list(obs_features.keys()))
    # Offline buffer: Created from dataset (pre-populated it with demonstrations)
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=offline_dataset, device=device, state_keys=list(obs_features.keys())
    )

    # Create communication channels between learner and actor processes
    transitions_queue = mp.Queue(maxsize=10)
    parameters_queue = mp.Queue(maxsize=2)
    shutdown_event = mp.Event()

    # Signal handler for graceful shutdown
    def signal_handler(sig):
        print(f"\nSignal {sig} received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create processes
    learner_process = mp.Process(
        target=run_learner,
        args=(
            transitions_queue,
            parameters_queue,
            shutdown_event,
            policy_learner,
            online_replay_buffer,
            offline_replay_buffer,
        ),
        kwargs={"device": device},  # can run on accelerated hardware for training
    )

    actor_process = mp.Process(
        target=run_actor,
        args=(
            transitions_queue,
            parameters_queue,
            shutdown_event,
            policy_actor,
            reward_classifier,
            env_cfg,
            output_directory,
        ),
        kwargs={"device": "cpu"},  # actor is frozen, can run on CPU or accelerate for inference
    )

    learner_process.start()
    actor_process.start()

    try:
        # Wait for actor to finish (it controls the episode loop)
        actor_process.join()
        shutdown_event.set()
        learner_process.join(timeout=10)

    except KeyboardInterrupt:
        print("Main process interrupted")
        shutdown_event.set()
        actor_process.join(timeout=5)
        learner_process.join(timeout=10)

    finally:
        if learner_process.is_alive():
            learner_process.terminate()
        if actor_process.is_alive():
            actor_process.terminate()


if __name__ == "__main__":
    main()
