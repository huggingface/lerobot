import logging
import multiprocessing as mp
import signal
import time
from collections import deque

import torch
import torch.optim as optim

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators.keyboard import KeyboardTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging
from tests.mocks.mock_robot import MockRobotConfig


def initialize_replay_buffer_with_offline_data(
    capacity: int,
    device: str,
    state_keys: list[str],
    offline_dataset_repo_id: str = None,
    max_offline_episodes: int = 5,
) -> ReplayBuffer:
    """
    Initialize replay buffer, optionally pre-populated with offline data like in HIL-SERL.

    Args:
        capacity: Replay buffer capacity
        device: Device to store tensors
        state_keys: Keys for state observations
        offline_dataset_repo_id: Optional dataset repo_id to pre-populate buffer
        max_offline_episodes: Maximum number of offline episodes to load

    Returns:
        ReplayBuffer: Initialized (possibly pre-populated) replay buffer
    """
    # Create empty replay buffer
    replay_buffer = ReplayBuffer(
        capacity=capacity,
        device=device,
        state_keys=state_keys,
        storage_device="cpu",
        optimize_memory=True,
    )

    # Pre-populate with offline data if provided (like HIL-SERL)
    if offline_dataset_repo_id:
        try:
            logging.info(f"[LEARNER] Loading offline dataset: {offline_dataset_repo_id}")

            # Load a small subset of the dataset for demonstration
            offline_dataset = LeRobotDataset(
                repo_id=offline_dataset_repo_id,
                episodes=list(range(min(max_offline_episodes, 2))),  # Load only 2 episodes
                download_videos=False,  # Skip videos for faster loading
                video_backend="pyav",  # Use pyav backend instead of torchcodec
            )

            # Create replay buffer from dataset (this pre-populates it)
            offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
                lerobot_dataset=offline_dataset,
                capacity=capacity,
                device=device,
                state_keys=state_keys,
                storage_device="cpu",
                optimize_memory=True,
            )

            logging.info(
                f"[LEARNER] Pre-populated replay buffer with {offline_replay_buffer.size} transitions from offline data"
            )
            return offline_replay_buffer

        except Exception as e:
            logging.warning(f"[LEARNER] Failed to load offline dataset: {e}")
            logging.info("[LEARNER] Continuing with empty replay buffer")

    return replay_buffer


def run_learner(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    device: str = "cpu",
):
    """The learner process - trains SAC policy and sends parameters to actor."""
    logging.info("[LEARNER] Starting learner process")

    # Configure policy features
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    features_map = {
        ACTION: ACTION,
        "observation.state": OBS_STATE,
    }

    # Create SAC policy configuration for learning
    policy_cfg = SACConfig(
        device=device,
        input_features=list(input_features.keys()),
        online_buffer_capacity=50000,  # Replay buffer size (increased for offline data)
        online_step_before_learning=100,  # Start learning after 100 steps
    )
    policy_cfg.input_features = input_features
    policy_cfg.output_features = output_features
    policy_cfg.features_map = features_map

    # Create SAC policy and optimizer
    policy = SACPolicy(policy_cfg)

    # Create Adam optimizer from scratch - simple and clean
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Create replay buffer, optionally pre-populated with offline data (like HIL-SERL)
    # You can uncomment the next line to pre-populate with a real dataset:
    # offline_repo_id = "lerobot/pusht"  # Example dataset - small and fast to download
    offline_repo_id = None  # Set to None for empty buffer, or provide a repo_id

    replay_buffer = initialize_replay_buffer_with_offline_data(
        capacity=policy_cfg.online_buffer_capacity,
        device=device,
        state_keys=list(input_features.keys()),
        offline_dataset_repo_id=offline_repo_id,
        max_offline_episodes=3,  # Load only a few episodes for demo
    )

    logging.info(f"[LEARNER] Replay buffer capacity: {replay_buffer.capacity}")

    training_step = 0

    while not shutdown_event.is_set():
        # Collect new transitions from actor
        try:
            transitions = transitions_queue.get(timeout=0.1)  # Wait briefly for transitions
            for transition in transitions:
                replay_buffer.add(**transition)
            logging.info(f"[LEARNER] Added {len(transitions)} transitions. Buffer size: {len(replay_buffer)}")
        except:
            # No transitions available, continue
            pass

        # Train if we have enough data
        if len(replay_buffer) >= policy_cfg.online_step_before_learning:
            # Sample batch and train
            batch = replay_buffer.sample(batch_size=32)  # Smaller batch size
            loss, _ = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_step += 1

            if training_step % 5 == 0:  # More frequent logging
                logging.info(f"[LEARNER] Training step {training_step}, Loss: {loss.item():.4f}")

            # Send updated parameters to actor every 10 training steps
            if training_step % 10 == 0:
                try:
                    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
                    parameters_queue.put_nowait(state_dict)
                    logging.info("[LEARNER] Sent updated parameters to actor")
                except:
                    pass
        else:
            # Not enough data yet, wait a bit
            time.sleep(0.1)

    logging.info("[LEARNER] Learner process finished")


def run_actor(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    device: str = "cpu",
):
    """The actor process - interacts with environment and collects data."""
    logging.info("[ACTOR] Starting actor process")

    # Robot and environment configuration
    robot_cfg = MockRobotConfig(n_motors=3)
    teleop_cfg = KeyboardTeleopConfig(mock=True)
    processor_cfg = HILSerlProcessorConfig(control_mode="keyboard")

    env_cfg = HILSerlRobotEnvConfig(robot=robot_cfg, teleop=teleop_cfg, processor=processor_cfg)

    # Create robot environment
    env, teleop_device = make_robot_env(env_cfg)

    # Configure policy features
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    features_map = {
        ACTION: ACTION,
        "observation.state": OBS_STATE,
    }

    # Create SAC policy for action selection
    policy_cfg = SACConfig(
        device=device,
        input_features=list(input_features.keys()),
    )
    policy_cfg.input_features = input_features
    policy_cfg.output_features = output_features
    policy_cfg.features_map = features_map

    policy = SACPolicy(policy_cfg)
    policy.eval()

    episode_rewards = deque(maxlen=5)
    total_steps = 0
    max_episodes = 10
    max_steps_per_episode = 100

    try:
        for episode in range(max_episodes):
            if shutdown_event.is_set():
                break

            obs, info = env.reset()
            episode_reward = 0.0
            step = 0
            episode_transitions = []

            logging.info(f"[ACTOR] Starting episode {episode + 1}")

            while step < max_steps_per_episode and not shutdown_event.is_set():
                # Check for updated parameters from learner
                try:
                    new_params = parameters_queue.get_nowait()
                    policy.load_state_dict(new_params)
                    logging.info("[ACTOR] Updated policy parameters from learner")
                except:
                    pass

                # Get action from policy
                with torch.no_grad():
                    state_obs = obs["agent_pos"]
                    policy_obs = {"observation.state": torch.from_numpy(state_obs).float().unsqueeze(0)}
                    action_tensor = policy.select_action(policy_obs)
                    action = action_tensor.squeeze(0).numpy()

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                transition = {
                    "state": {"observation.state": state_obs},
                    "action": action,
                    "reward": float(reward) if hasattr(reward, "item") else reward,
                    "next_state": {"observation.state": next_obs["agent_pos"]},
                    "done": done,
                    "truncated": truncated,
                }
                episode_transitions.append(transition)

                episode_reward += transition["reward"]
                step += 1
                total_steps += 1

                # Log progress
                if step % 25 == 0:
                    logging.info(
                        f"[ACTOR] Episode {episode + 1}, Step {step}: reward = {transition['reward']:.3f}"
                    )

                obs = next_obs

                if done:
                    break

                time.sleep(0.02)  # Small delay

            # Send episode transitions to learner
            try:
                transitions_queue.put_nowait(episode_transitions)
                logging.info(f"[ACTOR] Sent {len(episode_transitions)} transitions to learner")
            except:
                logging.warning("[ACTOR] Failed to send transitions - queue full")

            episode_rewards.append(episode_reward)
            avg_reward = sum(episode_rewards) / len(episode_rewards)

            logging.info(
                f"[ACTOR] Episode {episode + 1} finished: {step} steps, "
                f"reward = {episode_reward:.3f}, avg = {avg_reward:.3f}"
            )

    except KeyboardInterrupt:
        logging.info("[ACTOR] Interrupted by user")
    finally:
        # Clean up
        if hasattr(env, "robot") and env.robot.is_connected:
            env.robot.disconnect()
        if teleop_device and hasattr(teleop_device, "disconnect"):
            teleop_device.disconnect()
        logging.info("[ACTOR] Actor process finished")


def main():
    """Main function - coordinates actor and learner processes."""
    init_logging()

    print("INFO: Actor-Learner SAC with Mock Robot!")
    print("This demonstrates distributed RL with separate actor and learner processes.")
    print("Like HIL-SERL, the replay buffer can be pre-populated with offline data.")
    print("Press Ctrl+C to stop at any time.")
    print()
    print("💡 To enable offline data pre-population (like HIL-SERL):")
    print("   Edit the 'offline_repo_id' variable in run_learner() function")
    print("   Example: offline_repo_id = 'lerobot/aloha_static_coffee'")
    print("   This will download and pre-populate the replay buffer with demonstrations!")
    print()

    # Create communication channels
    transitions_queue = mp.Queue(maxsize=10)
    parameters_queue = mp.Queue(maxsize=2)
    shutdown_event = mp.Event()

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\nSignal {sig} received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create processes
    learner_process = mp.Process(
        target=run_learner, args=(transitions_queue, parameters_queue, shutdown_event, "cpu")
    )

    actor_process = mp.Process(
        target=run_actor, args=(transitions_queue, parameters_queue, shutdown_event, "cpu")
    )

    # Start processes
    learner_process.start()
    time.sleep(1)  # Give learner time to initialize
    actor_process.start()

    try:
        # Wait for actor to finish (it controls the episode loop)
        actor_process.join()
        shutdown_event.set()
        learner_process.join(timeout=10)
    except KeyboardInterrupt:
        logging.info("Main process interrupted")
        shutdown_event.set()
        actor_process.join(timeout=5)
        learner_process.join(timeout=10)
    finally:
        # Ensure processes are terminated
        if learner_process.is_alive():
            learner_process.terminate()
        if actor_process.is_alive():
            actor_process.terminate()

        logging.info("Demo completed! Actor-Learner SAC finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
