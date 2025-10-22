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
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators.keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging
try:
    from tests.mocks.mock_robot import MockRobotConfig
except ImportError:
    # Fallback if tests module is not available
    from dataclasses import dataclass
    
    @dataclass
    class MockRobotConfig:
        n_motors: int = 3


class DummyRewardClassifier:
    """
    Dummy reward classifier for testing HIL-SERL implementation.
    Always returns success (reward = 1.0) to test the reward classifier pipeline.
    """
    
    def __init__(self, always_success: bool = True, success_probability: float = 1.0):
        """
        Args:
            always_success: If True, always returns success
            success_probability: Probability of returning success (if always_success=False)
        """
        self.always_success = always_success
        self.success_probability = success_probability
        
    def to(self, device):
        """Mock method to match real classifier interface"""
        return self
        
    def eval(self):
        """Mock method to match real classifier interface"""
        return self
        
    def predict_reward(self, images, threshold=0.5):
        """
        Mock predict_reward method that always returns success for testing.
        
        Args:
            images: Dictionary of image observations (ignored in dummy)
            threshold: Success threshold (ignored in dummy)
            
        Returns:
            torch.Tensor: Always returns 1.0 (success) for testing
        """
        import torch
        
        if self.always_success:
            return torch.tensor(1.0)
        else:
            # For more realistic testing, you could add randomness here
            import random
            success = 1.0 if random.random() < self.success_probability else 0.0
            return torch.tensor(success)


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

    # HIL-SERL Dual Buffer System
    # 1. Online buffer: Gets ALL transitions (human + autonomous)
    online_buffer = initialize_replay_buffer_with_offline_data(
        capacity=policy_cfg.online_buffer_capacity,
        device=device,
        state_keys=list(input_features.keys()),
        offline_dataset_repo_id=None,  # Start empty for online learning
        max_offline_episodes=0,
    )

    # 2. Offline buffer: Gets ONLY human intervention transitions
    offline_buffer = ReplayBuffer(
        capacity=10000,  # Smaller capacity for human demonstrations
        device=device,
        state_keys=list(input_features.keys()),
        storage_device="cpu",
        optimize_memory=True,
    )

    # Optionally pre-populate offline buffer with demonstrations
    offline_repo_id = None  # Set to dataset repo_id to pre-populate, e.g. "lerobot/pusht"
    if offline_repo_id:
        try:
            logging.info(f"[LEARNER] Pre-populating offline buffer with: {offline_repo_id}")
            offline_dataset = LeRobotDataset(
                repo_id=offline_repo_id,
                episodes=list(range(2)),  # Load only 2 episodes
                download_videos=False,
                video_backend="pyav",
            )
            # Add offline data to the offline buffer
            offline_buffer_temp = ReplayBuffer.from_lerobot_dataset(
                lerobot_dataset=offline_dataset,
                capacity=10000,
                device=device,
                state_keys=list(input_features.keys()),
                storage_device="cpu",
                optimize_memory=True,
            )
            # Copy transitions from temp buffer to offline buffer
            for i in range(len(offline_buffer_temp)):
                transition = offline_buffer_temp.sample(1)
                offline_buffer.add(
                    state=transition["state"],
                    action=transition["action"],
                    reward=transition["reward"].item(),
                    next_state=transition["next_state"],
                    done=transition["done"].item(),
                    truncated=transition["truncated"].item(),
                )
            logging.info(f"[LEARNER] Pre-populated offline buffer with {len(offline_buffer)} transitions")
        except Exception as e:
            logging.warning(f"[LEARNER] Failed to pre-populate offline buffer: {e}")

    logging.info(f"[LEARNER] Online buffer capacity: {online_buffer.capacity}")
    logging.info(f"[LEARNER] Offline buffer capacity: {offline_buffer.capacity}")

    training_step = 0

    while not shutdown_event.is_set():
        # Collect new transitions from actor
        try:
            transitions = transitions_queue.get(timeout=0.1)  # Wait briefly for transitions
            for transition in transitions:
                # HIL-SERL: Add ALL transitions to online buffer
                online_buffer.add(**transition)
                
                # HIL-SERL: Add ONLY human intervention transitions to offline buffer
                is_intervention = transition.get("complementary_info", {}).get("is_intervention", False)
                if is_intervention:
                    offline_buffer.add(**transition)
                    logging.info(f"[LEARNER] 🤖 Human intervention detected! Added to offline buffer (now {len(offline_buffer)} transitions)")
            
            interventions_count = sum(1 for t in transitions if t.get("complementary_info", {}).get("is_intervention", False))
            logging.info(f"[LEARNER] ✅ Added {len(transitions)} transitions ({interventions_count} interventions). "
                        f"📊 Online buffer: {len(online_buffer)}, Offline buffer: {len(offline_buffer)}")
        except:
            # No transitions available, continue
            pass

        # Train if we have enough data
        if len(online_buffer) >= policy_cfg.online_step_before_learning:
            # HIL-SERL: Sample from BOTH buffers for training
            batch_size = 32
            
            # Sample from online buffer (autonomous + human data)
            online_batch = online_buffer.sample(batch_size // 2)
            
            # Sample from offline buffer if it has data (human demonstrations only)
            if len(offline_buffer) > 0:
                offline_batch = offline_buffer.sample(batch_size // 2)
                
                # Combine batches - this is the key HIL-SERL mechanism!
                batch = {}
                for key in online_batch.keys():
                    if key in offline_batch:
                        batch[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
                    else:
                        batch[key] = online_batch[key]
                logging.info(f"[LEARNER] 🎯 Training with mixed batch: {batch_size//2} online + {batch_size//2} offline")
            else:
                # No offline data yet, use only online data
                batch = online_buffer.sample(batch_size)
                logging.info(f"[LEARNER] 📈 Training with online-only batch: {batch_size} (no offline data yet)")
            
            loss, _ = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_step += 1

            if training_step % 5 == 0:  # More frequent logging
                logging.info(f"[LEARNER] 📊 Training step {training_step}, Loss: {loss.item():.4f}, "
                           f"Buffers: Online={len(online_buffer)}, Offline={len(offline_buffer)}")

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

    # Optional: Load reward classifier for automated reward assignment
    reward_classifier = None
    use_dummy_classifier = True  # Set to True to use dummy classifier for testing
    reward_classifier_path = None  # Set to your trained reward classifier path, e.g. "your_username/reward_classifier_model"
    success_threshold = 0.7  # Threshold for considering a state as successful
    success_reward = 1.0  # Reward to assign when success is detected

    if use_dummy_classifier:
        # Use dummy classifier for testing HIL-SERL implementation
        logging.info("[ACTOR] Using dummy reward classifier for testing")
        reward_classifier = DummyRewardClassifier(
            always_success=True,  # Always return success for testing
            success_probability=1.0  # 100% success rate
        )
        logging.info("[ACTOR] Dummy reward classifier loaded - will always detect success!")
    elif reward_classifier_path:
        try:
            logging.info(f"[ACTOR] Loading real reward classifier: {reward_classifier_path}")
            reward_classifier = Classifier.from_pretrained(reward_classifier_path)
            reward_classifier.to(device)
            reward_classifier.eval()
            logging.info("[ACTOR] Real reward classifier loaded successfully")
        except Exception as e:
            logging.warning(f"[ACTOR] Failed to load reward classifier: {e}")
            reward_classifier = None

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
    max_episodes = 20  # Run more episodes to see learning
    max_steps_per_episode = 10  # Shorter episodes since dummy classifier terminates them quickly

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

                # Optional: Use reward classifier to predict reward from visual observations
                if reward_classifier is not None:
                    try:
                        if use_dummy_classifier:
                            # Dummy classifier doesn't need images - just call predict_reward
                            with torch.no_grad():
                                predicted_success = reward_classifier.predict_reward({}, threshold=success_threshold)
                        else:
                            # Real classifier needs images
                            if "images" in next_obs:
                                # Extract images for reward classifier
                                images = {k: v for k, v in next_obs.items() if "image" in k}

                                if images:
                                    # Simple forward pass through reward classifier
                                    with torch.no_grad():
                                        # Convert images to the expected format
                                        image_batch = {}
                                        for img_key, img_tensor in images.items():
                                            if isinstance(img_tensor, torch.Tensor):
                                                # Ensure proper shape: [batch_size, channels, height, width]
                                                if img_tensor.dim() == 3:
                                                    img_tensor = img_tensor.unsqueeze(0)
                                                image_batch[f"observation.{img_key}"] = img_tensor

                                        # Predict reward using the classifier
                                        predicted_success = reward_classifier.predict_reward(
                                            image_batch, threshold=success_threshold
                                        )
                                else:
                                    continue  # Skip if no images available for real classifier
                            else:
                                continue  # Skip if no images available for real classifier

                        # Use classifier reward if success is detected
                        if predicted_success.item() > 0.5:  # Success detected
                            reward = success_reward
                            if not done:  # Only terminate if not already done
                                terminated = True
                                done = True
                            classifier_type = "Dummy" if use_dummy_classifier else "Real"
                            logging.info(f"[ACTOR] 🎯 {classifier_type} reward classifier detected success! Reward: {reward} (Episode will terminate)")

                    except Exception as e:
                        logging.debug(f"[ACTOR] Reward classifier error: {e}")
                        # Continue with original reward if classifier fails

                # Simulate human intervention detection (in real HIL-SERL this comes from teleop device)
                # For demo purposes, randomly simulate interventions 10% of the time
                is_intervention = False
                if hasattr(teleop_device, "get_teleop_events"):
                    # Real intervention detection from teleop device
                    teleop_events = teleop_device.get_teleop_events()
                    is_intervention = teleop_events.get(TeleopEvents.IS_INTERVENTION, False)
                else:
                    # Simulate interventions for demo (remove this in real implementation)
                    import random
                    is_intervention = random.random() < 0.1  # 10% intervention rate for demo

                # Store transition with intervention metadata
                transition = {
                    "state": {"observation.state": state_obs},
                    "action": action,
                    "reward": float(reward) if hasattr(reward, "item") else reward,
                    "next_state": {"observation.state": next_obs["agent_pos"]},
                    "done": done,
                    "truncated": truncated,
                    "complementary_info": {
                        "is_intervention": is_intervention,  # Key HIL-SERL metadata!
                    },
                }
                
                if is_intervention:
                    logging.info(f"[ACTOR] 🤖 Human intervention at step {step}! This will go to BOTH buffers.")
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

    print("=" * 80)
    print("🚀 HIL-SERL Actor-Learner Example with Reward Classifier Testing")
    print("=" * 80)
    print("This demonstrates:")
    print("• Distributed RL with separate actor and learner processes")
    print("• HIL-SERL dual buffer system (online + offline buffers)")
    print("• Reward classifier integration (dummy classifier for testing)")
    print("• Human intervention detection and routing")
    print()
    print("📊 TESTING INDICATORS TO WATCH FOR:")
    print("✅ Dummy reward classifier always detects success")
    print("✅ Episodes terminate early due to success detection")
    print("✅ Rewards are set to 1.0 when success is detected")
    print("✅ Human interventions are routed to both buffers")
    print("✅ Training uses mixed batches from both buffers")
    print("✅ Clear logging shows buffer sizes and training progress")
    print()
    print("Press Ctrl+C to stop at any time.")
    print("=" * 80)
    print()
    print("💡 Configuration Options:")
    print("   • Offline data: Edit 'offline_repo_id' in run_learner()")
    print("   • Dummy classifier: use_dummy_classifier = True (currently enabled)")
    print("   • Real classifier: Set reward_classifier_path to your model")
    print()
    print("🤖 HIL-SERL Dual Buffer System:")
    print("   • Online Buffer: Gets ALL transitions (human + autonomous)")
    print("   • Offline Buffer: Gets ONLY human intervention transitions")
    print("   • Training: Samples from BOTH buffers (50/50 mix)")
    print("   • Interventions are detected and routed to both buffers!")
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
