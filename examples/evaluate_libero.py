"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
"""

import collections
import dataclasses
import logging
import math
import pathlib

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """

    # --- Hugging Face arguments ---
    policy_path: str = "lerobot/smolvla_base"
    """Path to the pretrained policy on the Hugging Face Hub or local directory."""

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "data/libero/videos"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""


@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load Policy ---
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    # --- Initialize LIBERO task suite ---
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Available options are: {list(benchmark_dict.keys())}"
        )
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        # Fallback for custom task suites
        max_steps = 520

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        
        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(
            range(min(args.num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ):
            logging.info(f"\nTask: {task_description}")

            # Reset environment and policy
            env.reset()
            policy.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Setup
            t = 0
            frames = []
            done = False

            # Add initial frame
            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            # frames.append(agentview_image)
            # import ipdb; ipdb.set_trace()
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:
                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)

                    # Prepare observations dict
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(args.device).unsqueeze(0),
                        "observation.images.image2": torch.from_numpy(wrist_img / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }

                    # Query model to get action
                    with torch.inference_mode():
                        action_tensor = policy.select_action(observation)
                    action = action_tensor.cpu().numpy()[0]
                    action[-1] = 1 -  action[-1]

                    # Execute action in environment
                    obs, _, done, _ = env.step(action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            video_path = (
                pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
            )
            fps = 30
            writer = imageio.get_writer(video_path, fps=fps)

            for image in frames:
                writer.append_data(image)
            writer.close()
            logging.info(f"Saved video to {video_path}")

            # Log current results
            logging.info(f"Success: {done}")
            if total_episodes > 0:
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results for the task
        if task_episodes > 0:
            logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes):.2f}")
        if total_episodes > 0:
            logging.info(f"Cumulative success rate: {float(total_successes) / float(total_episodes):.2f}")

    logging.info("--- Evaluation finished ---")
    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.2f}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    cv2.destroyAllWindows()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero()