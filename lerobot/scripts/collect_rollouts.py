from pathlib import Path
import os
import click
import gym_pusht
import gym_pushany
import gymnasium as gym
import numpy as np
import imageio
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
import shortuuid

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import get_default_encoding
from lerobot.common.datasets.rollout_datasets.episode_stores import EpisodeVideoStore, EpisodeVideoStoreAsHDF5
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

#After update gym-pushany, alter the object name list to gym_pushany.
OBJECT_NAME_LIST = [
    't',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'ellipse',
    'rectangle',
    'reg3',
    'reg4',
    'reg5',
    'reg6',
    'reg7',
    'reg8',
    'reg9',
    'reg10'
]


def _resize_frame_tensors_to_frames(frames: list[torch.Tensor]) -> list[np.ndarray]:
    num_frames = len(frames)  # H W C

    frames_t = torch.stack(frames, dim=0).permute(0, 3, 1, 2)
    resized_frames_t = F.interpolate(frames_t, size=[224, 224], mode='bilinear', align_corners=False)
    resized_frames_t = resized_frames_t.permute(0, 2, 3, 1).numpy()
    return [resized_frames_t[i] for i in range(num_frames)]


def build_ep_dict(observation_states: list,
                  actions: list,
                  rewards: list,
                  dones: list,
                  successes: list,
                  frames: list[torch.Tensor],
                  videos_dir: Path,
                  fps=10):
    num_frames = len(frames)
    assert len(observation_states) == num_frames
    assert len(actions) == num_frames
    assert len(rewards) == num_frames
    assert len(dones) == num_frames
    assert len(successes) == num_frames

    ep_dict = {}

    frames = _resize_frame_tensors_to_frames(frames)

    #  observation.image
    img_key = 'observation.image'
    fname = f"{img_key}_episode_{shortuuid.ShortUUID().random(length=8)}.mp4"
    video_path = videos_dir / fname
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")
    ep_dict[img_key] = [
        {'path': f"videos/{fname}", 'timestamp': i / fps} for i in range(num_frames)
    ]

    # #  observation.state (b x n)
    # ep_dict['observation.state'] = torch.stack(observation_states)
    # # action (b x 2)
    # ep_dict['action'] = torch.stack(actions)
    # # frame_index
    # ep_dict['frame_index'] = torch.arange(0, num_frames, 1)
    # ep_dict['timestamp'] = torch.arange(0, num_frames, 1) / fps
    # ep_dict["next.reward"] = torch.tensor(rewards)
    # ep_dict["next.done"] = torch.tensor(dones, dtype=torch.int8)
    # ep_dict["next.success"] = torch.tensor(successes, dtype=torch.int8)
    
    # observation.state (b x n)
    ep_dict['observation.state'] = np.array([state.numpy() for state in observation_states])
    # action (b x 2)
    ep_dict['action'] = np.array([action.numpy() for action in actions])
    # frame_index
    ep_dict['frame_index'] = np.array(range(num_frames))
    ep_dict['timestamp'] = np.array([i / fps for i in range(num_frames)])
    ep_dict["next.reward"] = np.array(rewards)
    ep_dict["next.done"] = np.array([int(done) for done in dones])
    ep_dict["next.success"] = np.array([int(success) for success in successes])
    return ep_dict


def rollout_for_ep_dicts(policy, env, device, episode_video_store, num_episodes, videos_dir, object_name=None):
    for _ in range(num_episodes):
        policy.diffusion.num_inference_steps = np.random.randint(1, 20)
        policy.reset()

        numpy_observation, info = env.reset()

        frames = []
        observation_states = []
        actions = []
        rewards = []
        dones = []
        successes = []

        step = 0
        done = False
        while not done:
            frames.append(torch.from_numpy(env.render()))

            # Prepare observation for the policy running in Pytorch
            state = torch.from_numpy(numpy_observation["agent_pos"])
            image = torch.from_numpy(numpy_observation["pixels"])

            # Convert to float32 with image from channel first in [0,255]
            # to channel last in [0,1]
            state_t = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image_t = image.permute(2, 0, 1)  # c x h x w

            # Send data tensors from CPU to GPU
            state = state_t.to(device, non_blocking=True)
            image = image_t.to(device, non_blocking=True)

            # Add extra (empty) batch dimension, required to forward the policy
            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            # Create the policy input dictionary
            observation = {
                "observation.state": state,
                "observation.image": image,
            }

            # Predict the next action with respect to the current observation
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Prepare the action for the environment
            action_t = action.squeeze(0).to("cpu")
            numpy_action = action_t.numpy()

            # Step through the environment and receive a new observation
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            if step % 100 == 0:
                print(f"{step=} {reward=} {terminated=}")

            # The rollout is considered done when the success state is reach (i.e. terminated is True),
            # or the maximum number of iterations is reached (i.e. truncated is True)
            done = terminated | truncated | done

            # Keep track of all the rewards and frames
            observation_states.append(state_t)
            actions.append(action_t)
            rewards.append(reward)
            successes.append(terminated)
            dones.append(done)
            step += 1

        if terminated:
            print("Success!")
        else:
            print("Failure!")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.metadata["render_fps"]

        ep_dict = build_ep_dict(observation_states=observation_states,
                                actions=actions,
                                rewards=rewards,
                                dones=dones,
                                successes=successes,
                                frames=frames,
                                fps=fps,
                                videos_dir=videos_dir)
        episode_video_store.add_episode(ep_dict)

    return episode_video_store


@click.command()
@click.option('-o', '--output', required=True)
@click.option('-n', '--num_rollouts', required=True)
def main(output, num_rollouts):
    num_rollouts = int(num_rollouts)
    pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
        policy.diffusion.num_inference_steps = 10

    policy.diffusion.num_inference_steps = 1
    policy = policy.to(device)

    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)

    videos_dir = output_directory / 'videos'
    videos_dir.mkdir(parents=True, exist_ok=True)
    fps = 10
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": True,
        "encoding": get_default_encoding(),
        "videos_dir": str(videos_dir)
    }

    episode_video_store = EpisodeVideoStore.create_from_path(output_directory, info, mode='a')
    print(episode_video_store.num_episodes)
    print(episode_video_store.info)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
    )

    rollout_for_ep_dicts(policy, env, device, episode_video_store, num_rollouts, videos_dir)

@click.command()
@click.option('-p', '--pretrained_policy_path', required=True)
@click.option('-o', '--output', required=True)
@click.option('-n', '--num_rollouts', required=True)
@click.option('-t', '--task', required=True)
def main_pushany(pretrained_policy_path, output, num_rollouts, task):
    task_id = int(task)
    num_rollouts = int(num_rollouts)
    pretrained_policy_path = Path(pretrained_policy_path)
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
        policy.diffusion.num_inference_steps = 10

    policy.diffusion.num_inference_steps = 1
    policy = policy.to(device)

    output_directory = Path(output) / f"{OBJECT_NAME_LIST[task_id]}"
    output_directory.mkdir(parents=True, exist_ok=True)

    videos_dir = output_directory / 'videos'
    videos_dir.mkdir(parents=True, exist_ok=True)
    fps = 10
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": True,
        "encoding": get_default_encoding(),
        "videos_dir": str(videos_dir)
    }

    # episode_video_store = EpisodeVideoStore.create_from_path(output_directory, info, mode='a')
    
    hdf5_file_path = os.path.join(output_directory, 'data.h5')

    episode_video_store = EpisodeVideoStoreAsHDF5(hdf5_file_path, info)
    print(episode_video_store.num_episodes)
    print(episode_video_store.info)

    env = gym.make(
        "gym_pushany/PushAny-v0",
        object_name=OBJECT_NAME_LIST[task_id],
        max_episode_steps=300,
    )

    episode_video_store = rollout_for_ep_dicts(policy, env, device, episode_video_store, num_rollouts, videos_dir, object_name=OBJECT_NAME_LIST[task_id])




if __name__ == '__main__':
    main_pushany()
