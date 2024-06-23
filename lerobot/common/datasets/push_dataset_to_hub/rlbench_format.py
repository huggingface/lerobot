import shutil
from collections import defaultdict
from pathlib import Path

import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def _launch_env(dataset_root: str = ""):
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()), dataset_root, obs_config, headless=True
    )
    env.launch()
    return env


def check_format(raw_dir: Path) -> bool:
    # If demos can be loaded, the format is correct
    demos = load_from_raw(raw_dir)
    return bool(demos)


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path | None = None,
    fps: int | None = None,
    video: bool | None = None,
    episodes: list[int] | None = None,
) -> dict:
    env = _launch_env(str(raw_dir))
    # TODO: Automatically detect task
    task = env.get_task(ReachTarget)
    # TODO: Don't hardcode the number of demos
    # TODO: Maybe use rlbench.utils.get_stored_demos?
    demos = task.get_demos(100)
    env.shutdown()

    num_episodes = len(demos)
    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_dict = defaultdict(list)

        demo = demos[ep_idx]
        num_frames = len(demo)
        # Last two steps of demonstration is considered done
        # This is because last timestep does not have an action
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-2:] = True

        # Get camera attributes from rlbench.demo.Observation
        camera_attributes = []
        for attr in dir(demo[0]):
            if "rgb" in attr and getattr(demo[0], attr) is not None:
                camera_attributes.append(attr.replace("_rgb", ""))

        # Start from second timestep
        # Excludes observation from last timestep
        for timestep_idx, obs in enumerate(demo):
            # TODO: Add other low dim states
            ep_dict["observation.states.joint_positions"].append(torch.as_tensor(obs.joint_positions))
            ep_dict["observation.states.gripper_open"].append(torch.as_tensor(obs.gripper_open))

            for camera in camera_attributes:
                image = getattr(obs, f"{camera}_rgb")
                ep_dict[f"observation.images.{camera}"].append(torch.as_tensor(image))

            # First timestep doesn't have an action
            if timestep_idx > 0:
                ep_dict["action"].append(torch.as_tensor(obs.misc["joint_position_action"]))
        # len(action) == len(observation) - 1
        # Hence we add a dummy action to the last timestep
        ep_dict["action"].append(torch.zeros_like(ep_dict["action"][-1]))

        for key, value in ep_dict.items():
            ep_dict[key] = torch.stack(value)

        if video:
            for img_key in ep_dict:
                if "observation.images." not in img_key:
                    continue

                imgs_array = ep_dict[img_key].numpy()
                # save png images in temporary directory
                tmp_imgs_dir = videos_dir / "tmp_images"
                save_images_concurrently(imgs_array, tmp_imgs_dir)

                # encode images to a mp4 video
                fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                video_path = videos_dir / fname
                encode_video_frames(tmp_imgs_dir, video_path, fps)

                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)

                # store the reference to the video frame
                ep_dict[img_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                ]

        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["next.done"] = done

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict: dict, video: bool):
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    # Low dimensional states
    features["observation.states.joint_positions"] = Sequence(
        length=data_dict["observation.states.joint_positions"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["observation.states.gripper_open"] = Value(dtype="float32", id=None)

    # Only supports joint position actions for now
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
):
    # sanity check
    # check_format(raw_dir)

    if fps is None:
        fps = 10

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
