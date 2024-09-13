"""This script demonstrates how to record a LeRobot dataset of training data using a very simple gym environment.
"""

from pynput.keyboard import Key
from pynput import keyboard
import argparse
import copy
import os
import sys
import pathlib
import time
import importlib

import gymnasium as gym
import numpy as np
import torch

from datasets import Dataset, Features, Sequence, Value
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, DATA_DIR, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_videos_to_hub, save_meta_data
from tqdm import tqdm


def process_args():
    # parse the repo_id name via command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-name", type=str, default="gym_lowcostrobot")
    parser.add_argument("--env-name", type=str, default="ReachCube-v0")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--repo-id", type=str, default="myrepo")
    parser.add_argument("--push-to-hub", action="store_true")

    parser.add_argument(
        "--image-keys",
        type=str,
        default="image_top,image_front",
        help="The keys of the image observations to record.",
    )
    parser.add_argument(
        "--state-keys",
        type=str,
        default="arm_qpos,arm_qvel,cube_pos",
        help="The keys of the state observations to record.",
    )

    parser.add_argument(
        "--revision", type=str, default=CODEBASE_VERSION, help="Codebase version used to generate the dataset."
    )

    return parser.parse_args()


class Runner:
    def __init__(self, args):
        self.args = args
        self.image_keys = self.args.image_keys.split(",")
        self.state_keys = self.args.state_keys.split(",")
        self.dataset_counter = 0

        self.import_module_contining_gym()

        self.env = self.construct_and_set_up_env()

        self.set_up_output_path()

        if args.teleop_method == "keyboard":
            self.set_up_keyboard_teleop()
        else:
            raise Exception("only 'keyboard' teleop is suppored!")

        if (args.teleop_method == "") or (self.stop_teleoperation is callable(self.stop_teleoperation)):
            pass
        else:
            raise ValueError(
                "stop_teleoperation is not callable but a teleoperation system was initialized.")

    def increment_dataset_counter(self):
        self.dataset_counter += 1

    def construct_and_set_up_env(self):
        # Create the gym environment - check the kwargs in gym_real_world/gym_environment.py
        env = gym.make(self.args.env_name, disable_env_checker=True,
                       observation_mode="both", action_mode="ee", render_mode="human")

        # Reset the environment
        observation, info = self.env.reset()
        return env

    def set_up_keyboard_teleop(self):
        # Sample random action (usually positional).
        sample = self.env.action_space.sample()
        print(f"action type={type(sample)}")

        # Assign good initial action
        sample[0] = 0.0
        sample[1] = 0.14
        sample[2] = 0.17
        sample[3] = 0.0

        self.init_action = sample
        print(f"init_action={self.init_action}")
        assert self.env.action_space.contains(self.init_action)

        # TODO(samzapo): Use SharedMemoryManager.ShareableList for the following values.
        #                from multiprocessing.managers import SharedMemoryManager
        self.action = self.init_action * 1.0
        self.is_dropping_episode = False
        self.is_concluding_episode = False
        self.is_stopping = False

        # TODO(samzapo): Launch the following code in a separate process using a Process
        #                from multiprocessing import Process
        def on_press(key):
            # Y
            if key == Key.up:
                self.action[1] += 0.01
            elif key == Key.down:
                self.action[1] -= 0.01
            # X
            elif key == Key.left:
                self.action[0] -= 0.01
            elif key == Key.right:
                self.action[0] += 0.01
            # Z
            elif key == Key.page_down:
                self.action[2] -= 0.01
            elif key == Key.page_up:
                self.action[2] += 0.01
            # Gripper
            elif key == Key.cmd:
                self.action[3] -= 0.1
            elif key == Key.shift:
                self.action[3] += 0.1
            # Training
            elif key == Key.delete:
                self.is_dropping_episode = True
                self.is_concluding_episode = True
            elif key == Key.home:
                self.is_concluding_episode = True
            elif key == Key.end:
                self.is_concluding_episode = True
                self.is_stopping = True

        def on_release(key):
            # print('{0} released'.format(
            # key))
            if key == keyboard.Key.esc:
                self.is_stopping = True
                # Stop listener
                return False

        listener = keyboard.Listener(
            on_press=on_press, on_release=on_release)
        listener.start()

        # assign teleoperation shut-down function.
        def stop_teleoperation():
            listener.stop()

        self.stop_teleoperation = stop_teleoperation

    def output_data_path(self, dataset_index: int | None = None) -> str:
        if DATA_DIR == None:
            DATA_DIR = pathlib.Path("data_traces")
            print(
                "Warning: env variable DATA_DIR was not set, defaulting to './{}'.".format(DATA_DIR))

        if dataset_index is None:
            dataset_index = self.dataset_counter

        return DATA_DIR / self.args.repo_id / str(dataset_index)

    # Where to save the LeRobotDataset
    def hf_dataset_path(self, dataset_index: int | None = None) -> str:
        return self.output_data_path() / "train"

    # During data collection, frames are stored here as png images.
    def images_data_path(self, dataset_index: int | None = None) -> str:
        return self.output_data_path() / "images"

    # After data collection, png images of each episode are encoded into an mp4 file stored here.
    def videos_data_path(self, dataset_index: int | None = None) -> str:
        return self.output_data_path() / "videos"

    def meta_data_path(self, dataset_index: int | None) -> str:
        return self.output_data_path() / "meta_data"

    def set_up_output_path(self):
        # Create image and video directories
        if not os.path.exists(self.images_data_path()):
            os.makedirs(self.images_data_path(), exist_ok=True)
        if not os.path.exists(self.videos_data_path()):
            os.makedirs(self.videos_data_path(), exist_ok=True)

    def import_module_contining_gym(self):
        # import the gym module containing the environment
        try:
            # because we want to import using a variable, do it this way
            module_obj = __import__(self.args.module_name)
            # create a global object containging our module
            globals()[self.args.module_name] = module_obj
        except ImportError:
            sys.stderr.write("ERROR: missing python module: " +
                             self.args.module_name + "\n")
            sys.exit(1)

    def stop_simulating(self):
        self.env.close()
        if self.stop_teleoperation is not None:
            self.stop_teleoperation()

    def encode_video_frames(self, *, episode_fps):
        print("encode video frames")
        for ep_idx in range(len(episode_fps)):
            for img_key in self.image_keys:
                encode_video_frames(
                    imgs_dir=self.images_data_path(
                    ) / f"{img_key}_episode_{ep_idx:06d}",
                    video_path=self.videos_data_path(
                    ) / f"{img_key}_episode_{ep_idx:06d}.mp4",
                    fps=episode_fps[ep_idx],
                )

    def construct_hf_dataset(self, *, ep_dicts):
        print("concatenate episodes")
        # Since our fps varies we are sometimes off tolerance for the last frame
        data_dict = concatenate_episodes(ep_dicts)

        features = {}

        keys = [key for key in data_dict if "observation.image_" in key]
        for key in keys:
            features[key.replace("observation.image_",
                                 "observation.images.")] = VideoFrame()
            data_dict[key.replace("observation.image_",
                                  "observation.images.")] = data_dict[key]
            del data_dict[key]

        features["observation.state"] = Sequence(
            length=data_dict["observation.state"].shape[1], feature=Value(
                dtype="float32", id=None)
        )
        features["action"] = Sequence(
            length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None))
        features["episode_index"] = Value(dtype="int64", id=None)
        features["frame_index"] = Value(dtype="int64", id=None)
        features["timestamp"] = Value(dtype="float32", id=None)
        features["next.done"] = Value(dtype="bool", id=None)
        features["index"] = Value(dtype="int64", id=None)

        hf_dataset = Dataset.from_dict(
            data_dict, features=Features(features))
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def construct_lerobot_dataset_and_save_to_disk(self, *, hf_dataset, episode_data_index, episode_fps) -> LeRobotDataset:
        info = {
            # to have a good tolerance in data processing for the slowest video
            "fps": sum(episode_fps) / len(episode_fps),
            "video": 1,
        }

        print("from preloaded")
        lerobot_dataset = LeRobotDataset.from_preloaded(
            repo_id=self.args.repo_id,
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            info=info,
            videos_dir=self.videos_data_path(),
        )

        print("compute stats")
        stats = compute_stats(
            lerobot_dataset, num_workers=self.args.num_workers)

        print("save to disk")
        # to remove transforms that cant be saved
        hf_dataset = hf_dataset.with_format(None)
        hf_dataset.save_to_disk(self.hf_dataset_path())

        save_meta_data(info, stats, episode_data_index, self.meta_data_path())

        return lerobot_dataset

    def maybe_push_to_hub(self, *, hf_dataset, repo_id: str, revision: str):
        if not self.args.push_to_hub:
            return

        print(f"Pushing dataset to '{repo_id}'")
        hf_dataset.push_to_hub(repo_id, token=True, revision="main")
        hf_dataset.push_to_hub(repo_id, token=True, revision=revision)

        push_meta_data_to_hub(repo_id, self.meta_data_path(), revision="main")
        push_meta_data_to_hub(
            repo_id, self.meta_data_path(), revision=revision)

        push_videos_to_hub(repo_id, self.videos_data_path(), revision="main")
        push_videos_to_hub(repo_id, self.videos_data_path(), revision=revision)

    def run_all_episodes(self) -> LeRobotDataset:
        ep_dicts = []
        episode_data_index = {"from": [], "to": []}
        episode_fps = []
        id_from = 0
        id_to = 0

        def save_episode_data(*, observations, actions, timestamps):
            ep_dict = {}
            # store images in png and create the video
            for img_key in self.image_keys:
                save_images_concurrently(
                    observations[img_key],
                    self.images_data_path() /
                    f"{img_key}_episode_{episode_counter:06d}",
                    self.args.num_workers,
                )
                fname = f"{img_key}_episode_{episode_counter:06d}.mp4"

                # store the reference to the video frame
                ep_dict[f"observation.{img_key}"] = [
                    {"path": f"videos/{fname}", "timestamp": timestamp} for timestamp in timestamps]

            states = []
            for state_name in self.image_keys:
                states.append(np.array(observations[state_name]))
            state = torch.tensor(np.concatenate(states, axis=1))

            action = torch.tensor(np.array(actions))
            next_done = torch.zeros(step_counter, dtype=torch.bool)
            next_done[-1] = True

            ep_dict["observation.state"] = state
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor(
                [episode_counter] * step_counter, dtype=torch.int64)
            ep_dict["frame_index"] = torch.arange(0, step_counter, 1)
            ep_dict["timestamp"] = torch.tensor(timestamps)
            ep_dict["next.done"] = next_done

            print(f"step_counter={step_counter}")
            print(f"timestamps[-1]={timestamps[-1]}")
            episode_fps.append(step_counter / timestamps[-1])
            ep_dicts.append(ep_dict)
            print("Episode {} done, fps: {:.2f}".format(
                episode_counter, episode_fps[-1]))

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + step_counter)

            id_to = id_from + step_counter
            id_from = id_to

        print(f"gym environment created")

        episode_counter = 0

        self.is_stopping = False
        while not self.is_stopping:
            self.action = self.init_action * 1.0

            # bring the follower to the leader and start camera
            self.env.reset()

            print(f"go {episode_counter}")

            # init buffers
            observations = {k: [] for k in env.observation_space}
            actions = []
            timestamps = []

            self.is_dropping_episode = False
            self.is_concluding_episode = False
            step_counter = 0
            while not self.is_concluding_episode:
                # Apply the next action (provided by teleop)
                observation, reward, terminted, truncated, info = self.env.step(
                    action=self.action)

                # Render the simultion
                self.env.render()

                # store data
                for key in observation:
                    observations[key].append(copy.deepcopy(observation[key]))
                actions.append(copy.deepcopy(self.action))
                timestamps.append(info["timestamp"])

                step_counter += 1

            if self.is_dropping_episode:
                continue

            print(f"saving episode {episode_counter}...")

            save_episode_data(observations=observations,
                              actions=actions,
                              timestamps=timestamps)

            episode_counter += 1

        self.stop_simulating()

        self.encode_video_frames(episode_fps=episode_fps)

        hf_dataset = self.construct_hf_dataset(ep_dicts=ep_dicts)
        lerobot_dataset = self.construct_lerobot_dataset_and_save_to_disk(
            hf_dataset=hf_dataset, episode_data_index=episode_data_index, episode_fps=episode_fps)

        self.save_hf_dataset_to_disk(lerobot_dataset=lerobot_dataset)

        self.maybe_push_to_hub(hf_dataset=hf_dataset,
                               repo_id=self.args.repo_id,
                               revision=self.args.revision)

        return lerobot_dataset

    def teleop_robot_and_record_data() -> LeRobotDataset:
        new_dataset = self.run_all_episodes()
        return new_dataset

    def replay_episodes_and_record_data(self, *, source_dataset: LeRobotDataset | None = None) -> LeRobotDataset:
        if source_dataset is None:
            source_dataset = LeRobotDataset(
                repo_id=self.args.repo_id,
                root=self.hf_dataset_path()
            )

        new_dataset = self.run_all_episodes(source_dataset=source_dataset)
        return new_dataset


if __name__ == "__main__":
    args = process_args()
    runner = Runner(args)

    lerobot_dataset = record_initial_training_data()
    runner.increment_dataset_counter()

    print("Replaying from dataset")
    runner.replay_episodes(dataset=lerobot_dataset)

    print("Replaying teleop from ")
    runner.replay_episodes(source_)
