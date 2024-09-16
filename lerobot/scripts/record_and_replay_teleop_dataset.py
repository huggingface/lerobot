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
import types

import gymnasium as gym
import numpy as np
import torch
import lerobot

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
    parser.add_argument("--module-name", type=str, default="gym_drake_lca")
    parser.add_argument("--env-name", type=str, default="LiftCube-v0")
    parser.add_argument("--teleop-method", type=str, default="keyboard")
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


class ReplayHelper:
    def __init__(self, *, lerobot_datadet: LeRobotDataset):
        self.dataset = lerobot_datadet
        print("dataset={}".format(self.dataset))
        self.frame_index = 0

    def increment_frame_index(self):
        self.frame_index += 1

    def get_action_at_frame(self):
        return self.dataset[self.frame_index]["action"]

    @property
    def is_at_last_frame_in_episode(self):
        return self.dataset[self.frame_index]["episode_index"] != self.dataset[self.frame_index + 1]["episode_index"]

    @property
    def is_at_last_frame_in_dataset(self):
        return self.dataset.num_samples-1 == self.frame_index


class Runner:
    def __init__(self, args):
        self.args = args
        self.image_keys = self.args.image_keys.split(",")
        self.state_keys = self.args.state_keys.split(",")
        self.dataset_counter = 0
        self.prev_output_data_path = None
        self.set_up_teleop(args.teleop_method)
        self.sim_parameters = dict()

    def set_up_teleop(self, teleop_method: str):
        if teleop_method == "keyboard":
            self.set_up_keyboard_teleop()
        else:
            raise Exception(
                "A teleoperation method must be selected (Currently only 'keyboard' teleop is suppored)!")

        if not isinstance(self.stop_teleoperation_handler, types.FunctionType):
            raise ValueError(
                "stop_teleoperation_handler is not callable but a teleoperation system was initialized.")

    def increment_dataset_counter(self):
        self.prev_output_data_path = self.output_data_path
        self.dataset_counter += 1

    def construct_and_set_up_env(self):
        # Create the gym environment - check the kwargs in gym_real_world/gym_environment.py
        env = gym.make(self.args.env_name, disable_env_checker=True,
                       observation_mode="both", action_mode="ee", render_mode="human", parameters=self.sim_parameters)

        # Reset the environment
        observation, info = env.reset()
        return env

    def get_next_action_in_episode(self, replay_helper: ReplayHelper | None):
        if replay_helper is not None:
            action = replay_helper.get_action_at_frame()
            if replay_helper.is_at_last_frame_in_dataset:
                self.is_concluding_episode = True
                self.is_stopping = True
            elif replay_helper.is_at_last_frame_in_episode:
                self.is_concluding_episode = True
                replay_helper.increment_frame_index()
            else:
                replay_helper.increment_frame_index()

            return action
        else:
            assert self.teleoperation_action is not None
            return self.teleoperation_action

    def set_up_next_episode(self, *, is_teleoperating: bool, env):
        if is_teleoperating:
            # Sample random action (usually positional).
            sample = env.action_space.sample()
            print(f"action type={type(sample)}")

            # Assign good initial action
            sample[0] = 0.0
            sample[1] = 0.14
            sample[2] = 0.17
            sample[3] = 0.0

            print(f"init_action={sample}")
            assert env.action_space.contains(sample)

            self.teleoperation_action = sample * 1.0
        else:
            self.teleoperation_action = None

    def set_up_keyboard_teleop(self):
        print("Setting up keyboard teleop...")

        print("Keyboard controls:")
        print("up/dpwn: move hand forwards/backwards (+y/-y)")
        print("left/right: move hand left/right (-x/+x)")
        print("page_up/page_down: move hand up/down (+z/-z)")
        print("cmd/shift: Close/open gripper")
        print("delete: Discard Episode and reset env")
        print("home: Save episode and reset env")
        print("end: Save episode and end data collection")

        # TODO(samzapo): Use SharedMemoryManager.ShareableList for the following values.
        #                from multiprocessing.managers import SharedMemoryManager
        self.teleoperation_action = None
        self.is_dropping_episode = False
        self.is_concluding_episode = False
        self.is_stopping = False

        # TODO(samzapo): Launch the following code in a separate process using a Process
        #                from multiprocessing import Process
        def on_press(key):
            # Y
            if key == Key.up:
                self.teleoperation_action[1] += 0.01
            elif key == Key.down:
                self.teleoperation_action[1] -= 0.01
            # X
            elif key == Key.left:
                self.teleoperation_action[0] -= 0.01
            elif key == Key.right:
                self.teleoperation_action[0] += 0.01
            # Z
            elif key == Key.page_down:
                self.teleoperation_action[2] -= 0.01
            elif key == Key.page_up:
                self.teleoperation_action[2] += 0.01
            # Gripper
            elif key == Key.cmd:
                self.teleoperation_action[3] -= 0.1
            elif key == Key.shift:
                self.teleoperation_action[3] += 0.1
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
        self.stop_teleoperation_handler = lambda: listener.stop()

    # During data collection, all data written to disk is stored here.

    @property
    def output_data_path(self) -> str:
        # FIXME: Use DATA_DIR, if available
        print(f"DATA_DIR={lerobot.common.datasets.lerobot_dataset.DATA_DIR}")
        if lerobot.common.datasets.lerobot_dataset.DATA_DIR is None:
            lerobot.common.datasets.lerobot_dataset.DATA_DIR = pathlib.Path(
                "data_traces")
            print(
                "Warning: env variable DATA_DIR was not set, defaulting to './{}'.".format(lerobot.common.datasets.lerobot_dataset.DATA_DIR))

        dataset_index = self.dataset_counter

        return lerobot.common.datasets.lerobot_dataset.DATA_DIR / str(self.dataset_counter)

    # Where to save the HF Dataset

    @property
    def hf_dataset_path(self) -> str:
        return self.output_data_path / self.args.repo_id / "train"

    # During data collection, frames are stored here as png images.
    @property
    def images_data_path(self) -> str:
        return self.output_data_path / self.args.repo_id / "images"

    # After data collection, png images of each episode are encoded into an mp4 file stored here.
    @property
    def videos_data_path(self) -> str:
        return self.output_data_path / self.args.repo_id / "videos"

    # After data collection, meta data is stored here.
    @property
    def meta_data_path(self) -> str:
        return self.output_data_path / self.args.repo_id / "meta_data"

    def set_up_output_path(self):
        # Create image and video directories
        if not os.path.exists(self.images_data_path):
            os.makedirs(self.images_data_path, exist_ok=True)
        if not os.path.exists(self.videos_data_path):
            os.makedirs(self.videos_data_path, exist_ok=True)

    def stop_simulating(self):
        if self.stop_teleoperation_handler is not None:
            self.stop_teleoperation_handler()

    def encode_video_frames(self, *, episode_fps):
        print("encode video frames")
        for ep_idx in range(len(episode_fps)):
            for img_key in self.image_keys:
                encode_video_frames(
                    vcodec="libx265",
                    imgs_dir=self.images_data_path /
                    f"{img_key}_episode_{ep_idx:06d}",
                    video_path=self.videos_data_path /
                    f"{img_key}_episode_{ep_idx:06d}.mp4",
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

        print(f"features={features}")
        print(f"data_dict={data_dict}")

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
            videos_dir=self.videos_data_path,
        )

        print("compute stats")
        stats = compute_stats(
            lerobot_dataset, num_workers=self.args.num_workers)

        print("save to disk")
        # to remove transforms that cant be saved
        hf_dataset = hf_dataset.with_format(None)
        hf_dataset.save_to_disk(self.hf_dataset_path)

        save_meta_data(info, stats, episode_data_index, self.meta_data_path)

        return lerobot_dataset

    def maybe_push_to_hub(self, *, hf_dataset, repo_id: str, revision: str):
        if not self.args.push_to_hub:
            return

        print(f"Pushing dataset to '{repo_id}'")
        hf_dataset.push_to_hub(repo_id, token=True, revision="main")
        hf_dataset.push_to_hub(repo_id, token=True, revision=revision)

        push_meta_data_to_hub(repo_id, self.meta_data_path, revision="main")
        push_meta_data_to_hub(
            repo_id, self.meta_data_path, revision=revision)

        push_videos_to_hub(repo_id, self.videos_data_path, revision="main")
        push_videos_to_hub(repo_id, self.videos_data_path, revision=revision)

    def run_sim_while_recording_dataset(self, *, replay_helper: ReplayHelper | None = None) -> LeRobotDataset:
        self.set_up_output_path()

        ep_dicts = []
        episode_data_index = {"from": [], "to": []}
        episode_fps = []
        self.id_from = 0
        self.id_to = 0

        def save_episode_data(*, num_steps: int, episode_index: int, observations, actions, timestamps):
            ep_dict = {}
            # store images in png and create the video
            for img_key in self.image_keys:
                save_images_concurrently(
                    observations[img_key],
                    self.images_data_path /
                    f"{img_key}_episode_{episode_index:06d}",
                    self.args.num_workers,
                )
                fname = f"{img_key}_episode_{episode_index:06d}.mp4"

                # store the reference to the video frame
                ep_dict[f"observation.{img_key}"] = [
                    {"path": f"videos/{fname}", "timestamp": timestamp} for timestamp in timestamps]

            states = []
            for state_name in self.state_keys:
                states.append(np.array(observations[state_name]))
            state = torch.tensor(np.concatenate(states, axis=1))

            action = torch.tensor(np.array(actions))
            next_done = torch.zeros(num_steps, dtype=torch.bool)
            next_done[-1] = True

            ep_dict["observation.state"] = state
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor(
                [episode_index] * num_steps, dtype=torch.int64)
            ep_dict["frame_index"] = torch.arange(0, num_steps, 1)
            ep_dict["timestamp"] = torch.tensor(timestamps)
            ep_dict["next.done"] = next_done

            print(f"num_steps={num_steps}")
            print(f"timestamps[-1]={timestamps[-1]}")
            episode_fps.append(num_steps / timestamps[-1])
            ep_dicts.append(ep_dict)
            print("Episode {} done, fps: {:.2f}".format(
                episode_index, episode_fps[-1]))

            episode_data_index["from"].append(self.id_from)
            episode_data_index["to"].append(self.id_from + num_steps)

            self.id_to = self.id_from + num_steps
            self.id_from = self.id_to

        print(f"gym environment created")

        episode_counter = 0

        self.is_stopping = False
        while not self.is_stopping:
            env = self.construct_and_set_up_env()

            self.set_up_next_episode(
                is_teleoperating=(replay_helper is None), env=env)

            print(f"go {episode_counter}")

            # init buffers
            observations = {k: [] for k in env.observation_space}
            actions = []
            timestamps = []

            self.is_dropping_episode = False
            self.is_concluding_episode = False
            step_counter = 0
            while not self.is_concluding_episode:
                action = self.get_next_action_in_episode(replay_helper)

                # Apply the next action (provided by teleop)
                observation, reward, terminted, truncated, info = env.step(
                    action=action)

                # Render the simultion
                env.render()

                # store data
                for key in observation:
                    observations[key].append(copy.deepcopy(observation[key]))
                actions.append(copy.deepcopy(action))
                timestamps.append(info["timestamp"])

                step_counter += 1

            if self.is_dropping_episode:
                continue

            print(f"saving episode {episode_counter}...")

            save_episode_data(num_steps=step_counter,
                              episode_index=episode_counter,
                              observations=observations,
                              actions=actions,
                              timestamps=timestamps)

            episode_counter += 1

        self.stop_simulating()

        self.encode_video_frames(episode_fps=episode_fps)

        hf_dataset = self.construct_hf_dataset(ep_dicts=ep_dicts)
        lerobot_dataset = self.construct_lerobot_dataset_and_save_to_disk(
            hf_dataset=hf_dataset, episode_data_index=episode_data_index, episode_fps=episode_fps)

        self.maybe_push_to_hub(hf_dataset=hf_dataset,
                               repo_id=self.args.repo_id,
                               revision=self.args.revision)

        return lerobot_dataset

    def teleop_robot_and_record_data(self) -> LeRobotDataset:
        new_dataset = self.run_sim_while_recording_dataset()
        return new_dataset

    def replay_dataset_actions_in_sim(self, *, source_dataset: LeRobotDataset | None = None) -> LeRobotDataset:
        if source_dataset is None:
            source_dataset = LeRobotDataset(
                repo_id=self.args.repo_id,
                root=self.prev_output_data_path
            )

        new_dataset = self.run_sim_while_recording_dataset(
            replay_helper=ReplayHelper(lerobot_datadet=source_dataset))
        return new_dataset

    def augment_sim_parameters(self, new_sim_parameters: dict):
        for key, value in new_sim_parameters.items():
            self.sim_parameters[key] = value


def import_module_contining_gym(module_name: str):
    # import the gym module containing the environment
    try:
        # because we want to import using a variable, do it this way
        module_obj = __import__(module_name)
        # create a global object containging our module
        globals()[module_name] = module_obj
        return module_obj
    except ImportError:
        sys.stderr.write("ERROR: missing python module: " +
                         module_name + "\n")
        sys.exit(1)


if __name__ == "__main__":
    args = process_args()

    module_obj = import_module_contining_gym(args.module_name)
    print(f"Imported python module: '{args.module_name}'")
    assert hasattr(
        module_obj, 'ASSETS_PATH'), "Module should have the 'ASSETS_PATH' attribute!"

    runner = Runner(args)

    runner.augment_sim_parameters({
        "manipulands": [
            f"{module_obj.ASSETS_PATH}/red_cube.sdf",
        ],
    })
    lerobot_dataset = runner.teleop_robot_and_record_data()

    print("Replaying from dataset (in memory)")
    runner.increment_dataset_counter()
    runner.augment_sim_parameters({
        "manipulands": [
            f"{module_obj.ASSETS_PATH}/blue_cube.sdf",
        ],
    })
    runner.replay_dataset_actions_in_sim(source_dataset=lerobot_dataset)

    print("Replaying from previous dataset (from disk)")
    runner.increment_dataset_counter()
    runner.augment_sim_parameters({
        "manipulands": [
            f"{module_obj.ASSETS_PATH}/green_cube.sdf",
        ],
    })
    runner.replay_dataset_actions_in_sim()
