"""This script demonstrates how to record a LeRobot dataset of training data using a very simple gym environment.

Open your browser to: http://localhost:7000/ for visualization.

run
python3 lerobot/scripts/record_and_replay_teleop_dataset.py

Control with arrows, to end recording hit END.
Look in 'data_traces/{1,2,3}/<repo-id>' for videos

The following videos should result:
1) In "data_traces/1/<repo-id>/videos" recordings of the teleop session with a red block.
2) In "data_traces/2/<repo-id>/videos" recordings of a replay of session 1 (from disk) with a blue block.
3) In "data_traces/3/<repo-id>/videos" recordings of a replay of session 1 (from program memory) with a green block.

"""

import array
from turtle import position
from typing import NamedTuple
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
import logging

import gymnasium as gym
import numpy as np
import lerobot.common
import lerobot.common.utils
import lerobot.common.utils.utils
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

# Predeclarations
args = None
output_helper = None
episode_state = None

##########################################################################################
# Reading and writing datasets


class ReplayHelper:
    def __init__(self, *, lerobot_dataset: pathlib.PosixPath | LeRobotDataset):
        if isinstance(lerobot_dataset, pathlib.PosixPath):
            self.dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=lerobot_dataset
            )
        elif isinstance(lerobot_dataset, LeRobotDataset):
            self.dataset = lerobot_dataset
        else:
            raise ValueError(
                "Input value `lerobot_dataset` has an unexpected type: {}".format(type(lerobot_dataset)))

        print("dataset={}".format(self.dataset))
        self.frame_index = 0

    def increment_frame_index(self):
        self.frame_index += 1

    def get_action_at_frame(self):
        return self.dataset[self.frame_index]["action"]

    def get_state_observation_at_next_frame(self):
        if self.is_at_last_frame_in_dataset:
            return None
        if self.is_at_last_frame_in_episode:
            return None
        return self.dataset[self.frame_index+1]["observation.state"]

    @property
    def is_at_last_frame_in_episode(self):
        return self.dataset[self.frame_index]["episode_index"] != self.dataset[self.frame_index + 1]["episode_index"]

    @property
    def is_at_last_frame_in_dataset(self):
        return self.dataset.num_samples-1 == self.frame_index


class OutputHelper:
    def __init__(self, *, repo_id):
        self.repo_id = repo_id
        self.dataset_counter = -1

    def maybe_set_up_data_keys(self, observation):
        if len(self.image_keys) + len(self.state_keys) == 0:
            self.set_up_data_keys(observation)

    def set_up_data_keys(self, observation):
        """
        Preconditions:
            - len(self.state_keys) == 0
            - len(self.image_keys) == 0
        """
        assert len(self.state_keys) == 0
        assert len(self.image_keys) == 0
        for key, value in observation.items():
            if len(value.shape) == 1:
                self.state_keys.append(key)
            elif len(value.shape) == 3:
                self.image_keys.append(key)
            else:
                raise ValueError(
                    "{} has unrecognised shape {}!".format(key, value.shape))

    def reset_dataset_temp_data(self):
        self.ep_dicts = []
        self.episode_data_index = {"from": [], "to": []}
        self.episode_fps = []
        self.id_from = 0
        self.id_to = 0
        self.image_keys = []
        self.state_keys = []

    def vectorize_state_observations(self, observations):
        states = []
        for state_name in self.state_keys:
            states.append(np.array(observations[state_name]))
        return states

    def tensorize_state_observations(self, observations):
        states = self.vectorize_state_observations(observations)
        state = torch.tensor(np.concatenate(states, axis=1))
        return state

    def save_episode_data(self, *, num_steps: int, episode_index: int, observations, actions, timestamps):
        """
        Preconditions:
            - self.dataset_counter >= 0
        """
        assert self.dataset_counter >= 0
        ep_dict = {}
        # store images in png and create the video
        for img_key in self.image_keys:
            images = observations[img_key]
            save_images_concurrently(
                images,
                self.images_data_path /
                f"{img_key}_episode_{episode_index:06d}",
                args.num_workers,
            )
            fname = f"{img_key}_episode_{episode_index:06d}.mp4"

            # store the reference to the video frame
            ep_dict[f"observation.{img_key}"] = [
                {"path": f"videos/{fname}", "timestamp": timestamp} for timestamp in timestamps]

        action = torch.tensor(np.array(actions))
        next_done = torch.zeros(num_steps, dtype=torch.bool)
        next_done[-1] = True

        ep_dict["observation.state"] = self.tensorize_state_observations(
            observations)
        ep_dict["action"] = action
        ep_dict["episode_index"] = torch.tensor(
            [episode_index] * num_steps, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_steps, 1)
        ep_dict["timestamp"] = torch.tensor(timestamps)
        ep_dict["next.done"] = next_done

        print(f"num_steps={num_steps}")
        print(f"timestamps[-1]={timestamps[-1]}")
        self.episode_fps.append(num_steps / timestamps[-1])
        self.ep_dicts.append(ep_dict)
        print("Episode {} done, fps: {:.2f}".format(
            episode_index, self.episode_fps[-1]))

        self.episode_data_index["from"].append(self.id_from)
        self.episode_data_index["to"].append(self.id_from + num_steps)

        self.id_to = self.id_from + num_steps
        self.id_from = self.id_to

    def save_data_and_maybe_push_to_hub(self) -> LeRobotDataset:
        self.encode_video_frames(episode_fps=self.episode_fps)

        hf_dataset = self.construct_hf_dataset(ep_dicts=self.ep_dicts)
        lerobot_dataset = self.construct_lerobot_dataset_and_save_to_disk(
            hf_dataset=hf_dataset, episode_data_index=self.episode_data_index, episode_fps=self.episode_fps)

        self.maybe_push_to_hub(hf_dataset=hf_dataset,
                               repo_id=args.repo_id,
                               revision=args.revision)
        return lerobot_dataset

    def increment_dataset_counter(self):
        self.dataset_counter += 1
        self.set_up_output_path()
        self.reset_dataset_temp_data()

    # During data collection, all data written to disk is stored here.
    @property
    def output_data_path(self) -> str:
        # Use DATA_DIR, if available
        if lerobot.common.datasets.lerobot_dataset.DATA_DIR is None:
            lerobot.common.datasets.lerobot_dataset.DATA_DIR = pathlib.Path(
                "data_traces")
            print(
                "Warning: env variable DATA_DIR was not set, defaulting to './{}'."
                .format(lerobot.common.datasets.lerobot_dataset.DATA_DIR))

        dataset_index = self.dataset_counter

        return lerobot.common.datasets.lerobot_dataset.DATA_DIR / str(self.dataset_counter)

    # Where to save the HF Dataset

    @property
    def hf_dataset_path(self) -> str:
        return self.output_data_path / args.repo_id / "train"

    # During data collection, frames are stored here as png images.
    @property
    def images_data_path(self) -> str:
        return self.output_data_path / args.repo_id / "images"

    # After data collection, png images of each episode are encoded into an mp4 file stored here.
    @property
    def videos_data_path(self) -> str:
        return self.output_data_path / args.repo_id / "videos"

    # After data collection, meta data is stored here.
    @property
    def meta_data_path(self) -> str:
        return self.output_data_path / args.repo_id / "meta_data"

    def set_up_output_path(self):
        # Create image and video directories
        if not os.path.exists(self.images_data_path):
            os.makedirs(self.images_data_path, exist_ok=True)
        if not os.path.exists(self.videos_data_path):
            os.makedirs(self.videos_data_path, exist_ok=True)

    def encode_video_frames(self, *, episode_fps):
        print("encode video frames")
        for ep_idx in range(len(episode_fps)):
            for img_key in self.image_keys:
                encode_video_frames(
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
            repo_id=args.repo_id,
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            info=info,
            videos_dir=self.videos_data_path,
        )

        print("compute stats")
        stats = compute_stats(
            lerobot_dataset, num_workers=args.num_workers)

        print("save to disk")
        # to remove transforms that cant be saved
        hf_dataset = hf_dataset.with_format(None)
        hf_dataset.save_to_disk(self.hf_dataset_path)

        save_meta_data(info, stats, episode_data_index, self.meta_data_path)

        return lerobot_dataset

    def maybe_push_to_hub(self, *, hf_dataset, repo_id: str, revision: str):
        if not args.push_to_hub:
            return

        print(f"Pushing dataset to '{repo_id}'")
        hf_dataset.push_to_hub(repo_id, token=True, revision="main")
        hf_dataset.push_to_hub(repo_id, token=True, revision=revision)

        push_meta_data_to_hub(repo_id, self.meta_data_path, revision="main")
        push_meta_data_to_hub(
            repo_id, self.meta_data_path, revision=revision)

        push_videos_to_hub(repo_id, self.videos_data_path, revision="main")
        push_videos_to_hub(repo_id, self.videos_data_path, revision=revision)


##########################################################################################
# Controlling the robot (teleop) and data recording

class EpisodeState():
    def __init__(self):
        self.teleoperation_action = None
        self.is_dropping_episode: bool | None = None
        self.is_concluding_episode: bool | None = None
        self.is_stopping: bool | None = None


def set_up_keyboard_teleop():
    print("Setting up keyboard teleop...")

    print("Keyboard controls:")
    print("*      UP|DOWN       : move hand  forwards|backwards (+y|-y)")
    print("*    LEFT|RIGHT      : move hand      left|right     (-x|+x)")
    print("* PAGE UP|PAGE DOWN  : move hand        up|down      (+z|-z)")
    print("*     CMD|SHIFT      : move gripper   open|close")
    print("* DELETE             : Discard Episode and reset env")
    print("* HOME               : Save episode and reset env")
    print("* END                : Save episode and end data collection")

    episode_state.teleoperation_action = None
    episode_state.is_dropping_episode = False
    episode_state.is_concluding_episode = False
    episode_state.is_stopping = False

    def on_press(key):
        if np.isnan(np.linalg.norm(episode_state.teleoperation_action)):
            # Assign good initial action
            # FIXME: This is robot-specific (Low-Cost Robot Arm)
            episode_state.teleoperation_action[0] = 0.0
            episode_state.teleoperation_action[1] = 0.14
            episode_state.teleoperation_action[2] = 0.17
            episode_state.teleoperation_action[3] = 0.0
        assert np.isfinite(np.linalg.norm(episode_state.teleoperation_action))

        # Y
        if key == Key.up:
            episode_state.teleoperation_action[1] += 0.01
        elif key == Key.down:
            episode_state.teleoperation_action[1] -= 0.01
        # X
        elif key == Key.left:
            episode_state.teleoperation_action[0] -= 0.01
        elif key == Key.right:
            episode_state.teleoperation_action[0] += 0.01
        # Z
        elif key == Key.page_down:
            episode_state.teleoperation_action[2] -= 0.01
        elif key == Key.page_up:
            episode_state.teleoperation_action[2] += 0.01
        # Gripper
        elif key == Key.ctrl:
            episode_state.teleoperation_action[3] -= 0.1
        elif key == Key.shift:
            episode_state.teleoperation_action[3] += 0.1
        # Training
        elif key == Key.delete:
            episode_state.is_dropping_episode = True
            episode_state.is_concluding_episode = True
        elif key == Key.home:
            episode_state.is_concluding_episode = True
        elif key == Key.end:
            episode_state.is_concluding_episode = True
            episode_state.is_stopping = True

    listener = keyboard.Listener(on_press=on_press)

    def start_fn():
        listener.start()

    def stop_fn():
        listener.stop()

    # assign teleoperation shut-down function.
    return [start_fn, stop_fn]


def set_up_arm_teleop():
    episode_state.teleoperation_action = None
    episode_state.is_dropping_episode = False
    episode_state.is_concluding_episode = False
    episode_state.is_stopping = False

    print("Setting up keyboard & Arm teleop...")

    print("Keyboard controls:")
    print("* DELETE             : Discard Episode and reset env")
    print("* HOME               : Save episode and reset env")
    print("* END                : Save episode and end data collection")

    episode_state.teleoperation_action = None
    episode_state.is_dropping_episode = False
    episode_state.is_concluding_episode = False
    episode_state.is_stopping = False

    def on_press(key):
        # Training
        if key == Key.delete:
            episode_state.is_dropping_episode = True
            episode_state.is_concluding_episode = True
        elif key == Key.home:
            episode_state.is_concluding_episode = True
        elif key == Key.end:
            episode_state.is_concluding_episode = True
            episode_state.is_stopping = True

    listener = keyboard.Listener(on_press=on_press)

    # Separate process to get data from arm.
    import multiprocessing

    def update_controls(joint_commands: multiprocessing.Array):
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
        leader_arm = DynamixelMotorsBus(
            port=args.leader_arm_dev,
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        )

        if not leader_arm.is_connected:
            leader_arm.connect()

        counts_to_radians = np.pi * 2. / 4096.
        tare_positions = [2048, 2048, 2048, 3072, 2048, 2048]
        axis_direction = [-1, -1, -1, 1, -1, -1]
        while True:
            positions = leader_arm.read("Present_Position")
            assert len(joint_commands) == len(positions)
            for i in range(len(joint_commands)):
                joint_commands[i] = axis_direction[i] * \
                    (positions[i] - tare_positions[i]) * counts_to_radians

    joint_commands = multiprocessing.Array('d', [0, 0, 0, 0, 0, 0])
    run_robot_process = multiprocessing.Process(
        target=update_controls, args=(joint_commands,))

    # Separate thread to copy arm data to teleop_command (this shares compute time with simulation).
    from threading import Thread

    def copy_data(joint_commands: multiprocessing.Array, is_stopping: bool):
        while not is_stopping:
            while episode_state.teleoperation_action is None:
                print(f"Waiting for teleop to start...")
                time.sleep(1.0)
            assert len(joint_commands) == len(
                episode_state.teleoperation_action)
            for i in range(len(joint_commands)):
                episode_state.teleoperation_action[i] = joint_commands[i]

            time.sleep(0.01)

    copy_data_thread = Thread(target=copy_data, args=(
        joint_commands, episode_state.is_stopping))

    def start_fn():
        listener.start()
        run_robot_process.start()
        copy_data_thread.start()

    def stop_fn():
        listener.stop()
        run_robot_process.join()
        copy_data_thread.join()
        del robot

    return [start_fn, stop_fn]


def set_up_teleop(teleop_method: str):
    start_fn = None
    stop_fn = None

    if teleop_method == "keyboard":
        (start_fn, stop_fn) = set_up_keyboard_teleop()
    elif teleop_method == "arm":
        (start_fn, stop_fn) = set_up_arm_teleop()
    else:
        raise Exception(
            "A teleoperation method must be selected (Currently only 'keyboard' teleop is suppored)!")

    if not isinstance(start_fn, types.FunctionType):
        raise ValueError(
            "start_fn is not callable but a teleoperation system was initialized.")

    if not isinstance(stop_fn, types.FunctionType):
        raise ValueError(
            "stop_fn is not callable but a teleoperation system was initialized.")

    return [start_fn, stop_fn]


##########################################################################################
# Running the task
env_kwargs = None

def construct_and_set_up_env():
    lerobot.common.utils.utils.set_global_seed(args.random_seed)
    # Create the gym environment - check the kwargs in gym_real_world/gym_environment.py
    env = gym.make(args.env_name,
                   disable_env_checker=True,
                   **env_kwargs
                   )

    return env


def get_next_action_in_episode(replay_helper: ReplayHelper | None):
    if replay_helper is not None:
        action = replay_helper.get_action_at_frame()
        state_observation = replay_helper.get_state_observation_at_next_frame()
        if replay_helper.is_at_last_frame_in_dataset:
            episode_state.is_concluding_episode = True
            episode_state.is_stopping = True
        elif replay_helper.is_at_last_frame_in_episode:
            episode_state.is_concluding_episode = True
            replay_helper.increment_frame_index()
        else:
            replay_helper.increment_frame_index()

        return [action, state_observation]
    else:
        # Wait for first command to arrive.
        while np.isnan(np.linalg.norm(episode_state.teleoperation_action)):
            print(f"Waiting for first teleoperation command to arrive...")
            time.sleep(1)

        return [episode_state.teleoperation_action, None]


def set_up_next_episode(env):
    # Reset the environment
    observation, info = env.reset()
    output_helper.maybe_set_up_data_keys(observation)

    # Set up correctly-sized action.
    episode_state.teleoperation_action = env.action_space.sample() * np.nan


def run_sim_while_recording_dataset(*, replay_helper: ReplayHelper | None = None) -> LeRobotDataset:
    output_helper.increment_dataset_counter()
    print(f"gym environment created")

    episode_counter = 0
    env = construct_and_set_up_env()

    episode_state.is_stopping = False
    while not episode_state.is_stopping:
        set_up_next_episode(env)

        print(f"Starting episode #{episode_counter}")

        # init buffers
        observations = {k: [] for k in env.observation_space}
        actions = []
        timestamps = []

        episode_state.is_dropping_episode = False
        episode_state.is_concluding_episode = False
        step_counter = 0
        while not episode_state.is_concluding_episode:
            (action, expected_state_observation) = get_next_action_in_episode(
                replay_helper)

            # Apply the next action (provided by teleop)
            observation, reward, terminted, truncated, info = env.step(
                action=action)
            t = info["timestamp"]
            print("@t={}".format(t))

            if expected_state_observation is not None:
                state_observation = torch.tensor(np.concatenate(
                    output_helper.vectorize_state_observations(observation), axis=0))
                if not torch.norm(expected_state_observation - state_observation) < 1e-6:
                    print("action=\n{}".format(action))
                    print("diff =\n{}".format(
                        expected_state_observation - state_observation))
                    print("norm(diff) =\n{}".format(torch.norm(
                        expected_state_observation - state_observation)))
                    # print("expected_state_observation ({})=\n{}".format(
                    #     type(expected_state_observation), expected_state_observation))
                    # print("state_observation ({})=\n{}".format(
                    #     type(state_observation), state_observation))
                    # raise AssertionError(
                    #     "State observations should be equal between source dataset and replay.")

            # Render the simultion
            env.render()

            # store data
            for key in observation:
                observations[key].append(copy.deepcopy(observation[key]))
            actions.append(copy.deepcopy(action))
            timestamps.append(t)

            step_counter += 1

        if episode_state.is_dropping_episode:
            continue

        print(f"saving episode {episode_counter}...")

        output_helper.save_episode_data(num_steps=step_counter,
                                        episode_index=episode_counter,
                                        observations=observations,
                                        actions=actions,
                                        timestamps=timestamps)

        episode_counter += 1

    lerobot_dataset = output_helper.save_data_and_maybe_push_to_hub()

    return lerobot_dataset


def teleop_robot_and_record_data() -> LeRobotDataset:
    (start_teleoperation_fn, stop_teleoperation_fn) = set_up_teleop(args.teleop_method)

    # start teleoperation listener
    start_teleoperation_fn()

    new_dataset = run_sim_while_recording_dataset()

    # stop teleoperation listener
    stop_teleoperation_fn()

    return new_dataset


def replay_dataset_actions_in_sim(*, source_dataset: LeRobotDataset | None = None) -> LeRobotDataset:
    # use latest recording for replay, if missing.
    if source_dataset is None:
        source_dataset = output_helper.output_data_path

    new_dataset = run_sim_while_recording_dataset(
        replay_helper=ReplayHelper(lerobot_dataset=source_dataset),
    )
    return new_dataset


def import_module_containing_gym(module_name: str):
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

    # Check that module contains necessary attributes:
    print(f"Imported python module: '{module_name}'")
    assert hasattr(
        module_obj, 'ASSETS_PATH'), "Module must have the 'ASSETS_PATH' attribute!"


def process_args():
    # parse the repo_id name via command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default="0")

    parser.add_argument("--env-name", type=str, default="PickPlaceCube-v0")
    parser.add_argument("--teleop-method", type=str, default="keyboard")
    parser.add_argument("--num-workers", type=int, default=1)

    # Arguments for pushing to HF Dataset Hub
    parser.add_argument("--repo-id", type=str, default="myrepo")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--leader-arm-dev", type=str, default="/dev/ttyACM1")
    parser.add_argument("--gym-config-yaml", type=str, default="lerobot/configs/env/koch_drake_leader_arm_teleop.yaml")
    parser.add_argument("--gym-reconfig-yamls", type=str, default="lerobot/configs/env/koch_drake_leader_arm_teleop_replay_1.yaml;lerobot/configs/env/koch_drake_leader_arm_teleop_replay_2.yaml")

    parser.add_argument(
        "--revision", type=str, default=CODEBASE_VERSION, help="Codebase version used to generate the dataset."
    )

    return parser.parse_args()


if __name__ == "__main__":
    import yaml
    from pprint import pprint 
    args = process_args()
    logging.basicConfig(level=logging.CRITICAL)

    output_helper = OutputHelper(repo_id=args.repo_id)
    episode_state = EpisodeState()

    module_name = "gym_drake_lca"
    module_obj = import_module_containing_gym(module_name)

    print("Recording initial dataset w/ teleop")
    with open(args.gym_config_yaml, 'r') as f:
        env_kwargs = yaml.safe_load(f)
    pprint(env_kwargs)
    lerobot_dataset = teleop_robot_and_record_data()

    is_first_replay = True
    for reconfig_yaml in args.gym_reconfig_yamls.split(sep=';'):
        print("Replaying from previous dataset")
        with open(reconfig_yaml, 'r') as f:
            env_kwargs = yaml.safe_load(f)
        pprint(env_kwargs)

        if is_first_replay:
            print(" -- from disk (previous run)")
            replay_dataset_actions_in_sim()
            is_first_replay = False
        else:
            # Automatically re-record datasets with different models for the cube.
            print(" -- from dataset in memory")
            replay_dataset_actions_in_sim(source_dataset=lerobot_dataset)
