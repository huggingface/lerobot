import json
import logging
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    EPISODES_DIR,
    get_video_duration_in_s,
    get_video_size_in_mb,
    update_chunk_file_indices,
    write_info,
)
from lerobot.datasets.video_utils import concat_video_files
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

AGIBOT_FPS = 30
AGIBOT_ROBOT_TYPE = "AgiBot_A2D"
AGIBOT_FEATURES = {
    # gripper open range in mm (0 for pull open, 1 for full close)
    "observation.state.effector.position": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["left_gripper", "right_gripper"],
        },
    },
    # flange xyz in meters
    "observation.state.end.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"],
        },
    },
    # flange quaternion with xyzw
    "observation.state.end.orientation": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "axes": ["left_x", "left_y", "left_z", "left_w", "right_x", "right_y", "right_z", "right_w"],
        },
    },
    # in radians
    "observation.state.head.position": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["yaw", "pitch"],
        },
    },
    # in motor steps
    "observation.state.joint.current_value": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "axes": [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)],
        },
    },
    # same as current_value but in radians
    "observation.state.joint.position": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "axes": [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)],
        },
    },
    # pitch in radians, lift in meters
    "observation.state.waist.position": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["pitch", "lift"],
        },
    },
    # concatenation of head.position, joint.position, effector.position, waist.position
    "observation.state": {
        "dtype": "float32",
        "shape": (20,),
        "names": {
            "axes": ["head_yaw", "head_pitch"]
            + [f"left_joint_{i}" for i in range(7)]
            + ["left_gripper"]
            + [f"right_joint_{i}" for i in range(7)]
            + ["right_gripper"]
            + ["waist_pitch", "waist_lift"],
        },
    },
    # gripper open range in mm (0 for pull open, 1 for full close)
    "action.effector.position": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["left_gripper", "right_gripper"],
        },
    },
    # flange xyz in meters
    "action.end.position": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"],
        },
    },
    # flange quaternion with xyzw
    "action.end.orientation": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "axes": ["left_x", "left_y", "left_z", "left_w", "right_x", "right_y", "right_z", "right_w"],
        },
    },
    # in radians
    "action.head.position": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["yaw", "pitch"],
        },
    },
    # goal joint position in radians
    "action.joint.position": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "axes": [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)],
        },
    },
    "action.robot.velocity": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["velocity_x", "yaw_rate"],
        },
    },
    # pitch in radians, lift in meters
    "action.waist.position": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["pitch", "lift"],
        },
    },
    # concatenation of head.position, joint.position, effector.position, waist.position, robot.velocity
    "action": {
        "dtype": "float32",
        "shape": (22,),
        "names": {
            "axes": ["head_yaw", "head_pitch"]
            + [f"left_joint_{i}" for i in range(7)]
            + ["left_gripper"]
            + [f"right_joint_{i}" for i in range(7)]
            + ["right_gripper"]
            + ["waist_pitch", "waist_lift"]
            + ["velocity_x", "yaw_rate"],
        },
    },
    # episode level annotation
    "init_scene_text": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    # frame level annotation
    "action_text": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    # frame level annotation
    "skill": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
}

AGIBOT_IMAGES_FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.head_center_fisheye": {
        "dtype": "video",
        "shape": (748, 960, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.head_left_fisheye": {
        "dtype": "video",
        "shape": (748, 960, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.head_right_fisheye": {
        "dtype": "video",
        "shape": (748, 960, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.back_left_fisheye": {
        "dtype": "video",
        "shape": (748, 960, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.back_right_fisheye": {
        "dtype": "video",
        "shape": (748, 960, 3),
        "names": ["height", "width", "channel"],
    },
}


def load_info_per_task(raw_dir):
    info_per_task = {}
    task_info_dir = raw_dir / "task_info"
    for path in task_info_dir.glob("task_*.json"):
        task_index = int(path.name.replace("task_", "").replace(".json", ""))
        with open(path) as f:
            task_info = json.load(f)

        task_info = {ep["episode_id"]: ep for ep in task_info}
        info_per_task[task_index] = task_info

    return info_per_task


def create_frame_idx_to_frames_label_idx(ep_info):
    frame_idx_to_frames_label_idx = {}
    for label_idx, frames_label in enumerate(ep_info["label_info"]["action_config"]):
        for frame_idx in range(frames_label["start_frame"], frames_label["end_frame"]):
            frame_idx_to_frames_label_idx[frame_idx] = label_idx
    return frame_idx_to_frames_label_idx


def generate_lerobot_frames(raw_dir: Path, task_index: int, episode_index: int):
    r"""/!\ The frames dont contain observation.cameras.*"""
    info_per_task = load_info_per_task(raw_dir)
    ep_info = info_per_task[task_index][episode_index]
    frame_idx_to_frames_label_idx = create_frame_idx_to_frames_label_idx(ep_info)

    # Empty features are commented out.
    keys_mapping = {
        # STATE
        # "observation.state.effector.force": "state/effector/force",
        "observation.state.effector.position": "state/effector/position",
        # "observation.state.end.angular": "state/end/angular",
        "observation.state.end.position": "state/end/position",
        "observation.state.end.orientation": "state/end/orientation",
        # "observation.state.end.velocity": "state/end/velocity",
        # "observation.state.end.wrench": "state/end/wrench",
        # "observation.state.head.effort": "state/head/effort",
        "observation.state.head.position": "state/head/position",
        # "observation.state.head.velocity": "state/head/velocity",
        "observation.state.joint.current_value": "state/joint/current_value",
        # "observation.state.joint.effort": "state/joint/effort",
        "observation.state.joint.position": "state/joint/position",
        # "observation.state.joint.velocity": "state/joint/velocity",
        # "observation.state.robot.orientation": "state/robot/orientation",
        # "observation.state.robot.orientation_drift": "state/robot/orientation_drift",
        # "observation.state.robot.position": "state/robot/position",
        # "observation.state.robot.position_drift": "state/robot/position_drift",
        # "observation.state.waist.effort": "state/waist/effort",
        "observation.state.waist.position": "state/waist/position",
        # "observation.state.waist.velocity": "state/waist/velocity",
        # ----- ACTION (index are also commented out) -----
        # "action.effector.index": "action/effector/index",
        "action.effector.position": "action/effector/position",
        # "action.effector.force": "action/effector/force",
        # "action.end.index": "action/end/index",
        "action.end.position": "action/end/position",
        "action.end.orientation": "action/end/orientation",
        # "action.head.index": "action/head/index",
        "action.head.position": "action/head/position",
        # "action.joint.index": "action/joint/index",
        "action.joint.position": "action/joint/position",
        # "action.joint.effort": "action/joint/effort",
        # "action.joint.velocity": "action/joint/velocity",
        # "action.robot.index": "action/robot/index",
        # "action.robot.position": "action/robot/position",
        # "action.robot.orientation": "action/robot/orientation",
        # "action.robot.angular": "action/robot/angular",
        "action.robot.velocity": "action/robot/velocity",
        # "action.waist.index": "action/waist/index",
        "action.waist.position": "action/waist/position",
    }

    h5_path = raw_dir / f"proprio_stats/{task_index}/{episode_index}/proprio_stats.h5"
    with h5py.File(h5_path) as h5:
        num_frames = len(h5["state/joint/position"])

        for h5_key in keys_mapping.values():
            col_num_frames = h5[h5_key].shape[0]
            if col_num_frames != num_frames:
                raise ValueError(
                    f"HDF5 column '{h5_key}' is expected to have {num_frames} but has {col_num_frames}' frames instead."
                )

        for i in range(num_frames):
            # Create frame
            f = {new_key: h5[h5_key][i] for new_key, h5_key in keys_mapping.items()}

            for key in f:
                f[key] = np.array(f[key]).astype(np.float32)

            f["observation.state.end.position"] = f["observation.state.end.position"].reshape(6)
            f["observation.state.end.orientation"] = f["observation.state.end.orientation"].reshape(8)
            f["observation.state"] = np.concatenate(
                [
                    f["observation.state.head.position"],
                    f["observation.state.joint.position"][:7],  # left
                    f["observation.state.effector.position"][[0]],  # left
                    f["observation.state.joint.position"][7:],  # right
                    f["observation.state.effector.position"][[1]],  # right
                    f["observation.state.waist.position"],
                ]
            )

            f["action.end.position"] = f["action.end.position"].reshape(6)
            f["action.end.orientation"] = f["action.end.orientation"].reshape(8)
            f["action"] = np.concatenate(
                [
                    f["action.head.position"],
                    f["action.joint.position"][:7],  # left
                    f["action.effector.position"][[0]],  # left
                    f["action.joint.position"][7:],  # right
                    f["action.effector.position"][[1]],  # right
                    f["action.waist.position"],
                    f["action.robot.velocity"],
                ]
            )

            # episode level annotation
            f["task"] = ep_info["task_name"]
            f["init_scene_text"] = ep_info["init_scene_text"]

            # frame level annotation
            if i in frame_idx_to_frames_label_idx:
                frames_label_idx = frame_idx_to_frames_label_idx[i]
                frames_label = ep_info["label_info"]["action_config"][frames_label_idx]
                f["action_text"] = frames_label["action_text"]
                f["skill"] = frames_label["skill"]
            else:
                f["action_text"] = ""
                f["skill"] = ""

            yield f


def update_meta_data(
    df,
    ep_to_meta,
):
    def _update(row):
        ep_idx = row["episode_index"]
        for key, meta in ep_to_meta[ep_idx].items():
            row[f"videos/{key}/chunk_index"] = meta["chunk_index"]
            row[f"videos/{key}/file_index"] = meta["file_index"]
            row[f"videos/{key}/from_timestamp"] = meta["from_timestamp"]
            row[f"videos/{key}/to_timestamp"] = meta["to_timestamp"]
        return row

    return df.apply(_update, axis=1)


def move_videos_to_lerobot_directory(lerobot_dataset, raw_dir, task_index, episode_names):
    keys_mapping = {
        "observation.images.top_head": "head_color",
        "observation.images.hand_left": "hand_left_color",
        "observation.images.hand_right": "hand_right_color",
        "observation.images.head_center_fisheye": "head_center_fisheye_color",
        "observation.images.head_left_fisheye": "head_left_fisheye_color",
        "observation.images.head_right_fisheye": "head_right_fisheye_color",
        "observation.images.back_left_fisheye": "back_left_fisheye_color",
        "observation.images.back_right_fisheye": "back_right_fisheye_color",
    }

    # sanity check
    for key in keys_mapping:
        if key not in lerobot_dataset.meta.info["features"]:
            raise ValueError(f"Key '{key}' not found in features.")

    video_keys = keys_mapping.keys()
    chunk_idx = dict.fromkeys(video_keys, 0)
    file_idx = dict.fromkeys(video_keys, 0)
    latest_duration_in_s = dict.fromkeys(video_keys, 0)
    ep_to_meta = {}
    for ep_idx, ep_name in enumerate(episode_names):
        for key in video_keys:
            raw_videos_dir = raw_dir / f"observations/{task_index}/{ep_name}/videos"
            old_key = keys_mapping[key]
            ep_path = raw_videos_dir / f"{old_key}.mp4"
            ep_duration_in_s = get_video_duration_in_s(ep_path)

            aggr_path = lerobot_dataset.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx[key],
                file_index=file_idx[key],
            )
            if not aggr_path.exists():
                # First video
                aggr_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(ep_path), str(aggr_path))
            else:
                size_in_mb = get_video_size_in_mb(ep_path)
                aggr_size_in_mb = get_video_size_in_mb(aggr_path)

                if aggr_size_in_mb + size_in_mb >= DEFAULT_VIDEO_FILE_SIZE_IN_MB:
                    # Size limit is reached, prepare new parquet file
                    chunk_idx[key], file_idx[key] = update_chunk_file_indices(
                        chunk_idx[key], file_idx[key], DEFAULT_CHUNK_SIZE
                    )
                    aggr_path = lerobot_dataset.root / DEFAULT_VIDEO_PATH.format(
                        video_key=key,
                        chunk_index=chunk_idx[key],
                        file_index=file_idx[key],
                    )
                    aggr_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(ep_path), str(aggr_path))
                    latest_duration_in_s[key] = 0
                else:
                    # Update the existing parquet file with new rows
                    concat_video_files(
                        [aggr_path, ep_path],
                        lerobot_dataset.root,
                        key,
                        chunk_idx[key],
                        file_idx[key],
                    )

            if ep_idx not in ep_to_meta:
                ep_to_meta[ep_idx] = {}
            ep_to_meta[ep_idx][key] = {
                "chunk_index": chunk_idx[key],
                "file_index": file_idx[key],
                "from_timestamp": latest_duration_in_s[key],
                "to_timestamp": latest_duration_in_s[key] + ep_duration_in_s,
            }
            latest_duration_in_s[key] += ep_duration_in_s

    # Update episodes meta data
    for meta_path in (lerobot_dataset.root / EPISODES_DIR).glob("chunk-*/file-*.parquet"):
        df = pd.read_parquet(meta_path)
        df = update_meta_data(df, ep_to_meta)
        df.to_parquet(meta_path)


def port_agibot(
    raw_dir: Path, repo_id: str, task_index: int, episode_indices: list[int], push_to_hub: bool = False
):
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=AGIBOT_ROBOT_TYPE,
        fps=AGIBOT_FPS,
        features=AGIBOT_FEATURES,
    )

    start_time = time.time()
    num_episodes = len(episode_indices)
    logging.info(f"Number of episodes {num_episodes}")

    for i, episode_index in enumerate(episode_indices):
        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)

        logging.info(
            f"{i} / {num_episodes} episodes processed (after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)"
        )

        for frame in generate_lerobot_frames(raw_dir, task_index, episode_index):
            lerobot_dataset.add_frame(frame)

        lerobot_dataset.save_episode()
        logging.info("Save_episode")

    # Videos have already been encoded with the proper format, so we rely on hacks
    # HACK: Add extra images features
    lerobot_dataset.meta.info["features"].update(AGIBOT_IMAGES_FEATURES)
    write_info(lerobot_dataset.meta.info, lerobot_dataset.meta.root)
    move_videos_to_lerobot_directory(lerobot_dataset, raw_dir, task_index, episode_indices)

    if push_to_hub:
        lerobot_dataset.push_to_hub(
            # Add agibot tag, since it belongs to the agibot collection of datasets
            tags=["agibot"],
            private=False,
        )
