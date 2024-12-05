import json
import shutil
from pathlib import Path
import torch
from datasets import Dataset, Features, Image, Sequence, Value
import cv2
from PIL import Image as PILImage
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION

from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def extract_frames_from_video(video_file, frames):
    # Open the video file
    cap = cv2.VideoCapture(str(video_file))
    imgs_array = []

    for frame_info in frames:
        timestamp = frame_info['timestamp']
        # Set the video position to the specific timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert seconds to milliseconds

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at timestamp {timestamp}")
            continue

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = PILImage.fromarray(frame_rgb)
        imgs_array.append(img)

    # Release the video capture object
    cap.release()

    return imgs_array

def load_data(raw_dir: Path, videos_dir: Path, fps: int, video: bool, data_format: str = "json"):
    data_dir = raw_dir / "episodes"
    raw_videos_dir = raw_dir / "videos"
    if data_format == "json":
        data_files = list(data_dir.glob("*.json"))
        assert len(data_files) > 0, "No JSON files found in the directory."
    else:
        pt_dir = raw_dir / "episodes"
        data_files = list(pt_dir.glob("*.pt"))
        assert len(data_files) > 0, "No PT files found in the directory."

    ep_dicts = []
    for data_file in data_files:
        if data_format == "json":
            with open(data_file, "r") as f:
                data = json.load(f)
        else:
            data = torch.load(data_file)        

        ep_dict = {}
        # Process images from video
        for cam, frames in data['observation']['images'].items():
            if len(frames) == 0:
                continue
            img_key = f"observation.image.{cam}"
            cam_name = cam.split('_')[-1]
            raw_video_file = raw_videos_dir / data_file.stem / f"{cam_name}_{data_file.stem}.mp4"
            video_file = videos_dir / f"{cam}_{data_file.stem}.mp4"
            
            cap = cv2.VideoCapture(str(raw_video_file))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()


            if video:
                #copy raw_video to videos_dir
                shutil.copy(raw_video_file, video_file)
                # Store the reference to the video frame directly
                ep_dict[img_key] = [{"path": str(video_file), "timestamp": frame['timestamp']/1000} for frame in frames]
            else:
                # Extract frames from video if needed
                imgs_array = extract_frames_from_video(raw_video_file, frames)
                ep_dict[img_key] = imgs_array

        # Process state
        state_data = data['observation']['state']
        ep_dict["observation.state"] = torch.tensor(
            [list(map(float, state['absolute_position'] + [state['gripper_position']])) for state in state_data],
            dtype=torch.float32
        )

        # Process other fields
        ep_dict["action"] = torch.tensor(data['action'], dtype=torch.float32)
        ep_dict["timestamp"] =  torch.arange(0, num_frames, 1) / fps
        ep_dict["exec_time"] = torch.tensor(data['exec_time'], dtype=torch.float32)
        ep_dict["episode_index"] = torch.tensor(data['episode_index'], dtype=torch.int32)
        ep_dict["frame_index"] = torch.tensor(data['frame_index'], dtype=torch.int32)
        ep_dict["next.done"] = torch.tensor(data['next.done'], dtype=torch.bool)

        ep_dicts.append(ep_dict)

    return concatenate_episodes(ep_dicts)


def to_hf_dataset(data_dict, video):
    features = {}

    for key in data_dict.keys():
        if "observation.image" in key:
            features[key] = VideoFrame() if video else Image()
        elif "observation.state" in key:
            features[key] = Sequence(
                length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None)
            )
        elif "action" in key:
            features[key] = Sequence(
                length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None)
            )
        elif "episode_index" in key or "frame_index" in key or "index" in key:
            features[key] = Value(dtype="int64", id=None)
        elif "timestamp" in key or "next.reward" in key:
            features[key] = Value(dtype="float32", id=None)
        elif "next.done" in key:
            features[key] = Value(dtype="bool", id=None)
        else:
            # Default to float32 if type is unknown
            features[key] = Value(dtype="float32", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    if fps is None:
        fps = 30

    data_dict = load_data(raw_dir, videos_dir, fps, video, data_format='pt')
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info
