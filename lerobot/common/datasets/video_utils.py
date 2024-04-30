import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature


def load_from_videos(item, video_frame_keys, videos_dir):
    # since video path already contains "videos" (e.g. videos_dir="data/videos", path="videos/episode_0.mp4")
    data_dir = videos_dir.parent

    for key in video_frame_keys:
        ep_idx = item["episode_index"]
        video_path = data_dir / key / f"episode_{ep_idx:06d}.mp4"

        if isinstance(item[key], list):
            # load multiple frames at once
            timestamps = [frame["timestamp"] for frame in item[key]]
            paths = [frame["path"] for frame in item[key]]
            if len(set(paths)) == 1:
                raise NotImplementedError("All video paths are expected to be the same for now.")
            video_path = data_dir / paths[0]

            frames = decode_video_frames_torchvision(video_path, timestamps)
            assert len(frames) == len(timestamps)

            item[key] = frames
        else:
            # load one frame
            timestamps = [item[key]["timestamp"]]
            video_path = data_dir / item[key]["path"]

            frames = decode_video_frames_torchvision(video_path, timestamps)
            assert len(frames) == 1

            item[key] = frames[0]

    return item


def decode_video_frames_torchvision(
    video_path: str, timestamps: list[float], device: str = "cpu", log_loaded_timestamps: bool = False
):
    """Loads frames associated to the requested timestamps of a video

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceeding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    if device == "cpu":
        # explicitely use pyav
        torchvision.set_video_backend("pyav")
    elif device == "cuda":
        # TODO(rcadene, aliberts): implement video decoding with GPU
        # torchvision.set_video_backend("cuda")
        # torchvision.set_video_backend("video_reader")
        # requires installing torchvision from source, see: https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
        # check possible bug: https://github.com/pytorch/vision/issues/7745
        raise NotImplementedError()
    else:
        raise ValueError(device)

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    def round_timestamp(ts):
        # sanity preprocessing (e.g. 3.60000003 -> 3.6000, 0.0666666667 -> 0.0667)
        return round(ts, 4)

    timestamps = [round_timestamp(ts) for ts in timestamps]

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # access key frame of first requested frame, and load all frames until last requested frame
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts)
    frames = []
    for frame in reader:
        # get timestamp of the loaded frame
        ts = round_timestamp(frame["pts"])

        # if the loaded frame is not among the requested frames, we dont add it to the list of output frames
        is_frame_requested = ts in timestamps
        if is_frame_requested:
            frames.append(frame["data"])

        if log_loaded_timestamps:
            log = f"frame loaded at timestamp={ts:.4f}"
            if is_frame_requested:
                log += " requested"
            logging.info(log)

        if len(timestamps) == len(frames):
            break

        # hard stop
        assert (
            frame["pts"] >= last_ts
        ), f"Not enough frames have been loaded in [{first_ts}, {last_ts}]. {len(timestamps)} expected, but only {len(frames)} loaded."

    frames = torch.stack(frames)

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    frames = frames.type(torch.float32) / 255

    assert len(timestamps) == frames.shape[0]
    return frames


def encode_video_frames(imgs_dir: Path, video_path: Path, fps: int):
    # For more info this setting, see: `lerobot/common/datasets/_video_benchmark/README.md`
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = (
        f"ffmpeg -r {fps} "
        "-f image2 "
        "-loglevel error "
        f"-i {str(imgs_dir / 'frame_%06d.png')} "
        "-vcodec libx264 "
        "-pix_fmt yuv444p "
        f"{str(video_path)}"
    )
    subprocess.run(ffmpeg_cmd.split(" "), check=True)


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


# to make it available in HuggingFace `datasets`
register_feature(VideoFrame, "VideoFrame")
