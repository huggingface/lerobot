import itertools
import subprocess
from pathlib import Path

import torch
import torchvision


def load_from_videos(item, video_frame_keys, videos_dir):
    for key in video_frame_keys:
        ep_idx = item["episode_index"]
        video_path = videos_dir / key / f"episode_{ep_idx:06d}.mp4"

        if f"{key}_timestamp" in item:
            # load multiple frames at once
            timestamps = item[f"{key}_timestamp"]
            item[key] = decode_video_frames_torchvision(video_path, timestamps)
        else:
            # load one frame
            timestamps = [item["timestamp"]]
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
    reader = torchvision.io.VideoReader(str(video_path), "video")

    # sanity preprocessing (e.g. 3.60000003 -> 3.6)
    timestamps = [round(ts, 4) for ts in timestamps]

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # access key frame of first requested frame, and load all frames until last requested frame
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    frames = []
    for frame in itertools.takewhile(lambda x: x["pts"] <= last_ts, reader.seek(first_ts)):
        # get timestamp of the loaded frame
        ts = frame["pts"]

        # if the loaded frame is not among the requested frames, we dont add it to the list of output frames
        is_frame_requested = ts in timestamps
        if is_frame_requested:
            frames.append(frame["data"])

        if log_loaded_timestamps:
            log = f"frame loaded at timestamp={ts:.4f}"
            if is_frame_requested:
                log += " requested"
            print(log)

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
        f"ffmpeg -r {fps} -f image2 "
        f"-i {str(imgs_dir / 'frame_%06d.png')} "
        "-vcodec libx264 "
        "-pix_fmt yuv444p "
        f"{str(video_path)}"
    )
    subprocess.run(ffmpeg_cmd.split(" "), check=True)
