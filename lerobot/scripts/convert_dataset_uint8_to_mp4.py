"""
    usage: `python lerobot/scripts/convert_dataset_uint8_to_mp4.py --in-data-dir data/pusht --out-data-dir tests/data/pusht`
"""

import argparse
import shutil
from pathlib import Path

import torch
from tensordict import TensorDict


def convert_dataset_uint8_to_mp4(in_data_dir, out_data_dir, fps, overwrite_num_frames=None):
    assert fps is not None and isinstance(fps, float)
    # load full dataset as a tensor dict
    in_td_data = TensorDict.load_memmap(in_data_dir)

    out_data_dir = Path(out_data_dir)
    # use 1 frame to know the specification of the dataset
    # and copy it over `n` frames in the test artifact directory
    out_rb_dir = out_data_dir / "replay_buffer"
    if out_rb_dir.exists():
        shutil.rmtree(out_rb_dir)

    num_frames = len(in_td_data) if overwrite_num_frames is None else overwrite_num_frames

    # del in_td_data["observation", "image"]
    # del in_td_data["next", "observation", "image"]

    out_td_data = in_td_data[0].memmap_().clone()

    out_td_data["observation", "frame", "video_id"] = torch.zeros(1, dtype=torch.int)
    out_td_data["observation", "frame", "timestamp"] = torch.zeros(1)
    out_td_data["next", "observation", "frame", "video_id"] = torch.zeros(1, dtype=torch.int)
    out_td_data["next", "observation", "frame", "timestamp"] = torch.zeros(1)

    out_td_data = out_td_data.expand(num_frames)
    out_td_data = out_td_data.memmap_like(out_rb_dir)

    out_vid_dir = out_data_dir / "videos"
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    video_id_to_path = {}

    for key in out_td_data.keys(include_nested=True, leaves_only=True):
        if in_td_data.get(key, None) is None:
            continue
        if overwrite_num_frames is None:
            out_td_data[key].copy_(in_td_data[key].clone())
        else:
            out_td_data[key][:num_frames].copy_(in_td_data[key][:num_frames].clone())

    for i in range(num_frames):
        video_id = in_td_data["episode"][i]
        frame_id = in_td_data["frame_id"][i]

        out_td_data["observation", "frame", "video_id"][i] = video_id
        out_td_data["observation", "frame", "timestamp"][i] = frame_id / fps
        out_td_data["next", "observation", "frame", "video_id"][i] = video_id
        out_td_data["next", "observation", "frame", "timestamp"][i] = (frame_id + 1) / fps

        video_id = video_id.item()
        if video_id not in video_id_to_path:
            video_id_to_path[video_id] = f"videos/episode_{video_id}.mp4"

    # copy the first `n` frames so that we have real data

    # make sure everything has been properly written
    out_td_data.lock_()

    # copy the full statistics of dataset since it's pretty small
    in_stats_path = Path(in_data_dir) / "stats.pth"

    out_stats_path = Path(out_data_dir) / "stats.pth"
    shutil.copy(in_stats_path, out_stats_path)

    meta_data = {
        "video_id_to_path": video_id_to_path,
    }
    torch.save(meta_data, out_data_dir / "meta_data.pth")


# def write_to_mp4():
#     buffer = io.BytesIO()
#     swriter = StreamWriter(buffer, format="mp4")

#     device = "cuda"

#     c,h,w = in_td_data[0]["observation", "image"].shape

#     swriter.add_video_stream(
#         frame_rate=fps,
#         width=w,
#         height=h,
#         # frame_rate=30000 / 1001,
#         format="yuv444p",
#         encoder="h264_nvenc",
#         encoder_format="yuv444p",
#         hw_accel=device,
#     )

#     for i in range(num_frames):
#         ep_id = in_td_data[i]["episode"]
#         data = in_td_data[i]["observation", "image"]
#         with swriter.open():
#             t0 = time.monotonic()
#             data = data.to(device)
#             swriter.write_video_chunk(0, data)
#             elapsed = time.monotonic() - t0
#             size = buffer.tell()
#             print(f"{elapsed=}")
#             print(f"{size=}")
#             buffer.seek(0)
#         video = buffer.read()

#         vid_path = out_vid_dir / f"episode_{ep_id}.mp4"
#         with open(vid_path, 'wb+') as f:
#             f.write(video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset")

    parser.add_argument("--in-data-dir", type=str, help="Path to input data")
    parser.add_argument("--out-data-dir", type=str, help="Path to save the output data")
    parser.add_argument("--fps", type=float)

    args = parser.parse_args()

    convert_dataset_uint8_to_mp4(args.in_data_dir, args.out_data_dir, args.fps)
