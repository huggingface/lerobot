#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Assess the performance of video decoding in various configurations.

This script will run different video decoding benchmarks where one parameter varies at a time.
These parameters and theirs values are specified in the BENCHMARKS dict.

All of these benchmarks are evaluated within different timestamps modes corresponding to different frame-loading scenarios:
    - `1_frame`: 1 single frame is loaded.
    - `2_frames`: 2 consecutive frames are loaded.
    - `2_frames_4_space`: 2 frames separated by 4 frames are loaded.
    - `6_frames`: 6 consecutive frames are loaded.

These values are more or less arbitrary and based on possible future usage.

These benchmarks are run on the first episode of each dataset specified in DATASET_REPO_IDS.
Note: These datasets need to be image datasets, not video datasets.
"""

import json
import random
from pathlib import Path

import einops
import numpy as np
import PIL
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import (
    decode_video_frames_torchvision,
    encode_video_frames,
)
from lerobot.common.utils.benchmark import TimeBenchmark

OUTPUT_DIR = Path("outputs/video_benchmark")
DRY_RUN = False
NUM_SAMPLES = 2

DATASET_REPO_IDS = [
    "lerobot/pusht_image",
    "aliberts/aloha_mobile_shrimp_image",
    "aliberts/paris_street",
    "aliberts/kitchen",
]
TIMESTAMPS_MODES = [
    "1_frame",
    "2_frames",
    "2_frames_4_space",
    "6_frames",
]
BENCHMARKS = {
    "pixel_format": ["yuv444p", "yuv420p"],
    "codec": ["libx264"],  # TODO(aliberts): add "libaom-av1" (need to build ffmpeg with "--enable-libaom")
    "gop_size": [1, 2, 3, 4, 5, 6, 10, 15, 20, 40, 100, None],
    "crf": [0, 5, 10, 15, 20, None, 25, 30, 40, 50],
}
DECODING_BACKENDS = {
    "backend": ["pyav", "video_reader"],
}
BASE_ENCODING = {
    "pixel_format": "yuv444p",
    "codec": "libx264",
    "gop_size": 2,
    "crf": None,
}


def get_directory_size(directory: Path) -> int:
    total_size = 0
    for item in directory.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size


def load_original_frames(imgs_dir: Path, timestamps: list[float], fps: int) -> torch.Tensor:
    frames = []
    for ts in timestamps:
        idx = int(ts * fps)
        frame = PIL.Image.open(imgs_dir / f"frame_{idx:06d}.png")
        frame = torch.from_numpy(np.array(frame))
        frame = frame.type(torch.float32) / 255
        frame = einops.rearrange(frame, "h w c -> c h w")
        frames.append(frame)
    return frames


def save_first_episode(dataset: LeRobotDataset, imgs_dir: Path) -> None:
    imgs_dir.mkdir(parents=True, exist_ok=True)
    ep_num_images = dataset.episode_data_index["to"][0].item()
    hf_dataset = dataset.hf_dataset.with_format(None)

    # We only save images from the first camera
    img_keys = [key for key in hf_dataset.features if key.startswith("observation.image")]
    imgs_dataset = hf_dataset.select_columns(img_keys[0])

    for i, item in enumerate(imgs_dataset):
        img = item[img_keys[0]]
        img.save(str(imgs_dir / f"frame_{i:06d}.png"), quality=100)

        if i >= ep_num_images - 1:
            break


def sample_timestamps(timestamps_mode: str, ep_num_images: int, fps: int):
    # Start at 5 to allow for 2_frames_4_space and 6_frames
    idx = random.randint(5, ep_num_images - 1)
    match timestamps_mode:
        case "1_frame":
            frame_indexes = [idx]
        case "2_frames":
            frame_indexes = [idx - 1, idx]
        case "2_frames_4_space":
            frame_indexes = [idx - 5, idx]
        case "6_frames":
            frame_indexes = [idx - i for i in range(6)][::-1]
        case _:
            raise ValueError(timestamps_mode)

    return [idx / fps for idx in frame_indexes]


def run_video_benchmark(
    dataset: LeRobotDataset,
    output_dir: Path,
    imgs_dir: Path,
    encoding_cfg: dict,
    seed: int = 1337,
):
    fps = dataset.fps
    sum_original_frames_size_bytes = get_directory_size(imgs_dir)

    # Encode images into video
    video_path = output_dir / "episode_0.mp4"

    encode_video_frames(
        imgs_dir=imgs_dir,
        video_path=video_path,
        fps=fps,
        video_codec=encoding_cfg["codec"],
        pixel_format=encoding_cfg["pixel_format"],
        group_of_pictures_size=encoding_cfg.get("gop_size"),
        constant_rate_factor=encoding_cfg.get("crf"),
    )

    video_size_bytes = video_path.stat().st_size

    list_avg_load_time = []
    list_avg_load_time_from_images = []
    per_pixel_l2_errors = []
    psnr_values = []
    ssim_values = []
    mse_values = []

    time_benchmark = TimeBenchmark()
    random.seed(seed)
    ep_num_images = dataset.episode_data_index["to"][0].item()

    for backend in DECODING_BACKENDS:
        for timestamps_mode in TIMESTAMPS_MODES:
            for t in range(NUM_SAMPLES):
                timestamps = sample_timestamps(timestamps_mode, ep_num_images, fps)
                num_frames = len(timestamps)

                with time_benchmark:
                    frames = decode_video_frames_torchvision(
                        video_path, timestamps=timestamps, tolerance_s=1e-4, backend=backend
                    )
                list_avg_load_time.append(time_benchmark.result / num_frames)

                with time_benchmark:
                    original_frames = load_original_frames(imgs_dir, timestamps, fps)
                list_avg_load_time_from_images.append(time_benchmark.result / num_frames)

                # Estimate reconstruction error between original frames and decoded frames with various metrics
                for i, ts in enumerate(timestamps):
                    num_pixels = original_frames[i].numel()
                    per_pixel_l2_error = torch.norm(frames[i] - original_frames[i], p=2).item() / num_pixels
                    per_pixel_l2_errors.append(per_pixel_l2_error)

                    frame_np, original_frame_np = frames[i].numpy(), original_frames[i].numpy()
                    psnr_values.append(peak_signal_noise_ratio(original_frame_np, frame_np, data_range=1.0))
                    ssim_values.append(
                        structural_similarity(original_frame_np, frame_np, data_range=1.0, channel_axis=0)
                    )
                    mse_values.append(mean_squared_error(original_frame_np, frame_np))

                    # save decoded frames
                    if t == 0:
                        frame_hwc = (frames[i].permute((1, 2, 0)) * 255).type(torch.uint8).cpu().numpy()
                        PIL.Image.fromarray(frame_hwc).save(output_dir / f"frame_{i:06d}.png")

                    # save original_frames
                    idx = int(ts * fps)
                    if t == 0:
                        original_frame = PIL.Image.open(imgs_dir / f"frame_{idx:06d}.png")
                        original_frame.save(output_dir / f"original_frame_{i:06d}.png")

    image_size = tuple(dataset[0][dataset.camera_keys[0]].shape[-2:])
    avg_load_time = float(np.array(list_avg_load_time).mean())
    avg_load_time_from_images = float(np.array(list_avg_load_time_from_images).mean())
    avg_per_pixel_l2_error = float(np.array(per_pixel_l2_errors).mean())
    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))
    avg_mse = float(np.mean(mse_values))

    # Save benchmark info

    info = {
        "image_size": image_size,
        "sum_original_frames_size_bytes": sum_original_frames_size_bytes,
        "video_size_bytes": video_size_bytes,
        "avg_load_time_from_images": avg_load_time_from_images,
        "avg_load_time": avg_load_time,
        "compression_factor": sum_original_frames_size_bytes / video_size_bytes,
        "load_time_factor": avg_load_time_from_images / avg_load_time,
        "avg_per_pixel_l2_error": avg_per_pixel_l2_error,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_mse": avg_mse,
    }

    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f)

    return info


def display_markdown_table(headers, rows):
    for i, row in enumerate(rows):
        new_row = []
        for col in row:
            if col is None:
                new_col = "None"
            elif isinstance(col, float):
                new_col = f"{col:.3f}"
                if new_col == "0.000":
                    new_col = f"{col:.7f}"
            elif isinstance(col, int):
                new_col = f"{col}"
            else:
                new_col = col
            new_row.append(new_col)
        rows[i] = new_row

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---" for _ in headers]) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    markdown_table = "\n".join([header_line, separator_line] + body_lines)
    print(markdown_table)
    print()


def load_info(out_dir):
    with open(out_dir / "info.json") as f:
        info = json.load(f)
    return info


# def one_variable_study(
#     var_name: str, var_value: any, repo_id: str, dataset: LeRobotDataset, output_dir: Path
# ):
#     rows = []
#     repo_dir = output_dir / repo_id
#     study_dir = repo_dir / var_name
#     encoding_cfg = BASE_ENCODING
#     encoding_cfg[var_name] = var_value
#     study_dir = repo_dir / var_name / str(var_value)
#     run_video_benchmark(
#         dataset,
#         output_dir,
#         imgs_dir,
#         encoding_cfg,
#     )
#     info = load_info(study_dir)
#     width, height = info["image_size"][0], info["image_size"][1]
#     rows.append(
#         [
#             repo_id,
#             f"{width} x {height}",
#             var_value,
#             backend,
#             info["compression_factor"],
#             info["load_time_factor"],
#             info["avg_load_time"] * 1e3,
#             info["avg_per_pixel_l2_error"],
#             info["avg_psnr"],
#             info["avg_ssim"],
#             info["avg_mse"],
#         ]
#     )
#     display_markdown_table(headers, rows)


# def best_study(repo_ids: list, bench_dir: Path, timestamps_mode: str, dry_run: bool):
#     """Change the config once you deciced what's best based on one-variable-studies"""
#     print("**best**")
#     headers = [
#         "repo_id",
#         "image_size",
#         "compression_factor",
#         "load_time_factor",
#         "avg_load_time_ms",
#         "avg_per_pixel_l2_error",
#         "avg_psnr",
#         "avg_ssim",
#         "avg_mse",
#     ]
#     rows = []
#     for repo_id in repo_ids:
#         study_dir = bench_dir / repo_id / "best"
#         if not dry_run:
#             run_video_benchmark(study_dir, repo_id, BASE_ENCODING, timestamps_mode)
#         info = load_info(study_dir)
#         width, height = info["image_size"][0], info["image_size"][1]
#         rows.append(
#             [
#                 repo_id,
#                 f"{width} x {height}",
#                 info["compression_factor"],
#                 info["load_time_factor"],
#                 info["avg_load_time"] * 1e3,
#                 info["avg_per_pixel_l2_error"],
#                 info["avg_psnr"],
#                 info["avg_ssim"],
#                 info["avg_mse"],
#             ]
#         )
#     display_markdown_table(headers, rows)


def main(output_dir: Path = OUTPUT_DIR):
    # TODO(aliberts): args from argparse
    # columns = [
    #     "repo_id",
    #     "image_size",
    # ]
    # columns += list(BENCHMARKS.keys())
    # columns += [
    #     "decoder",
    #     "timestamps_mode",
    #     "compression_factor",
    #     "load_time_factor",
    #     "avg_load_time_ms",
    #     "avg_per_pixel_l2_error",
    #     "avg_psnr",
    #     "avg_ssim",
    #     "avg_mse",
    # ]
    # benchmark_df = pd.DataFrame([], columns=columns)
    for repo_id in DATASET_REPO_IDS:
        repo_dir = output_dir / repo_id
        repo_dir.mkdir(parents=True, exist_ok=True)

        dataset = LeRobotDataset(repo_id)
        if dataset.video:
            raise ValueError(
                f"Use only image dataset for running this benchmark. Video dataset provided: {repo_id}"
            )

        # We only use the first episode
        imgs_dir = repo_dir / "images_episode_000000"
        if not imgs_dir.exists():
            save_first_episode(dataset, imgs_dir)

        for var_name, values in BENCHMARKS.items():
            for value in values:
                study_dir = repo_dir / var_name / value
                encoding_cfg = BASE_ENCODING
                encoding_cfg[var_name] = value
                run_video_benchmark(
                    dataset,
                    study_dir,
                    imgs_dir,
                    encoding_cfg,
                )

        # best_study(DATASET_REPO_IDS, bench_dir, timestamps_mode, DRY_RUN)


if __name__ == "__main__":
    main()
