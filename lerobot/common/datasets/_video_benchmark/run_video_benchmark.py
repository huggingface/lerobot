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

import datetime as dt
import random
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import PIL
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import (
    decode_video_frames_torchvision,
    encode_video_frames,
)
from lerobot.common.utils.benchmark import TimeBenchmark

OUTPUT_DIR = Path("outputs/video_benchmark")
NUM_SAMPLES = 50

DATASET_REPO_IDS = [
    "lerobot/pusht_image",
    "aliberts/aloha_mobile_shrimp_image",
    "aliberts/paris_street",
    "aliberts/kitchen",
]
BENCHMARKS = {
    "pixel_format": ["yuv444p", "yuv420p"],
    "gop_size": [1, 2, 3, 4, 5, 6, 10, 15, 20, 40, 100, None],
    "crf": [0, 5, 10, 15, 20, None, 25, 30, 40, 50],
    # TODO(aliberts): add "libaom-av1" (need to build ffmpeg with "--enable-libaom")
    # "codec": ["libx264", "libaom-av1"],
}
BASE_ENCODING = {
    "pixel_format": "yuv444p",
    "codec": "libx264",
    "gop_size": 2,
    "crf": None,
}
TIMESTAMPS_MODES = [
    "1_frame",
    "2_frames",
    "2_frames_4_space",
    "6_frames",
]
DECODING_BACKENDS = ["pyav", "video_reader"]


def check_datasets_formats(repo_ids: list) -> None:
    for repo_id in repo_ids:
        dataset = LeRobotDataset(repo_id)
        if dataset.video:
            raise ValueError(
                f"Use only image dataset for running this benchmark. Video dataset provided: {repo_id}"
            )


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
    return torch.stack(frames)


def save_first_episode(dataset: LeRobotDataset, output_dir: Path) -> Path:
    imgs_dir = output_dir / "images" / dataset.repo_id
    ep_num_images = dataset.episode_data_index["to"][0].item()
    if imgs_dir.exists() and len(list(imgs_dir.glob("frame_*.png"))) == ep_num_images:
        return imgs_dir

    imgs_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset = dataset.hf_dataset.with_format(None)

    # We only save images from the first camera
    img_keys = [key for key in hf_dataset.features if key.startswith("observation.image")]
    imgs_dataset = hf_dataset.select_columns(img_keys[0])

    for i, item in enumerate(
        tqdm(imgs_dataset, desc=f"saving {dataset.repo_id} first episode images", leave=False)
    ):
        img = item[img_keys[0]]
        img.save(str(imgs_dir / f"frame_{i:06d}.png"), quality=100)

        if i >= ep_num_images - 1:
            break

    return imgs_dir


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


def benchmark_video_decoding(
    imgs_dir: Path,
    video_path: Path,
    timestamps_mode: str,
    backend: str,
    ep_num_images: int,
    fps: int,
    save_frames: bool = False,
) -> dict:
    load_times_video_ms = []
    load_times_images_ms = []
    mse_values = []
    psnr_values = []
    ssim_values = []
    # per_pixel_l2_errors = []

    time_benchmark = TimeBenchmark()
    for t in tqdm(range(NUM_SAMPLES), desc="samples", leave=False):
        timestamps = sample_timestamps(timestamps_mode, ep_num_images, fps)
        num_frames = len(timestamps)

        with time_benchmark:
            frames = decode_video_frames_torchvision(
                video_path, timestamps=timestamps, tolerance_s=1e-4, backend=backend
            )
        load_times_video_ms.append(time_benchmark.result_ms / num_frames)

        with time_benchmark:
            original_frames = load_original_frames(imgs_dir, timestamps, fps)
        load_times_images_ms.append(time_benchmark.result_ms / num_frames)

        # Estimate reconstruction error between original frames and decoded frames with various metrics
        frames_np, original_frames_np = frames.numpy(), original_frames.numpy()
        for i in range(num_frames):
            psnr_values.append(peak_signal_noise_ratio(original_frames_np[i], frames_np[i], data_range=1.0))
            ssim_values.append(
                structural_similarity(original_frames_np[i], frames_np[i], data_range=1.0, channel_axis=0)
            )
            mse_values.append(mean_squared_error(original_frames_np[i], frames_np[i]))

            if save_frames and t == 0:
                save_dir = video_path.parent / "saved" / timestamps_mode / backend
                save_dir.mkdir(parents=True, exist_ok=True)
                frame_hwc = (frames[i].permute((1, 2, 0)) * 255).type(torch.uint8).cpu().numpy()
                PIL.Image.fromarray(frame_hwc).save(save_dir / f"frame_{i:06d}.png")

    avg_load_time_video_ms = float(np.array(load_times_video_ms).mean())
    avg_load_time_images_ms = float(np.array(load_times_images_ms).mean())
    video_images_load_time_ratio = avg_load_time_video_ms / avg_load_time_images_ms

    return {
        "avg_load_time_video_ms": avg_load_time_video_ms,
        "avg_load_time_images_ms": avg_load_time_images_ms,
        "video_images_load_time_ratio": video_images_load_time_ratio,
        "avg_mse": float(np.mean(mse_values)),
        "avg_psnr": float(np.mean(psnr_values)),
        "avg_ssim": float(np.mean(ssim_values)),
    }


def run_video_benchmark(
    dataset: LeRobotDataset,
    video_path: Path,
    imgs_dir: Path,
    encoding_cfg: dict,
    overwrite: bool = False,
    seed: int = 1337,
):
    fps = dataset.fps

    if overwrite or not video_path.is_file():
        tqdm.write(f"encoding {video_path}")
        encode_video_frames(
            imgs_dir=imgs_dir,
            video_path=video_path,
            fps=fps,
            video_codec=encoding_cfg["codec"],
            pixel_format=encoding_cfg["pixel_format"],
            group_of_pictures_size=encoding_cfg.get("gop_size"),
            constant_rate_factor=encoding_cfg.get("crf"),
            overwrite=True,
        )

    ep_num_images = dataset.episode_data_index["to"][0].item()
    width, height = tuple(dataset[0][dataset.camera_keys[0]].shape[-2:])
    num_pixels = width * height
    video_size_bytes = video_path.stat().st_size
    images_size_bytes = get_directory_size(imgs_dir)
    video_images_size_ratio = video_size_bytes / images_size_bytes

    random.seed(seed)
    benchmark_table = []
    for timestamps_mode in tqdm(TIMESTAMPS_MODES, desc="timestamps modes", leave=False):
        for backend in tqdm(DECODING_BACKENDS, desc="backends", leave=False):
            benchmark_row = benchmark_video_decoding(
                imgs_dir, video_path, timestamps_mode, backend, ep_num_images, fps
            )
            benchmark_row.update(
                **{
                    "repo_id": dataset.repo_id,
                    "resolution": f"{width} x {height}",
                    "num_pixels": num_pixels,
                    "video_size_bytes": video_size_bytes,
                    "images_size_bytes": images_size_bytes,
                    "video_images_size_ratio": video_images_size_ratio,
                    "timestamps_mode": timestamps_mode,
                    "backend": backend,
                },
                **encoding_cfg,
            )
            benchmark_table.append(benchmark_row)

    return benchmark_table


def main(output_dir: Path = OUTPUT_DIR):
    # TODO(aliberts): args from argparse
    check_datasets_formats(DATASET_REPO_IDS)
    benchmark_table = []
    for repo_id in tqdm(DATASET_REPO_IDS, desc="datasets"):
        # We only use the first episode
        dataset = LeRobotDataset(repo_id)
        imgs_dir = save_first_episode(dataset, output_dir)

        for var_name, values in tqdm(BENCHMARKS.items(), desc="encodings", leave=False):
            for value in values:
                video_path = output_dir / "videos" / repo_id / var_name / f"{value}.mp4"
                encoding_cfg = BASE_ENCODING.copy()
                encoding_cfg[var_name] = value
                benchmark_table += run_video_benchmark(dataset, video_path, imgs_dir, encoding_cfg)

    columns_order = ["repo_id", "resolution", "num_pixels"]
    columns_order += list(BASE_ENCODING.keys())
    columns_order += [
        "video_size_bytes",
        "images_size_bytes",
        "video_images_size_ratio",
        "timestamps_mode",
        "backend",
        "avg_load_time_video_ms",
        "avg_load_time_images_ms",
        "video_images_load_time_ratio",
        "avg_mse",
        "avg_psnr",
        "avg_ssim",
    ]
    benchmark_df = pd.DataFrame(benchmark_table, columns=columns_order)
    now = dt.datetime.now()
    csv_path = output_dir / f"{now:%Y-%m-%d}_{now:%H-%M-%S}_{NUM_SAMPLES}-samples.csv"
    benchmark_df.to_csv(csv_path, header=True, index=False)


if __name__ == "__main__":
    main()
