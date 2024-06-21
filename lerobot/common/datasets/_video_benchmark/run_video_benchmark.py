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
import shutil
import subprocess
import time
from pathlib import Path

import einops
import numpy as np
import PIL
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import (
    decode_video_frames_torchvision,
)

OUTPUT_DIR = Path("tmp/run_video_benchmark")
DRY_RUN = False

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
    # "pix_fmt": ["yuv420p", "yuv444p"],
    # "g": [1, 2, 3, 4, 5, 6, 10, 15, 20, 40, 100, None],
    # "crf": [0, 5, 10, 15, 20, None, 25, 30, 40, 50],
    "backend": ["pyav", "video_reader"],
}


def get_directory_size(directory):
    total_size = 0
    # Iterate over all files and subdirectories recursively
    for item in directory.rglob("*"):
        if item.is_file():
            # Add the file size to the total
            total_size += item.stat().st_size
    return total_size


def run_video_benchmark(
    output_dir,
    cfg,
    timestamps_mode,
    seed=1337,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = cfg["repo_id"]

    # TODO(rcadene): rewrite with hardcoding of original images and episodes
    dataset = LeRobotDataset(repo_id)
    if dataset.video:
        raise ValueError(
            f"Use only image dataset for running this benchmark. Video dataset provided: {repo_id}"
        )

    # Get fps
    fps = dataset.fps

    # we only load first episode
    ep_num_images = dataset.episode_data_index["to"][0].item()

    # Save/Load image directory for the first episode
    imgs_dir = Path(f"tmp/data/images/{repo_id}/observation.image_episode_000000")
    if not imgs_dir.exists():
        imgs_dir.mkdir(parents=True, exist_ok=True)
        hf_dataset = dataset.hf_dataset.with_format(None)
        img_keys = [key for key in hf_dataset.features if key.startswith("observation.image")]
        imgs_dataset = hf_dataset.select_columns(img_keys[0])

        for i, item in enumerate(imgs_dataset):
            img = item[img_keys[0]]
            img.save(str(imgs_dir / f"frame_{i:06d}.png"), quality=100)

            if i >= ep_num_images - 1:
                break

    sum_original_frames_size_bytes = get_directory_size(imgs_dir)

    # Encode images into video
    video_path = output_dir / "episode_0.mp4"

    g = cfg.get("g")
    crf = cfg.get("crf")
    pix_fmt = cfg["pix_fmt"]

    cmd = f"ffmpeg -r {fps} "
    cmd += "-f image2 "
    cmd += "-loglevel error "
    cmd += f"-i {str(imgs_dir / 'frame_%06d.png')} "
    cmd += "-vcodec libx264 "
    if g is not None:
        cmd += f"-g {g} "  # ensures at least 1 keyframe every 10 frames
    # cmd += "-keyint_min 10 " set a minimum of 10 frames between 2 key frames
    # cmd += "-sc_threshold 0 " disable scene change detection to lower the number of key frames
    if crf is not None:
        cmd += f"-crf {crf} "
    cmd += f"-pix_fmt {pix_fmt} "
    cmd += f"{str(video_path)}"
    subprocess.run(cmd.split(" "), check=True)

    video_size_bytes = video_path.stat().st_size

    # Set decoder

    decoder = cfg["decoder"]
    decoder_kwgs = cfg["decoder_kwgs"]
    backend = cfg["backend"]

    if decoder == "torchvision":
        decode_frames_fn = decode_video_frames_torchvision
    else:
        raise ValueError(decoder)

    # Estimate average loading time

    def load_original_frames(imgs_dir, timestamps) -> torch.Tensor:
        frames = []
        for ts in timestamps:
            idx = int(ts * fps)
            frame = PIL.Image.open(imgs_dir / f"frame_{idx:06d}.png")
            frame = torch.from_numpy(np.array(frame))
            frame = frame.type(torch.float32) / 255
            frame = einops.rearrange(frame, "h w c -> c h w")
            frames.append(frame)
        return frames

    list_avg_load_time = []
    list_avg_load_time_from_images = []
    per_pixel_l2_errors = []
    psnr_values = []
    ssim_values = []
    mse_values = []

    random.seed(seed)

    for t in range(50):
        # test loading 2 frames that are 4 frames appart, which might be a common setting
        ts = random.randint(fps, ep_num_images - fps) / fps

        if timestamps_mode == "1_frame":
            timestamps = [ts]
        elif timestamps_mode == "2_frames":
            timestamps = [ts - 1 / fps, ts]
        elif timestamps_mode == "2_frames_4_space":
            timestamps = [ts - 5 / fps, ts]
        elif timestamps_mode == "6_frames":
            timestamps = [ts - i / fps for i in range(6)][::-1]
        else:
            raise ValueError(timestamps_mode)

        num_frames = len(timestamps)

        start_time_s = time.monotonic()
        frames = decode_frames_fn(
            video_path, timestamps=timestamps, tolerance_s=1e-4, backend=backend, **decoder_kwgs
        )
        avg_load_time = (time.monotonic() - start_time_s) / num_frames
        list_avg_load_time.append(avg_load_time)

        start_time_s = time.monotonic()
        original_frames = load_original_frames(imgs_dir, timestamps)
        avg_load_time_from_images = (time.monotonic() - start_time_s) / num_frames
        list_avg_load_time_from_images.append(avg_load_time_from_images)

        # Estimate reconstruction error between original frames and decoded frames with various metrics
        for i, ts in enumerate(timestamps):
            # are_close = torch.allclose(frames[i], original_frames[i], atol=0.02)
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


def one_variable_study(
    var_name: str, var_values: list, repo_ids: list, bench_dir: Path, timestamps_mode: str, dry_run: bool
):
    print(f"**`{var_name}`**")
    headers = [
        "repo_id",
        "image_size",
        var_name,
        "compression_factor",
        "load_time_factor",
        "avg_per_pixel_l2_error",
        "avg_psnr",
        "avg_ssim",
        "avg_mse",
    ]
    rows = []
    base_cfg = {
        "repo_id": None,
        # video encoding
        "g": 2,
        "crf": None,
        "pix_fmt": "yuv444p",
        # video decoding
        "backend": "pyav",
        "decoder": "torchvision",
        "decoder_kwgs": {},
    }
    for repo_id in repo_ids:
        for val in var_values:
            cfg = base_cfg.copy()
            cfg["repo_id"] = repo_id
            cfg[var_name] = val
            if not dry_run:
                run_video_benchmark(
                    bench_dir / repo_id / f"torchvision_{var_name}_{val}", cfg, timestamps_mode
                )
            info = load_info(bench_dir / repo_id / f"torchvision_{var_name}_{val}")
            width, height = info["image_size"][0], info["image_size"][1]
            rows.append(
                [
                    repo_id,
                    f"{width} x {height}",
                    val,
                    info["compression_factor"],
                    info["load_time_factor"],
                    info["avg_per_pixel_l2_error"],
                    info["avg_psnr"],
                    info["avg_ssim"],
                    info["avg_mse"],
                ]
            )
    display_markdown_table(headers, rows)


def best_study(repo_ids: list, bench_dir: Path, timestamps_mode: str, dry_run: bool):
    """Change the config once you deciced what's best based on one-variable-studies"""
    print("**best**")
    headers = [
        "repo_id",
        "image_size",
        "compression_factor",
        "load_time_factor",
        "avg_per_pixel_l2_error",
        "avg_psnr",
        "avg_ssim",
        "avg_mse",
    ]
    rows = []
    for repo_id in repo_ids:
        cfg = {
            "repo_id": repo_id,
            # video encoding
            "g": 2,
            "crf": None,
            "pix_fmt": "yuv444p",
            # video decoding
            "backend": "video_reader",
            "decoder": "torchvision",
            "decoder_kwgs": {},
        }
        if not dry_run:
            run_video_benchmark(bench_dir / repo_id / "torchvision_best", cfg, timestamps_mode)
        info = load_info(bench_dir / repo_id / "torchvision_best")
        width, height = info["image_size"][0], info["image_size"][1]
        rows.append(
            [
                repo_id,
                f"{width} x {height}",
                info["compression_factor"],
                info["load_time_factor"],
                info["avg_per_pixel_l2_error"],
            ]
        )
    display_markdown_table(headers, rows)


def main():
    for timestamps_mode in TIMESTAMPS_MODES:
        bench_dir = OUTPUT_DIR / timestamps_mode

        print(f"### `{timestamps_mode}`")
        print()

        for name, values in BENCHMARKS.items():
            one_variable_study(name, values, DATASET_REPO_IDS, bench_dir, timestamps_mode, DRY_RUN)

        # best_study(DATASET_REPO_IDS, bench_dir, timestamps_mode, DRY_RUN)


if __name__ == "__main__":
    main()
