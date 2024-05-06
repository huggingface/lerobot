import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import einops
import numpy
import PIL
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import (
    decode_video_frames_torchvision,
)


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

    # Get fps
    fps = dataset.fps

    # we only load first episode
    ep_num_images = dataset.episode_data_index["to"][0].item()

    # Save/Load image directory for the first episode
    imgs_dir = Path(f"tmp/data/images/{repo_id}/observation.image_episode_000000")
    if not imgs_dir.exists():
        imgs_dir.mkdir(parents=True, exist_ok=True)
        hf_dataset = dataset.hf_dataset.with_format(None)
        imgs_dataset = hf_dataset.select_columns("observation.image")

        for i, item in enumerate(imgs_dataset):
            img = item["observation.image"]
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
    device = cfg["device"]

    if decoder == "torchvision":
        decode_frames_fn = decode_video_frames_torchvision
    else:
        raise ValueError(decoder)

    # Estimate average loading time

    def load_original_frames(imgs_dir, timestamps):
        frames = []
        for ts in timestamps:
            idx = int(ts * fps)
            frame = PIL.Image.open(imgs_dir / f"frame_{idx:06d}.png")
            frame = torch.from_numpy(numpy.array(frame))
            frame = frame.type(torch.float32) / 255
            frame = einops.rearrange(frame, "h w c -> c h w")
            frames.append(frame)
        return frames

    list_avg_load_time = []
    list_avg_load_time_from_images = []
    per_pixel_l2_errors = []

    random.seed(seed)

    for t in range(50):
        # test loading 2 frames that are 4 frames appart, which might be a common setting
        ts = random.randint(fps, ep_num_images - fps) / fps

        if timestamps_mode == "1_frame":
            timestamps = [ts]
        elif timestamps_mode == "2_frames":
            timestamps = [ts - 1 / fps, ts]
        elif timestamps_mode == "2_frames_4_space":
            timestamps = [ts - 4 / fps, ts]
        elif timestamps_mode == "6_frames":
            timestamps = [ts - i / fps for i in range(6)][::-1]
        else:
            raise ValueError(timestamps_mode)

        num_frames = len(timestamps)

        start_time_s = time.monotonic()
        frames = decode_frames_fn(
            video_path, timestamps=timestamps, tolerance_s=1e-4, device=device, **decoder_kwgs
        )
        avg_load_time = (time.monotonic() - start_time_s) / num_frames
        list_avg_load_time.append(avg_load_time)

        start_time_s = time.monotonic()
        original_frames = load_original_frames(imgs_dir, timestamps)
        avg_load_time_from_images = (time.monotonic() - start_time_s) / num_frames
        list_avg_load_time_from_images.append(avg_load_time_from_images)

        # Estimate average L2 error between original frames and decoded frames
        for i, ts in enumerate(timestamps):
            # are_close = torch.allclose(frames[i], original_frames[i], atol=0.02)
            num_pixels = original_frames[i].numel()
            per_pixel_l2_error = torch.norm(frames[i] - original_frames[i], p=2).item() / num_pixels

            # save decoded frames
            if t == 0:
                frame_hwc = (frames[i].permute((1, 2, 0)) * 255).type(torch.uint8).cpu().numpy()
                PIL.Image.fromarray(frame_hwc).save(output_dir / f"frame_{i:06d}.png")

            # save original_frames
            idx = int(ts * fps)
            if t == 0:
                original_frame = PIL.Image.open(imgs_dir / f"frame_{idx:06d}.png")
                original_frame.save(output_dir / f"original_frame_{i:06d}.png")

            per_pixel_l2_errors.append(per_pixel_l2_error)

    avg_load_time = float(numpy.array(list_avg_load_time).mean())
    avg_load_time_from_images = float(numpy.array(list_avg_load_time_from_images).mean())
    avg_per_pixel_l2_error = float(numpy.array(per_pixel_l2_errors).mean())

    # Save benchmark info

    info = {
        "sum_original_frames_size_bytes": sum_original_frames_size_bytes,
        "video_size_bytes": video_size_bytes,
        "avg_load_time_from_images": avg_load_time_from_images,
        "avg_load_time": avg_load_time,
        "compression_factor": sum_original_frames_size_bytes / video_size_bytes,
        "load_time_factor": avg_load_time_from_images / avg_load_time,
        "avg_per_pixel_l2_error": avg_per_pixel_l2_error,
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


def main():
    out_dir = Path("tmp/run_video_benchmark")
    dry_run = False
    repo_ids = ["lerobot/pusht", "lerobot/umi_cup_in_the_wild"]
    timestamps_modes = [
        "1_frame",
        "2_frames",
        "2_frames_4_space",
        "6_frames",
    ]
    for timestamps_mode in timestamps_modes:
        bench_dir = out_dir / timestamps_mode

        print(f"### `{timestamps_mode}`")
        print()

        print("**`pix_fmt`**")
        headers = ["repo_id", "pix_fmt", "compression_factor", "load_time_factor", "avg_per_pixel_l2_error"]
        rows = []
        for repo_id in repo_ids:
            for pix_fmt in ["yuv420p", "yuv444p"]:
                cfg = {
                    "repo_id": repo_id,
                    # video encoding
                    "g": 2,
                    "crf": None,
                    "pix_fmt": pix_fmt,
                    # video decoding
                    "device": "cpu",
                    "decoder": "torchvision",
                    "decoder_kwgs": {},
                }
                if not dry_run:
                    run_video_benchmark(bench_dir / repo_id / f"torchvision_{pix_fmt}", cfg, timestamps_mode)
                info = load_info(bench_dir / repo_id / f"torchvision_{pix_fmt}")
                rows.append(
                    [
                        repo_id,
                        pix_fmt,
                        info["compression_factor"],
                        info["load_time_factor"],
                        info["avg_per_pixel_l2_error"],
                    ]
                )
        display_markdown_table(headers, rows)

        print("**`g`**")
        headers = ["repo_id", "g", "compression_factor", "load_time_factor", "avg_per_pixel_l2_error"]
        rows = []
        for repo_id in repo_ids:
            for g in [1, 2, 3, 4, 5, 6, 10, 15, 20, 40, 100, None]:
                cfg = {
                    "repo_id": repo_id,
                    # video encoding
                    "g": g,
                    "pix_fmt": "yuv444p",
                    # video decoding
                    "device": "cpu",
                    "decoder": "torchvision",
                    "decoder_kwgs": {},
                }
                if not dry_run:
                    run_video_benchmark(bench_dir / repo_id / f"torchvision_g_{g}", cfg, timestamps_mode)
                info = load_info(bench_dir / repo_id / f"torchvision_g_{g}")
                rows.append(
                    [
                        repo_id,
                        g,
                        info["compression_factor"],
                        info["load_time_factor"],
                        info["avg_per_pixel_l2_error"],
                    ]
                )
        display_markdown_table(headers, rows)

        print("**`crf`**")
        headers = ["repo_id", "crf", "compression_factor", "load_time_factor", "avg_per_pixel_l2_error"]
        rows = []
        for repo_id in repo_ids:
            for crf in [0, 5, 10, 15, 20, None, 25, 30, 40, 50]:
                cfg = {
                    "repo_id": repo_id,
                    # video encoding
                    "g": 2,
                    "crf": crf,
                    "pix_fmt": "yuv444p",
                    # video decoding
                    "device": "cpu",
                    "decoder": "torchvision",
                    "decoder_kwgs": {},
                }
                if not dry_run:
                    run_video_benchmark(bench_dir / repo_id / f"torchvision_crf_{crf}", cfg, timestamps_mode)
                info = load_info(bench_dir / repo_id / f"torchvision_crf_{crf}")
                rows.append(
                    [
                        repo_id,
                        crf,
                        info["compression_factor"],
                        info["load_time_factor"],
                        info["avg_per_pixel_l2_error"],
                    ]
                )
        display_markdown_table(headers, rows)

        print("**best**")
        headers = ["repo_id", "compression_factor", "load_time_factor", "avg_per_pixel_l2_error"]
        rows = []
        for repo_id in repo_ids:
            cfg = {
                "repo_id": repo_id,
                # video encoding
                "g": 2,
                "crf": None,
                "pix_fmt": "yuv444p",
                # video decoding
                "device": "cpu",
                "decoder": "torchvision",
                "decoder_kwgs": {},
            }
            if not dry_run:
                run_video_benchmark(bench_dir / repo_id / "torchvision_best", cfg, timestamps_mode)
            info = load_info(bench_dir / repo_id / "torchvision_best")
            rows.append(
                [
                    repo_id,
                    info["compression_factor"],
                    info["load_time_factor"],
                    info["avg_per_pixel_l2_error"],
                ]
            )
        display_markdown_table(headers, rows)


if __name__ == "__main__":
    main()
