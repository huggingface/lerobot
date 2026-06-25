#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Precompute CLIP image features for a LeRobot dataset.

Writes a (N_total_frames, model.config.projection_dim) float32 .npy memmap indexed by absolute global
frame index, ready to be passed to SARM via SARMConfig.precomputed_image_features_path.

Usage:

```bash
lerobot-sarm-precompute-clip \\
    --repo-id={hf_username}/{repo_name} \\
    --camera-key=observation.images.top \\
    --output=/path/to/clip_features.npy
```
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.utils.import_utils import _transformers_available, require_package
from lerobot.utils.utils import init_logging

if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPImageProcessor, CLIPModel
else:
    CLIPModel = None  # type: ignore[assignment, misc]
    CLIPImageProcessor = None  # type: ignore[assignment, misc]


@torch.no_grad()
def _preprocess_on_device(
    raw_uint8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, crop_size: int
) -> torch.Tensor:
    """GPU-batched equivalent of HF CLIPImageProcessor preprocessing:
    resize shorter edge to `crop_size` (bicubic, antialiased), center-crop, /255, normalize.

    Args:
        raw_uint8: (B, H, W, 3) uint8 tensor on the active device.
        mean, std: (1, 3, 1, 1) normalization constants on the same device.
        crop_size: target square crop size (pulled from CLIPImageProcessor at runtime).

    Returns:
        (B, 3, crop_size, crop_size) float32 tensor ready for the CLIP vision tower.
    """
    x = raw_uint8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    _, _, H, W = x.shape
    if H < W:
        new_h, new_w = crop_size, int(round(W * crop_size / H))
    else:
        new_h, new_w = int(round(H * crop_size / W)), crop_size
    # Resize while preserving the aspect ratio
    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", antialias=True, align_corners=False)
    # crop in the center
    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2
    x = x[:, :, top : top + crop_size, left : left + crop_size]
    return (x - mean) / std



def precompute_for_camera(
    repo_id: str,
    camera_key: str,
    output: Path,
    root: Path | None = None,
    revision: str | None = None,
    clip_model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    chunk_frames: int = 512,
) -> None:
    """Precompute CLIP image features for one camera of a LeRobot dataset.

    Output is a (N_total_frames, proj_dim) float32 .npy memmap indexed by absolute
    global frame index.
    """
    require_package("transformers", extra="sarm")

    logging.info(f"Loading dataset metadata for {repo_id}...")
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    if camera_key not in meta.video_keys:
        raise ValueError(
            f"camera_key={camera_key!r} not found in dataset video keys: {meta.video_keys}"
        )

    n_frames_total = meta.total_frames
    logging.info(f"  episodes={meta.total_episodes}  total_frames={n_frames_total}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    device = torch.device(device)

    logging.info(f"Loading CLIP model + image processor: {clip_model_id}")
    # Pull preprocessing constants (crop size, mean, std) from the model's
    # CLIPImageProcessor rather than hardcoding them. This keeps the precompute
    # in lockstep with whichever CLIP checkpoint is used and matches the
    # online sarm processor's behavior.
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_id)
    crop_size_val = image_processor.crop_size
    # Newer transformers versions return a SizeDict like {"height": 224, "width": 224},
    # older versions return a plain int. SizeDict isn't always a dict subclass so we
    # duck-type on dict-style access instead of `isinstance`.
    try:
        h, w = crop_size_val["height"], crop_size_val["width"]
        assert h == w, f"non-square crop_size from CLIPImageProcessor: {crop_size_val}"
        crop_size = int(h)
    except (TypeError, KeyError, AttributeError):
        crop_size = int(crop_size_val)
    mean = torch.tensor(image_processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, device=device).view(1, 3, 1, 1)
    logging.info(
        f"  crop_size={crop_size}  mean={image_processor.image_mean}  std={image_processor.image_std}"
    )

    model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    vision = model.vision_model
    proj = model.visual_projection

    proj_dim = int(model.config.projection_dim)
    # Create the output memmap that will store the clip features for all the frames
    out = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(n_frames_total, proj_dim)
    )
    logging.info(f"Output memmap: {output}  shape={out.shape}  dtype={out.dtype}")

    ### Enumerate episodes and group by mp4 file ###
    # In LeRobot v3+, multiple episodes can share one mp4 file. Sort by position
    # in the video files, then group by (chunk_index, file_index) so we open
    # ffmpeg once per mp4 and stream all of its frames in order.
    episodes_df = (
        meta.episodes.to_pandas()
        .sort_values(
            [
                f"videos/{camera_key}/chunk_index",
                f"videos/{camera_key}/file_index",
                f"videos/{camera_key}/from_timestamp",
            ]
        )
        .reset_index(drop=True)
    )
    file_groups = episodes_df.groupby(
        [f"videos/{camera_key}/chunk_index", f"videos/{camera_key}/file_index"],
        sort=True,
    )

    # Probe video dimensions from the first mp4 (assume all camera mp4s share W,H).
    sample_ep_idx = int(episodes_df.iloc[0]["episode_index"])
    sample_path = meta.root / meta.get_video_file_path(sample_ep_idx, camera_key)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(sample_path),
        ],
        capture_output=True, text=True, check=True,
    )
    W, H = map(int, probe.stdout.strip().split(","))
    frame_bytes = H * W * 3
    logging.info(f"  video: {W}x{H} (frame_bytes={frame_bytes})")

    ### Stream each mp4 with ffmpeg, batch through CLIP on GPU ###
    t0 = time.time()
    n_done = 0
    n_files = len(file_groups)

    with torch.no_grad():
        for file_idx, ((chunk_i, file_i), group) in enumerate(file_groups, start=1):
            mp4_ep_idx = int(group.iloc[0]["episode_index"])
            mp4_path = meta.root / meta.get_video_file_path(mp4_ep_idx, camera_key)
            group = group.sort_values(f"videos/{camera_key}/from_timestamp").reset_index(drop=True)

            # Build mp4_frame_to_global: within-mp4 frame index -> absolute global frame index.
            n_frames_in_file = int(group["length"].sum())
            mp4_frame_to_global = np.empty(n_frames_in_file, dtype=np.int64)
            cursor = 0
            for _, ep in group.iterrows():
                L = int(ep["length"])
                g0 = int(ep["dataset_from_index"])
                mp4_frame_to_global[cursor : cursor + L] = np.arange(g0, g0 + L)
                cursor += L

            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-i", str(mp4_path),
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-vsync", "0", "-loglevel", "error", "-",
                ],
                stdout=subprocess.PIPE,
                bufsize=128 * 1024 * 1024,
            )
            try:
                file_frame_idx = 0
                while True:
                    n_want = min(chunk_frames, n_frames_in_file - file_frame_idx)
                    if n_want <= 0:
                        break
                    buf = ffmpeg.stdout.read(n_want * frame_bytes)
                    if not buf:
                        break
                    n_got = len(buf) // frame_bytes
                    if n_got == 0:
                        break
                    np_raw = np.frombuffer(buf[: n_got * frame_bytes], dtype=np.uint8).reshape(n_got, H, W, 3)
                    raw = torch.from_numpy(np_raw.copy()).to(device, non_blocking=True)
                    pixel_values = _preprocess_on_device(raw, mean, std, crop_size)
                    out_vision = vision(pixel_values=pixel_values)
                    pooled = out_vision.pooler_output if hasattr(out_vision, "pooler_output") else out_vision[1]
                    embs = proj(pooled).float().cpu().numpy()
                    gidx = mp4_frame_to_global[file_frame_idx : file_frame_idx + n_got]
                    out[gidx] = embs
                    file_frame_idx += n_got
                    n_done += n_got
                if file_frame_idx != n_frames_in_file:
                    logging.warning(
                        f"file-{file_i:03d}: expected {n_frames_in_file} frames, got {file_frame_idx}"
                    )
            finally:
                try:
                    ffmpeg.stdout.close()
                except Exception:
                    pass
                ffmpeg.wait(timeout=5)

            now = time.time()
            rate = n_done / max(now - t0, 1e-6)
            eta = (n_frames_total - n_done) / max(rate, 1e-6)
            logging.info(
                f"  [file {file_idx}/{n_files}] {n_done}/{n_frames_total} "
                f"({100 * n_done / n_frames_total:.1f}%)  rate={rate:.0f} fps  eta={eta:.0f}s"
            )

    out.flush()
    elapsed = time.time() - t0
    logging.info(f"Done in {elapsed:.1f}s ({n_frames_total / elapsed:.0f} fps). Output: {output}")



def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP image features for a LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo ID (e.g. lerobot/pusht).")
    parser.add_argument("--camera-key", type=str, required=True, help="Video feature key (e.g. observation.images.top).")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path for the feature memmap.")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root (optional).")
    parser.add_argument("--revision", type=str, default=None, help="Dataset revision (optional).")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model ID.")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda / cpu).")
    parser.add_argument("--chunk-frames", type=int, default=512, help="Batch size per CLIP forward.")
    args = parser.parse_args()

    init_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    precompute_for_camera(
        repo_id=args.repo_id,
        camera_key=args.camera_key,
        output=args.output,
        root=args.root,
        revision=args.revision,
        clip_model_id=args.clip_model,
        device=device,
        chunk_frames=args.chunk_frames,
    )


if __name__ == "__main__":
    main()
