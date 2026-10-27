#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Encode human demonstration videos into Zero-WAM in-context latents for training.

Reads a ``pairs.jsonl`` (``{"episode_index", "start_frame", "end_frame", "human_video",
"human_text"}`` per robot segment, e.g. from ``humangen/build_manifest.py``), encodes each distinct
video once with the Wan2.2 VAE at the policy's ICL resolution and frame rate, embeds its caption
with UMT5, and writes ``.pth`` files in the upstream HumanGen layout plus a pairs file for
``--policy.icl_pairs_path``:

    python examples/zero_wam/encode_human_videos.py \
        --pairs humangen/outputs/pairs.jsonl --output-dir outputs/human_latents
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.policies.lingbot_va.utils import clean_prompt, load_text_encoder, load_tokenizer, load_vae
from lerobot.policies.zero_wam.modeling_zero_wam import read_video_at_fps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-components-path", default="robbyant-research/zero-wam-pretrain")
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    videos = {p["human_video"]: p.get("human_text", "") for p in pairs}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    path = args.frozen_components_path
    vae = load_vae(path, torch_dtype=torch.bfloat16, torch_device=args.device, subfolder="vae").eval()
    tokenizer = load_tokenizer(path, subfolder="tokenizer")
    encoder = load_text_encoder(
        path, torch_dtype=torch.bfloat16, torch_device=args.device, subfolder="text_encoder"
    )
    mean = torch.tensor(vae.config.latents_mean, device=args.device).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=args.device).view(1, -1, 1, 1, 1)

    latent_paths = {}
    with torch.no_grad():
        for video, text in tqdm(sorted(videos.items())):
            source = Path(video).stat()
            identity = json.dumps(
                {
                    "video": str(Path(video).resolve()),
                    "size": source.st_size,
                    "mtime_ns": source.st_mtime_ns,
                    "text": text,
                    "fps": args.fps,
                    "height": args.height,
                    "width": args.width,
                    "max_sequence_length": args.max_sequence_length,
                    "frozen_components_path": args.frozen_components_path,
                },
                sort_keys=True,
            )
            out = args.output_dir / f"{hashlib.sha256(identity.encode()).hexdigest()[:24]}.pth"
            latent_paths[video] = str(out)
            if out.exists():
                continue
            frames = read_video_at_fps(video, args.fps)
            frames = F.interpolate(
                frames, size=(args.height, args.width), mode="bilinear", align_corners=False
            )
            clip = (frames.permute(1, 0, 2, 3)[None].contiguous() * 2.0 - 1.0).to(args.device, torch.bfloat16)
            mu = vae.encode(clip).latent_dist.mean
            latent = ((mu.float() - mean) * inv_std)[0]  # [48, F, H, W]
            tokens = tokenizer(
                [clean_prompt(text)],
                padding="max_length",
                max_length=args.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            hidden = encoder(
                tokens.input_ids.to(args.device), tokens.attention_mask.to(args.device)
            ).last_hidden_state
            length = int(tokens.attention_mask.gt(0).sum())
            text_emb = torch.zeros_like(hidden[0])
            text_emb[:length] = hidden[0, :length]
            _, f, h, w = latent.shape
            torch.save(
                {
                    "latent": latent.permute(1, 2, 3, 0).reshape(f * h * w, -1).to(torch.bfloat16).cpu(),
                    "latent_num_frames": f,
                    "latent_height": h,
                    "latent_width": w,
                    "text_emb": text_emb.to(torch.bfloat16).cpu(),
                    "text": text,
                    "fps": args.fps,
                    "source_video": video,
                },
                out,
            )

    pairs_out = args.output_dir / "pairs.jsonl"
    with pairs_out.open("w") as f:
        for pair in pairs:
            f.write(json.dumps({**pair, "latent_path": latent_paths[pair["human_video"]]}) + "\n")
    print(f"Encoded {len(latent_paths)} videos; wrote {len(pairs)} pairs to {pairs_out}")


if __name__ == "__main__":
    main()
