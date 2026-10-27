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

"""Encode recorded demonstration videos into Zero-WAM in-context latents.

Reads JSONL entries with ``prompt_video``, optional ``prompt_text``, ``prompt_start_s`` and
``prompt_end_s``. ``prepare_episode_prompts.py`` resolves these from a LeRobot episode or pairing
manifest. The legacy ``human_video`` / ``human_text`` fields are also supported. Each distinct
video/crop/caption is encoded once. Outputs are normalized Wan latents and UMT5 embeddings, a
``prompts.jsonl`` index for rollout, and training ``pairs.jsonl`` for entries with a target
``episode_index``, ``start_frame`` and ``end_frame``:

    python examples/zero_wam/encode_human_videos.py \
        --pairs outputs/episode_prompts.jsonl --output-dir outputs/demo_latents
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
from lerobot.policies.zero_wam.prompt_cache import atomic_torch_save, save_shared_prompt_text
from lerobot.policies.zero_wam.prompt_video import (
    PromptVideoReader,
    canonical_prompt_video,
    prompt_source_identity,
)
from lerobot.streaming.location import StorageLocation


def prompt_spec(pair: dict, max_frames: int | None = None) -> dict:
    """One encoding identity per video, caption and crop; legacy human-video keys remain valid."""
    video = pair.get("prompt_video", pair.get("human_video"))
    if not video:
        raise ValueError("Each prompt requires prompt_video (or human_video)")
    return {
        "video": canonical_prompt_video(video),
        "text": pair.get("prompt_text", pair.get("human_text", "")),
        "start_time_s": float(pair.get("prompt_start_s", 0)),
        "end_time_s": pair.get("prompt_end_s"),
        "max_frames": pair.get("prompt_max_frames", max_frames),
    }


def encoded_pools(pairs: list[dict], pair_keys: list[str], latent_paths: dict[str, str]) -> list[dict]:
    """Attach encoded member paths while retaining compact pool and split identities."""
    pools = {}
    for pair, key in zip(pairs, pair_keys, strict=True):
        if "pool_id" not in pair:
            continue
        identity = (pair["pool_id"], pair["dataset"], pair["split"])
        pool = pools.setdefault(
            identity,
            {"pool_id": pair["pool_id"], "dataset": pair["dataset"], "split": pair["split"], "members": []},
        )
        pool["members"].append(
            {
                "episode_index": pair["prompt_episode_index"],
                "start_frame": pair["prompt_start_frame"],
                "end_frame": pair["prompt_end_frame"],
                "latent_path": latent_paths[key],
            }
        )
    return list(pools.values())


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
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mp4-sidecar", type=Path, help="Reuse the streaming dataset's video byte index")
    parser.add_argument("--max-prompt-video-mb", type=int, default=512, help="Per-clip compressed-byte limit")
    parser.add_argument(
        "--text-embedding-cache", type=Path, help="Reuse precompute_text_embeddings.py output"
    )
    parser.add_argument(
        "--share-text-embeddings", action="store_true", help="Store repeated padded captions once per cache"
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    with args.pairs.open() as source:
        pairs = [json.loads(line) for line in source if line.strip()]
    pair_keys = [json.dumps(prompt_spec(p, args.max_video_frames), sort_keys=True) for p in pairs]
    videos = {key: json.loads(key) for key in pair_keys}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cached_text = None
    text_cache_identity = {}
    if args.text_embedding_cache is not None:
        cached_text = torch.load(args.text_embedding_cache, map_location="cpu", weights_only=True)
        if (
            not isinstance(cached_text, dict)
            or {spec["text"] for spec in videos.values()} - cached_text.keys()
        ):
            raise ValueError("Text cache must contain every prompt's exact instruction")
        for text in {spec["text"] for spec in videos.values()}:
            value = cached_text[text]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or not 0 < value.shape[0] <= args.max_sequence_length
                or value.shape[1] < 1
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Cached text must be finite [tokens, hidden] within max-sequence-length")
        if len({cached_text[spec["text"]].shape[1] for spec in videos.values()}) != 1:
            raise ValueError("Cached prompt embeddings must have one shared hidden dimension")
        text_cache_identity = {
            "text_cache_sha256": hashlib.sha256(args.text_embedding_cache.read_bytes()).hexdigest()
        }
    path = args.frozen_components_path
    vae = load_vae(path, torch_dtype=torch.bfloat16, torch_device=args.device, subfolder="vae").eval()
    if cached_text is None:
        tokenizer = load_tokenizer(path, subfolder="tokenizer")
        encoder = load_text_encoder(
            path, torch_dtype=torch.bfloat16, torch_device=args.device, subfolder="text_encoder"
        )
    mean = torch.tensor(vae.config.latents_mean, device=args.device).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=args.device).view(1, -1, 1, 1, 1)

    def encode_text(text: str) -> torch.Tensor:
        if cached_text is not None:
            embedding = cached_text[text].to(args.device, torch.bfloat16)
            padded = embedding.new_zeros(args.max_sequence_length, embedding.shape[1])
            padded[: len(embedding)] = embedding
        else:
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
            padded = torch.zeros_like(hidden[0])
            padded[:length] = hidden[0, :length]
        return padded.to(torch.bfloat16).cpu()

    latent_paths = {}
    shared_text_paths = {}
    with (
        torch.no_grad(),
        PromptVideoReader(
            [spec["video"] for spec in videos.values()],
            sidecar_path=args.mp4_sidecar,
            max_video_bytes=args.max_prompt_video_mb * 1024**2,
        ) as reader,
    ):
        # Visit shared containers together when no full-source sidecar is supplied.
        ordered = sorted(videos.items(), key=lambda item: (item[1]["video"], item[1]["start_time_s"]))
        for key, spec in tqdm(ordered):
            video, text = spec["video"], spec["text"]
            identity = json.dumps(
                {
                    **spec,
                    **prompt_source_identity(video),
                    **text_cache_identity,
                    "text": text,
                    "fps": args.fps,
                    "height": args.height,
                    "width": args.width,
                    "max_sequence_length": args.max_sequence_length,
                    "frozen_components_path": args.frozen_components_path,
                    **({"text_storage": "shared-v1"} if args.share_text_embeddings else {}),
                },
                sort_keys=True,
            )
            out = args.output_dir / f"{hashlib.sha256(identity.encode()).hexdigest()[:24]}.pth"
            latent_paths[key] = out.name
            # Also repair a deleted shared caption during resume, without re-encoding video.
            if args.share_text_embeddings and text not in shared_text_paths:
                shared_text_paths[text] = save_shared_prompt_text(encode_text(text), args.output_dir)
            if out.exists():
                continue
            decode = reader.read if StorageLocation.parse(video).is_remote else read_video_at_fps
            frames = decode(
                video,
                args.fps,
                start_time_s=spec["start_time_s"],
                end_time_s=spec["end_time_s"],
                max_frames=spec["max_frames"],
            )
            frames = F.interpolate(
                frames, size=(args.height, args.width), mode="bilinear", align_corners=False
            )
            clip = (frames.permute(1, 0, 2, 3)[None].contiguous() * 2.0 - 1.0).to(args.device, torch.bfloat16)
            mu = vae.encode(clip).latent_dist.mean
            latent = ((mu.float() - mean) * inv_std)[0]  # [48, F, H, W]
            _, f, h, w = latent.shape
            payload = {
                "latent": latent.permute(1, 2, 3, 0).reshape(f * h * w, -1).to(torch.bfloat16).cpu(),
                "latent_num_frames": f,
                "latent_height": h,
                "latent_width": w,
                "text": text,
                "fps": args.fps,
                "source_video": video,
                "source_start_s": spec["start_time_s"],
                "source_end_s": spec["end_time_s"],
                "source_max_frames": spec["max_frames"],
            }
            if args.share_text_embeddings:
                payload["text_emb_path"] = shared_text_paths[text]
            else:
                payload["text_emb"] = encode_text(text)
            atomic_torch_save(payload, out)

    pairs_out = args.output_dir / "pairs.jsonl"
    with pairs_out.open("w") as f:
        for pair, key in zip(pairs, pair_keys, strict=True):
            if "episode_index" in pair:
                f.write(json.dumps({**pair, "latent_path": latent_paths[key]}) + "\n")
    prompts_out = args.output_dir / "prompts.jsonl"
    with prompts_out.open("w") as f:
        for key, spec in videos.items():
            f.write(json.dumps({**spec, "latent_path": latent_paths[key]}) + "\n")
    pools = encoded_pools(pairs, pair_keys, latent_paths)
    if pools:
        pools_out = args.output_dir / "pools.jsonl"
        with pools_out.open("w") as f:
            for pool in pools:
                f.write(json.dumps(pool) + "\n")
        print(f"Wrote {len(pools)} compact prompt pools to {pools_out}")
    num_pairs = sum("episode_index" in pair for pair in pairs)
    print(f"Encoded {len(latent_paths)} prompts; wrote {num_pairs} training pairs to {pairs_out}")
    print(f"Prompt latent paths: {prompts_out}")


if __name__ == "__main__":
    main()
