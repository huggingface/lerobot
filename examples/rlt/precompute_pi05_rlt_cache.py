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

"""Precompute PI0.5-derived RLT cache from an existing LeRobotDataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tqdm import trange

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rlt.vla_adapter import OBS_VLA_EMBEDDINGS, PI05PrefixRLTAdapter
from lerobot.utils.constants import (
    ACTION,
    DONE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    REWARD,
    TRUNCATED,
)


def _save_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported save dtype: {name}")


def _collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif key == "task":
            batch[key] = values
        elif isinstance(first, (int, float, bool)):
            batch[key] = torch.tensor(values)
    return batch


def _policy_input_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed by the PI0.5 preprocessor."""
    excluded = {
        ACTION,
        REWARD,
        DONE,
        TRUNCATED,
        "index",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    }
    return {key: value for key, value in batch.items() if key not in excluded}


def _optional_tensor(batch: dict[str, Any], key: str) -> torch.Tensor | None:
    value = batch.get(key)
    return value if isinstance(value, torch.Tensor) else None


@torch.no_grad()
def _add_pi05_embeddings_only(policy: PI05Policy, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = dict(batch)
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    prefix_embs, _, _ = policy.model.embed_prefix(images, img_masks, tokens, masks)
    batch[OBS_VLA_EMBEDDINGS] = prefix_embs
    return batch


def _flush_shard(
    output_dir: Path,
    shard_id: int,
    tensors: dict[str, list[torch.Tensor]],
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    shard = {}
    for key, values in tensors.items():
        if not values:
            continue
        tensor = torch.cat(values, dim=0).contiguous().cpu()
        if tensor.is_floating_point() and key in {"vla_embeddings", "reference_action"}:
            tensor = tensor.to(dtype=dtype)
        shard[key] = tensor

    filename = f"shard-{shard_id:06d}.safetensors"
    save_file(shard, output_dir / filename)

    num_frames = int(next(iter(shard.values())).shape[0])
    shapes = {key: list(value.shape) for key, value in shard.items()}
    return {"file": filename, "num_frames": num_frames, "shapes": shapes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--policy-path", required=True, help="PI0.5 checkpoint or Hub repo.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rlt-chunk-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--start-frame", type=int, default=0, help="Absolute frame index to start from.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Local tokenizer directory or Hugging Face tokenizer repo. Defaults to the PI0.5 config value.",
    )
    parser.add_argument(
        "--skip-reference-action",
        action="store_true",
        help="Only save PI0.5 prefix embeddings. This is enough for RLT Stage 1 and uses less memory/time.",
    )
    parser.add_argument("--save-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"Loading dataset {args.dataset_repo_id} from {args.dataset_root or 'default cache'}...", flush=True)
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    if args.start_frame < 0 or args.start_frame >= len(dataset):
        raise ValueError(f"--start-frame must be in [0, {len(dataset) - 1}], got {args.start_frame}.")
    end_frame = len(dataset) if args.max_frames is None else min(args.start_frame + args.max_frames, len(dataset))
    num_frames = end_frame - args.start_frame
    print(
        f"Loaded dataset with {len(dataset)} frames; processing frames "
        f"[{args.start_frame}, {end_frame}) ({num_frames} frames).",
        flush=True,
    )

    print(f"Loading PI0.5 config from {args.policy_path}...", flush=True)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    if args.policy_dtype is not None and hasattr(policy_cfg, "dtype"):
        policy_cfg.dtype = args.policy_dtype
    if args.num_inference_steps is not None and hasattr(policy_cfg, "num_inference_steps"):
        policy_cfg.num_inference_steps = args.num_inference_steps
    if args.tokenizer_path is not None and hasattr(policy_cfg, "tokenizer_name"):
        policy_cfg.tokenizer_name = args.tokenizer_path

    dtype_msg = getattr(policy_cfg, "dtype", "checkpoint-default")
    steps_msg = getattr(policy_cfg, "num_inference_steps", "checkpoint-default")
    print(f"Loading PI0.5 policy on {args.device} with dtype={dtype_msg}, num_inference_steps={steps_msg}...", flush=True)
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    if not isinstance(policy, PI05Policy):
        raise TypeError(f"Expected a PI05Policy, got {type(policy).__name__}.")

    print("Building PI0.5 preprocessor and RLT adapter...", flush=True)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=dataset.meta.stats,
    )
    adapter = PI05PrefixRLTAdapter(policy, rlt_chunk_size=args.rlt_chunk_size)
    dtype = _save_dtype(args.save_dtype)

    pending: dict[str, list[torch.Tensor]] = {}
    shard_infos = []
    shard_id = 0
    frames_in_pending = 0

    for start in trange(args.start_frame, end_frame, args.batch_size, desc="precomputing RLT cache"):
        end = min(start + args.batch_size, end_frame)
        samples = [dataset[i] for i in range(start, end)]
        raw_batch = _collate_samples(samples)
        pi05_input = preprocessor(_policy_input_batch(raw_batch))

        with torch.inference_mode():
            if args.skip_reference_action:
                rlt_batch = _add_pi05_embeddings_only(policy, pi05_input)
            else:
                rlt_batch = adapter(pi05_input)

        batch_tensors = {
            "vla_embeddings": rlt_batch[OBS_VLA_EMBEDDINGS].detach(),
            "index": raw_batch["index"].to(torch.int64),
            "episode_index": raw_batch["episode_index"].to(torch.int64),
            "frame_index": raw_batch["frame_index"].to(torch.int64),
        }
        task_index = _optional_tensor(raw_batch, "task_index")
        if task_index is not None:
            batch_tensors["task_index"] = task_index.to(torch.int64)
        if not args.skip_reference_action:
            batch_tensors["reference_action"] = rlt_batch["observation.reference_action"].detach()

        for key in (ACTION, REWARD, DONE, TRUNCATED):
            value = _optional_tensor(raw_batch, key)
            if value is not None:
                batch_tensors[key] = value.detach()

        for key, value in batch_tensors.items():
            pending.setdefault(key, []).append(value.cpu())

        frames_in_pending += end - start
        if frames_in_pending >= args.shard_size:
            shard_infos.append(_flush_shard(output_dir, shard_id, pending, dtype=dtype))
            shard_id += 1
            pending = {}
            frames_in_pending = 0

    if frames_in_pending > 0:
        shard_infos.append(_flush_shard(output_dir, shard_id, pending, dtype=dtype))

    metadata = {
        "format": "lerobot_rlt_pi05_prefix_cache_v1",
        "embedding_source": "pi05.model.embed_prefix",
        "policy_path": str(args.policy_path),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root) if args.dataset_root is not None else None,
        "source_num_frames": len(dataset),
        "start_frame": args.start_frame,
        "end_frame": end_frame,
        "num_frames": num_frames,
        "rlt_chunk_size": args.rlt_chunk_size,
        "reference_action_saved": not args.skip_reference_action,
        "save_dtype": args.save_dtype,
        "fields": {
            "vla_embeddings": "Tensor[num_frames, prefix_tokens, embed_dim]",
            "reference_action": "Tensor[num_frames, rlt_chunk_size * action_dim]",
            "index": "Original LeRobotDataset absolute frame index",
            "episode_index": "Original episode index",
            "frame_index": "Frame index inside episode",
            "task_index": "Optional original task index",
            ACTION: "Optional original dataset action",
            REWARD: "Optional original dataset next-step reward",
            DONE: "Optional original dataset next.done",
            TRUNCATED: "Optional original dataset next.truncated",
        },
        "shards": shard_infos,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved RLT cache with {num_frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
