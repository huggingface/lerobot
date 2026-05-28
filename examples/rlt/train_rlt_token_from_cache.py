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

"""Train the RLT RL-token encoder/decoder from a precomputed PI0.5 cache."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rlt.configuration_rlt import RLTConfig, RLTokenConfig
from lerobot.policies.rlt.modeling_rlt import RLTPolicy
from lerobot.rl.algorithms.rlt.configuration_rlt import RLTAlgorithmConfig
from lerobot.utils.constants import ACTION

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _read_unique_split_values_worker(args: tuple[str, dict[str, Any], str]) -> list[int]:
    cache_dir, shard_info, key = args
    with safe_open(str(Path(cache_dir) / shard_info["file"]), framework="pt", device="cpu") as shard:
        if key not in shard.keys():
            raise KeyError(f"Cannot split by {key}; {shard_info['file']} does not contain that field.")
        values = shard.get_tensor(key).to(torch.int64).unique().tolist()
    return [int(value) for value in values]


def _read_split_ranges_worker(
    args: tuple[str, dict[str, Any], str, list[int]],
) -> list[tuple[dict[str, Any], int, int]]:
    cache_dir, shard_info, key, allowed_values = args
    allowed = torch.tensor(allowed_values, dtype=torch.int64)
    with safe_open(str(Path(cache_dir) / shard_info["file"]), framework="pt", device="cpu") as shard:
        split_values = shard.get_tensor(key).to(torch.int64)
        mask = torch.isin(split_values, allowed)
    return [(shard_info, start, end) for start, end in _contiguous_true_ranges(mask)]


class RLTCacheBatchIterator:
    """Infinite iterator yielding batches in the format expected by RLTAlgorithm."""

    def __init__(
        self,
        cache_dir: Path,
        shards: list[dict[str, Any]],
        *,
        batch_size: int,
        device: str,
        shuffle: bool,
        skip_zero_embeddings: bool,
        sampling_mode: str,
        split_key: str | None = None,
        allowed_split_values: set[int] | None = None,
        ranges: list[tuple[dict[str, Any], int, int]] | None = None,
    ):
        self.cache_dir = cache_dir
        self.shards = shards
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.skip_zero_embeddings = skip_zero_embeddings
        self.sampling_mode = sampling_mode
        self.split_key = split_key
        self.allowed_split_values = allowed_split_values
        self.ranges = ranges
        if self.ranges is None and split_key is not None:
            self.ranges = _build_split_ranges(cache_dir, shards, key=split_key, allowed_values=allowed_split_values)
        self._batch_iter = self._iter_batches()

    def __iter__(self) -> RLTCacheBatchIterator:
        return self

    def __next__(self) -> dict[str, dict[str, torch.Tensor]]:
        return next(self._batch_iter)

    def _yield_batch(self, shard_info: dict[str, Any], start: int, end: int):
        shard_path = self.cache_dir / shard_info["file"]
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            batch = shard.get_slice("vla_embeddings")[start:end].float().to(self.device, non_blocking=True)
        if self.skip_zero_embeddings:
            nonzero = batch.flatten(1).abs().sum(dim=1) > 0
            batch = batch[nonzero]
            if batch.numel() == 0:
                return None
        return {"state": {"observation.vla_embeddings": batch}}

    def _global_batch_refs(self) -> list[tuple[dict[str, Any], int, int]]:
        ranges = self.ranges
        if ranges is None:
            ranges = []
            for shard_info in self.shards:
                shard_path = self.cache_dir / shard_info["file"]
                with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                    num_frames = shard.get_slice("vla_embeddings").get_shape()[0]
                ranges.append((shard_info, 0, num_frames))

        refs: list[tuple[dict[str, Any], int, int]] = []
        for shard_info, range_start, range_end in ranges:
            if range_end <= range_start:
                continue
            for start in range(range_start, range_end, self.batch_size):
                refs.append((shard_info, start, min(start + self.batch_size, range_end)))
        return refs

    def _iter_batches(self):
        while True:
            if self.sampling_mode == "global":
                batch_refs = self._global_batch_refs()
                if self.shuffle:
                    random.shuffle(batch_refs)
                for shard_info, start, end in batch_refs:
                    batch = self._yield_batch(shard_info, start, end)
                    if batch is not None:
                        yield batch
                continue

            if self.ranges is not None:
                range_order = list(range(len(self.ranges)))
                if self.shuffle:
                    random.shuffle(range_order)

                for range_idx in range_order:
                    shard_info, range_start, range_end = self.ranges[range_idx]
                    if range_end <= range_start:
                        continue
                    shard_path = self.cache_dir / shard_info["file"]
                    batch_starts = list(range(range_start, range_end, self.batch_size))
                    if self.shuffle:
                        random.shuffle(batch_starts)

                    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                        embeddings_slice = shard.get_slice("vla_embeddings")
                        for start in batch_starts:
                            end = min(start + self.batch_size, range_end)
                            batch = embeddings_slice[start:end].float().to(self.device, non_blocking=True)
                            if self.skip_zero_embeddings:
                                nonzero = batch.flatten(1).abs().sum(dim=1) > 0
                                batch = batch[nonzero]
                                if batch.numel() == 0:
                                    continue
                            yield {"state": {"observation.vla_embeddings": batch}}
                continue

            shard_order = list(range(len(self.shards)))
            if self.shuffle:
                random.shuffle(shard_order)

            for shard_idx in shard_order:
                shard_info = self.shards[shard_idx]
                shard_path = self.cache_dir / shard_info["file"]
                with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                    embeddings = shard.get_tensor("vla_embeddings").float()

                frame_order = torch.randperm(embeddings.shape[0]) if self.shuffle else torch.arange(embeddings.shape[0])
                for start in range(0, embeddings.shape[0], self.batch_size):
                    indices = frame_order[start : start + self.batch_size]
                    batch = embeddings.index_select(0, indices).to(self.device, non_blocking=True)
                    if self.skip_zero_embeddings:
                        nonzero = batch.flatten(1).abs().sum(dim=1) > 0
                        batch = batch[nonzero]
                        if batch.numel() == 0:
                            continue
                    yield {"state": {"observation.vla_embeddings": batch}}


def _load_metadata(cache_dir: Path) -> dict[str, Any]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        shards = [{"file": path.name} for path in sorted(cache_dir.glob("shard-*.safetensors"))]
        if not shards:
            raise FileNotFoundError(f"No metadata.json or shard-*.safetensors found in {cache_dir}.")
        return {"shards": shards}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _split_shards(
    shards: list[dict[str, Any]],
    *,
    val_ratio: float,
    val_shards: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1), got {val_ratio}.")
    if val_shards is not None and val_shards < 0:
        raise ValueError(f"--val-shards must be non-negative, got {val_shards}.")
    if len(shards) < 2:
        return shards, []

    num_val = val_shards if val_shards is not None else int(round(len(shards) * val_ratio))
    if val_ratio > 0 and num_val == 0:
        num_val = 1
    num_val = min(num_val, len(shards) - 1)
    if num_val == 0:
        return shards, []

    order = list(range(len(shards)))
    rng = random.Random(seed)
    rng.shuffle(order)
    val_indices = set(order[:num_val])
    train_shards = [shard for idx, shard in enumerate(shards) if idx not in val_indices]
    val_split = [shard for idx, shard in enumerate(shards) if idx in val_indices]
    return train_shards, val_split


def _split_values_from_cache(
    cache_dir: Path,
    shards: list[dict[str, Any]],
    *,
    key: str,
    val_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    values: set[int] = set()
    tasks = [(str(cache_dir), shard_info, key) for shard_info in shards]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=8) as pool:
        iterator = pool.imap(_read_unique_split_values_worker, tasks, chunksize=1)
        for shard_values in tqdm(iterator, total=len(shards), desc=f"scanning {key}", unit="shard"):
            values.update(shard_values)
            gc.collect()

    if len(values) < 2:
        return values, set()

    ordered = sorted(values)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    num_val = int(round(len(ordered) * val_ratio))
    if val_ratio > 0 and num_val == 0:
        num_val = 1
    num_val = min(num_val, len(ordered) - 1)
    val_values = set(ordered[:num_val])
    train_values = set(ordered[num_val:])
    return train_values, val_values


def _contiguous_true_ranges(mask: torch.Tensor) -> list[tuple[int, int]]:
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not indices:
        return []

    ranges = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev + 1))
        start = prev = idx
    ranges.append((start, prev + 1))
    return ranges


def _build_split_ranges(
    cache_dir: Path,
    shards: list[dict[str, Any]],
    *,
    key: str,
    allowed_values: set[int] | None,
) -> list[tuple[dict[str, Any], int, int]]:
    if allowed_values is None:
        raise ValueError(f"allowed_values is required when splitting by {key}.")

    ranges: list[tuple[dict[str, Any], int, int]] = []
    tasks = [(str(cache_dir), shard_info, key, sorted(allowed_values)) for shard_info in shards]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=8) as pool:
        iterator = pool.imap(_read_split_ranges_worker, tasks, chunksize=1)
        for shard_ranges in tqdm(iterator, total=len(shards), desc=f"building {key} ranges", unit="shard"):
            ranges.extend(shard_ranges)
            gc.collect()
    return ranges


def _split_index_default_path(output_dir: Path, split_by: str, seed: int, val_ratio: float) -> Path:
    ratio = str(val_ratio).replace(".", "p")
    return output_dir / f"split_index_{split_by}_seed{seed}_val{ratio}.json"


def _serialize_ranges(ranges: list[tuple[dict[str, Any], int, int]]) -> list[dict[str, Any]]:
    return [{"file": shard_info["file"], "start": int(start), "end": int(end)} for shard_info, start, end in ranges]


def _deserialize_ranges(items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int, int]]:
    return [({"file": item["file"]}, int(item["start"]), int(item["end"])) for item in items]


def _load_or_build_split_index(
    cache_dir: Path,
    shards: list[dict[str, Any]],
    *,
    split_by: str,
    split_key: str,
    val_ratio: float,
    seed: int,
    index_file: Path,
    rebuild: bool,
) -> tuple[set[int], set[int], list[tuple[dict[str, Any], int, int]], list[tuple[dict[str, Any], int, int]]]:
    if index_file.exists() and not rebuild:
        print(f"Loading split index from {index_file}", flush=True)
        data = json.loads(index_file.read_text(encoding="utf-8"))
        expected = {
            "cache_dir": str(cache_dir),
            "split_by": split_by,
            "split_key": split_key,
            "val_ratio": val_ratio,
            "seed": seed,
            "num_shards": len(shards),
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(
                    f"Split index {index_file} does not match current run for {key}: "
                    f"{data.get(key)!r} != {value!r}. Pass --rebuild-split-index to rebuild."
                )
        return (
            set(int(v) for v in data["train_values"]),
            set(int(v) for v in data["val_values"]),
            _deserialize_ranges(data["train_ranges"]),
            _deserialize_ranges(data["val_ranges"]),
        )

    print(f"Building split index at {index_file}", flush=True)
    train_values, val_values = _split_values_from_cache(
        cache_dir,
        shards,
        key=split_key,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_ranges = _build_split_ranges(cache_dir, shards, key=split_key, allowed_values=train_values)
    val_ranges = _build_split_ranges(cache_dir, shards, key=split_key, allowed_values=val_values)
    data = {
        "format": "lerobot_rlt_split_index_v1",
        "cache_dir": str(cache_dir),
        "split_by": split_by,
        "split_key": split_key,
        "val_ratio": val_ratio,
        "seed": seed,
        "num_shards": len(shards),
        "train_values": sorted(int(v) for v in train_values),
        "val_values": sorted(int(v) for v in val_values),
        "train_ranges": _serialize_ranges(train_ranges),
        "val_ranges": _serialize_ranges(val_ranges),
    }
    index_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = index_file.with_suffix(index_file.suffix + ".tmp")
    tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_file.replace(index_file)
    return train_values, val_values, train_ranges, val_ranges


def _read_cache_shapes(cache_dir: Path, first_shard: dict[str, Any]) -> tuple[int, int | None, int | None, int | None]:
    shard = load_file(str(cache_dir / first_shard["file"]), device="cpu")
    if "vla_embeddings" not in shard:
        raise KeyError(f"{first_shard['file']} does not contain vla_embeddings.")

    embeddings = shard["vla_embeddings"]
    if embeddings.dim() != 3:
        raise ValueError(f"Expected vla_embeddings to have shape (N, M, D), got {tuple(embeddings.shape)}.")

    reference_action = shard.get("reference_action")
    reference_action_dim = None if reference_action is None else int(reference_action.shape[-1])
    action = shard.get(ACTION)
    action_dim = None if action is None else int(action.shape[-1])
    return int(embeddings.shape[-1]), int(embeddings.shape[1]), reference_action_dim, action_dim


@torch.no_grad()
def _validation_loss(
    policy: RLTPolicy,
    cache_dir: Path,
    shards: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    max_batches: int | None,
    skip_zero_embeddings: bool,
    split_key: str | None = None,
    allowed_split_values: set[int] | None = None,
    ranges: list[tuple[dict[str, Any], int, int]] | None = None,
) -> float | None:
    if not shards:
        return None

    was_training = policy.training
    policy.eval()

    if ranges is None:
        ranges = (
            _build_split_ranges(cache_dir, shards, key=split_key, allowed_values=allowed_split_values)
            if split_key is not None
            else [(shard_info, 0, None) for shard_info in shards]
        )

    losses = []
    batches_seen = 0
    for shard_info, range_start, range_end in ranges:
        with safe_open(str(cache_dir / shard_info["file"]), framework="pt", device="cpu") as shard:
            embeddings_slice = shard.get_slice("vla_embeddings")
            if range_end is None:
                range_end = embeddings_slice.get_shape()[0]
            for start in range(range_start, range_end, batch_size):
                end = min(start + batch_size, range_end)
                batch = embeddings_slice[start:end].float().to(device, non_blocking=True)
                if skip_zero_embeddings:
                    nonzero = batch.flatten(1).abs().sum(dim=1) > 0
                    batch = batch[nonzero]
                    if batch.numel() == 0:
                        continue
                z_rl = policy.rl_token_encoder(batch)
                z_reconstructed = policy.rl_token_decoder(z_rl, batch)
                losses.append(torch.nn.functional.mse_loss(z_reconstructed, batch).item())
                batches_seen += 1
                if max_batches is not None and batches_seen >= max_batches:
                    if was_training:
                        policy.train()
                    return sum(losses) / len(losses)

    if was_training:
        policy.train()
    return sum(losses) / len(losses) if losses else None


def _split_key(split_by: str) -> str | None:
    if split_by == "shard":
        return None
    if split_by == "episode":
        return "episode_index"
    if split_by == "task":
        return "task_index"
    raise ValueError(f"Unsupported split mode: {split_by}")


def _build_policy_config(
    *,
    input_dim: int,
    action_dim: int,
    chunk_size: int,
    device: str,
    args: argparse.Namespace,
) -> RLTConfig:
    rl_token_dim = args.rl_token_dim or input_dim
    ff_dim = args.ff_dim or rl_token_dim * 4
    if rl_token_dim % args.num_heads != 0:
        raise ValueError(f"rl_token_dim={rl_token_dim} must be divisible by num_heads={args.num_heads}.")

    return RLTConfig(
        input_features={},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        device=device,
        chunk_size=chunk_size,
        rl_token=RLTokenConfig(
            input_dim=input_dim,
            rl_token_dim=rl_token_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            num_heads=args.num_heads,
            ff_dim=ff_dim,
            dropout=args.dropout,
        ),
        rl_token_lr=args.lr,
        clip_grad_norm=args.clip_grad_norm,
    )


def _load_or_create_policy(
    *,
    policy_cfg: RLTConfig,
    device: str,
    resume_policy_path: str | None,
) -> RLTPolicy:
    if resume_policy_path is None:
        return RLTPolicy(policy_cfg).to(device)

    resume_path = Path(resume_policy_path)
    if not resume_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {resume_path}. "
            "Use an existing checkpoint-* directory and do not use checkpoint-*.tmp."
        )
    if resume_path.name.endswith(".tmp"):
        raise ValueError(f"Refusing to resume from temporary checkpoint directory: {resume_path}")
    print(f"Resuming RLTPolicy from {resume_path}", flush=True)
    policy = RLTPolicy.from_pretrained(resume_path, local_files_only=True, strict=True)
    policy.config.device = device
    return policy.to(device)


def _save_policy_checkpoint(policy: RLTPolicy, checkpoint_dir: Path) -> None:
    tmp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    policy.save_pretrained(tmp_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def _save_training_metadata(
    output_dir: Path,
    *,
    cache_dir: Path,
    source_metadata: dict[str, Any],
    args: argparse.Namespace,
    input_dim: int,
    prefix_tokens: int,
    action_dim: int,
    chunk_size: int,
) -> None:
    metadata = {
        "format": "lerobot_rlt_token_stage1_v1",
        "cache_dir": str(cache_dir),
        "source_cache_format": source_metadata.get("format"),
        "source_embedding": source_metadata.get("embedding_source"),
        "start_step": args.start_step,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "val_shards": args.val_shards,
        "split_by": args.split_by,
        "sampling_mode": args.sampling_mode,
        "skip_zero_embeddings": not args.include_zero_embeddings,
        "val_freq": args.val_freq,
        "val_max_batches": args.val_max_batches,
        "input_dim": input_dim,
        "prefix_tokens": prefix_tokens,
        "action_dim": action_dim,
        "rlt_chunk_size": chunk_size,
    }
    (output_dir / "rlt_token_training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, help="Directory produced by precompute_pi05_rlt_cache.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Global step offset for resumed runs. Training runs from start-step + 1 through steps.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rlt-chunk-size", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--rl-token-dim", type=int, default=None)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--val-shards", type=int, default=None)
    parser.add_argument(
        "--split-by",
        default="shard",
        choices=["shard", "episode", "task"],
        help="Validation split unit. Use task only for caches generated with task_index.",
    )
    parser.add_argument("--val-freq", type=int, default=100)
    parser.add_argument("--val-max-batches", type=int, default=64)
    parser.add_argument(
        "--sampling-mode",
        default="episode_block",
        choices=["episode_block", "global"],
        help="episode_block consumes one split range at a time; global shuffles all train batches across ranges.",
    )
    parser.add_argument(
        "--split-index-file",
        default=None,
        help="Optional cached split/range index JSON. Defaults to <output-dir>/split_index_<split>_<seed>_<val>.json.",
    )
    parser.add_argument(
        "--rebuild-split-index",
        action="store_true",
        help="Rebuild the cached split index even if it already exists.",
    )
    parser.add_argument(
        "--resume-policy-path",
        default=None,
        help="Optional RLTPolicy checkpoint directory to initialize from.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        default=None,
        help="Optional TensorBoard log directory. Defaults to <output-dir>/runs when TensorBoard is installed.",
    )
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event writing.")
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Optional JSONL metrics path. Defaults to <output-dir>/train_metrics.jsonl.",
    )
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--include-zero-embeddings",
        action="store_true",
        help="Train on all-zero embedding rows. By default they are skipped because they usually indicate bad cache rows.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.start_step < 0:
        raise ValueError(f"--start-step must be non-negative, got {args.start_step}.")
    if args.start_step >= args.steps:
        raise ValueError(f"--start-step must be smaller than --steps, got {args.start_step} >= {args.steps}.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.resume_policy_path is not None:
            pass
        elif not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_file) if args.metrics_file is not None else output_dir / "train_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tb_log_dir = Path(args.tensorboard_log_dir) if args.tensorboard_log_dir is not None else output_dir / "runs"
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=str(tb_log_dir)) if SummaryWriter is not None else None
    if writer is None:
        print("TensorBoard is not installed; writing JSONL metrics only.", flush=True)
    else:
        print(f"Writing TensorBoard logs to {tb_log_dir}", flush=True)

    source_metadata = _load_metadata(cache_dir)
    shards = source_metadata["shards"]
    if not shards:
        raise ValueError(f"No shards listed in {cache_dir / 'metadata.json'}.")
    split_key = _split_key(args.split_by)
    train_values = None
    val_values = None
    train_ranges = None
    val_ranges = None
    if split_key is None:
        train_shards, val_shards = _split_shards(
            shards,
            val_ratio=args.val_ratio,
            val_shards=args.val_shards,
            seed=args.seed,
        )
    else:
        train_shards = shards
        val_shards = shards
        split_index_file = (
            Path(args.split_index_file)
            if args.split_index_file is not None
            else _split_index_default_path(output_dir, args.split_by, args.seed, args.val_ratio)
        )
        train_values, val_values, train_ranges, val_ranges = _load_or_build_split_index(
            cache_dir,
            shards,
            split_by=args.split_by,
            split_key=split_key,
            val_ratio=args.val_ratio,
            seed=args.seed,
            index_file=split_index_file,
            rebuild=args.rebuild_split_index,
        )

    input_dim, prefix_tokens, reference_action_dim, action_dim_from_cache = _read_cache_shapes(cache_dir, shards[0])
    chunk_size = args.rlt_chunk_size or source_metadata.get("rlt_chunk_size")
    if chunk_size is None:
        raise ValueError("Cannot infer RLT chunk size. Pass --rlt-chunk-size.")

    action_dim = args.action_dim
    if action_dim is None and reference_action_dim is not None:
        if reference_action_dim % int(chunk_size) != 0:
            raise ValueError(
                f"reference_action dim {reference_action_dim} is not divisible by chunk size {chunk_size}."
            )
        action_dim = reference_action_dim // int(chunk_size)
    if action_dim is None and action_dim_from_cache is not None:
        action_dim = action_dim_from_cache
    if action_dim is None:
        raise ValueError("Cannot infer action dim. Pass --action-dim.")

    print(
        f"Training RLT token with split_by={args.split_by} from "
        f"{len(train_shards)} train shard(s), {len(val_shards)} val shard(s): "
        f"prefix_tokens={prefix_tokens}, input_dim={input_dim}, action_dim={action_dim}, chunk_size={chunk_size}",
        flush=True,
    )
    if split_key is not None:
        print(
            f"Train {split_key} values={len(train_values or [])}, val {split_key} values={len(val_values or [])}",
            flush=True,
        )
    print(f"Sampling mode: {args.sampling_mode}", flush=True)
    if not args.include_zero_embeddings:
        print("Skipping all-zero vla_embeddings rows during train/validation.", flush=True)

    policy_cfg = _build_policy_config(
        input_dim=input_dim,
        action_dim=action_dim,
        chunk_size=int(chunk_size),
        device=args.device,
        args=args,
    )
    policy = _load_or_create_policy(
        policy_cfg=policy_cfg,
        device=args.device,
        resume_policy_path=args.resume_policy_path,
    )
    policy.train()

    algorithm_cfg = RLTAlgorithmConfig.from_policy_config(policy_cfg)
    algorithm = algorithm_cfg.build_algorithm(policy)
    algorithm.make_optimizers()

    batch_iterator = RLTCacheBatchIterator(
        cache_dir,
        train_shards,
        batch_size=args.batch_size,
        device=args.device,
        shuffle=not args.no_shuffle,
        skip_zero_embeddings=not args.include_zero_embeddings,
        sampling_mode=args.sampling_mode,
        split_key=split_key,
        allowed_split_values=train_values,
        ranges=train_ranges,
    )

    for step in range(args.start_step + 1, args.steps + 1):
        stats = algorithm.offline_update(batch_iterator)
        loss = stats.losses["loss_rl_token"]
        if writer is not None:
            writer.add_scalar("train/loss_rl_token", loss, step)

        if step == 1 or step % args.log_freq == 0:
            print(f"step {step}/{args.steps} loss_rl_token={loss:.6f}", flush=True)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "loss_rl_token": loss, "split": "train"}) + "\n")

        if val_shards and args.val_freq > 0 and step % args.val_freq == 0:
            val_loss = _validation_loss(
                policy,
                cache_dir,
                val_shards,
                batch_size=args.batch_size,
                device=args.device,
                max_batches=args.val_max_batches,
                skip_zero_embeddings=not args.include_zero_embeddings,
                split_key=split_key,
                allowed_split_values=val_values,
                ranges=val_ranges,
            )
            if val_loss is not None:
                if writer is not None:
                    writer.add_scalar("val/loss_rl_token", val_loss, step)
                print(f"step {step}/{args.steps} val_loss_rl_token={val_loss:.6f}", flush=True)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "loss_rl_token": val_loss, "split": "val"}) + "\n")

        if args.save_freq > 0 and step % args.save_freq == 0:
            checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
            _save_policy_checkpoint(policy, checkpoint_dir)

    policy.save_pretrained(output_dir)
    if writer is not None:
        writer.close()
    _save_training_metadata(
        output_dir,
        cache_dir=cache_dir,
        source_metadata=source_metadata,
        args=args,
        input_dim=input_dim,
        prefix_tokens=prefix_tokens,
        action_dim=action_dim,
        chunk_size=int(chunk_size),
    )
    print(f"Saved trained RLT token policy to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
