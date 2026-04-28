#!/usr/bin/env python
"""Train a lightweight LIBERO progress reward model for tree search.

The default setup trains a frozen SigLIP2 encoder plus a small MLP head. Frames
from successful demonstrations are labeled by normalized episode progress:
0.0 at the first sampled frame and 1.0 at the final frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets import LeRobotDatasetMetadata

from reward_model import (
    MultiModalRewardModel,
    RewardBatchProcessor,
    RewardModelConfig,
    default_model_id,
    move_batch_to_device,
)
from vlm_sequence_prompt_probe import (
    EpisodeFrameReader,
    _episode_tasks,
    _get_libero_task,
    _image_to_uint8_hwc,
    _load_episode_reader,
    _normalize_text,
    _sample_local_indices,
    _to_jsonable,
)

logger = logging.getLogger(__name__)


@dataclass
class RewardSampleSpec:
    task: str
    episode_index: int
    local_index: int
    label: float


def parse_task_orders(raw: str) -> list[int]:
    text = raw.strip()
    if text.lower() == "all":
        return list(range(10))
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def find_episodes_for_task(
    meta: LeRobotDatasetMetadata,
    *,
    task_metadata: dict[str, Any],
    limit: int,
) -> list[int]:
    targets = {
        _normalize_text(str(task_metadata.get("language", ""))),
        _normalize_text(str(task_metadata.get("name", ""))),
    }
    targets.discard("")
    matches: list[int] = []
    for ep_idx in range(meta.total_episodes):
        episode = meta.episodes[ep_idx]
        if any(_normalize_text(task) in targets for task in _episode_tasks(meta, episode)):
            matches.append(ep_idx)
            if len(matches) >= limit:
                break
    if not matches:
        raise SystemExit(f"No dataset episodes matched task targets: {sorted(targets)}")
    return matches


def _reader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        force_cache_sync=args.force_cache_sync,
        download_videos=args.download_videos,
        video_backend=args.video_backend,
    )


class LiberoRewardFrameDataset(Dataset):
    def __init__(
        self,
        *,
        specs: list[RewardSampleSpec],
        meta: LeRobotDatasetMetadata,
        reader_args: SimpleNamespace,
        scene_camera_key: str,
        wrist_camera_key: str | None,
        state_key: str,
        use_proprioception: bool,
        reader_cache_size: int = 3,
    ) -> None:
        self.specs = specs
        self.meta = meta
        self.reader_args = reader_args
        self.scene_camera_key = scene_camera_key
        self.wrist_camera_key = wrist_camera_key
        self.state_key = state_key
        self.use_proprioception = use_proprioception
        self.reader_cache_size = max(1, reader_cache_size)
        self._reader_cache: OrderedDict[int, EpisodeFrameReader] = OrderedDict()
        self._logged_first_sample = False

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.specs[int(index)]
        reader = self._reader_for_episode(spec.episode_index)
        item = reader[spec.local_index]
        if self.scene_camera_key not in item:
            raise KeyError(f"Missing scene camera key '{self.scene_camera_key}' in dataset item.")

        sample: dict[str, Any] = {
            "task": spec.task,
            "episode_index": spec.episode_index,
            "local_index": spec.local_index,
            "label": spec.label,
            "scene_image": _image_to_uint8_hwc(item[self.scene_camera_key]),
        }
        if self.wrist_camera_key is not None and self.wrist_camera_key in item:
            sample["wrist_image"] = _image_to_uint8_hwc(item[self.wrist_camera_key])
        if self.use_proprioception:
            if self.state_key not in item:
                raise KeyError(f"Missing proprioception key '{self.state_key}' in dataset item.")
            sample["proprioception"] = item[self.state_key]
        if not self._logged_first_sample:
            self._logged_first_sample = True
            logger.info(
                "Loaded first sample episode=%s local_index=%s keys=%s",
                spec.episode_index,
                spec.local_index,
                sorted(sample),
            )
        return sample

    def _reader_for_episode(self, episode_index: int) -> EpisodeFrameReader:
        if episode_index in self._reader_cache:
            reader = self._reader_cache.pop(episode_index)
            self._reader_cache[episode_index] = reader
            return reader

        start = time.perf_counter()
        logger.info("Opening episode reader episode=%s", episode_index)
        reader = _load_episode_reader(self.reader_args, self.meta, episode_index)
        logger.info(
            "Opened episode reader episode=%s len=%s elapsed=%.2fs",
            episode_index,
            len(reader),
            time.perf_counter() - start,
        )
        self._reader_cache[episode_index] = reader
        while len(self._reader_cache) > self.reader_cache_size:
            self._reader_cache.popitem(last=False)
        return reader


def build_sample_specs(args: argparse.Namespace, meta: LeRobotDatasetMetadata) -> tuple[list[RewardSampleSpec], dict[str, Any]]:
    all_specs: list[RewardSampleSpec] = []
    task_summaries: dict[str, Any] = {}
    task_orders = parse_task_orders(args.task_orders)
    logger.info("Building sample specs for task_orders=%s", task_orders)
    for task_order in task_orders:
        start = time.perf_counter()
        logger.info("Resolving task_order=%s", task_order)
        task_metadata = _get_libero_task(args.suite, task_order)
        task_language = str(task_metadata["language"])
        logger.info("Finding episodes for task_order=%s task=%r", task_order, task_language)
        episodes = find_episodes_for_task(meta, task_metadata=task_metadata, limit=args.episodes_per_task)
        logger.info(
            "Found episodes for task_order=%s episodes=%s elapsed=%.2fs",
            task_order,
            episodes,
            time.perf_counter() - start,
        )
        task_summaries[str(task_order)] = {
            "task": task_language,
            "episodes": episodes,
        }

        for episode_index in episodes:
            episode = meta.episodes[episode_index]
            length = max(0, int(episode["dataset_to_index"]) - int(episode["dataset_from_index"]))
            local_indices = _sample_local_indices(
                length,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames_per_episode,
            )
            denom = max(1, length - 1)
            before_count = len(all_specs)
            for local_index in local_indices:
                all_specs.append(
                    RewardSampleSpec(
                        task=task_language,
                        episode_index=episode_index,
                        local_index=local_index,
                        label=float(local_index) / float(denom),
                    )
                )
            logger.info(
                "Added samples task_order=%s episode=%s length=%s sampled=%s total=%s",
                task_order,
                episode_index,
                length,
                len(all_specs) - before_count,
                len(all_specs),
            )

    return all_specs, task_summaries


def split_specs(
    specs: list[RewardSampleSpec],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[RewardSampleSpec], list[RewardSampleSpec]]:
    by_episode: dict[int, list[RewardSampleSpec]] = defaultdict(list)
    for spec in specs:
        by_episode[spec.episode_index].append(spec)
    episode_indices = list(by_episode)
    rng = random.Random(seed)
    rng.shuffle(episode_indices)
    val_count = max(1, round(len(episode_indices) * val_fraction)) if len(episode_indices) > 1 else 0
    val_episodes = set(episode_indices[:val_count])
    train = [spec for spec in specs if spec.episode_index not in val_episodes]
    val = [spec for spec in specs if spec.episode_index in val_episodes]
    if not train:
        train, val = specs, []
    return train, val


def model_forward(model: MultiModalRewardModel, batch: dict[str, Tensor]) -> Tensor:
    return model(
        scene_pixel_values=batch["scene_pixel_values"],
        wrist_pixel_values=batch.get("wrist_pixel_values"),
        input_ids=batch.get("input_ids"),
        attention_mask=batch.get("attention_mask"),
        proprioception=batch.get("proprioception"),
    )


def ranking_loss(pred: Tensor, labels: Tensor, episode_indices: Tensor, *, margin: float) -> Tensor:
    losses: list[Tensor] = []
    for episode_index in episode_indices.unique():
        mask = episode_indices == episode_index
        if int(mask.sum()) < 2:
            continue
        p = pred[mask]
        y = labels[mask]
        diff_y = y[:, None] - y[None, :]
        pair_mask = diff_y > 0.05
        if pair_mask.any():
            diff_p = p[:, None] - p[None, :]
            losses.append(torch.relu(margin - diff_p[pair_mask]).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def run_epoch(
    *,
    model: MultiModalRewardModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    ranking_weight: float,
    ranking_margin: float,
    epoch_name: str,
    log_every_batches: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    start = time.perf_counter()
    total_batches = len(loader)
    logger.info("%s start batches=%s samples=%s", epoch_name, total_batches, len(loader.dataset))
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            logger.info("%s first batch loaded keys=%s", epoch_name, sorted(batch))
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        with torch.enable_grad() if is_train else torch.no_grad():
            pred = model_forward(model, batch)
            mse = torch.nn.functional.mse_loss(pred, labels)
            rank = ranking_loss(pred, labels, batch["episode_indices"], margin=ranking_margin)
            loss = mse + ranking_weight * rank

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        count = int(labels.numel())
        total_loss += float(loss.detach()) * count
        total_mse += float(mse.detach()) * count
        total_mae += float(torch.mean(torch.abs(pred.detach() - labels))) * count
        total_count += count
        if log_every_batches > 0 and (batch_idx == 0 or (batch_idx + 1) % log_every_batches == 0):
            elapsed = time.perf_counter() - start
            logger.info(
                "%s batch=%s/%s samples=%s loss=%.4f mse=%.4f rank=%.4f elapsed=%.1fs",
                epoch_name,
                batch_idx + 1,
                total_batches,
                total_count,
                float(loss.detach()),
                float(mse.detach()),
                float(rank.detach()),
                elapsed,
            )

    denom = max(1, total_count)
    return {
        "loss": total_loss / denom,
        "mse": total_mse / denom,
        "mae": total_mae / denom,
        "count": float(total_count),
    }


@torch.no_grad()
def evaluate_records(
    *,
    model: MultiModalRewardModel,
    dataset: LiberoRewardFrameDataset,
    processor: RewardBatchProcessor,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=processor)
    model.eval()
    records: list[dict[str, Any]] = []
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = model_forward(model, batch)
        labels = batch["labels"]
        total_mse += float(torch.nn.functional.mse_loss(pred, labels, reduction="sum"))
        total_mae += float(torch.sum(torch.abs(pred - labels)))
        total_count += int(labels.numel())
        for ix in range(int(labels.numel())):
            records.append(
                {
                    "episode_index": int(batch["episode_indices"][ix].cpu()),
                    "local_index": int(batch["local_indices"][ix].cpu()),
                    "label": float(labels[ix].cpu()),
                    "prediction": float(pred[ix].cpu()),
                }
            )

    violations = 0
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[int(record["episode_index"])].append(record)
    for episode_records in by_episode.values():
        ordered = sorted(episode_records, key=lambda item: int(item["local_index"]))
        for prev, cur in zip(ordered, ordered[1:], strict=False):
            if float(cur["prediction"]) + 0.02 < float(prev["prediction"]):
                violations += 1

    denom = max(1, total_count)
    return {
        "mse": total_mse / denom,
        "mae": total_mae / denom,
        "count": float(total_count),
        "monotonic_violations": float(violations),
    }, records


def write_eval_outputs(records: list[dict[str, Any]], metrics: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "eval_metrics.json").write_text(json.dumps(_to_jsonable(metrics), indent=2))
    with (output_dir / "eval_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_index", "local_index", "label", "prediction"],
        )
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task_orders", default="all")
    parser.add_argument("--episodes_per_task", type=int, default=5)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_frames_per_episode", type=int, default=32)
    parser.add_argument("--scene_camera_key", default="observation.images.image")
    parser.add_argument("--wrist_camera_key", default="observation.images.image2")
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--download_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_cache_sync", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument("--encoder_type", choices=["siglip2", "siglip", "clip", "resnet18", "resnet34"], default="siglip2")
    parser.add_argument("--encoder_model_id", default=None)
    parser.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_proprioception", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proprioception_dim", type=int, default=8)
    parser.add_argument("--proprioception_hidden_dim", type=int, default=64)
    parser.add_argument("--head_hidden_dim", type=int, default=512)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ranking_weight", type=float, default=0.2)
    parser.add_argument("--ranking_margin", type=float, default=0.05)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--log_every_batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training args: %s", vars(args))
    metadata_start = time.perf_counter()
    logger.info("Loading metadata for %s", args.dataset_repo_id)
    meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
    )
    logger.info(
        "Loaded metadata total_episodes=%s elapsed=%.2fs",
        meta.total_episodes,
        time.perf_counter() - metadata_start,
    )
    specs, task_summaries = build_sample_specs(args, meta)
    train_specs, val_specs = split_specs(specs, val_fraction=args.val_fraction, seed=args.seed)
    logger.info("Built %d samples: train=%d val=%d", len(specs), len(train_specs), len(val_specs))

    model_id = args.encoder_model_id or default_model_id(args.encoder_type)
    model_cfg = RewardModelConfig(
        encoder_type=args.encoder_type,
        encoder_model_id=model_id,
        freeze_encoder=args.freeze_encoder,
        use_proprioception=args.use_proprioception,
        proprioception_dim=args.proprioception_dim,
        proprioception_hidden_dim=args.proprioception_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    )
    logger.info("Model config: %s", model_cfg.to_dict())
    processor = RewardBatchProcessor(model_cfg)
    reader_args = _reader_args(args)
    train_ds = LiberoRewardFrameDataset(
        specs=train_specs,
        meta=meta,
        reader_args=reader_args,
        scene_camera_key=args.scene_camera_key,
        wrist_camera_key=args.wrist_camera_key,
        state_key=args.state_key,
        use_proprioception=args.use_proprioception,
    )
    val_ds = LiberoRewardFrameDataset(
        specs=val_specs,
        meta=meta,
        reader_args=reader_args,
        scene_camera_key=args.scene_camera_key,
        wrist_camera_key=args.wrist_camera_key,
        state_key=args.state_key,
        use_proprioception=args.use_proprioception,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=processor)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=processor)

    device = torch.device(args.device)
    logger.info("Creating model on device=%s", device)
    model = MultiModalRewardModel(model_cfg).to(device)
    logger.info("Model ready.")
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
            epoch_name=f"train epoch={epoch}",
            log_every_batches=args.log_every_batches,
        )
        val_metrics = (
            run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                ranking_weight=args.ranking_weight,
                ranking_margin=args.ranking_margin,
                epoch_name=f"val epoch={epoch}",
                log_every_batches=args.log_every_batches,
            )
            if len(val_ds)
            else {"loss": 0.0, "mse": 0.0, "mae": 0.0, "count": 0.0}
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        logger.info("epoch=%s train=%s val=%s", epoch, train_metrics, val_metrics)
        if val_metrics["mse"] <= best_val:
            best_val = val_metrics["mse"]
            torch.save(
                {
                    "model_config": model_cfg.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "task_summaries": task_summaries,
                    "history": history,
                },
                args.output_dir / "best_reward_model.pt",
            )

    eval_metrics, eval_records = (
        evaluate_records(
            model=model,
            dataset=val_ds if len(val_ds) else train_ds,
            processor=processor,
            batch_size=args.batch_size,
            device=device,
        )
        if len(specs)
        else ({}, [])
    )
    summary = {
        "model_config": model_cfg.to_dict(),
        "task_summaries": task_summaries,
        "train_sample_count": len(train_specs),
        "val_sample_count": len(val_specs),
        "history": history,
        "eval": eval_metrics,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2))
    write_eval_outputs(eval_records, eval_metrics, args.output_dir)
    logger.info("Saved reward model outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
