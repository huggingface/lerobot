#!/usr/bin/env python
"""Train a lightweight LIBERO progress reward model for tree search.

The default setup trains a frozen SigLIP2 encoder plus a small MLP head. Frames
from successful demonstrations are labeled by normalized episode progress:
0.0 at the first sampled frame and 1.0 at the final frame. It also adds
wrong-instruction negatives by pairing each frame with another task text and a
low reward label, so the rewarder is forced to use language conditioning.
Scene-camera inputs can be temporal windows while wrist/proprioception remain
current-frame inputs. Multiple comma-separated dataset repo ids can be mixed
when they share the same LeRobot/LIBERO schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from lerobot.datasets import LeRobotDatasetMetadata

from reward_model import (
    MultiModalRewardModel,
    RewardBatchProcessor,
    RewardModelConfig,
    default_model_id,
    move_batch_to_device,
)
from policy_inference_api import _load_task_language_map, _translate_task_language
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


def stage(message: str) -> None:
    print(f"[reward-train] {message}", flush=True)
    logger.info(message)


def _language_map_stop_words(translations: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = translations.get("__stop_words__")
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError(f"Unsupported __stop_words__ value in task language map: {value!r}")


@dataclass
class RewardSampleSpec:
    dataset_id: int
    dataset_repo_id: str
    task: str
    episode_index: int
    local_index: int
    label: float
    source_task_order: int | None = None
    text_task_order: int | None = None
    is_text_mismatch: bool = False
    is_bad_sequence: bool = False
    source_task: str | None = None


@dataclass
class DatasetSource:
    dataset_id: int
    repo_id: str
    meta: LeRobotDatasetMetadata
    reader_args: SimpleNamespace


def parse_task_orders(raw: str) -> list[int]:
    text = raw.strip()
    if text.lower() == "all":
        return list(range(10))
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_dataset_repo_ids(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return [item.strip() for item in text.split(",") if item.strip()]


def find_episodes_for_task(
    meta: LeRobotDatasetMetadata,
    *,
    task_metadata: dict[str, Any],
    limit: int,
    allowed_frame_indices: set[int] | None = None,
    require_matches: bool = True,
) -> list[int]:
    targets = {
        _normalize_text(str(task_metadata.get("language", ""))),
        _normalize_text(str(task_metadata.get("name", ""))),
    }
    targets.discard("")
    matches: list[int] = []
    for ep_idx in range(meta.total_episodes):
        episode = meta.episodes[ep_idx]
        episode_from = int(episode["dataset_from_index"])
        episode_to = int(episode["dataset_to_index"])
        if allowed_frame_indices is not None and not any(
            frame_index in allowed_frame_indices for frame_index in range(episode_from, episode_to)
        ):
            continue
        if any(_normalize_text(task) in targets for task in _episode_tasks(meta, episode)):
            matches.append(ep_idx)
            if len(matches) >= limit:
                break
    if not matches and require_matches:
        raise SystemExit(f"No dataset episodes matched task targets: {sorted(targets)}")
    return matches


def _parse_split_range(value: str, *, total_count: int) -> set[int]:
    text = str(value).strip()
    if not text:
        return set()
    if "," in text:
        indices: set[int] = set()
        for part in text.split(","):
            indices.update(_parse_split_range(part, total_count=total_count))
        return indices
    if ":" in text:
        start_raw, end_raw = text.split(":", 1)
        start = int(start_raw) if start_raw else 0
        end = int(end_raw) if end_raw else total_count
        start = max(0, start)
        end = min(total_count, end)
        if end < start:
            raise ValueError(f"Invalid split range {value!r}: end < start")
        return set(range(start, end))
    index = int(text)
    if index < 0 or index >= total_count:
        raise ValueError(f"Split index {index} out of range [0, {total_count - 1}]")
    return {index}


def _episode_indices_to_frame_indices(meta: LeRobotDatasetMetadata, episode_indices: set[int]) -> set[int]:
    frame_indices: set[int] = set()
    for episode_index in episode_indices:
        episode = meta.episodes[int(episode_index)]
        frame_indices.update(range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"])))
    return frame_indices


def _fallback_frame_splits(total_frames: int, *, val_fraction: float) -> dict[str, set[int]]:
    if val_fraction <= 0 or total_frames <= 1:
        return {"train": set(range(total_frames)), "validation": set()}
    val_count = max(1, int(round(total_frames * val_fraction)))
    val_count = min(val_count, total_frames - 1)
    train_count = total_frames - val_count
    return {
        "train": set(range(0, train_count)),
        "validation": set(range(train_count, total_frames)),
    }


def dataset_split_frames(
    meta: LeRobotDatasetMetadata,
    *,
    repo_id: str,
    fallback_val_fraction: float,
) -> tuple[dict[str, set[int]], str]:
    total_frames = int(meta.info.get("total_frames", 0))
    raw_splits = meta.info.get("splits")
    if not isinstance(raw_splits, dict):
        return _fallback_frame_splits(total_frames, val_fraction=fallback_val_fraction), "fallback_frame_fraction"
    split_unit = str(meta.info.get("split_unit", "episode"))
    if split_unit == "frame":
        splits = {
            name: _parse_split_range(value, total_count=total_frames)
            for name, value in raw_splits.items()
        }
        split_source = "dataset_meta_info"
    elif split_unit == "episode":
        episode_splits = {
            name: _parse_split_range(value, total_count=meta.total_episodes)
            for name, value in raw_splits.items()
        }
        splits = {
            name: _episode_indices_to_frame_indices(meta, episode_indices)
            for name, episode_indices in episode_splits.items()
        }
        split_source = "dataset_meta_info_episode_ranges"
    else:
        raise ValueError(
            f"Dataset {repo_id} must define split_unit as 'frame' or omit it for legacy episode ranges; "
            f"got split_unit={split_unit!r}."
        )
    if "train" not in raw_splits:
        raise ValueError(f"Dataset {repo_id} must define a train split if meta/info.json splits are present.")
    if "validation" not in splits:
        train = splits["train"]
        validation = set(range(total_frames)) - train
        if not validation:
            fallback = _fallback_frame_splits(total_frames, val_fraction=fallback_val_fraction)
            train = fallback["train"]
            validation = fallback["validation"]
        splits["train"] = train
        splits["validation"] = validation
        split_source = "fallback_frame_fraction"
    train = splits["train"]
    validation = splits["validation"]
    if not train:
        raise ValueError(f"Dataset {repo_id} has an empty train frame split.")
    if not validation:
        stage(f"Dataset {repo_id} has no validation frames; final eval will be skipped for this dataset.")
    overlap = train & validation
    if overlap:
        raise ValueError(f"Dataset {repo_id} train/validation frame splits overlap: {sorted(overlap)[:10]}")
    return {"train": train, "validation": validation}, split_source


def _reader_args(args: argparse.Namespace, *, dataset_repo_id: str, dataset_root: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        force_cache_sync=args.force_cache_sync,
        download_videos=args.download_videos,
        video_backend=args.video_backend,
    )


class LiberoRewardFrameDataset(Dataset):
    def __init__(
        self,
        *,
        specs: list[RewardSampleSpec],
        sources: list[DatasetSource],
        split_frame_indices_by_dataset: dict[int, set[int]],
        scene_camera_key: str,
        wrist_camera_key: str | None,
        state_key: str,
        use_proprioception: bool,
        scene_temporal_window: int,
        scene_temporal_stride: int,
        bad_sequence_max_reward: float,
        bad_sequence_decay: float,
        reader_cache_size: int = 128,
        log_reader_opens: bool = False,
    ) -> None:
        self.specs = specs
        self.sources = {source.dataset_id: source for source in sources}
        self.split_frame_indices_by_dataset = split_frame_indices_by_dataset
        self.scene_camera_key = scene_camera_key
        self.wrist_camera_key = wrist_camera_key
        self.state_key = state_key
        self.use_proprioception = use_proprioception
        self.scene_temporal_window = max(1, int(scene_temporal_window))
        self.scene_temporal_stride = max(1, int(scene_temporal_stride))
        self.bad_sequence_max_reward = float(bad_sequence_max_reward)
        self.bad_sequence_decay = float(bad_sequence_decay)
        self.reader_cache_size = max(1, reader_cache_size)
        self.log_reader_opens = bool(log_reader_opens)
        self._reader_cache: OrderedDict[tuple[int, int], EpisodeFrameReader] = OrderedDict()
        self._logged_first_sample = False

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.specs[int(index)]
        source = self.sources[spec.dataset_id]
        reader = self._reader_for_episode(spec.dataset_id, spec.episode_index)
        item = reader[spec.local_index]
        if self.scene_camera_key not in item:
            raise KeyError(f"Missing scene camera key '{self.scene_camera_key}' in dataset item.")
        episode = source.meta.episodes[spec.episode_index]
        episode_from = int(episode["dataset_from_index"])
        temporal_indices = self._temporal_indices(
            spec.local_index,
            dataset_id=spec.dataset_id,
            episode_from=episode_from,
        )
        scene_images = [
            _image_to_uint8_hwc(reader[temporal_index][self.scene_camera_key])
            for temporal_index in temporal_indices
        ]
        is_bad_sequence = self._coerce_bool(item.get("is_bad_sequence", False))
        label = self._label_for_sample(spec=spec, reader=reader, is_bad_sequence=is_bad_sequence)

        sample: dict[str, Any] = {
            "dataset_id": spec.dataset_id,
            "dataset_repo_id": spec.dataset_repo_id,
            "task": spec.task,
            "episode_index": spec.episode_index,
            "local_index": spec.local_index,
            "temporal_local_indices": temporal_indices,
            "label": label,
            "is_bad_sequence": is_bad_sequence,
            "spec_is_bad_sequence": spec.is_bad_sequence,
            "source_task_order": spec.source_task_order if spec.source_task_order is not None else -1,
            "text_task_order": spec.text_task_order if spec.text_task_order is not None else -1,
            "is_text_mismatch": spec.is_text_mismatch,
            "source_task": spec.source_task or spec.task,
            "scene_image": _image_to_uint8_hwc(item[self.scene_camera_key]),
            "scene_images": scene_images,
        }
        if self.wrist_camera_key is not None and self.wrist_camera_key in item:
            sample["wrist_image"] = _image_to_uint8_hwc(item[self.wrist_camera_key])
        if self.use_proprioception:
            if self.state_key not in item:
                raise KeyError(f"Missing proprioception key '{self.state_key}' in dataset item.")
            sample["proprioception"] = item[self.state_key]
        if not self._logged_first_sample:
            self._logged_first_sample = True
            stage(
                f"Loaded first sample dataset={source.repo_id} episode={spec.episode_index} "
                f"local_index={spec.local_index} temporal_indices={temporal_indices} keys={sorted(sample)}"
            )
        return sample

    def _temporal_indices(self, local_index: int, *, dataset_id: int, episode_from: int) -> list[int]:
        allowed = self.split_frame_indices_by_dataset.get(dataset_id)
        current = int(local_index)
        indices: list[int] = []
        for offset in reversed(range(self.scene_temporal_window)):
            candidate = max(0, current - self.scene_temporal_stride * offset)
            if allowed is not None:
                while candidate < current and episode_from + candidate not in allowed:
                    candidate += 1
                if episode_from + candidate not in allowed:
                    candidate = current
            indices.append(candidate)
        return indices

    def _label_for_sample(
        self,
        *,
        spec: RewardSampleSpec,
        reader: EpisodeFrameReader,
        is_bad_sequence: bool,
    ) -> float:
        if spec.is_text_mismatch:
            return float(spec.label)
        if not is_bad_sequence:
            return float(spec.label)
        if len(reader) <= 1:
            return float(self.bad_sequence_max_reward)
        progress = min(1.0, max(0.0, float(spec.local_index) / float(len(reader) - 1)))
        if self.bad_sequence_decay <= 0:
            return float(self.bad_sequence_max_reward * (1.0 - progress))
        decay_end = math.exp(-float(self.bad_sequence_decay))
        numerator = math.exp(-float(self.bad_sequence_decay) * progress) - decay_end
        denominator = max(1e-8, 1.0 - decay_end)
        return float(max(0.0, self.bad_sequence_max_reward * numerator / denominator))

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return False
            return bool(value.reshape(-1)[0])
        try:
            if np.isnan(value):
                return False
        except TypeError:
            pass
        return bool(value)

    def _reader_for_episode(self, dataset_id: int, episode_index: int) -> EpisodeFrameReader:
        key = (dataset_id, episode_index)
        if key in self._reader_cache:
            reader = self._reader_cache.pop(key)
            self._reader_cache[key] = reader
            return reader

        source = self.sources[dataset_id]
        start = time.perf_counter()
        if self.log_reader_opens:
            stage(f"Opening episode reader dataset={source.repo_id} episode={episode_index}")
        reader = _load_episode_reader(source.reader_args, source.meta, episode_index)
        if self.log_reader_opens:
            stage(
                f"Opened episode reader dataset={source.repo_id} episode={episode_index} "
                f"len={len(reader)} elapsed={time.perf_counter() - start:.2f}s"
            )
        self._reader_cache[key] = reader
        while len(self._reader_cache) > self.reader_cache_size:
            self._reader_cache.popitem(last=False)
        return reader


def build_sample_specs(
    args: argparse.Namespace,
    source: DatasetSource,
    split_frame_indices: set[int],
    split_name: str,
    task_language_translations: Mapping[str, Any],
    require_task_matches: bool = False,
) -> tuple[list[RewardSampleSpec], dict[str, Any]]:
    meta = source.meta
    all_specs: list[RewardSampleSpec] = []
    task_summaries: dict[str, Any] = {}
    task_orders = parse_task_orders(args.task_orders)
    task_languages: dict[int, str] = {}
    stage(
        f"Building sample specs for dataset={source.repo_id} split={split_name} "
        f"frames={len(split_frame_indices)} task_orders={task_orders}"
    )
    for task_order in task_orders:
        start = time.perf_counter()
        stage(f"Resolving task_order={task_order}")
        task_metadata = _get_libero_task(args.suite, task_order)
        original_task_language = str(task_metadata["language"])
        task_language = _translate_task_language(
            original_task_language,
            suite=args.suite,
            task_id=task_order,
            translations=task_language_translations,
        )
        task_languages[task_order] = task_language
        stage(
            f"Finding episodes for task_order={task_order} task={original_task_language!r} "
            f"train_text={task_language!r}"
        )
        episodes = find_episodes_for_task(
            meta,
            task_metadata=task_metadata,
            limit=args.episodes_per_task,
            allowed_frame_indices=split_frame_indices,
            require_matches=require_task_matches,
        )
        if not episodes:
            stage(
                f"No episodes for task_order={task_order} split={split_name}; skipping this task in this split."
            )
            task_summaries[str(task_order)] = {
                "task": task_language,
                "original_task": original_task_language,
                "dataset_repo_id": source.repo_id,
                "split": split_name,
                "episodes": [],
                "skipped": True,
            }
            continue
        stage(
            f"Found episodes for task_order={task_order} split={split_name} "
            f"episodes={episodes} elapsed={time.perf_counter() - start:.2f}s"
        )
        task_summaries[str(task_order)] = {
            "task": task_language,
            "original_task": original_task_language,
            "dataset_repo_id": source.repo_id,
            "split": split_name,
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
            episode_from = int(episode["dataset_from_index"])
            local_indices = [
                local_index
                for local_index in local_indices
                if episode_from + int(local_index) in split_frame_indices
            ]
            if not local_indices:
                continue
            denom = max(1, length - 1)
            before_count = len(all_specs)
            reader_for_flags: EpisodeFrameReader | None = None
            for local_index in local_indices:
                is_bad_sequence = False
                if "is_bad_sequence" in getattr(meta, "features", {}):
                    if reader_for_flags is None:
                        reader_for_flags = _load_episode_reader(source.reader_args, meta, episode_index)
                    is_bad_sequence = LiberoRewardFrameDataset._coerce_bool(
                        reader_for_flags[int(local_index)].get("is_bad_sequence", False)
                    )
                all_specs.append(
                    RewardSampleSpec(
                        dataset_id=source.dataset_id,
                        dataset_repo_id=source.repo_id,
                        task=task_language,
                        episode_index=episode_index,
                        local_index=local_index,
                        label=float(local_index) / float(denom),
                        source_task_order=task_order,
                        text_task_order=task_order,
                        is_text_mismatch=False,
                        is_bad_sequence=is_bad_sequence,
                        source_task=task_language,
                    )
                )
            stage(
                f"Added samples task_order={task_order} episode={episode_index} "
                f"length={length} sampled={len(all_specs) - before_count} total={len(all_specs)}"
            )

    negative_count = int(args.wrong_text_negatives_per_sample) if args.use_wrong_text_negatives else 0
    if negative_count > 0:
        if len(task_languages) < 2:
            stage("Skipping wrong-text negatives because fewer than two task texts are available.")
        else:
            rng = random.Random(args.seed)
            positives = list(all_specs)
            negative_specs: list[RewardSampleSpec] = []
            for spec in positives:
                if spec.is_text_mismatch or spec.is_bad_sequence:
                    continue
                candidates = [order for order in task_orders if order != spec.source_task_order]
                rng.shuffle(candidates)
                for wrong_order in candidates[:negative_count]:
                    negative_specs.append(
                        RewardSampleSpec(
                            dataset_id=spec.dataset_id,
                            dataset_repo_id=spec.dataset_repo_id,
                            task=task_languages[wrong_order],
                            episode_index=spec.episode_index,
                            local_index=spec.local_index,
                            label=float(args.wrong_text_negative_label),
                            source_task_order=spec.source_task_order,
                            text_task_order=wrong_order,
                            is_text_mismatch=True,
                            is_bad_sequence=False,
                            source_task=spec.source_task or spec.task,
                        )
                    )
            all_specs.extend(negative_specs)
            stage(
                "Added wrong-text negatives "
                f"count={len(negative_specs)} per_positive={negative_count} "
                f"negative_label={args.wrong_text_negative_label} total={len(all_specs)}"
            )

    return all_specs, task_summaries


def model_forward(model: MultiModalRewardModel, batch: dict[str, Any]) -> Tensor:
    return model(
        scene_pixel_values=batch["scene_pixel_values"],
        wrist_pixel_values=batch.get("wrist_pixel_values"),
        input_ids=batch.get("input_ids"),
        attention_mask=batch.get("attention_mask"),
        query_input_ids=batch.get("query_input_ids"),
        query_attention_mask=batch.get("query_attention_mask"),
        text_query_mask=batch.get("text_query_mask"),
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
    stage(f"{epoch_name} start batches={total_batches} samples={len(loader.dataset)}")
    progress = tqdm(
        enumerate(loader),
        total=total_batches,
        desc=epoch_name,
        dynamic_ncols=True,
        unit="batch",
        leave=True,
    )
    for batch_idx, batch in progress:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        with torch.enable_grad() if is_train else torch.no_grad():
            pred = model_forward(model, batch)
            mse = torch.nn.functional.mse_loss(pred, labels)
            loss = mse

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
            progress.set_postfix(
                {
                    "loss": f"{float(loss.detach()):.4f}",
                    "mse": f"{float(mse.detach()):.4f}",
                    "samples": total_count,
                    "elapsed": f"{elapsed:.1f}s",
                },
                refresh=True,
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
    progress = tqdm(loader, total=len(loader), desc="final eval", dynamic_ncols=True, unit="batch", leave=True)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        pred = model_forward(model, batch)
        labels = batch["labels"]
        total_mse += float(torch.nn.functional.mse_loss(pred, labels, reduction="sum"))
        total_mae += float(torch.sum(torch.abs(pred - labels)))
        total_count += int(labels.numel())
        progress.set_postfix({"samples": total_count}, refresh=True)
        for ix in range(int(labels.numel())):
            records.append(
                {
                    "dataset_id": int(batch["dataset_ids"][ix].cpu()),
                    "dataset_repo_id": batch["dataset_repo_ids"][ix],
                    "episode_index": int(batch["raw_episode_indices"][ix].cpu()),
                    "local_index": int(batch["local_indices"][ix].cpu()),
                    "temporal_local_indices": batch["temporal_local_indices"][ix],
                    "is_bad_sequence": bool(batch["is_bad_sequence"][ix].cpu()),
                    "source_task_order": int(batch["source_task_orders"][ix].cpu()),
                    "text_task_order": int(batch["text_task_orders"][ix].cpu()),
                    "is_text_mismatch": bool(batch["is_text_mismatch"][ix].cpu()),
                    "source_task": batch["source_tasks"][ix],
                    "task": batch["tasks"][ix],
                    "label": float(labels[ix].cpu()),
                    "prediction": float(pred[ix].cpu()),
                }
            )

    violations = 0
    by_episode: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[(int(record["dataset_id"]), int(record["episode_index"]))].append(record)
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
            fieldnames=[
                "episode_index",
                "dataset_id",
                "dataset_repo_id",
                "local_index",
                "temporal_local_indices",
                "is_bad_sequence",
                "source_task_order",
                "text_task_order",
                "is_text_mismatch",
                "source_task",
                "task",
                "label",
                "prediction",
            ],
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
    parser.add_argument("--scene_temporal_window", type=int, default=1)
    parser.add_argument("--scene_temporal_stride", type=int, default=10)
    parser.add_argument("--task_language_map", type=Path, default=None)
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
    parser.add_argument("--use_patch_text_fusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text_query_ngram_window", type=int, default=3)
    parser.add_argument("--text_query_ngram_stride", type=int, default=2)
    parser.add_argument("--text_query_include_full", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text_query_include_tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text_query_max_count", type=int, default=12)
    parser.add_argument("--patch_attention_dim", type=int, default=256)
    parser.add_argument("--scene_summary_dim", type=int, default=512)
    parser.add_argument("--wrist_summary_dim", type=int, default=128)
    parser.add_argument("--head_hidden_dim", type=int, default=512)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ranking_weight", type=float, default=0.2)
    parser.add_argument("--ranking_margin", type=float, default=0.05)
    parser.add_argument("--use_wrong_text_negatives", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wrong_text_negatives_per_sample", type=int, default=1)
    parser.add_argument("--wrong_text_negative_label", type=float, default=0.0)
    parser.add_argument("--bad_sequence_max_reward", type=float, default=0.2)
    parser.add_argument("--bad_sequence_decay", type=float, default=4.0)
    parser.add_argument("--fallback_val_fraction", type=float, default=0.2)
    parser.add_argument("--log_every_batches", type=int, default=5)
    parser.add_argument("--reader_cache_size", type=int, default=128)
    parser.add_argument("--log_reader_opens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stage("Process started")
    args = parse_args()
    stage("Args parsed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    task_language_translations = _load_task_language_map(args.task_language_map)
    map_stop_words = _language_map_stop_words(task_language_translations)
    if task_language_translations:
        stage(
            f"Loaded task language map path={args.task_language_map} keys={len(task_language_translations)}"
        )
    if map_stop_words:
        stage(f"Loaded text-query stop_words from language map count={len(map_stop_words)}")
    stage(f"Creating output_dir={args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage(f"Training args={vars(args)}")
    dataset_repo_ids = parse_dataset_repo_ids(args.dataset_repo_id)
    if len(dataset_repo_ids) > 1 and args.dataset_root is not None:
        raise ValueError("--dataset_root can only be used with a single --dataset_repo_id.")

    sources: list[DatasetSource] = []
    train_specs: list[RewardSampleSpec] = []
    val_specs: list[RewardSampleSpec] = []
    train_split_frames_by_dataset: dict[int, set[int]] = {}
    val_split_frames_by_dataset: dict[int, set[int]] = {}
    task_summaries: dict[str, Any] = {}
    for dataset_id, dataset_repo_id in enumerate(dataset_repo_ids):
        metadata_start = time.perf_counter()
        stage(f"Loading metadata for dataset_id={dataset_id} dataset_repo_id={dataset_repo_id}")
        meta = LeRobotDatasetMetadata(
            dataset_repo_id,
            root=args.dataset_root if len(dataset_repo_ids) == 1 else None,
            revision=args.dataset_revision,
        )
        stage(
            f"Loaded metadata dataset_id={dataset_id} total_episodes={meta.total_episodes} "
            f"elapsed={time.perf_counter() - metadata_start:.2f}s"
        )
        source = DatasetSource(
            dataset_id=dataset_id,
            repo_id=dataset_repo_id,
            meta=meta,
            reader_args=_reader_args(
                args,
                dataset_repo_id=dataset_repo_id,
                dataset_root=args.dataset_root if len(dataset_repo_ids) == 1 else None,
            ),
        )
        sources.append(source)
        split_frames, split_source = dataset_split_frames(
            meta,
            repo_id=dataset_repo_id,
            fallback_val_fraction=args.fallback_val_fraction,
        )
        train_split_frames_by_dataset[dataset_id] = split_frames["train"]
        val_split_frames_by_dataset[dataset_id] = split_frames["validation"]
        stage(
            f"Dataset splits dataset_id={dataset_id} "
            f"train_frames={len(split_frames['train'])} "
            f"validation_frames={len(split_frames['validation'])}"
        )
        stage(f"Building train sample specs for dataset_id={dataset_id}")
        source_train_specs, source_train_task_summaries = build_sample_specs(
            args,
            source,
            split_frame_indices=split_frames["train"],
            split_name="train",
            task_language_translations=task_language_translations,
            require_task_matches=False,
        )
        stage(f"Building validation sample specs for dataset_id={dataset_id}")
        source_val_specs, source_val_task_summaries = build_sample_specs(
            args,
            source,
            split_frame_indices=split_frames["validation"],
            split_name="validation",
            task_language_translations=task_language_translations,
            require_task_matches=False,
        )
        train_specs.extend(source_train_specs)
        val_specs.extend(source_val_specs)
        task_summaries[str(dataset_id)] = {
            "dataset_repo_id": dataset_repo_id,
            "split_unit": "frame",
            "split_source": split_source,
            "split_frame_counts": {name: len(indices) for name, indices in split_frames.items()},
            "tasks": {
                "train": source_train_task_summaries,
                "validation": source_val_task_summaries,
            },
            "train_sample_count": len(source_train_specs),
            "val_sample_count": len(source_val_specs),
        }

    if not train_specs:
        raise ValueError("No training samples were built from dataset train splits.")
    if not val_specs:
        stage("No validation samples were built; validation metrics and final eval will be skipped.")
    stage(f"Built samples total={len(train_specs) + len(val_specs)} train={len(train_specs)} val={len(val_specs)}")

    model_id = args.encoder_model_id or default_model_id(args.encoder_type)
    stage(f"Preparing model config encoder_type={args.encoder_type} model_id={model_id}")
    model_cfg = RewardModelConfig(
        encoder_type=args.encoder_type,
        encoder_model_id=model_id,
        freeze_encoder=args.freeze_encoder,
        use_proprioception=args.use_proprioception,
        proprioception_dim=args.proprioception_dim,
        proprioception_hidden_dim=args.proprioception_hidden_dim,
        scene_temporal_window=args.scene_temporal_window,
        use_patch_text_fusion=args.use_patch_text_fusion,
        text_query_ngram_window=args.text_query_ngram_window,
        text_query_ngram_stride=args.text_query_ngram_stride,
        text_query_include_full=args.text_query_include_full,
        text_query_include_tail=args.text_query_include_tail,
        text_query_max_count=args.text_query_max_count,
        text_query_stop_words=map_stop_words
        if map_stop_words is not None
        else RewardModelConfig().text_query_stop_words,
        patch_attention_dim=args.patch_attention_dim,
        scene_summary_dim=args.scene_summary_dim,
        wrist_summary_dim=args.wrist_summary_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    )
    stage(f"Model config={model_cfg.to_dict()}")
    stage("Creating RewardBatchProcessor")
    processor = RewardBatchProcessor(model_cfg)
    stage("RewardBatchProcessor ready")
    stage("Creating datasets")
    train_ds = LiberoRewardFrameDataset(
        specs=train_specs,
        sources=sources,
        split_frame_indices_by_dataset=train_split_frames_by_dataset,
        scene_camera_key=args.scene_camera_key,
        wrist_camera_key=args.wrist_camera_key,
        state_key=args.state_key,
        use_proprioception=args.use_proprioception,
        scene_temporal_window=args.scene_temporal_window,
        scene_temporal_stride=args.scene_temporal_stride,
        bad_sequence_max_reward=args.bad_sequence_max_reward,
        bad_sequence_decay=args.bad_sequence_decay,
        reader_cache_size=args.reader_cache_size,
        log_reader_opens=args.log_reader_opens,
    )
    val_ds = LiberoRewardFrameDataset(
        specs=val_specs,
        sources=sources,
        split_frame_indices_by_dataset=val_split_frames_by_dataset,
        scene_camera_key=args.scene_camera_key,
        wrist_camera_key=args.wrist_camera_key,
        state_key=args.state_key,
        use_proprioception=args.use_proprioception,
        scene_temporal_window=args.scene_temporal_window,
        scene_temporal_stride=args.scene_temporal_stride,
        bad_sequence_max_reward=args.bad_sequence_max_reward,
        bad_sequence_decay=args.bad_sequence_decay,
        reader_cache_size=args.reader_cache_size,
        log_reader_opens=args.log_reader_opens,
    )
    stage(f"Datasets ready train_len={len(train_ds)} val_len={len(val_ds)}")

    stage("Creating DataLoaders")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=processor)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=processor)
    stage(f"DataLoaders ready train_batches={len(train_loader)} val_batches={len(val_loader)}")

    device = torch.device(args.device)
    stage(f"Creating model on device={device}")
    model = MultiModalRewardModel(model_cfg).to(device)
    stage("Model ready")
    stage("Creating optimizer")
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    stage("Optimizer ready; starting training")

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        stage(f"Starting epoch={epoch}")
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_name=f"train epoch={epoch}",
            log_every_batches=args.log_every_batches,
        )
        val_metrics = (
            run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                epoch_name=f"val epoch={epoch}",
                log_every_batches=args.log_every_batches,
            )
            if len(val_ds)
            else {"loss": 0.0, "mse": 0.0, "mae": 0.0, "count": 0.0}
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        stage(f"Finished epoch={epoch} train={train_metrics} val={val_metrics}")
        if val_metrics["mse"] <= best_val:
            best_val = val_metrics["mse"]
            stage(f"Saving best checkpoint epoch={epoch} best_val_mse={best_val:.6f}")
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

    stage("Running final evaluation records")
    eval_metrics, eval_records = (
        evaluate_records(
            model=model,
            dataset=val_ds if len(val_ds) else train_ds,
            processor=processor,
            batch_size=args.batch_size,
            device=device,
        )
        if len(val_specs)
        else ({}, [])
    )
    summary = {
        "model_config": model_cfg.to_dict(),
        "dataset_repo_ids": dataset_repo_ids,
        "split_source": "dataset_meta_info_or_fallback_frame_fraction",
        "split_unit": "frame",
        "task_language_map": str(args.task_language_map) if args.task_language_map is not None else None,
        "task_language_map_keys": len(task_language_translations),
        "task_summaries": task_summaries,
        "train_sample_count": len(train_specs),
        "val_sample_count": len(val_specs),
        "use_wrong_text_negatives": args.use_wrong_text_negatives,
        "train_text_mismatch_count": sum(int(spec.is_text_mismatch) for spec in train_specs),
        "val_text_mismatch_count": sum(int(spec.is_text_mismatch) for spec in val_specs),
        "scene_temporal_window": args.scene_temporal_window,
        "scene_temporal_stride": args.scene_temporal_stride,
        "bad_sequence_max_reward": args.bad_sequence_max_reward,
        "bad_sequence_decay": args.bad_sequence_decay,
        "history": history,
        "eval": eval_metrics,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2))
    write_eval_outputs(eval_records, eval_metrics, args.output_dir)
    stage(f"Saved reward model outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
