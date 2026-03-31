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
"""Train/val script for the standalone RECAP distributional value network.

Fixed CSV schema expected by this script:
- `episode_index`: episode identifier matching LeRobot dataset episode indices
- `success`: binary episode outcome label (1 for success, 0 for failure)
"""

import json
import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

from lerobot.configs import parser
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.rl.algorithms.recap_utils import collect_images
from lerobot.rl.algorithms.recap_value_network import RECAPValueNetwork, RECAPValueNetworkConfig

CSV_EPISODE_INDEX_COLUMN = "episode_index"
CSV_SUCCESS_COLUMN = "success"
DEFAULT_EPISODE_LABELS_FILENAME = "meta/episode_labels.csv"


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    task: str
    start_index: int
    end_index: int
    length: int


@dataclass(frozen=True)
class FrameTarget:
    frame_index: int
    episode_index: int
    success: int
    task: str
    target_value: float
    target_bin: int


@dataclass(frozen=True)
class ValidationFramePrediction:
    frame_index: int
    success: int
    target_value: float
    target_bin: int
    predicted_bin: int
    reconstructed_value: float
    predicted_probs: np.ndarray


@dataclass
class RECAPValueTrainingConfig:
    """Configuration for RECAP value-network train/val."""

    repo_id: str
    output_dir: str
    labels_csv_path: str | None = None
    root: str | None = None
    revision: str | None = None
    episodes: list[int] | None = None

    epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None
    log_every_n_steps: int = 100
    validate_every_n_train_steps: int = 0
    plot_every_n_train_steps: int = 0
    max_val_steps_per_step_validation: int | None = None
    val_plot_num_episodes: int = 4
    val_plot_num_frames: int = 8
    val_plot_every_n_epochs: int = 1

    # Value target construction
    c_fail: float = 24.0
    num_value_bins: int = 50

    # Input processing
    tokenizer_max_length: int = 200
    image_size: int = 224

    # PaliGemma VLM backbone
    paligemma_variant: str = "gemma_300m"
    tokenizer_name: str = "google/paligemma-3b-pt-224"
    model_precision: str = "float32"
    freeze_vision_encoder: bool = False
    freeze_backbone: bool = False
    num_unfrozen_backbone_layers: int = 0
    num_vlm_layers: int = 18
    value_head_depth: int = 1
    dropout: float = 0.1

    # Pretrained VLM initialisation (e.g. "lerobot/pi05_base")
    pretrained_path: str | None = None

    # Weights & Biases (optional; set wandb_project to enable)
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to 0."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins > 0:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


def _is_known_video_validation_error(error: Exception) -> bool:
    message = str(error)
    return (
        "Could not push packet to decoder" in message
        or "Invalid data found when processing input" in message
        or "FrameTimestampError" in message
        or "tolerance_s=" in message
        or "Failed to decode frame" in message
    )


def _load_episode_success_map(labels_csv_path: Path) -> dict[int, int]:
    labels_df = pd.read_csv(labels_csv_path)
    missing_columns = [
        column
        for column in (CSV_EPISODE_INDEX_COLUMN, CSV_SUCCESS_COLUMN)
        if column not in labels_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"CSV {labels_csv_path} is missing required columns: {missing_columns}. "
            f"Expected schema: [{CSV_EPISODE_INDEX_COLUMN}, {CSV_SUCCESS_COLUMN}]"
        )

    success_map: dict[int, int] = {}
    for _, row in labels_df.iterrows():
        episode_index = _to_int(row[CSV_EPISODE_INDEX_COLUMN])
        success = _to_int(row[CSV_SUCCESS_COLUMN])
        if success not in (0, 1):
            raise ValueError(
                f"CSV success labels must be 0/1. Found {success} for episode_index={episode_index}."
            )
        success_map[episode_index] = success
    return success_map


def _resolve_labels_csv_path(cfg: RECAPValueTrainingConfig) -> Path:
    """Return the path to the episode-labels CSV.

    When ``labels_csv_path`` is provided explicitly, that path is used directly.
    Otherwise the file is expected at ``<dataset_root>/meta/episode_labels.csv``.

    If the file isn't in the local cache yet (e.g. the dataset was cached before
    the labels were pushed), the resolver attempts to download it from HuggingFace.
    """
    if cfg.labels_csv_path is not None:
        resolved = Path(cfg.labels_csv_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Provided --labels_csv_path does not exist: {resolved}")
        return resolved

    dataset_root = Path(cfg.root) / cfg.repo_id if cfg.root else HF_LEROBOT_HOME / cfg.repo_id
    default_path = dataset_root / DEFAULT_EPISODE_LABELS_FILENAME
    if default_path.is_file():
        return default_path

    # The local cache may predate the labels upload — try fetching the file.
    try:
        from huggingface_hub import hf_hub_download

        logging.info(
            f"Episode labels not found locally at {default_path}; "
            f"attempting to download from {cfg.repo_id} ..."
        )
        hf_hub_download(
            repo_id=cfg.repo_id,
            filename=DEFAULT_EPISODE_LABELS_FILENAME,
            repo_type="dataset",
            revision=cfg.revision,
            local_dir=str(dataset_root),
        )
    except Exception:  # noqa: BLE001
        pass

    if default_path.is_file():
        return default_path

    raise FileNotFoundError(
        f"No episode labels CSV found at the default location: {default_path}\n"
        "Either:\n"
        "  1. Push a labels CSV to your HuggingFace dataset under meta/episode_labels.csv, or\n"
        "  2. Pass --labels_csv_path explicitly."
    )


def _selected_episode_indices(dataset: LeRobotDataset) -> list[int]:
    if dataset.episodes is None:
        return list(range(dataset.meta.total_episodes))
    return [_to_int(ep_idx) for ep_idx in dataset.episodes]


def _build_episode_infos(dataset: LeRobotDataset) -> dict[int, EpisodeInfo]:
    episode_infos: dict[int, EpisodeInfo] = {}
    for ep_idx in _selected_episode_indices(dataset):
        ep_data = dataset.meta.episodes[ep_idx]
        start_index = _to_int(ep_data["dataset_from_index"])
        end_index = _to_int(ep_data["dataset_to_index"])
        length = max(1, end_index - start_index)

        task: str | None = None
        if "task_index" in ep_data and dataset.meta.tasks is not None:
            task_index = _to_int(ep_data["task_index"])
            task = str(dataset.meta.tasks.iloc[task_index].name)
        elif "tasks" in ep_data:
            tasks = ep_data["tasks"]
            if isinstance(tasks, str):
                task = tasks
            elif tasks:
                task = str(tasks[0])

        if task is None:
            available_fields = sorted(ep_data.keys())
            raise ValueError(
                f"Episode {ep_idx} metadata is missing task information. "
                f"Expected either 'task_index' or non-empty 'tasks'. Available fields: {available_fields}"
            )

        episode_infos[ep_idx] = EpisodeInfo(
            episode_index=ep_idx,
            task=task,
            start_index=start_index,
            end_index=end_index,
            length=length,
        )
    return episode_infos


def _compute_task_max_episode_len(episode_infos: dict[int, EpisodeInfo]) -> dict[str, int]:
    by_task: dict[str, int] = {}
    for info in episode_infos.values():
        by_task[info.task] = max(by_task.get(info.task, 1), info.length)
    return by_task


def _discretize_values(normalized_returns: torch.Tensor, num_value_bins: int) -> torch.Tensor:
    bin_edges = torch.linspace(
        -1.0,
        0.0,
        num_value_bins + 1,
        dtype=normalized_returns.dtype,
        device=normalized_returns.device,
    )
    bin_ids = torch.bucketize(normalized_returns, bin_edges[1:], right=False)
    return bin_ids.clamp(min=0, max=num_value_bins - 1).to(torch.long)


def _build_frame_targets(
    dataset: LeRobotDataset,
    success_by_episode: dict[int, int],
    c_fail: float,
    num_value_bins: int,
) -> list[FrameTarget]:
    episode_infos = _build_episode_infos(dataset)
    task_max_episode_len = _compute_task_max_episode_len(episode_infos)

    dataset._ensure_hf_dataset_loaded()
    abs_to_rel_idx: dict[int, int] = {}
    for rel_idx in range(len(dataset.hf_dataset)):
        row = dataset.hf_dataset[rel_idx]
        abs_index = _to_int(row["index"])
        abs_to_rel_idx[abs_index] = rel_idx

    missing_episode_labels = sorted(set(episode_infos) - set(success_by_episode))
    if missing_episode_labels:
        raise ValueError(
            f"CSV is missing success labels for episode indices: {missing_episode_labels[:20]}"
        )

    frame_targets: list[FrameTarget] = []
    for ep_idx, info in episode_infos.items():
        success = bool(success_by_episode[ep_idx])
        rewards = torch.full((info.length,), -1.0, dtype=torch.float32)
        rewards[-1] = 0.0 if success else -float(c_fail)

        returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=[0]), dim=0), dims=[0])
        max_len_for_task = float(task_max_episode_len[info.task])
        normalized_returns = torch.clamp(returns / max_len_for_task, min=-1.0, max=0.0)
        target_bins = _discretize_values(normalized_returns, num_value_bins=num_value_bins)

        for offset, abs_index in enumerate(range(info.start_index, info.end_index)):
            if abs_index not in abs_to_rel_idx:
                continue
            rel_idx = abs_to_rel_idx[abs_index]
            frame_targets.append(
                FrameTarget(
                    frame_index=rel_idx,
                    episode_index=ep_idx,
                    success=int(success),
                    task=info.task,
                    target_value=_to_float(normalized_returns[offset]),
                    target_bin=_to_int(target_bins[offset]),
                )
            )

    if not frame_targets:
        raise ValueError("No frame targets were constructed. Check dataset episodes and CSV labels.")

    return frame_targets


def _split_train_val_targets(
    frame_targets: list[FrameTarget],
    val_ratio: float,
    seed: int,
) -> tuple[list[FrameTarget], list[FrameTarget]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_split_ratio must be in [0, 1). Got {val_ratio}.")

    episode_ids = sorted({target.episode_index for target in frame_targets})
    if len(episode_ids) < 2:
        raise ValueError("At least 2 labeled episodes are required for a train/val split.")

    episode_success: dict[int, int] = {}
    for target in frame_targets:
        existing = episode_success.get(target.episode_index)
        if existing is not None and existing != target.success:
            raise ValueError(
                f"Episode {target.episode_index} has inconsistent success labels in frame targets: "
                f"{existing} vs {target.success}."
            )
        episode_success[target.episode_index] = target.success

    success_episode_ids = [ep_id for ep_id in episode_ids if episode_success[ep_id] == 1]
    failure_episode_ids = [ep_id for ep_id in episode_ids if episode_success[ep_id] == 0]

    rng = random.Random(seed)
    rng.shuffle(success_episode_ids)
    rng.shuffle(failure_episode_ids)

    val_count = max(1, int(round(len(episode_ids) * val_ratio)))
    val_count = min(val_count, len(episode_ids) - 1)

    class_to_ids = {
        1: success_episode_ids,
        0: failure_episode_ids,
    }
    val_per_class = {1: 0, 0: 0}

    # If possible, keep both classes represented in validation.
    eligible_classes = [label for label, ids in class_to_ids.items() if len(ids) > 1]
    if val_count >= len(eligible_classes):
        for label in eligible_classes:
            val_per_class[label] = 1
    remaining_val_slots = val_count - sum(val_per_class.values())

    if remaining_val_slots > 0:
        total_episodes = len(episode_ids)
        fractional_parts: list[tuple[float, int]] = []
        for label, ids in class_to_ids.items():
            class_size = len(ids)
            max_for_class = max(0, class_size - 1)
            available = max_for_class - val_per_class[label]
            if available <= 0:
                continue
            raw_target = val_count * (class_size / total_episodes)
            additional = int(raw_target)
            to_add = min(available, additional)
            val_per_class[label] += to_add
            remaining_val_slots -= to_add
            fractional_parts.append((raw_target - additional, label))

        if remaining_val_slots > 0:
            fractional_parts.sort(reverse=True)
            for _, label in fractional_parts:
                if remaining_val_slots <= 0:
                    break
                max_for_class = max(0, len(class_to_ids[label]) - 1)
                if val_per_class[label] >= max_for_class:
                    continue
                val_per_class[label] += 1
                remaining_val_slots -= 1

        if remaining_val_slots > 0:
            for label, ids in class_to_ids.items():
                if remaining_val_slots <= 0:
                    break
                max_for_class = max(0, len(ids) - 1)
                while remaining_val_slots > 0 and val_per_class[label] < max_for_class:
                    val_per_class[label] += 1
                    remaining_val_slots -= 1

    val_episode_ids = set()
    val_episode_ids.update(success_episode_ids[: val_per_class[1]])
    val_episode_ids.update(failure_episode_ids[: val_per_class[0]])
    if not val_episode_ids:
        shuffled_episode_ids = episode_ids.copy()
        rng.shuffle(shuffled_episode_ids)
        val_episode_ids = {shuffled_episode_ids[0]}

    train_episode_ids = set(episode_ids) - val_episode_ids
    if not train_episode_ids:
        moved_episode = next(iter(val_episode_ids))
        val_episode_ids.remove(moved_episode)
        train_episode_ids.add(moved_episode)

    train_targets = [target for target in frame_targets if target.episode_index in train_episode_ids]
    val_targets = [target for target in frame_targets if target.episode_index in val_episode_ids]

    if not train_targets or not val_targets:
        raise ValueError("Train/val split produced an empty partition. Adjust val_split_ratio.")

    return train_targets, val_targets


def _to_chw_float_tensor(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        img = image.detach().clone().float()
    else:
        img = torch.as_tensor(np.asarray(image), dtype=torch.float32)

    if img.ndim == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    elif img.ndim == 3:
        if img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if img.shape[0] > 3:
            img = img[:3]
    else:
        raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")

    if img.max() > 1.0:
        img = img / 255.0
    return img.clamp(0.0, 1.0)


def _collect_images(frame: dict, camera_keys: list[str], image_size: int) -> torch.Tensor:
    image_tensors: list[torch.Tensor] = []
    for key in camera_keys:
        if key in frame:
            image_tensors.append(_to_chw_float_tensor(frame[key]))

    if not image_tensors:
        image_tensors = [torch.zeros(3, image_size, image_size, dtype=torch.float32)]

    resized = []
    for image in image_tensors:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        resized.append(image)

    return torch.stack(resized, dim=0)


class RECAPFrameSupervisionDataset(Dataset):
    """Frame-level dataset with paper-faithful return-bin supervision.

    Returns raw LeRobot frames (with per-camera image keys) augmented with
    value-training metadata.  Preprocessing (normalisation, state
    discretisation, tokenisation) is handled externally by the pi05
    preprocessor pipeline.
    """

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        frame_targets: list[FrameTarget],
    ):
        self.base_dataset = base_dataset
        self.frame_targets = frame_targets

    def __len__(self) -> int:
        return len(self.frame_targets)

    _MAX_DECODE_RETRIES = 5
    _RETRY_BASE_DELAY_S = 0.1

    def _decode_frame(self, frame_index: int) -> dict | None:
        """Try to decode a single frame, returning None on persistent failure."""
        for attempt in range(self._MAX_DECODE_RETRIES):
            try:
                return self.base_dataset[frame_index]
            except RuntimeError as exc:
                if not _is_known_video_validation_error(exc):
                    raise
                time.sleep(self._RETRY_BASE_DELAY_S * (2 ** attempt))
        return None

    def __getitem__(self, index: int) -> dict:
        target = self.frame_targets[index]

        frame = self._decode_frame(target.frame_index)
        if frame is None:
            logging.warning(
                f"Permanently failed to decode frame {target.frame_index} "
                f"(episode {target.episode_index}); substituting a random frame."
            )
            for _ in range(self._MAX_DECODE_RETRIES):
                alt_index = random.randint(0, len(self.frame_targets) - 1)
                alt_target = self.frame_targets[alt_index]
                frame = self._decode_frame(alt_target.frame_index)
                if frame is not None:
                    target = alt_target
                    break
            else:
                raise RuntimeError(
                    f"Failed to decode frame {target.frame_index} and "
                    f"{self._MAX_DECODE_RETRIES} random substitutes"
                )

        frame["target_bin"] = target.target_bin
        frame["target_value"] = target.target_value
        frame["success"] = target.success
        frame["frame_index"] = target.frame_index
        return frame


_TRAINING_METADATA_KEYS = ("target_bin", "target_value", "success", "frame_index")


def _build_preprocessor(
    dataset: LeRobotDataset,
    paligemma_variant: str,
    model_precision: str,
    device: str,
):
    """Build a pi05-compatible preprocessor for the value network.

    This constructs a PI05Config with the dataset's features and uses
    ``make_pi05_pre_post_processors`` to create the same preprocessing
    pipeline used by the pi05 policy (normalise state, discretise state,
    build prompt, tokenise, move to device).
    """
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    policy_cfg = PI05Config(
        input_features=input_features,
        output_features=output_features,
        paligemma_variant=paligemma_variant,
        dtype=model_precision,
        device=device,
    )
    preprocessor, _ = make_pi05_pre_post_processors(
        config=policy_cfg,
        dataset_stats=dataset.meta.stats,
    )
    return preprocessor


def _preprocess_batch(batch: dict, preprocessor) -> dict:
    """Apply the preprocessor while preserving training metadata keys."""
    preserved = {}
    for k in _TRAINING_METADATA_KEYS:
        if k in batch:
            v = batch[k]
            preserved[k] = v
    batch = preprocessor(batch)
    for k, v in preserved.items():
        if isinstance(v, torch.Tensor) and v.device != batch.get("observation.state", v).device:
            device = next(
                (bv.device for bv in batch.values() if isinstance(bv, torch.Tensor)),
                v.device,
            )
            v = v.to(device)
        batch[k] = v
    return batch


def _select_validation_plot_episode_ids(frame_targets: list[FrameTarget], max_episodes: int) -> list[int]:
    if max_episodes <= 0:
        return []

    episode_success: dict[int, int] = {}
    for target in frame_targets:
        episode_success[target.episode_index] = target.success

    success_ids = sorted(ep_id for ep_id, success in episode_success.items() if success == 1)
    failure_ids = sorted(ep_id for ep_id, success in episode_success.items() if success == 0)

    selected: list[int] = []
    if success_ids:
        selected.append(success_ids[0])
    if failure_ids and len(selected) < max_episodes:
        selected.append(failure_ids[0])

    all_ids = sorted(episode_success.keys())
    for ep_id in all_ids:
        if len(selected) >= max_episodes:
            break
        if ep_id not in selected:
            selected.append(ep_id)
    return selected


def _sample_preview_frame_indices(frame_indices: list[int], num_frames: int) -> list[int]:
    if not frame_indices or num_frames <= 0:
        return []
    if len(frame_indices) <= num_frames:
        return frame_indices

    sample_positions = np.linspace(0, len(frame_indices) - 1, num_frames, dtype=int)
    return [frame_indices[pos] for pos in sample_positions.tolist()]


def _prepare_plot_frame(frame: dict, camera_keys: list[str], image_size: int) -> np.ndarray:
    image_stack = _collect_images(frame=frame, camera_keys=camera_keys, image_size=image_size)
    image = image_stack[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(image, 0.0, 1.0)


def _save_validation_episode_plot(
    dataset: LeRobotDataset,
    episode_index: int,
    predictions: list[ValidationFramePrediction],
    output_path: Path,
    num_preview_frames: int,
    num_value_bins: int,
    image_size: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is unavailable; skipping validation plots.")
        return False

    if not predictions:
        return False

    predictions = sorted(predictions, key=lambda x: x.frame_index)
    success = predictions[0].success
    status = "SUCCESS" if success == 1 else "FAIL"

    frame_indices = [pred.frame_index for pred in predictions]
    target_values = np.array([pred.target_value for pred in predictions], dtype=np.float32)
    reconstructed_values = np.array([pred.reconstructed_value for pred in predictions], dtype=np.float32)

    preview_indices = _sample_preview_frame_indices(frame_indices=frame_indices, num_frames=num_preview_frames)
    preview_images: list[np.ndarray] = []
    preview_steps: list[int] = []
    camera_keys = list(dataset.meta.camera_keys)
    frame_to_step = {frame_idx: idx for idx, frame_idx in enumerate(frame_indices)}
    for frame_idx in preview_indices:
        frame = dataset[frame_idx]
        preview_images.append(_prepare_plot_frame(frame=frame, camera_keys=camera_keys, image_size=image_size))
        preview_steps.append(frame_to_step[frame_idx])

    time_steps = np.arange(len(predictions), dtype=np.int64)

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6])

    top_grid = grid[0].subgridspec(1, max(1, len(preview_images)))
    if preview_images:
        for idx, image in enumerate(preview_images):
            ax = fig.add_subplot(top_grid[0, idx])
            ax.imshow(image)
            ax.set_title(f"t={preview_steps[idx]}", fontsize=9)
            ax.axis("off")
    else:
        ax = fig.add_subplot(top_grid[0, 0])
        ax.text(0.5, 0.5, "No preview frames", ha="center", va="center")
        ax.axis("off")

    ax_return = fig.add_subplot(grid[1])
    ax_return.plot(time_steps, target_values, color="red", linewidth=2, label="Labeled expected return")
    ax_return.plot(
        time_steps,
        reconstructed_values,
        color="green",
        linewidth=2,
        label="Reconstructed return E[v|p(bin)]",
    )
    ax_return.set_ylim(-1.05, 0.05)
    ax_return.set_ylabel("Normalized return")
    ax_return.set_xlabel("Trajectory step")
    ax_return.grid(alpha=0.3)
    ax_return.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Validation Episode {episode_index} ({status})", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return True


def _init_wandb(cfg: RECAPValueTrainingConfig) -> Any:
    """Initialise a W&B run if ``wandb_project`` is set, otherwise return ``None``."""
    if cfg.wandb_project is None:
        return None
    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config=asdict(cfg),
    )
    logging.info(f"W&B run: {run.url}")
    return run


def _run_epoch(
    model: RECAPValueNetwork,
    loader: DataLoader,
    preprocessor: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float,
    epoch_index: int,
    total_epochs: int,
    max_steps: int | None = None,
    log_every_n_steps: int = 0,
    on_train_step_end: Callable[[int], None] | None = None,
    collect_episode_ids: set[int] | None = None,
    value_bin_support: torch.Tensor | None = None,
    collected_predictions: dict[int, list[ValidationFramePrediction]] | None = None,
    wandb_run: Any = None,
    global_step_offset: int = 0,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    phase = "train" if training else "val"
    if log_every_n_steps < 0:
        raise ValueError(f"log_every_n_steps must be >= 0, got {log_every_n_steps}")

    total_loss = 0.0
    total_mae = 0.0
    total_acc = 0.0
    total_samples = 0
    total_steps = len(loader)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    window_loss = 0.0
    window_mae = 0.0
    window_acc = 0.0
    window_samples = 0
    window_steps = 0
    epoch_start_time = time.perf_counter()
    window_start_time = epoch_start_time

    grad_accum = max(1, gradient_accumulation_steps) if training else 1
    if training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        step_num = step + 1

        batch = _preprocess_batch(batch, preprocessor)
        images = collect_images(batch, model.config.image_size)

        outputs = model(batch, images)
        value_logits = outputs["value_logits"]
        expected_value = outputs["expected_value"].squeeze(-1)

        loss = F.cross_entropy(value_logits, batch["target_bin"])

        if training:
            assert optimizer is not None
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            is_accumulation_boundary = (step_num % grad_accum == 0) or (step_num == total_steps)
            if is_accumulation_boundary:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        with torch.no_grad():
            batch_size = batch["target_bin"].shape[0]
            pred_bins = value_logits.argmax(dim=-1)
            acc = (pred_bins == batch["target_bin"]).float().mean()
            mae = torch.abs(expected_value - batch["target_value"]).mean()

            if not training and collected_predictions is not None:
                probs = outputs["value_probs"].detach()
                if value_bin_support is None:
                    support = torch.linspace(-1.0, 0.0, probs.shape[-1], device=probs.device, dtype=probs.dtype)
                else:
                    support = value_bin_support.to(device=probs.device, dtype=probs.dtype)
                reconstructed = (probs * support.unsqueeze(0)).sum(dim=-1)

                for idx in range(batch_size):
                    episode_index = _to_int(batch["episode_index"][idx])
                    if collect_episode_ids is not None and episode_index not in collect_episode_ids:
                        continue
                    prediction = ValidationFramePrediction(
                        frame_index=_to_int(batch["frame_index"][idx]),
                        success=_to_int(batch["success"][idx]),
                        target_value=_to_float(batch["target_value"][idx]),
                        target_bin=_to_int(batch["target_bin"][idx]),
                        predicted_bin=_to_int(pred_bins[idx]),
                        reconstructed_value=_to_float(reconstructed[idx]),
                        predicted_probs=probs[idx].float().cpu().numpy().copy(),
                    )
                    collected_predictions.setdefault(episode_index, []).append(prediction)

            total_loss += float(loss.item()) * batch_size
            total_acc += float(acc.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            total_samples += batch_size
            window_loss += float(loss.item()) * batch_size
            window_acc += float(acc.item()) * batch_size
            window_mae += float(mae.item()) * batch_size
            window_samples += batch_size
            window_steps += 1

        should_log_step = (
            log_every_n_steps > 0
            and window_samples > 0
            and (
                step_num == 1
                or step_num % log_every_n_steps == 0
                or (total_steps > 0 and step_num == total_steps)
            )
        )
        if should_log_step:
            now = time.perf_counter()
            elapsed = max(now - epoch_start_time, 1e-9)
            window_elapsed = max(now - window_start_time, 1e-9)
            avg_loss = total_loss / total_samples
            avg_acc = total_acc / total_samples
            avg_mae = total_mae / total_samples
            window_loss_avg = window_loss / window_samples
            window_acc_avg = window_acc / window_samples
            window_mae_avg = window_mae / window_samples
            avg_sec_per_step = elapsed / max(step_num, 1)
            steps_per_sec = window_steps / window_elapsed
            samples_per_sec = window_samples / window_elapsed
            eta_seconds = avg_sec_per_step * max(total_steps - step_num, 0)
            progress = f"{step_num}/{total_steps}" if total_steps > 0 else f"{step_num}/?"

            logging.info(
                f"[Epoch {epoch_index}/{total_epochs}][{phase}] step={progress} "
                f"loss={window_loss_avg:.5f} acc={window_acc_avg:.4f} mae={window_mae_avg:.5f} "
                f"avg_loss={avg_loss:.5f} avg_acc={avg_acc:.4f} avg_mae={avg_mae:.5f} "
                f"it/s={steps_per_sec:.2f} samples/s={samples_per_sec:.2f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)}"
            )
            if wandb_run is not None:
                wb_step = global_step_offset + step_num
                wb_prefix = "train" if training else "val"
                wandb_run.log(
                    {
                        f"{wb_prefix}/loss": avg_loss,
                        f"{wb_prefix}/bin_acc": avg_acc,
                        f"{wb_prefix}/value_mae": avg_mae,
                        f"{wb_prefix}/window_loss": window_loss_avg,
                        f"{wb_prefix}/window_bin_acc": window_acc_avg,
                        f"{wb_prefix}/window_value_mae": window_mae_avg,
                        f"{wb_prefix}/samples_per_sec": samples_per_sec,
                        "global_step": wb_step,
                    },
                    step=wb_step,
                )
            window_loss = 0.0
            window_mae = 0.0
            window_acc = 0.0
            window_samples = 0
            window_steps = 0
            window_start_time = now

        if training and on_train_step_end is not None:
            callback_start_time = time.perf_counter()
            on_train_step_end(step_num)
            callback_elapsed = time.perf_counter() - callback_start_time
            # Keep throughput metrics focused on training work (exclude callback wall time).
            epoch_start_time += callback_elapsed
            window_start_time += callback_elapsed

    if total_samples == 0:
        return {"loss": float("nan"), "bin_acc": float("nan"), "value_mae": float("nan")}

    return {
        "loss": total_loss / total_samples,
        "bin_acc": total_acc / total_samples,
        "value_mae": total_mae / total_samples,
    }


def _save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    requested = torch.device(device_str)
    if requested.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return requested


@parser.wrap()
def run_recap_value_train_val(cfg: RECAPValueTrainingConfig) -> None:
    """Train and validate RECAPValueNetwork with distributional return-bin supervision."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _save_json(output_dir / "train_config.json", asdict(cfg))

    device = _resolve_device(cfg.device)
    logging.info(f"Using device: {device}")

    wandb_run = _init_wandb(cfg)

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=cfg.episodes,
    )

    labels_csv_path = _resolve_labels_csv_path(cfg)
    logging.info(f"Using episode labels from: {labels_csv_path}")
    success_by_episode = _load_episode_success_map(labels_csv_path)
    frame_targets = _build_frame_targets(
        dataset=dataset,
        success_by_episode=success_by_episode,
        c_fail=cfg.c_fail,
        num_value_bins=cfg.num_value_bins,
    )
    train_targets, val_targets = _split_train_val_targets(
        frame_targets=frame_targets,
        val_ratio=cfg.val_split_ratio,
        seed=cfg.seed,
    )

    preprocessor = _build_preprocessor(
        dataset=dataset,
        paligemma_variant=cfg.paligemma_variant,
        model_precision=cfg.model_precision,
        device=str(device),
    )
    logging.info("Created pi05 preprocessor for value network training")

    train_dataset = RECAPFrameSupervisionDataset(
        base_dataset=dataset,
        frame_targets=train_targets,
    )
    val_dataset = RECAPFrameSupervisionDataset(
        base_dataset=dataset,
        frame_targets=val_targets,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    step_val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if cfg.model_precision not in ("float32", "bfloat16"):
        raise ValueError(f"model_precision must be one of ['float32', 'bfloat16'], got {cfg.model_precision}")
    model_precision = cast(Literal["float32", "bfloat16"], cfg.model_precision)

    model_config = RECAPValueNetworkConfig(
        paligemma_variant=cfg.paligemma_variant,
        precision=model_precision,
        image_size=cfg.image_size,
        tokenizer_name=cfg.tokenizer_name,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        freeze_backbone=cfg.freeze_backbone,
        num_unfrozen_backbone_layers=cfg.num_unfrozen_backbone_layers,
        num_vlm_layers=cfg.num_vlm_layers,
        num_value_bins=cfg.num_value_bins,
        value_head_depth=cfg.value_head_depth,
        dropout=cfg.dropout,
        pretrained_path=cfg.pretrained_path,
    )
    model = RECAPValueNetwork(model_config).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} "
        f"/ {sum(p.numel() for p in model.parameters()):,} total"
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    if cfg.max_train_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_steps_per_epoch)
    grad_accum = max(1, cfg.gradient_accumulation_steps)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * cfg.epochs
    scheduler = _build_warmup_cosine_scheduler(
        optimizer, total_steps=total_optimizer_steps, warmup_ratio=cfg.warmup_ratio,
    )
    logging.info(
        f"Using warmup+cosine schedule: {int(total_optimizer_steps * cfg.warmup_ratio)} warmup steps "
        f"/ {total_optimizer_steps} total optimizer steps (grad_accum={grad_accum}), "
        f"peak lr={cfg.learning_rate}"
    )

    best_val_loss = float("inf")
    history: list[dict] = []
    plot_episode_ids = _select_validation_plot_episode_ids(
        frame_targets=val_targets,
        max_episodes=cfg.val_plot_num_episodes,
    )
    if plot_episode_ids:
        logging.info(f"Validation plots will track episodes: {plot_episode_ids}")
    plot_episode_id_set = set(plot_episode_ids)
    plot_targets = [target for target in val_targets if target.episode_index in plot_episode_id_set]
    val_plot_loader: DataLoader | None = None
    if plot_targets:
        val_plot_dataset = RECAPFrameSupervisionDataset(
            base_dataset=dataset,
            frame_targets=plot_targets,
        )
        val_plot_loader = DataLoader(
            val_plot_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    logging.info(
        "Starting training: "
        f"{len(train_targets)} train frames / {len(val_targets)} val frames "
        f"from {len(set(t.episode_index for t in train_targets))} train episodes and "
        f"{len(set(t.episode_index for t in val_targets))} val episodes."
    )

    if cfg.validate_every_n_train_steps < 0:
        raise ValueError(
            f"validate_every_n_train_steps must be >= 0, got {cfg.validate_every_n_train_steps}"
        )
    if cfg.plot_every_n_train_steps < 0:
        raise ValueError(
            f"plot_every_n_train_steps must be >= 0, got {cfg.plot_every_n_train_steps}"
        )
    if cfg.max_val_steps_per_step_validation is not None and cfg.max_val_steps_per_step_validation <= 0:
        raise ValueError(
            "max_val_steps_per_step_validation must be > 0 when provided, "
            f"got {cfg.max_val_steps_per_step_validation}"
        )
    if cfg.plot_every_n_train_steps > 0 and cfg.val_plot_num_episodes <= 0:
        logging.warning(
            "plot_every_n_train_steps is set but val_plot_num_episodes <= 0, "
            "so step-based plotting is disabled."
        )

    def _run_validation_and_maybe_plot(
        *,
        epoch_index: int,
        trigger_tag: str,
        max_steps: int | None,
        plot_subdir: str | None,
        loader: DataLoader,
    ) -> dict[str, float]:
        should_plot = bool(plot_subdir) and bool(plot_episode_ids) and val_plot_loader is not None
        try:
            val_metrics_local = _run_epoch(
                model=model,
                loader=loader,
                preprocessor=preprocessor,
                device=device,
                optimizer=None,
                max_grad_norm=cfg.max_grad_norm,
                epoch_index=epoch_index,
                total_epochs=cfg.epochs,
                max_steps=max_steps,
                log_every_n_steps=cfg.log_every_n_steps,
                collect_episode_ids=None,
                value_bin_support=model.value_bin_support,
                collected_predictions=None
            )
        except Exception as error:  # noqa: BLE001
            if loader is step_val_loader or not _is_known_video_validation_error(error):
                raise
            logging.warning(
                f"[{trigger_tag}] Validation failed with video worker decoding/timestamp issue; "
                "retrying validation with num_workers=0."
            )
            try:
                collected_predictions = {} if should_plot else None
                val_metrics_local = _run_epoch(
                    model=model,
                    loader=step_val_loader,
                    preprocessor=preprocessor,
                    device=device,
                    optimizer=None,
                    max_grad_norm=cfg.max_grad_norm,
                    epoch_index=epoch_index,
                    total_epochs=cfg.epochs,
                    max_steps=max_steps,
                    log_every_n_steps=cfg.log_every_n_steps,
                    collect_episode_ids=None,
                    value_bin_support=model.value_bin_support,
                    collected_predictions=None,
                )
            except Exception as retry_error:  # noqa: BLE001
                if not _is_known_video_validation_error(retry_error):
                    raise
                logging.warning(
                    f"[{trigger_tag}] Validation skipped due to persistent video decoding/timestamp errors: "
                    f"{retry_error}"
                )
                return {"loss": float("nan"), "bin_acc": float("nan"), "value_mae": float("nan")}
        logging.info(
            f"[{trigger_tag}] "
            f"val_loss={val_metrics_local['loss']:.5f} "
            f"val_acc={val_metrics_local['bin_acc']:.4f} "
            f"val_mae={val_metrics_local['value_mae']:.5f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/loss": val_metrics_local["loss"],
                    "val/bin_acc": val_metrics_local["bin_acc"],
                    "val/value_mae": val_metrics_local["value_mae"],
                    "global_step": global_train_step,
                },
                step=global_train_step,
            )

        if should_plot and plot_subdir is not None and val_plot_loader is not None:
            collected_predictions: dict[int, list[ValidationFramePrediction]] = {}
            try:
                _run_epoch(
                    model=model,
                    loader=val_plot_loader,
                    preprocessor=preprocessor,
                    device=device,
                    optimizer=None,
                    max_grad_norm=cfg.max_grad_norm,
                    epoch_index=epoch_index,
                    total_epochs=cfg.epochs,
                    max_steps=None,
                    log_every_n_steps=0,
                    collect_episode_ids=plot_episode_id_set,
                    value_bin_support=model.value_bin_support,
                    collected_predictions=collected_predictions,
                )
            except Exception as error:  # noqa: BLE001
                if _is_known_video_validation_error(error):
                    logging.warning(
                        f"[{trigger_tag}] Plot generation skipped due to video decode/timestamp issue: {error}"
                    )
                    return val_metrics_local
                raise
            plot_dir = output_dir / "validation_plots" / plot_subdir
            saved_paths: list[Path] = []
            for episode_index in plot_episode_ids:
                episode_predictions = collected_predictions.get(episode_index, [])
                plot_path = plot_dir / f"episode_{episode_index:05d}.png"
                did_save = _save_validation_episode_plot(
                    dataset=dataset,
                    episode_index=episode_index,
                    predictions=episode_predictions,
                    output_path=plot_path,
                    num_preview_frames=cfg.val_plot_num_frames,
                    num_value_bins=cfg.num_value_bins,
                    image_size=cfg.image_size,
                )
                if did_save:
                    saved_paths.append(plot_path)
            if saved_paths:
                logging.info(f"[{trigger_tag}] Saved {len(saved_paths)} validation plot(s) under {plot_dir}")
                if wandb_run is not None:
                    import wandb as _wandb

                    plot_images = {
                        f"val_plots/episode_{p.stem.split('_')[-1]}": _wandb.Image(str(p))
                        for p in saved_paths
                    }
                    wandb_run.log(plot_images, step=global_train_step)

        return val_metrics_local

    global_train_step = 0

    for epoch in range(1, cfg.epochs + 1):
        on_train_step_end: Callable[[int], None] | None = None
        if cfg.validate_every_n_train_steps > 0 or cfg.plot_every_n_train_steps > 0:
            step_val_max_steps = (
                cfg.max_val_steps_per_step_validation
                if cfg.max_val_steps_per_step_validation is not None
                else cfg.max_val_steps_per_epoch
            )

            def _on_train_step_end(step_num: int) -> None:
                nonlocal global_train_step
                global_train_step += 1
                should_validate_now = (
                    cfg.validate_every_n_train_steps > 0
                    and global_train_step % cfg.validate_every_n_train_steps == 0
                )
                should_plot_now = (
                    cfg.plot_every_n_train_steps > 0
                    and global_train_step % cfg.plot_every_n_train_steps == 0
                    and bool(plot_episode_ids)
                )
                if should_plot_now:
                    should_validate_now = True
                if not should_validate_now:
                    return

                trigger = (
                    f"Epoch {epoch}/{cfg.epochs} step-validate "
                    f"(global_step={global_train_step}, epoch_step={step_num})"
                )
                _run_validation_and_maybe_plot(
                    epoch_index=epoch,
                    trigger_tag=trigger,
                    max_steps=step_val_max_steps,
                    plot_subdir=f"step_{global_train_step:08d}" if should_plot_now else None,
                    loader=step_val_loader,
                )

            on_train_step_end = _on_train_step_end

        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            preprocessor=preprocessor,
            device=device,
            optimizer=optimizer,
            max_grad_norm=cfg.max_grad_norm,
            epoch_index=epoch,
            total_epochs=cfg.epochs,
            max_steps=cfg.max_train_steps_per_epoch,
            log_every_n_steps=cfg.log_every_n_steps,
            on_train_step_end=on_train_step_end,
            wandb_run=wandb_run,
            global_step_offset=global_train_step,
            scheduler=scheduler,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )
        should_plot_validation = (
            cfg.val_plot_num_episodes > 0
            and cfg.val_plot_every_n_epochs > 0
            and (epoch % cfg.val_plot_every_n_epochs == 0)
            and bool(plot_episode_ids)
        )
        val_metrics = _run_validation_and_maybe_plot(
            epoch_index=epoch,
            trigger_tag=f"Epoch {epoch}/{cfg.epochs} epoch-end",
            max_steps=cfg.max_val_steps_per_epoch,
            plot_subdir=f"epoch_{epoch:03d}" if should_plot_validation else None,
            loader=val_loader,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bin_acc": train_metrics["bin_acc"],
            "train_value_mae": train_metrics["value_mae"],
            "val_loss": val_metrics["loss"],
            "val_bin_acc": val_metrics["bin_acc"],
            "val_value_mae": val_metrics["value_mae"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        logging.info(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"train_loss={epoch_metrics['train_loss']:.5f} "
            f"val_loss={epoch_metrics['val_loss']:.5f} "
            f"train_acc={epoch_metrics['train_bin_acc']:.4f} "
            f"val_acc={epoch_metrics['val_bin_acc']:.4f} "
            f"val_mae={epoch_metrics['val_value_mae']:.5f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {f"epoch/{k}": v for k, v in epoch_metrics.items()},
                step=global_train_step,
            )

        _save_json(output_dir / "metrics_history.json", history)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(cfg),
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")
        torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:04d}.pt")
        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            torch.save(checkpoint, checkpoints_dir / "best.pt")

    logging.info(f"Training complete. Best val loss: {best_val_loss:.5f}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run_recap_value_train_val()  # ty: ignore[missing-argument]
