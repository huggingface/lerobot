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
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.algorithms.RECAPValueNetwork import RECAPValueNetwork, RECAPValueNetworkConfig
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
else:
    AutoTokenizer = None
    PreTrainedTokenizerBase = object


CSV_EPISODE_INDEX_COLUMN = "episode_index"
CSV_SUCCESS_COLUMN = "success"


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
    task: str
    target_value: float
    target_bin: int


@dataclass
class RECAPValueTrainingConfig:
    """Configuration for RECAP value-network train/val."""

    repo_id: str
    labels_csv_path: str
    output_dir: str
    root: str | None = None
    revision: str | None = None
    episodes: list[int] | None = None

    epochs: int = 10
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None

    # Value target construction
    c_fail: float = 24.0
    num_value_bins: int = 201

    # Input processing
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"
    tokenizer_max_length: int = 96
    image_size: int = 224
    max_state_dim: int = 32

    # Backbone sizing (paper-faithful PI0.5/PI05 base model without expert head)
    paligemma_variant: str = "gemma_300m"
    model_precision: str = "float32"
    freeze_vision_encoder: bool = False
    dropout: float = 0.1


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
        task_index = _to_int(ep_data["task_index"])
        task = str(dataset.meta.tasks.iloc[task_index].name)
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

    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    val_count = max(1, int(round(len(episode_ids) * val_ratio)))
    val_count = min(val_count, len(episode_ids) - 1)

    val_episode_ids = set(episode_ids[:val_count])
    train_episode_ids = set(episode_ids[val_count:])

    train_targets = [target for target in frame_targets if target.episode_index in train_episode_ids]
    val_targets = [target for target in frame_targets if target.episode_index in val_episode_ids]

    if not train_targets or not val_targets:
        raise ValueError("Train/val split produced an empty partition. Adjust val_split_ratio.")

    return train_targets, val_targets


def _prepare_state_vector(state, max_state_dim: int) -> torch.Tensor:
    if state is None:
        state_tensor = torch.zeros(max_state_dim, dtype=torch.float32)
    else:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).flatten()
        if state_tensor.numel() >= max_state_dim:
            state_tensor = state_tensor[:max_state_dim]
        else:
            pad_size = max_state_dim - state_tensor.numel()
            state_tensor = F.pad(state_tensor, (0, pad_size), mode="constant", value=0.0)
    return state_tensor


def _build_critic_prompt(task: str, state_vector: torch.Tensor) -> str:
    cleaned_task = task.strip().replace("_", " ").replace("\n", " ").lower()
    state_np = state_vector.cpu().numpy()
    clipped = np.clip(state_np, -1.0, 1.0)
    discretized = np.digitize(clipped, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized.tolist()))
    return f"Task: {cleaned_task}, State: {state_str};\n"


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
    """Frame-level dataset with paper-faithful return-bin supervision."""

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        frame_targets: list[FrameTarget],
        max_state_dim: int,
        image_size: int,
    ):
        self.base_dataset = base_dataset
        self.frame_targets = frame_targets
        self.max_state_dim = max_state_dim
        self.image_size = image_size
        self.camera_keys = list(base_dataset.meta.camera_keys)

    def __len__(self) -> int:
        return len(self.frame_targets)

    def __getitem__(self, index: int) -> dict:
        target = self.frame_targets[index]
        frame = self.base_dataset[target.frame_index]

        state_vector = _prepare_state_vector(frame.get(OBS_STATE), max_state_dim=self.max_state_dim)
        task = str(frame.get("task", target.task))
        prompt = _build_critic_prompt(task=task, state_vector=state_vector)
        images = _collect_images(frame=frame, camera_keys=self.camera_keys, image_size=self.image_size)

        return {
            "images": images,
            "prompt": prompt,
            "target_bin": target.target_bin,
            "target_value": target.target_value,
        }


class RECAPBatchCollator:
    """Collates frame samples and tokenizes prompts in batch."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        prompts = [sample["prompt"] for sample in samples]
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "images": torch.stack([sample["images"] for sample in samples], dim=0),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"].bool(),
            "target_bin": torch.tensor([sample["target_bin"] for sample in samples], dtype=torch.long),
            "target_value": torch.tensor([sample["target_value"] for sample in samples], dtype=torch.float32),
        }


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _run_epoch(
    model: RECAPValueNetwork,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float,
    max_steps: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_mae = 0.0
    total_acc = 0.0
    total_samples = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = _move_batch_to_device(batch, device=device)

        outputs = model(
            images=batch["images"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        value_logits = outputs["value_logits"]
        expected_value = outputs["expected_value"].squeeze(-1)

        loss = F.cross_entropy(value_logits, batch["target_bin"])

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        with torch.no_grad():
            batch_size = batch["target_bin"].shape[0]
            pred_bins = value_logits.argmax(dim=-1)
            acc = (pred_bins == batch["target_bin"]).float().mean()
            mae = torch.abs(expected_value - batch["target_value"]).mean()

            total_loss += float(loss.item()) * batch_size
            total_acc += float(acc.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            total_samples += batch_size

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
    if AutoTokenizer is None:
        raise ImportError("transformers is required to run RECAPTrainValueNetwork.")

    logging.basicConfig(level=logging.INFO)
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _save_json(output_dir / "train_config.json", asdict(cfg))

    device = _resolve_device(cfg.device)
    logging.info(f"Using device: {device}")

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=cfg.episodes,
    )

    success_by_episode = _load_episode_success_map(Path(cfg.labels_csv_path))
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    collator = RECAPBatchCollator(
        tokenizer=tokenizer,
        max_length=cfg.tokenizer_max_length,
    )

    train_dataset = RECAPFrameSupervisionDataset(
        base_dataset=dataset,
        frame_targets=train_targets,
        max_state_dim=cfg.max_state_dim,
        image_size=cfg.image_size,
    )
    val_dataset = RECAPFrameSupervisionDataset(
        base_dataset=dataset,
        frame_targets=val_targets,
        max_state_dim=cfg.max_state_dim,
        image_size=cfg.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
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
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        num_value_bins=cfg.num_value_bins,
        dropout=cfg.dropout,
    )
    model = RECAPValueNetwork(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
    )

    best_val_loss = float("inf")
    history: list[dict] = []

    logging.info(
        "Starting training: "
        f"{len(train_targets)} train frames / {len(val_targets)} val frames "
        f"from {len(set(t.episode_index for t in train_targets))} train episodes and "
        f"{len(set(t.episode_index for t in val_targets))} val episodes."
    )

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_grad_norm=cfg.max_grad_norm,
            max_steps=cfg.max_train_steps_per_epoch,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            max_grad_norm=cfg.max_grad_norm,
            max_steps=cfg.max_val_steps_per_epoch,
        )
        scheduler.step()

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
        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            torch.save(checkpoint, checkpoints_dir / "best.pt")

    logging.info(f"Training complete. Best val loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    run_recap_value_train_val()
