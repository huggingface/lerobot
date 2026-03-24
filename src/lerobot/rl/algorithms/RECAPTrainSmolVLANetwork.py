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
"""Train/val script for the standalone RECAP SmolVLA distributional value network."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.rl.algorithms.RECAPSmolVLAValueNetwork import (
    RECAPSmolVLAValueNetwork,
    RECAPSmolVLAValueNetworkConfig,
)
from lerobot.rl.algorithms import RECAPTrainValueNetwork as base


@dataclass
class RECAPSmolVLAValueTrainingConfig:
    """Configuration for RECAP SmolVLA value-network train/val."""

    repo_id: str
    output_dir: str
    labels_csv_path: str | None = None
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
    text_tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    tokenizer_max_length: int = 96
    image_size: int = 512
    max_state_dim: int = 32

    # SmolVLM backbone settings
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    num_vlm_layers: int = 16
    model_precision: str = "float32"
    freeze_vision_encoder: bool = False
    freeze_backbone: bool = False
    dropout: float = 0.1


@parser.wrap()
def run_recap_smolvla_train_val(cfg: RECAPSmolVLAValueTrainingConfig) -> None:
    """Train and validate RECAPSmolVLAValueNetwork with distributional return-bin supervision."""
    if base.AutoTokenizer is None:
        raise ImportError("transformers is required to run RECAPTrainSmolVLANetwork.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    base._set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base._save_json(output_dir / "train_config.json", asdict(cfg))

    device = base._resolve_device(cfg.device)
    logging.info(f"Using device: {device}")

    dataset = base.LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=cfg.episodes,
    )

    labels_csv_path = base._resolve_labels_csv_path(cfg)
    logging.info(f"Using episode labels from: {labels_csv_path}")
    success_by_episode = base._load_episode_success_map(labels_csv_path)
    frame_targets = base._build_frame_targets(
        dataset=dataset,
        success_by_episode=success_by_episode,
        c_fail=cfg.c_fail,
        num_value_bins=cfg.num_value_bins,
    )
    train_targets, val_targets = base._split_train_val_targets(
        frame_targets=frame_targets,
        val_ratio=cfg.val_split_ratio,
        seed=cfg.seed,
    )

    tokenizer = base.AutoTokenizer.from_pretrained(cfg.text_tokenizer_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    collator = base.RECAPBatchCollator(
        tokenizer=tokenizer,
        max_length=cfg.tokenizer_max_length,
    )

    train_dataset = base.RECAPFrameSupervisionDataset(
        base_dataset=dataset,
        frame_targets=train_targets,
        max_state_dim=cfg.max_state_dim,
        image_size=cfg.image_size,
    )
    val_dataset = base.RECAPFrameSupervisionDataset(
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
    step_val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if cfg.model_precision not in ("float32", "bfloat16"):
        raise ValueError(
            f"model_precision must be one of ['float32', 'bfloat16'], got {cfg.model_precision}"
        )
    model_precision = cast(Literal["float32", "bfloat16"], cfg.model_precision)

    model_config = RECAPSmolVLAValueNetworkConfig(
        vlm_model_name=cfg.vlm_model_name,
        load_vlm_weights=cfg.load_vlm_weights,
        precision=model_precision,
        image_size=cfg.image_size,
        max_state_dim=cfg.max_state_dim,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        freeze_backbone=cfg.freeze_backbone,
        num_vlm_layers=cfg.num_vlm_layers,
        num_value_bins=cfg.num_value_bins,
        dropout=cfg.dropout,
    )
    model = RECAPSmolVLAValueNetwork(model_config).to(device)

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
    plot_episode_ids = base._select_validation_plot_episode_ids(
        frame_targets=val_targets,
        max_episodes=cfg.val_plot_num_episodes,
    )
    if plot_episode_ids:
        logging.info(f"Validation plots will track episodes: {plot_episode_ids}")
    plot_episode_id_set = set(plot_episode_ids)
    plot_targets = [target for target in val_targets if target.episode_index in plot_episode_id_set]
    val_plot_loader: DataLoader | None = None
    if plot_targets:
        val_plot_dataset = base.RECAPFrameSupervisionDataset(
            base_dataset=dataset,
            frame_targets=plot_targets,
            max_state_dim=cfg.max_state_dim,
            image_size=cfg.image_size,
        )
        val_plot_loader = DataLoader(
            val_plot_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
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
            val_metrics_local = base._run_epoch(
                model=model,
                loader=loader,
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
        except Exception as error:  # noqa: BLE001
            if loader is step_val_loader or not base._is_known_video_validation_error(error):
                raise
            logging.warning(
                f"[{trigger_tag}] Validation failed with video worker decoding/timestamp issue; "
                "retrying validation with num_workers=0."
            )
            try:
                val_metrics_local = base._run_epoch(
                    model=model,
                    loader=step_val_loader,
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
                if not base._is_known_video_validation_error(retry_error):
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

        if should_plot and plot_subdir is not None and val_plot_loader is not None:
            collected_predictions: dict[int, list[base.ValidationFramePrediction]] = {}
            try:
                base._run_epoch(
                    model=model,
                    loader=val_plot_loader,
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
                if base._is_known_video_validation_error(error):
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
                did_save = base._save_validation_episode_plot(
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

        return val_metrics_local

    global_train_step = 0

    for epoch in range(1, cfg.epochs + 1):
        on_train_step_end = None
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

        train_metrics = base._run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_grad_norm=cfg.max_grad_norm,
            epoch_index=epoch,
            total_epochs=cfg.epochs,
            max_steps=cfg.max_train_steps_per_epoch,
            log_every_n_steps=cfg.log_every_n_steps,
            on_train_step_end=on_train_step_end,
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

        base._save_json(output_dir / "metrics_history.json", history)

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
    run_recap_smolvla_train_val()
