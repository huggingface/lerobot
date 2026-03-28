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
"""Train/val script for the SmolStar06 advantage-conditioned policy.

Standalone training loop with an episode-level train/val split (reusing the
value-network splitting logic) and validation metrics that measure how well
the model learns to differentiate positive vs negative advantage distributions.

Validation metrics (computed via two forward passes with shared noise/time):
  - Stratified flow-matching loss (val_loss, val_loss_pos, val_loss_neg)
  - Conditioning accuracy: fraction of samples where the correct advantage
    label yields lower flow-matching loss than the flipped label
  - Conditioning gap: mean(loss_wrong - loss_correct)
"""

import logging
import time as time_module
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.rl.algorithms import RECAPTrainValueNetwork as base
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS


@dataclass
class RECAPSmolStarTrainingConfig:
    """Configuration for RECAP SmolStar06 advantage-conditioned policy training."""

    repo_id: str
    output_dir: str
    value_network_checkpoint: str
    episode_labels_path: str | None = None
    root: str | None = None
    revision: str | None = None
    episodes: list[int] | None = None

    epochs: int = 5
    batch_size: int = 6
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    val_split_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None
    log_every_n_steps: int = 10
    validate_every_n_train_steps: int = 0
    max_val_steps_per_step_validation: int | None = None

    # RECAP advantage conditioning
    c_fail: float = 500.0
    num_value_bins: int = 56
    advantage_threshold: float = 0.0
    advantage_dropout: float = 0.3
    cfg_beta: float = 1.0

    # SmolStar06 model settings
    load_vlm_weights: bool = True
    train_expert_only: bool = False
    freeze_vision_encoder: bool = True

    # Weights & Biases (optional; set wandb_project to enable)
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


def _init_wandb(cfg: RECAPSmolStarTrainingConfig):
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


def _resolve_labels_csv(cfg: RECAPSmolStarTrainingConfig) -> Path:
    """Resolve the episode labels CSV path, reusing the value-network logic."""
    if cfg.episode_labels_path is not None:
        resolved = Path(cfg.episode_labels_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Provided --episode_labels_path does not exist: {resolved}")
        return resolved

    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

    dataset_root = Path(cfg.root) / cfg.repo_id if cfg.root else HF_LEROBOT_HOME / cfg.repo_id
    default_path = dataset_root / "meta" / "episode_labels.csv"
    if default_path.is_file():
        return default_path

    try:
        from huggingface_hub import hf_hub_download

        logging.info(
            f"Episode labels not found locally at {default_path}; "
            f"attempting to download from {cfg.repo_id} ..."
        )
        hf_hub_download(
            repo_id=cfg.repo_id,
            filename="meta/episode_labels.csv",
            repo_type="dataset",
            revision=cfg.revision,
            local_dir=str(dataset_root),
        )
    except Exception:  # noqa: BLE001
        pass

    if default_path.is_file():
        return default_path

    raise FileNotFoundError(
        f"No episode labels CSV found at {default_path}\n"
        "Either push meta/episode_labels.csv to your dataset or pass --episode_labels_path."
    )


def _build_policy_config(
    cfg: RECAPSmolStarTrainingConfig,
    train_dataset: LeRobotDataset,
):
    """Build a SmolStar06Config from the training config and dataset metadata."""
    from lerobot.policies.smolstar06.configuration_smolstar06 import SmolStar06Config

    features = dataset_to_policy_features(train_dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    policy_cfg = SmolStar06Config(
        input_features=input_features,
        output_features=output_features,
        value_network_checkpoint=cfg.value_network_checkpoint,
        episode_labels_path=str(_resolve_labels_csv(cfg)),
        c_fail=cfg.c_fail,
        advantage_threshold=cfg.advantage_threshold,
        advantage_dropout=cfg.advantage_dropout,
        cfg_beta=cfg.cfg_beta,
        load_vlm_weights=cfg.load_vlm_weights,
        train_expert_only=cfg.train_expert_only,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
    )
    return policy_cfg


@torch.no_grad()
def _run_validation(
    policy,
    loader: DataLoader,
    preprocessor,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Two-pass validation computing stratified loss and conditioning accuracy.

    For each batch (using suffix-based advantage embedding, not language tokens):
      Pass 1 -- correct advantage embedding  -> per-sample loss_correct
      Pass 2 -- flipped advantage embedding  -> per-sample loss_wrong
    Both passes share identical noise and flow time for fair comparison.
    """
    policy.eval()
    has_episode_info = policy._episode_info is not None

    total_loss = 0.0
    total_loss_pos = 0.0
    total_loss_neg = 0.0
    total_n_pos = 0
    total_n_neg = 0
    total_correct_wins = 0.0
    total_gap = 0.0
    total_gap_pos = 0.0
    total_gap_neg = 0.0
    total_samples = 0

    total_aligned = 0.0
    total_aligned_success = 0.0
    total_aligned_failure = 0.0
    total_success_samples = 0
    total_failure_samples = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = preprocessor(batch)

        advantages, _ = policy._compute_advantages(batch)
        true_indicator = advantages > policy.config.advantage_threshold
        pos_mask = true_indicator
        neg_mask = ~true_indicator

        B = batch[OBS_LANGUAGE_TOKENS].shape[0]

        if has_episode_info:
            ep_indices = batch["episode_index"]
            episode_success = torch.tensor(
                [policy._episode_info[int(idx)]["success"] for idx in ep_indices],
                device=device,
                dtype=torch.bool,
            )
            adv_positive = advantages > policy.config.advantage_threshold
            aligned = (adv_positive == episode_success).float()
            total_aligned += aligned.sum().item()

            success_mask = episode_success
            failure_mask = ~episode_success
            n_success = success_mask.sum().item()
            n_failure = failure_mask.sum().item()
            total_success_samples += n_success
            total_failure_samples += n_failure
            if n_success > 0:
                total_aligned_success += aligned[success_mask].sum().item()
            if n_failure > 0:
                total_aligned_failure += aligned[failure_mask].sum().item()

        padded_actions = policy.prepare_action(batch)
        noise = torch.randn_like(padded_actions)
        fm_time = torch.rand(B, device=device)

        # Pass 1: correct advantage embedding (no dropout)
        losses_correct = policy._forward_with_advantage(
            batch, true_indicator, dropout_mask=None, noise=noise, time=fm_time
        )
        original_action_dim = policy.config.action_feature.shape[0]
        losses_correct = losses_correct[:, :, :original_action_dim]
        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            losses_correct = losses_correct * (~actions_is_pad).unsqueeze(-1)
        losses_correct = losses_correct[:, :, : policy.config.max_action_dim]
        loss_correct = losses_correct.mean(dim=(1, 2))

        # Pass 2: flipped advantage embedding (no dropout)
        losses_wrong = policy._forward_with_advantage(
            batch, ~true_indicator, dropout_mask=None, noise=noise, time=fm_time
        )
        losses_wrong = losses_wrong[:, :, :original_action_dim]
        if actions_is_pad is not None:
            losses_wrong = losses_wrong * (~actions_is_pad).unsqueeze(-1)
        losses_wrong = losses_wrong[:, :, : policy.config.max_action_dim]
        loss_wrong = losses_wrong.mean(dim=(1, 2))

        total_loss += loss_correct.sum().item()
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        total_n_pos += n_pos
        total_n_neg += n_neg
        if n_pos > 0:
            total_loss_pos += loss_correct[pos_mask].sum().item()
        if n_neg > 0:
            total_loss_neg += loss_correct[neg_mask].sum().item()

        correct_wins = (loss_correct < loss_wrong).float()
        total_correct_wins += correct_wins.sum().item()

        gap = loss_wrong - loss_correct
        total_gap += gap.sum().item()
        if n_pos > 0:
            total_gap_pos += gap[pos_mask].sum().item()
        if n_neg > 0:
            total_gap_neg += gap[neg_mask].sum().item()

        total_samples += B

    if total_samples == 0:
        return {
            "val_loss": float("nan"),
            "val_loss_pos": float("nan"),
            "val_loss_neg": float("nan"),
            "val_n_pos": 0,
            "val_n_neg": 0,
            "val_conditioning_accuracy": float("nan"),
            "val_conditioning_gap": float("nan"),
            "val_conditioning_gap_pos": float("nan"),
            "val_conditioning_gap_neg": float("nan"),
            "val_adv_episode_alignment": float("nan"),
            "val_alignment_on_success": float("nan"),
            "val_alignment_on_failure": float("nan"),
        }

    return {
        "val_loss": total_loss / total_samples,
        "val_loss_pos": total_loss_pos / total_n_pos if total_n_pos > 0 else float("nan"),
        "val_loss_neg": total_loss_neg / total_n_neg if total_n_neg > 0 else float("nan"),
        "val_n_pos": total_n_pos,
        "val_n_neg": total_n_neg,
        "val_conditioning_accuracy": total_correct_wins / total_samples,
        "val_conditioning_gap": total_gap / total_samples,
        "val_conditioning_gap_pos": total_gap_pos / total_n_pos if total_n_pos > 0 else float("nan"),
        "val_conditioning_gap_neg": total_gap_neg / total_n_neg if total_n_neg > 0 else float("nan"),
        "val_adv_episode_alignment": total_aligned / total_samples if has_episode_info else float("nan"),
        "val_alignment_on_success": (
            total_aligned_success / total_success_samples
            if has_episode_info and total_success_samples > 0
            else float("nan")
        ),
        "val_alignment_on_failure": (
            total_aligned_failure / total_failure_samples
            if has_episode_info and total_failure_samples > 0
            else float("nan")
        ),
    }


def _log_val_metrics(tag: str, metrics: dict[str, float]) -> None:
    logging.info(
        f"[{tag}] "
        f"val_loss={metrics['val_loss']:.5f} "
        f"(pos={metrics['val_loss_pos']:.5f}, neg={metrics['val_loss_neg']:.5f}) "
        f"cond_acc={metrics['val_conditioning_accuracy']:.4f} "
        f"cond_gap={metrics['val_conditioning_gap']:.5f} "
        f"(gap_pos={metrics['val_conditioning_gap_pos']:.5f}, gap_neg={metrics['val_conditioning_gap_neg']:.5f}) "
        f"adv_ep_align={metrics['val_adv_episode_alignment']:.4f} "
        f"(success={metrics['val_alignment_on_success']:.4f}, failure={metrics['val_alignment_on_failure']:.4f}) "
        f"n_pos={metrics['val_n_pos']} n_neg={metrics['val_n_neg']}"
    )


@parser.wrap()
def run_recap_smolstar_train_val(cfg: RECAPSmolStarTrainingConfig) -> None:
    """Train and validate SmolStar06 with RECAP advantage conditioning and stratified metrics."""
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

    wandb_run = _init_wandb(cfg)

    # ── 1. Load dataset and build episode-level train/val split ──────────
    full_dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=cfg.episodes,
    )

    labels_csv_path = _resolve_labels_csv(cfg)
    logging.info(f"Using episode labels from: {labels_csv_path}")
    success_by_episode = base._load_episode_success_map(labels_csv_path)

    frame_targets = base._build_frame_targets(
        dataset=full_dataset,
        success_by_episode=success_by_episode,
        c_fail=cfg.c_fail,
        num_value_bins=cfg.num_value_bins,
    )
    train_targets, val_targets = base._split_train_val_targets(
        frame_targets=frame_targets,
        val_ratio=cfg.val_split_ratio,
        seed=cfg.seed,
    )

    train_ep_ids = sorted({t.episode_index for t in train_targets})
    val_ep_ids = sorted({t.episode_index for t in val_targets})
    logging.info(
        f"Split: {len(train_ep_ids)} train episodes ({len(train_targets)} frames), "
        f"{len(val_ep_ids)} val episodes ({len(val_targets)} frames)"
    )

    # ── 2. Create separate datasets for train and val ────────────────────
    policy_cfg = _build_policy_config(cfg, full_dataset)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, full_dataset.meta)

    train_dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=train_ep_ids,
        delta_timestamps=delta_timestamps,
    )
    val_dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        revision=cfg.revision,
        episodes=val_ep_ids,
        delta_timestamps=delta_timestamps,
    )

    # ── 3. Create policy ─────────────────────────────────────────────────
    from lerobot.policies.smolstar06.modeling_smolstar06 import SmolStar06Policy

    policy = SmolStar06Policy(config=policy_cfg, dataset_meta=train_dataset.meta)
    policy.to(device)

    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    logging.info(f"Trainable parameters: {num_trainable:,} / {num_total:,} total")

    # ── 4. Create preprocessor ───────────────────────────────────────────
    from lerobot.policies.smolstar06.processor_smolstar06 import make_smolstar06_pre_post_processors

    preprocessor, _postprocessor = make_smolstar06_pre_post_processors(
        config=policy_cfg,
        dataset_stats=train_dataset.meta.stats,  # ty: ignore[invalid-argument-type]
    )

    # ── 5. Create dataloaders ────────────────────────────────────────────
    train_sampler = None
    train_shuffle = True
    if hasattr(policy_cfg, "drop_n_last_frames"):
        train_shuffle = False
        train_sampler = EpisodeAwareSampler(
            train_dataset.meta.episodes["dataset_from_index"],
            train_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=train_dataset.episodes,
            drop_n_last_frames=policy_cfg.drop_n_last_frames,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    step_val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── 6. Optimizer and scheduler ───────────────────────────────────────
    trainable_params = policy.get_optim_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
    )

    # ── 7. Training loop ─────────────────────────────────────────────────
    best_val_cond_acc = -1.0
    history: list[dict] = []
    global_train_step = 0

    if cfg.validate_every_n_train_steps < 0:
        raise ValueError(
            f"validate_every_n_train_steps must be >= 0, got {cfg.validate_every_n_train_steps}"
        )

    logging.info(
        f"Starting training: {cfg.epochs} epochs, "
        f"{len(train_dataset)} train frames, {len(val_dataset)} val frames"
    )

    for epoch in range(1, cfg.epochs + 1):
        policy.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_start = time_module.perf_counter()

        for step, batch in enumerate(train_loader):
            if cfg.max_train_steps_per_epoch is not None and step >= cfg.max_train_steps_per_epoch:
                break

            batch = preprocessor(batch)
            loss, output_dict = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm > 0:
                clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item() * batch[ACTION].shape[0]
            epoch_samples += batch[ACTION].shape[0]
            global_train_step += 1

            wandb_step_metrics: dict[str, float] = {}

            if cfg.log_every_n_steps > 0 and global_train_step % cfg.log_every_n_steps == 0:
                avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time_module.perf_counter() - epoch_start
                logging.info(
                    f"[Epoch {epoch}/{cfg.epochs} step {step+1}] "
                    f"train_loss={avg_loss:.5f} lr={lr:.2e} elapsed={elapsed:.1f}s "
                    f"global_step={global_train_step}"
                )
                wandb_step_metrics.update({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/step_loss": loss.item(),
                    "global_step": global_train_step,
                })

            # Step-based validation
            if (
                cfg.validate_every_n_train_steps > 0
                and global_train_step % cfg.validate_every_n_train_steps == 0
            ):
                step_val_max = (
                    cfg.max_val_steps_per_step_validation
                    if cfg.max_val_steps_per_step_validation is not None
                    else cfg.max_val_steps_per_epoch
                )
                step_val_metrics = _run_validation(
                    policy, step_val_loader, preprocessor, device, max_steps=step_val_max
                )
                tag = (
                    f"Epoch {epoch}/{cfg.epochs} step-validate "
                    f"(global_step={global_train_step})"
                )
                _log_val_metrics(tag, step_val_metrics)
                wandb_step_metrics.update(
                    {f"val/{k}": v for k, v in step_val_metrics.items()}
                )
                policy.train()

            if wandb_run is not None and wandb_step_metrics:
                wandb_run.log(wandb_step_metrics, step=global_train_step)

        train_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")

        # End-of-epoch validation
        try:
            val_metrics = _run_validation(
                policy, val_loader, preprocessor, device, max_steps=cfg.max_val_steps_per_epoch
            )
        except Exception as error:  # noqa: BLE001
            if not base._is_known_video_validation_error(error):
                raise
            logging.warning(
                f"[Epoch {epoch}] Validation failed with video error; "
                "retrying with num_workers=0."
            )
            try:
                val_metrics = _run_validation(
                    policy, step_val_loader, preprocessor, device,
                    max_steps=cfg.max_val_steps_per_epoch,
                )
            except Exception as retry_error:  # noqa: BLE001
                if not base._is_known_video_validation_error(retry_error):
                    raise
                logging.warning(f"[Epoch {epoch}] Validation skipped: {retry_error}")
                val_metrics = {
                    "val_loss": float("nan"),
                    "val_loss_pos": float("nan"),
                    "val_loss_neg": float("nan"),
                    "val_n_pos": 0,
                    "val_n_neg": 0,
                    "val_conditioning_accuracy": float("nan"),
                    "val_conditioning_gap": float("nan"),
                    "val_conditioning_gap_pos": float("nan"),
                    "val_conditioning_gap_neg": float("nan"),
                    "val_adv_episode_alignment": float("nan"),
                    "val_alignment_on_success": float("nan"),
                    "val_alignment_on_failure": float("nan"),
                }

        scheduler.step()

        _log_val_metrics(f"Epoch {epoch}/{cfg.epochs} epoch-end", val_metrics)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **val_metrics,
        }
        history.append(epoch_metrics)

        logging.info(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['val_loss']:.5f} "
            f"cond_acc={val_metrics['val_conditioning_accuracy']:.4f} "
            f"cond_gap={val_metrics['val_conditioning_gap']:.5f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {f"epoch/{k}": v for k, v in epoch_metrics.items()},
                step=global_train_step,
            )

        base._save_json(output_dir / "metrics_history.json", history)

        checkpoint = {
            "epoch": epoch,
            "global_train_step": global_train_step,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_config": policy_cfg,
            "train_config": asdict(cfg),
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")

        cond_acc = val_metrics["val_conditioning_accuracy"]
        if not (cond_acc != cond_acc) and cond_acc > best_val_cond_acc:  # noqa: PLR0124 (NaN check)
            best_val_cond_acc = cond_acc
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            logging.info(f"New best conditioning accuracy: {best_val_cond_acc:.4f}")

    logging.info(
        f"Training complete. Best val conditioning accuracy: {best_val_cond_acc:.4f}"
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run_recap_smolstar_train_val()  # ty: ignore[missing-argument]
