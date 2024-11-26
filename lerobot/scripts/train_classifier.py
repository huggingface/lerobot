#!/usr/bin/env python

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat

import hydra
import torch
import torch.nn as nn
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

import wandb
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.logger import Logger
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    set_global_seed,
)


def create_balanced_sampler(dataset, cfg):
    labels = []
    for item in dataset:
        labels.append(item[cfg.training.label_key])
    labels = torch.tensor(labels)

    _, counts = torch.unique(labels, return_counts=True)
    class_weights = 1.0 / counts.float()
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def train_epoch(model, train_loader, criterion, optimizer, grad_scaler, device, logger, step, cfg):
    model.train()
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        start_time = time.perf_counter()
        images = batch[cfg.training.image_key].to(device)
        labels = batch[cfg.training.label_key].float().to(device)

        with torch.autocast(device_type=device.type) if cfg.training.use_amp else nullcontext():
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        if cfg.training.use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        current_acc = 100 * correct / total
        train_info = {
            "loss": loss.item(),
            "accuracy": current_acc,
            "dataloading_s": time.perf_counter() - start_time,
        }

        logger.log_dict(train_info, step + batch_idx, mode="train")

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"})


def validate(model, val_loader, criterion, device, logger, cfg, use_amp=False, num_samples_to_log=8):
    model.eval()
    correct = 0
    total = 0
    batch_start_time = time.perf_counter()
    samples = []
    running_loss = 0

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.training.use_amp else nullcontext():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch[cfg.training.image_key].to(device)
            labels = batch[cfg.training.label_key].float().to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            if len(samples) < num_samples_to_log:
                for i in range(min(num_samples_to_log - len(samples), len(images))):
                    samples.append(
                        {
                            "image": wandb.Image(images[i].cpu()),
                            "true_label": labels[i].item(),
                            "predicted": predictions[i].item(),
                            "confidence": torch.sigmoid(outputs.logits[i]).item(),
                        }
                    )

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(val_loader)

    eval_info = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "eval_s": time.perf_counter() - batch_start_time,
        "eval/prediction_samples": wandb.Table(
            data=[[s["image"], s["true_label"], s["predicted"], f"{s['confidence']:.3f}"] for s in samples],
            columns=["Image", "True Label", "Predicted", "Confidence"],
        )
        if logger._cfg.wandb.enable
        else None,
    }

    return accuracy, eval_info


@hydra.main(version_base="1.2", config_path="../configs", config_name="classifier")
def train(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))

    # Setup device and seeds
    device = get_safe_torch_device(cfg.device, log=True)
    set_global_seed(cfg.seed)

    # Create output directory and logger
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(cfg, out_dir, cfg.wandb.job_name if cfg.wandb.enable else None)

    # Load dataset
    dataset = LeRobotDataset(cfg.dataset_repo_id)
    logging.info(f"Dataset size: {len(dataset)}")

    # Split dataset
    train_size = int(cfg.train_split_proportion * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    sampler = create_balanced_sampler(train_dataset, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler=sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,  # Using eval batch size for validation
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Training loop
    step = 0
    best_val_acc = 0

    if cfg.resume:
        if not Logger.get_last_checkpoint_dir(out_dir).exists():
            raise RuntimeError(
                "You have set resume=True, but there is no model checkpoint in "
                f"{Logger.get_last_checkpoint_dir(out_dir)}"
            )
        checkpoint_cfg_path = str(Logger.get_last_pretrained_model_dir(out_dir) / "config.yaml")
        logging.info(
            colored(
                "You have set resume=True, indicating that you wish to resume a run",
                color="yellow",
                attrs=["bold"],
            )
        )
        # Get the configuration file from the last checkpoint.
        checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)
        # Check for differences between the checkpoint configuration and provided configuration.
        # Hack to resolve the delta_timestamps ahead of time in order to properly diff.
        resolve_delta_timestamps(cfg)
        diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))
        # Ignore the `resume` and parameters.
        if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
            del diff["values_changed"]["root['resume']"]
        # Log a warning about differences between the checkpoint configuration and the provided
        # configuration.
        if len(diff) > 0:
            logging.warning(
                "At least one difference was detected between the checkpoint configuration and "
                f"the provided configuration: \n{pformat(diff)}\nNote that the checkpoint configuration "
                "takes precedence.",
            )
        # Use the checkpoint config instead of the provided config (but keep `resume` parameter).
        cfg = checkpoint_cfg
        cfg.resume = True
    model = make_policy(
        hydra_cfg=cfg,
        dataset_stats=dataset.meta.stats if not cfg.resume else None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir) if cfg.resume else None,
    ).to(device)

    # Setup training components
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    grad_scaler = GradScaler(enabled=cfg.training.use_amp)

    # Log model info
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Learnable parameters: {format_big_number(num_learnable_params)}")
    logging.info(f"Total parameters: {format_big_number(num_total_params)}")

    if cfg.resume:
        step = logger.load_last_training_state(optimizer, None)

    for epoch in range(cfg.training.num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{cfg.training.num_epochs}")

        train_epoch(model, train_loader, criterion, optimizer, grad_scaler, device, logger, step, cfg)

        if cfg.training.eval_freq > 0 and (epoch + 1) % cfg.training.eval_freq == 0:
            val_acc, eval_info = validate(
                model,
                val_loader,
                criterion,
                device,
                logger,
                cfg,
            )
            logger.log_dict(eval_info, step + len(train_loader), mode="eval")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.save_checkpoint(
                    train_step=step + len(train_loader),
                    policy=model,
                    optimizer=optimizer,
                    scheduler=None,
                    identifier="best",
                )

        if cfg.training.save_checkpoint and (epoch + 1) % cfg.training.save_freq == 0:
            logger.save_checkpoint(
                train_step=step + len(train_loader),
                policy=model,
                optimizer=optimizer,
                scheduler=None,
                identifier=f"{epoch+1:06d}",
            )

        step += len(train_loader)

    logging.info("Training completed")


if __name__ == "__main__":
    train()
