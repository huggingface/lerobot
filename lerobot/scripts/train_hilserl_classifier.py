#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat

import hydra
import torch
import torch.nn as nn
import wandb
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.logger import Logger
from lerobot.common.policies.factory import _policy_cfg_from_hydra_cfg
from lerobot.common.policies.hilserl.classifier.configuration_classifier import ClassifierConfig
from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    set_global_seed,
)


def get_model(cfg, logger):  # noqa I001
    classifier_config = _policy_cfg_from_hydra_cfg(ClassifierConfig, cfg)
    model = Classifier(classifier_config)
    if cfg.resume:
        model.load_state_dict(Classifier.from_pretrained(str(logger.last_pretrained_model_dir)).state_dict())
    return model


def create_balanced_sampler(dataset, cfg):
    # Creates a weighted sampler to handle class imbalance

    labels = torch.tensor([item[cfg.training.label_key] for item in dataset])
    _, counts = torch.unique(labels, return_counts=True)
    class_weights = 1.0 / counts.float()
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def support_amp(device: torch.device, cfg: DictConfig) -> bool:
    # Check if the device supports AMP
    # Here is an example of the issue that says that MPS doesn't support AMP properply
    return cfg.training.use_amp and device.type in ("cuda", "cpu")


def train_epoch(model, train_loader, criterion, optimizer, grad_scaler, device, logger, step, cfg):
    # Single epoch training loop with AMP support and progress tracking
    model.train()
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        start_time = time.perf_counter()
        images = [batch[img_key].to(device) for img_key in cfg.training.image_keys]
        labels = batch[cfg.training.label_key].float().to(device)

        # Forward pass with optional AMP
        with torch.autocast(device_type=device.type) if support_amp(device, cfg) else nullcontext():
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

        # Backward pass with gradient scaling if AMP enabled
        optimizer.zero_grad()
        if cfg.training.use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Track metrics
        if model.config.num_classes == 2:
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            predictions = torch.argmax(outputs.logits, dim=1)
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


def validate(model, val_loader, criterion, device, logger, cfg, num_samples_to_log=8):
    # Validation loop with metric tracking and sample logging
    model.eval()
    correct = 0
    total = 0
    batch_start_time = time.perf_counter()
    samples = []
    running_loss = 0

    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if support_amp(device, cfg) else nullcontext(),
    ):
        for batch in tqdm(val_loader, desc="Validation"):
            images = [batch[img_key].to(device) for img_key in cfg.training.image_keys]
            labels = batch[cfg.training.label_key].float().to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            # Track metrics
            if model.config.num_classes == 2:
                predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
            else:
                predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            # Log sample predictions for visualization
            if len(samples) < num_samples_to_log:
                for i in range(min(num_samples_to_log - len(samples), len(images))):
                    if model.config.num_classes == 2:
                        confidence = round(outputs.probabilities[i].item(), 3)
                    else:
                        confidence = [round(prob, 3) for prob in outputs.probabilities[i].tolist()]
                    samples.append(
                        {
                            "image": wandb.Image(images[i].cpu()),
                            "true_label": labels[i].item(),
                            "predicted": predictions[i].item(),
                            "confidence": confidence,
                        }
                    )

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(val_loader)
    print(f"Average validation loss {avg_loss}, and accuracy {accuracy}")

    eval_info = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "eval_s": time.perf_counter() - batch_start_time,
        "eval/prediction_samples": wandb.Table(
            data=[[s["image"], s["true_label"], s["predicted"], f"{s['confidence']}"] for s in samples],
            columns=["Image", "True Label", "Predicted", "Confidence"],
        )
        if logger._cfg.wandb.enable
        else None,
    }

    return accuracy, eval_info


@hydra.main(version_base="1.2", config_path="../configs/policy", config_name="hilserl_classifier")
def train(cfg: DictConfig) -> None:
    # Main training pipeline with support for resuming training
    logging.info(OmegaConf.to_yaml(cfg))

    # Initialize training environment
    device = get_safe_torch_device(cfg.device, log=True)
    set_global_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(cfg, out_dir, cfg.wandb.job_name if cfg.wandb.enable else None)

    # Setup dataset and dataloaders
    dataset = LeRobotDataset(cfg.dataset_repo_id)
    logging.info(f"Dataset size: {len(dataset)}")

    train_size = int(cfg.train_split_proportion * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Resume training if requested
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
        # Load and validate checkpoint configuration
        checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)
        # Check for differences between the checkpoint configuration and provided configuration.
        # Hack to resolve the delta_timestamps ahead of time in order to properly diff.
        resolve_delta_timestamps(cfg)
        diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))
        # Ignore the `resume` and parameters.
        if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
            del diff["values_changed"]["root['resume']"]
        if len(diff) > 0:
            logging.warning(
                "At least one difference was detected between the checkpoint configuration and "
                f"the provided configuration: \n{pformat(diff)}\nNote that the checkpoint configuration "
                "takes precedence.",
            )
        # Use the checkpoint config instead of the provided config (but keep `resume` parameter).
        cfg = checkpoint_cfg
        cfg.resume = True

    # Initialize model and training components
    model = get_model(cfg=cfg, logger=logger).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    # Use BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multi-class
    criterion = nn.BCEWithLogitsLoss() if model.config.num_classes == 2 else nn.CrossEntropyLoss()
    grad_scaler = GradScaler(enabled=cfg.training.use_amp)

    # Log model parameters
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Learnable parameters: {format_big_number(num_learnable_params)}")
    logging.info(f"Total parameters: {format_big_number(num_total_params)}")

    if cfg.resume:
        step = logger.load_last_training_state(optimizer, None)

    # Training loop with validation and checkpointing
    for epoch in range(cfg.training.num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{cfg.training.num_epochs}")

        train_epoch(model, train_loader, criterion, optimizer, grad_scaler, device, logger, step, cfg)

        # Periodic validation
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

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.save_checkpoint(
                    train_step=step + len(train_loader),
                    policy=model,
                    optimizer=optimizer,
                    scheduler=None,
                    identifier="best",
                )

        # Periodic checkpointing
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
