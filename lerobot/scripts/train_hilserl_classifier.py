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
from pprint import pformat

import numpy as np
import torch
import torch.nn as nn
import wandb
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from torch import optim
from torch.autograd import profiler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm import tqdm

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.hilserl.classifier.configuration_classifier import ClassifierConfig
from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.server.buffer import random_shift


def get_model(cfg, logger):  # noqa I001
    classifier_config = _policy_cfg_from_hydra_cfg(ClassifierConfig, cfg)
    model = Classifier(classifier_config)
    if cfg.resume:
        model.load_state_dict(Classifier.from_pretrained(str(logger.last_pretrained_model_dir)).state_dict())
    return model


def create_balanced_sampler(dataset, cfg):
    # Get underlying dataset if using Subset
    original_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset

    # Get indices if using Subset (for slicing)
    indices = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else None

    # Get labels from Hugging Face dataset
    if indices is not None:
        # Get subset of labels using Hugging Face's select()
        hf_subset = original_dataset.hf_dataset.select(indices)
        labels = hf_subset[cfg.training.label_key]
    else:
        # Get all labels directly
        labels = original_dataset.hf_dataset[cfg.training.label_key]

    labels = torch.stack(labels)
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
        images = [random_shift(img, 4) for img in images]
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


def validate(model, val_loader, criterion, device, logger, cfg):
    # Validation loop with metric tracking and sample logging
    model.eval()
    correct = 0
    total = 0
    batch_start_time = time.perf_counter()
    samples = []
    running_loss = 0
    inference_times = []

    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if support_amp(device, cfg) else nullcontext(),
    ):
        for batch in tqdm(val_loader, desc="Validation"):
            images = [batch[img_key].to(device) for img_key in cfg.training.image_keys]
            labels = batch[cfg.training.label_key].float().to(device)

            if cfg.training.profile_inference_time and logger._cfg.wandb.enable:
                with (
                    profiler.profile(record_shapes=True) as prof,
                    profiler.record_function("model_inference"),
                ):
                    outputs = model(images)
                inference_times.append(
                    next(x for x in prof.key_averages() if x.key == "model_inference").cpu_time
                )
            else:
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
            if len(samples) < cfg.eval.num_samples_to_log:
                for i in range(min(cfg.eval.num_samples_to_log - len(samples), len(images))):
                    if model.config.num_classes == 2:
                        confidence = round(outputs.probabilities[i].item(), 3)
                    else:
                        confidence = [round(prob, 3) for prob in outputs.probabilities[i].tolist()]
                    samples.append(
                        {
                            **{
                                f"image_{img_key}": wandb.Image(images[img_idx][i].cpu())
                                for img_idx, img_key in enumerate(cfg.training.image_keys)
                            },
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
            data=[list(s.values()) for s in samples],
            columns=list(samples[0].keys()),
        )
        if logger._cfg.wandb.enable
        else None,
    }

    if len(inference_times) > 0:
        eval_info["inference_time_avg"] = np.mean(inference_times)
        eval_info["inference_time_median"] = np.median(inference_times)
        eval_info["inference_time_std"] = np.std(inference_times)
        eval_info["inference_time_batch_size"] = val_loader.batch_size

        print(
            f"Inference mean time: {eval_info['inference_time_avg']:.2f} us, median: {eval_info['inference_time_median']:.2f} us, std: {eval_info['inference_time_std']:.2f} us, with {len(inference_times)} iterations on {device.type} device, batch size: {eval_info['inference_time_batch_size']}"
        )

    return accuracy, eval_info


def benchmark_inference_time(model, dataset, logger, cfg, device, step):
    if not cfg.training.profile_inference_time:
        return

    iters = cfg.training.profile_inference_time_iters
    inference_times = []

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.training.num_workers,
        sampler=RandomSampler(dataset),
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(iters), desc="Benchmarking inference time"):
            x = next(iter(loader))
            x = [x[img_key].to(device) for img_key in cfg.training.image_keys]

            # Warm up
            for _ in range(10):
                _ = model(x)

            # sync the device
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            with (
                profiler.profile(record_shapes=True) as prof,
                profiler.record_function("model_inference"),
            ):
                _ = model(x)

            inference_times.append(
                next(x for x in prof.key_averages() if x.key == "model_inference").cpu_time
            )

    inference_times = np.array(inference_times)
    avg, median, std = (
        inference_times.mean(),
        np.median(inference_times),
        inference_times.std(),
    )
    print(
        f"Inference time mean: {avg:.2f} us, median: {median:.2f} us, std: {std:.2f} us, with {iters} iterations on {device.type} device"
    )
    if logger._cfg.wandb.enable:
        logger.log_dict(
            {
                "inference_time_benchmark_avg": avg,
                "inference_time_benchmark_median": median,
                "inference_time_benchmark_std": std,
            },
            step + 1,
            mode="eval",
        )

    return avg, median, std


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None) -> None:
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    # Main training pipeline with support for resuming training
    init_logging()
    logging.info(OmegaConf.to_yaml(cfg))

    # Initialize training environment
    device = get_safe_torch_device(cfg.device, log=True)
    set_global_seed(cfg.seed)

    # Setup dataset and dataloaders
    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        root=cfg.dataset_root,
        local_files_only=cfg.local_files_only,
    )
    logging.info(f"Dataset size: {len(dataset)}")

    n_total = len(dataset)
    n_train = int(cfg.train_split_proportion * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_total))

    sampler = create_balanced_sampler(train_dataset, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
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
        logging.info(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")

        train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            grad_scaler,
            device,
            logger,
            step,
            cfg,
        )

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
                identifier=f"{epoch + 1:06d}",
            )

        step += len(train_loader)

    benchmark_inference_time(model, dataset, logger, cfg, device, step)

    logging.info("Training completed")


@hydra.main(
    version_base="1.2",
    config_name="hilserl_classifier",
    config_path="../configs/policy",
)
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train_notebook(
    out_dir=None,
    job_name=None,
    config_name="hilserl_classifier",
    config_path="../configs/policy",
):
    from hydra import compose, initialize

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    train(cfg, out_dir=out_dir, job_name=job_name)


if __name__ == "__main__":
    train_cli()
