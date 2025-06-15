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
import re
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def slugify_string(text: str, max_length: int = 50) -> str:
    """Convert a string to a safe filename slug.

    Args:
        text: Input text to slugify
        max_length: Maximum length of the resulting slug

    Returns:
        Safe filename string
    """
    # Remove or replace unsafe characters
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[-\s]+", "-", slug)
    slug = slug.strip("-")

    # Truncate if too long
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")

    return slug if slug else "untitled"


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to PIL Image.

    Args:
        tensor: Tensor of shape (C, H, W) with values in [0, 1]

    Returns:
        PIL Image
    """
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor[0]

    # Ensure tensor is on CPU and in correct format
    if tensor.device != torch.device("cpu"):
        tensor = tensor.cpu()

    # Convert to PIL
    to_pil = ToPILImage()
    return to_pil(tensor)


def create_text_image(text: str, width: int, height: int, font_size: int = 16) -> Image.Image:
    """Create an image with text.

    Args:
        text: Text to render
        width: Image width
        height: Image height
        font_size: Font size for text

    Returns:
        PIL Image with text
    """
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        # Try to use a better font if available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        # Fallback to default font
        font = ImageFont.load_default()

    # Wrap text to fit in image
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= width - 20:  # Leave some margin
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    # Draw text lines
    y_offset = 10
    line_height = font_size + 2

    for line in lines:
        if y_offset + line_height > height - 10:  # Stop if we run out of space
            break
        draw.text((10, y_offset), line, fill="black", font=font)
        y_offset += line_height

    return img


def create_concatenated_visualization(
    input_images: dict[str, torch.Tensor],
    output_data: dict[str, Any],
    task_text: str,
    step: int,
    additional_info: dict[str, Any] = None,
) -> Image.Image:
    """Create a concatenated visualization of input images, output data, and task text.

    Args:
        input_images: Dictionary of input images (camera_key -> tensor)
        output_data: Dictionary containing output data (e.g., predicted actions)
        task_text: Language instruction/task description
        step: Training step number
        additional_info: Additional information to display

    Returns:
        Concatenated PIL Image
    """
    images_to_concat = []

    # Add task text as an image
    if input_images:
        first_img_tensor = next(iter(input_images.values()))
        img_height = first_img_tensor.shape[-2]
        img_width = first_img_tensor.shape[-1]
    else:
        img_height, img_width = 224, 224

    task_img = create_text_image(f"Task: {task_text}\nStep: {step}", img_width, img_height)
    images_to_concat.append(task_img)

    # Add input images
    for camera_key, img_tensor in input_images.items():
        try:
            pil_img = tensor_to_pil(img_tensor)
            # Resize to match first image if needed
            pil_img = pil_img.resize((img_width, img_height))
            images_to_concat.append(pil_img)
        except Exception as e:
            logging.warning(f"Failed to convert image for {camera_key}: {e}")
            # Create placeholder image
            placeholder = Image.new("RGB", (img_width, img_height), color="gray")
            draw = ImageDraw.Draw(placeholder)
            draw.text((10, 10), f"Failed to load\n{camera_key}", fill="white")
            images_to_concat.append(placeholder)

    # Add output information as text image
    if output_data:
        output_text = "Outputs:\n"
        for key, value in output_data.items():
            if isinstance(value, torch.Tensor):
                if value.numel() <= 10:  # Only show small tensors
                    output_text += f"{key}: {value.detach().cpu().numpy()}\n"
                else:
                    output_text += f"{key}: shape {value.shape}\n"
            else:
                output_text += f"{key}: {value}\n"

        output_img = create_text_image(output_text, img_width, img_height)
        images_to_concat.append(output_img)

    # Add additional info if provided
    if additional_info:
        info_text = "Additional Info:\n"
        for key, value in additional_info.items():
            info_text += f"{key}: {value}\n"

        info_img = create_text_image(info_text, img_width, img_height)
        images_to_concat.append(info_img)

    # Concatenate images horizontally
    if images_to_concat:
        total_width = sum(img.width for img in images_to_concat)
        max_height = max(img.height for img in images_to_concat)

        concat_img = Image.new("RGB", (total_width, max_height), color="white")
        x_offset = 0

        for img in images_to_concat:
            concat_img.paste(img, (x_offset, 0))
            x_offset += img.width

        return concat_img
    else:
        # Return empty image if no images to concatenate
        return Image.new("RGB", (200, 100), color="gray")


def log_training_samples(
    batch: dict[str, torch.Tensor],
    output_dict: dict[str, Any],
    step: int,
    output_dir: Path,
    num_samples: int = 4,
) -> None:
    """Log training samples with visualizations.

    Args:
        batch: Training batch data
        output_dict: Model outputs
        step: Training step number
        output_dir: Output directory for saving logs
        num_samples: Number of samples to log
    """
    try:
        logging.debug(f"Logging training samples for step {step}")

        # Create samples directory
        samples_dir = output_dir / "training_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Get batch size
        batch_size = batch["action"].shape[0] if "action" in batch else 1
        num_samples = min(num_samples, batch_size)

        # Get camera keys from batch - filter out padding indicators
        camera_keys = [
            key for key in batch if key.startswith("observation.images.") and not key.endswith("_is_pad")
        ]

        for i in range(num_samples):
            try:
                # Extract input images for this sample
                input_images = {}
                for camera_key in camera_keys:
                    if camera_key in batch and isinstance(batch[camera_key], torch.Tensor):
                        tensor = batch[camera_key][i]
                        # Check if this is actually an image tensor (should have at least 2 dimensions)
                        if tensor.dim() >= 2 and tensor.numel() > 0:
                            input_images[camera_key] = tensor

                # Get task text
                task_text = ""
                if "task" in batch:
                    if isinstance(batch["task"], list):
                        task_text = batch["task"][i] if i < len(batch["task"]) else ""
                    else:
                        task_text = str(batch["task"][i]) if batch["task"].numel() > i else ""

                # Extract relevant output data
                sample_output_data = {}
                if output_dict:
                    for key, value in output_dict.items():
                        if isinstance(value, torch.Tensor) and value.shape[0] > i:
                            sample_output_data[key] = value[i]
                        elif not isinstance(value, torch.Tensor):
                            sample_output_data[key] = value

                # Additional info
                additional_info = {}
                if "action" in batch:
                    action = batch["action"][i].detach().cpu().numpy()
                    additional_info["action"] = action[:6] if len(action) > 6 else action  # Show first 6 dims

                # Create visualization
                concat_img = create_concatenated_visualization(
                    input_images=input_images,
                    output_data=sample_output_data,
                    task_text=task_text,
                    step=step,
                    additional_info=additional_info,
                )

                # Create filename
                task_slug = slugify_string(task_text) if task_text else "no_task"
                filename = f"step_{step:06d}_sample_{i:02d}_{task_slug}.jpg"

                # Save image
                filepath = samples_dir / filename
                concat_img.save(filepath, "JPEG", quality=95)

                logging.debug(f"Saved training sample: {filepath}")

            except Exception as e:
                logging.warning(f"Failed to log training sample {i} for step {step}: {e}")

    except Exception as e:
        logging.error(f"Failed to log training samples for step {step}: {e}")


def log_test_inferences(
    policy: PreTrainedPolicy, dataset, device: torch.device, step: int, output_dir: Path, num_samples: int = 8
) -> None:
    """Log test inferences with visualizations.

    Args:
        policy: Trained policy model
        dataset: Dataset for getting test samples
        device: Device for inference
        step: Training step number
        output_dir: Output directory for saving logs
        num_samples: Number of test samples to generate
    """
    try:
        logging.debug(f"Generating test inferences for step {step}")

        # Create test inferences directory
        test_dir = output_dir / "test_inferences"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Get some test samples from dataset
        test_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

        policy.eval()
        with torch.no_grad():
            for i, idx in enumerate(test_indices):
                # Convert numpy int64 to Python int for dataset indexing
                idx = int(idx)
                try:
                    # Get sample from dataset
                    sample = dataset[idx]

                    # Prepare batch (add batch dimension)
                    batch = {}
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() > 0:  # Check if tensor is not empty
                                batch[key] = value.unsqueeze(0).to(device)
                            else:
                                logging.warning(f"Skipping empty tensor for key: {key}")
                                continue
                        elif isinstance(value, str):
                            batch[key] = [value]
                        else:
                            batch[key] = value

                    # Check if we have essential data
                    if "action" not in batch:
                        logging.warning(f"Sample {idx} missing 'action' key, skipping")
                        continue

                    # Get camera keys - filter out padding indicators
                    camera_keys = [
                        key
                        for key in batch
                        if key.startswith("observation.images.") and not key.endswith("_is_pad")
                    ]

                    # Extract input images
                    input_images = {}
                    for camera_key in camera_keys:
                        if camera_key in batch and isinstance(batch[camera_key], torch.Tensor):
                            tensor = batch[camera_key][0]
                            # Check if this is actually an image tensor (should have at least 2 dimensions)
                            if tensor.dim() >= 2 and tensor.numel() > 0:
                                input_images[camera_key] = tensor

                    # Get task text
                    task_text = ""
                    if "task" in batch:
                        if isinstance(batch["task"], list):
                            task_text = batch["task"][0] if len(batch["task"]) > 0 else ""
                        else:
                            task_text = str(batch["task"]) if hasattr(batch["task"], "__str__") else ""

                    # Run inference
                    if hasattr(policy, "select_action"):
                        # For policies with select_action method
                        action = policy.select_action(batch)
                        output_data = {"predicted_action": action}
                    else:
                        # For policies with forward method
                        with (
                            torch.autocast(device_type=device.type)
                            if hasattr(policy.config, "use_amp") and policy.config.use_amp
                            else nullcontext()
                        ):
                            loss, output_dict = policy.forward(batch)
                        output_data = output_dict if output_dict else {"loss": loss}

                    # Ground truth action
                    additional_info = {}
                    if "action" in batch:
                        gt_action = batch["action"][0].detach().cpu().numpy()
                        additional_info["ground_truth_action"] = (
                            gt_action[:6] if len(gt_action) > 6 else gt_action
                        )

                    # Add inference info
                    additional_info["dataset_index"] = idx
                    additional_info["inference_step"] = step

                    # Create visualization
                    concat_img = create_concatenated_visualization(
                        input_images=input_images,
                        output_data=output_data,
                        task_text=task_text,
                        step=step,
                        additional_info=additional_info,
                    )

                    # Create filename
                    task_slug = slugify_string(task_text) if task_text else "no_task"
                    filename = f"step_{step:06d}_test_{i:02d}_{task_slug}.jpg"

                    # Save image
                    filepath = test_dir / filename
                    concat_img.save(filepath, "JPEG", quality=95)

                    logging.debug(f"Saved test inference: {filepath}")

                except Exception as e:
                    logging.warning(f"Failed to generate test inference {i} for step {step}: {e}")

        policy.train()  # Return to training mode

    except Exception as e:
        logging.error(f"Failed to log test inferences for step {step}: {e}")


def push_checkpoint_to_hub(
    checkpoint_dir: Path, step: int, repo_id: str, private: bool = False, token: str = None
) -> None:
    """Push checkpoint to Hugging Face Hub with step-based branch naming.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        step: Training step number
        repo_id: Repository ID on the Hub (e.g., "username/model-name")
        private: Whether to create a private repository
        token: Hugging Face token for authentication
    """
    try:
        from huggingface_hub import HfApi

        logging.info(f"Pushing checkpoint for step {step} to Hub: {repo_id}")

        # Create step-based branch name
        branch_name = f"step-{step:04d}"

        # Initialize Hub API
        api = HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)
            logging.debug(f"Repository {repo_id} created or already exists")
        except Exception as e:
            logging.warning(f"Could not create repository {repo_id}: {e}")

        # Create branch if it doesn't exist
        try:
            api.create_branch(repo_id=repo_id, branch=branch_name, repo_type="model", exist_ok=True)
            logging.debug(f"Branch {branch_name} created or already exists")
        except Exception as e:
            logging.warning(f"Could not create branch {branch_name}: {e}")

        # Get the pretrained model directory
        pretrained_dir = checkpoint_dir / "pretrained_model"

        if not pretrained_dir.exists():
            logging.error(f"Pretrained model directory not found: {pretrained_dir}")
            return

        # Upload the pretrained model to the branch
        commit_message = f"Upload checkpoint at step {step}"

        commit_info = api.upload_folder(
            repo_id=repo_id,
            folder_path=pretrained_dir,
            repo_type="model",
            revision=branch_name,
            commit_message=commit_message,
            ignore_patterns=["*.git*", "*.DS_Store"],
        )

        logging.info(f"Successfully pushed checkpoint to branch '{branch_name}': {commit_info.commit_url}")

    except ImportError:
        logging.error(
            "huggingface_hub is required for pushing to Hub. Install with: pip install huggingface_hub"
        )
    except Exception as e:
        logging.error(f"Failed to push checkpoint for step {step} to Hub: {e}")


def get_hub_repo_id_from_config(cfg: TrainPipelineConfig) -> str:
    """Generate a reasonable repo_id from the configuration.

    Args:
        cfg: Training pipeline configuration

    Returns:
        A suggested repo_id string
    """
    # Try to construct a meaningful repo_id from config
    parts = []

    if hasattr(cfg, "policy") and cfg.policy:
        parts.append(cfg.policy.type)

    if hasattr(cfg, "dataset") and cfg.dataset and hasattr(cfg.dataset, "repo_id"):
        if isinstance(cfg.dataset.repo_id, list):
            dataset_name = cfg.dataset.repo_id[0].split("/")[-1] if cfg.dataset.repo_id else "dataset"
        else:
            dataset_name = cfg.dataset.repo_id.split("/")[-1] if cfg.dataset.repo_id else "dataset"
        parts.append(dataset_name)

    if hasattr(cfg, "env") and cfg.env:
        parts.append(cfg.env.type)

    # Join parts with underscores and limit length
    repo_name = "_".join(parts)[:50]  # Limit to 50 chars

    # If no meaningful name could be constructed, use a default
    if not repo_name:
        repo_name = "lerobot_model"

    # Note: This assumes the user wants to push to their own namespace
    # In practice, you might want to get the username from HF Hub
    return f"lerobot/{repo_name}"


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def create_policy_with_memory_management(
    cfg: TrainPipelineConfig,
    dataset,
    device: torch.device,
) -> PreTrainedPolicy:
    """Create policy with Accelerate memory management for big models.

    Args:
        cfg: Training pipeline configuration
        dataset: Dataset containing metadata
        device: Target device for training

    Returns:
        PreTrainedPolicy: Policy instance with memory-efficient loading
    """
    # Define memory constraints as specified by user
    # 130GB on GPU 0, 2035843657728 bytes on CPU (exact value provided)
    max_memory = {
        0: "130GB",  # GPU 0
        "cpu": 2035843657728,  # CPU (exact bytes as specified)
    }

    logging.info(f"Creating policy with memory constraints: {max_memory}")

    # Log current memory state before policy creation
    if torch.cuda.is_available():
        logging.info("Memory state before policy creation:")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            logging.info(
                f"  GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved, {mem_total:.2f}GB total"
            )

    # Check if we're loading a pretrained model that might need memory management
    if hasattr(cfg.policy, "pretrained_path") and cfg.policy.pretrained_path:
        logging.info(f"Loading pretrained policy from: {cfg.policy.pretrained_path}")

        # For very large models, we might need to use Accelerate's memory management
        # First, try standard loading with memory monitoring
        try:
            # Clear any existing cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cleared CUDA cache before policy loading")

            policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
            logging.info("Successfully loaded policy with standard method")

            # Log memory after successful loading
            if torch.cuda.is_available():
                logging.info("Memory state after policy creation:")
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    logging.info(f"  GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

            return policy

        except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
            logging.warning(f"Standard loading failed due to memory constraints: {e}")
            logging.info("Memory constraints exceeded - consider using model sharding or CPU offloading")

            # Clear memory and try again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Cleared CUDA cache after failed loading attempt")

            # Log memory constraint recommendation
            logging.info("For large models, consider:")
            logging.info("1. Using model sharding across multiple GPUs")
            logging.info("2. Enabling CPU offloading for model weights")
            logging.info("3. Using gradient checkpointing to reduce memory usage")
            logging.info("4. Reducing batch size or sequence length")

            # Re-raise the original exception with additional context
            raise RuntimeError(
                f"Failed to load policy due to memory constraints. "
                f"Original error: {e}. "
                f"Memory limits: GPU 0: 130GB, CPU: {2035843657728} bytes. "
                f"Consider using model sharding or reducing model size."
            ) from e

    else:
        # For non-pretrained models, use standard creation with memory monitoring
        logging.info("Creating fresh policy (no pretrained weights)")

        # Clear cache before creating new model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.debug("Cleared CUDA cache before fresh policy creation")

        policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

        # Log memory after creation
        if torch.cuda.is_available():
            logging.info("Memory state after fresh policy creation:")
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                logging.info(f"  GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        return policy


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # Configure memory management settings
    # Set max memory constraints: 130GB on GPU 0, ~1.9TB on CPU
    max_memory_config = {
        0: "130GB",  # GPU 0 limit
        "cpu": 2035843657728,  # CPU limit in bytes as specified by user
    }
    logging.info(
        f"Memory constraints configured: GPU 0: 130GB, CPU: {max_memory_config['cpu']} bytes (~1.9TB)"
    )

    # Log current memory usage
    if torch.cuda.is_available():
        logging.info("CUDA memory before training:")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            logging.info(f"  GPU {i}: Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB")

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    # Only change repo_id if it's set to "all/datasets"
    if cfg.dataset.repo_id == "all/datasets":
        cfg.dataset.repo_id = [
            "danielkorth/whiteboard-marker",
            # "HovorunB/pick-data-merged",
            "danielkorth/usbc-cable-2",
            "danielkorth/bike-light",
            "danielkorth/usb-stick",
            "danielkorth/bike-light4am-part2",
            "danielkorth/bike-light4am",
            "danielkorth/usb-C-cable",
            "danielkorth/green-marker-part2",
            "danielkorth/supadummytest2",
            "danielkorth/green-marker4am",
            "danielkorth/green-marker2",
            "danielkorth/green-marker",
            "danielkorth/green-pe",
            "danielkorth/green-pen4",
            # "vectorcrumb/trash_pickup_v1",
            # "islexu/eval_record_test2_orange"
        ]
    else:
        cfg.dataset.repo_id = ["danielkorth/whiteboard-marker", "danielkorth/bike-light"]
    dataset = make_dataset(cfg)

    # Log comprehensive dataset size information
    logging.info("=== DATASET SIZE INFORMATION ===")
    logging.info(f"Total frames: {dataset.num_frames:,} ({format_big_number(dataset.num_frames)})")
    logging.info(f"Total episodes: {dataset.num_episodes:,}")
    logging.info(f"Dataset length (samples): {len(dataset):,}")

    # Log individual dataset contributions if multiple repos
    if hasattr(cfg.dataset, "repo_id") and isinstance(cfg.dataset.repo_id, list):
        logging.info(f"Combined from {len(cfg.dataset.repo_id)} datasets:")
        for i, repo_id in enumerate(cfg.dataset.repo_id):
            logging.info(f"  {i + 1}. {repo_id}")

    # Calculate approximate memory usage if possible
    try:
        # Get a sample to estimate memory usage
        sample = dataset[0]
        sample_size_bytes = 0

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                tensor_bytes = value.element_size() * value.numel()
                sample_size_bytes += tensor_bytes
                logging.debug(f"  {key}: {value.shape} -> {tensor_bytes:,} bytes")
            elif isinstance(value, str):
                sample_size_bytes += len(value.encode("utf-8"))

        total_dataset_size_gb = (sample_size_bytes * len(dataset)) / (1024**3)
        logging.info(f"Estimated dataset size in memory: {total_dataset_size_gb:.2f} GB")
        logging.info(f"Average sample size: {sample_size_bytes / (1024**2):.2f} MB")

    except Exception as e:
        logging.warning(f"Could not estimate dataset memory usage: {e}")

    # Log dataset metadata if available
    if hasattr(dataset, "meta") and dataset.meta:
        logging.info("Dataset metadata:")
        try:
            # Handle MultiDatasetMeta objects
            if hasattr(dataset.meta, "__dict__"):
                meta_dict = vars(dataset.meta)
            elif hasattr(dataset.meta, "items"):
                meta_dict = dict(dataset.meta.items())
            else:
                meta_dict = {"meta_type": type(dataset.meta).__name__}

            for key, value in meta_dict.items():
                if isinstance(value, (int, float, str, bool)) or isinstance(value, dict) and len(value) < 10:
                    logging.info(f"  {key}: {value}")
                else:
                    logging.info(
                        f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})"
                    )
        except Exception as e:
            logging.info(f"  Could not parse metadata: {type(dataset.meta)} - {e}")

    logging.info("=== END DATASET SIZE INFO ===")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = create_policy_with_memory_management(cfg, dataset, device)

    # Final memory status after policy creation
    logging.info("=== MEMORY MANAGEMENT SUMMARY ===")
    logging.info(f"Memory constraints: GPU 0: 130GB, CPU: {max_memory_config['cpu']} bytes")
    if torch.cuda.is_available():
        total_gpu_memory = 0
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            total_gpu_memory += mem_allocated
            logging.info(
                f"GPU {i}: {mem_allocated:.2f}GB/{mem_total:.2f}GB used ({mem_allocated / mem_total * 100:.1f}%)"
            )

        logging.info(f"Total GPU memory allocated: {total_gpu_memory:.2f}GB")

        # Check if we're approaching the 130GB limit on GPU 0
        gpu0_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu0_limit = 130  # GB
        if gpu0_allocated > gpu0_limit * 0.8:  # 80% threshold
            logging.warning(f"GPU 0 memory usage ({gpu0_allocated:.2f}GB) is approaching the 130GB limit")
        else:
            logging.info(f"GPU 0 memory usage ({gpu0_allocated:.2f}GB) is within the 130GB limit")

    logging.info("=== END MEMORY SUMMARY ===")

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    logging.debug(f"Training loop starting from step {step} to step {cfg.steps}")

    # Create progress bar for training loop
    pbar = tqdm(
        range(step, cfg.steps),
        desc="Training",
        initial=step,
        total=cfg.steps,
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )
    logging.debug("Progress bar created for training loop")

    for _ in pbar:
        logging.debug(f"Starting training step {step}")
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        logging.debug(f"Data loading completed in {train_tracker.dataloading_s.val:.4f}s for step {step}")

        # Move batch to device
        logging.debug(f"Moving batch data to device {device} for step {step}")
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        logging.debug(f"Starting policy update for step {step}")
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )
        logging.debug(f"Policy update completed for step {step}, loss: {train_tracker.loss.val:.6f}")

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_sample_log_step = step % 500 == 0  # Log training samples every 500 steps
        is_test_inference_step = step % 500 == 0  # Log test inferences every 500 steps
        is_hub_push_step = cfg.push_to_hub and is_saving_step  # Push to hub every time we save a checkpoint

        logging.debug(
            f"Step {step} completed. log_step={is_log_step}, saving_step={is_saving_step}, eval_step={is_eval_step}, sample_log={is_sample_log_step}, hub_push={is_hub_push_step}"
        )

        # Update progress bar with current metrics
        pbar.set_postfix(
            {
                "loss": f"{train_tracker.loss.val:.4f}",
                "lr": f"{train_tracker.lr.val:.2e}",
                "grad_norm": f"{train_tracker.grad_norm.val:.3f}",
            }
        )

        # Log training samples every 500 steps
        if is_sample_log_step and step > 0:
            logging.debug(f"Logging training samples for step {step}")
            log_training_samples(
                batch=batch, output_dict=output_dict, step=step, output_dir=cfg.output_dir, num_samples=4
            )

        # Log test inferences every 500 steps
        if is_test_inference_step and step > 0:
            logging.debug(
                f"Skipping test inferences for step {step} - temporarily disabled due to tensor issues"
            )
            # Temporarily disabled to avoid tensor stacking errors
            # log_test_inferences(
            #     policy=policy,
            #     dataset=dataset,
            #     device=device,
            #     step=step,
            #     output_dir=cfg.output_dir,
            #     num_samples=8
            # )

        if is_log_step:
            logging.debug(f"Logging metrics for step {step}")
            logging.info(train_tracker)

            # Log memory usage
            if torch.cuda.is_available():
                logging.debug(f"CUDA memory at step {step}:")
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    logging.debug(
                        f"  GPU {i}: Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB"
                    )

            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)

                # Add memory metrics to wandb logging
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                        wandb_log_dict[f"memory/gpu_{i}_allocated_gb"] = mem_allocated
                        wandb_log_dict[f"memory/gpu_{i}_reserved_gb"] = mem_reserved

                wandb_logger.log_dict(wandb_log_dict, step)
                logging.debug(f"Logged metrics to wandb for step {step}")
            train_tracker.reset_averages()
            logging.debug(f"Reset training metrics averages for step {step}")

        if cfg.save_checkpoint and is_saving_step:
            logging.debug(f"Starting checkpoint save for step {step}")
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            logging.debug(f"Checkpoint directory: {checkpoint_dir}")
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            logging.debug(f"Checkpoint saved successfully for step {step}")
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
                logging.debug(f"Checkpoint logged to wandb for step {step}")

            # Push checkpoint to hub if configured
            if is_hub_push_step:
                logging.debug(f"Starting hub push for step {step}")
                # Get repo_id from config or generate one
                if hasattr(cfg, "hub_repo_id") and cfg.hub_repo_id:
                    repo_id = cfg.hub_repo_id
                else:
                    repo_id = get_hub_repo_id_from_config(cfg)
                    logging.info(f"Generated hub repo_id: {repo_id}")

                # Get hub token from config if available
                hub_token = getattr(cfg, "hub_token", None) if hasattr(cfg, "hub_token") else None
                hub_private = getattr(cfg, "hub_private", False) if hasattr(cfg, "hub_private") else False

                push_checkpoint_to_hub(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    repo_id=repo_id,
                    private=hub_private,
                    token=hub_token,
                )
                logging.debug(f"Hub push completed for step {step}")

        if cfg.env and is_eval_step:
            logging.debug(f"Starting evaluation for step {step}")
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            logging.debug(f"Running evaluation with {cfg.eval.n_episodes} episodes")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            logging.debug(f"Evaluation completed for step {step}")

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.debug(
                f"Evaluation metrics calculated: reward={eval_tracker.avg_sum_reward.val:.3f}, success={eval_tracker.pc_success.val:.1f}%, time={eval_tracker.eval_s.val:.3f}s"
            )
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
                logging.debug(f"Evaluation results logged to wandb for step {step}")

    logging.debug("Training loop completed, cleaning up resources")
    if eval_env:
        eval_env.close()
        logging.debug("Evaluation environment closed")
    logging.debug(f"Training completed after {step} steps")
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
