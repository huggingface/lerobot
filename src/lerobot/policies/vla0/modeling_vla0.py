#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""
VLA-0: Vision-Language-Action Model

VLA-0 directly represents robot actions as discretized text tokens,
enabling the use of pretrained VLMs without architectural modifications.

Reference: https://github.com/NVlabs/vla0

Example usage:
```bash
lerobot-train \
    --policy.type=vla0 \
    --dataset.repo_id=your_dataset \
    --batch_size=8 \
    --steps=10000
```
"""

import logging
import re
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    LogitsProcessor,
)

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.policies.vla0.configuration_vla0 import VLA0Config
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


class NumberSpaceOnlyProcessor(LogitsProcessor):
    """
    Logits processor that constrains generation to only output digits (0-9) and spaces.
    This ensures the model outputs valid action tokens.
    """

    def __init__(self, tokenizer):
        self.allowed_tokens = set()
        # Add digit tokens
        for digit in "0123456789":
            token_ids = tokenizer.encode(digit, add_special_tokens=False)
            self.allowed_tokens.update(token_ids)
        # Add space token
        space_ids = tokenizer.encode(" ", add_special_tokens=False)
        self.allowed_tokens.update(space_ids)
        # Add EOS token to allow stopping
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            if token_id < scores.shape[-1]:
                mask[:, token_id] = 0
        return scores + mask


class VLA0Model(nn.Module):
    """
    VLA-0 Model that uses Qwen2.5-VL for action prediction.

    The model converts continuous actions to discretized text tokens and back,
    using the VLM's text generation capabilities for action prediction.
    """

    def __init__(self, config: VLA0Config):
        super().__init__()
        self.config = config

        # Load processor and model
        self._load_model_and_processor()

        # Store action bounds as tensors
        # Note: action_min/action_max may be updated by validate_features() based on dataset
        action_dim = config.action_dim
        action_min = config.action_min[:action_dim] if len(config.action_min) >= action_dim else config.action_min + [-1.0] * (action_dim - len(config.action_min))
        action_max = config.action_max[:action_dim] if len(config.action_max) >= action_dim else config.action_max + [1.0] * (action_dim - len(config.action_max))

        self.register_buffer(
            "action_min", torch.tensor(action_min, dtype=torch.float32)
        )
        self.register_buffer(
            "action_max", torch.tensor(action_max, dtype=torch.float32)
        )

        # Create number-only logits processor for constrained decoding
        self.number_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

    def _load_model_and_processor(self):
        """Load the VLM model and processor."""
        model_name = self.config.vlm_model_name

        # Quantization config for QLoRA
        quantization_config = None
        if self.config.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }

        if self.config.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        # Load model
        if self.config.load_vlm_weights:
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **model_kwargs,
            )
        else:
            # Initialize from config only (for training from scratch)
            from transformers import AutoConfig

            vlm_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.vlm = AutoModelForVision2Seq.from_config(vlm_config)

        # Apply LoRA if configured
        if self.config.use_lora and not self.config.use_qlora:
            self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
            logger.info("Applied LoRA to the model")
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install with: pip install peft")

    def action_to_text(self, actions: Tensor) -> list[str]:
        """
        Convert continuous actions to discretized text representation.

        Args:
            actions: Tensor of shape (batch_size, action_dim) or (batch_size, chunk_size, action_dim)

        Returns:
            List of action strings, each containing space-separated discretized values
        """
        # Handle different input shapes
        if actions.ndim == 3:
            # (batch_size, chunk_size, action_dim) -> flatten to (batch_size * chunk_size, action_dim)
            batch_size, chunk_size, action_dim = actions.shape
            actions = actions.reshape(-1, action_dim)
            needs_reshape = True
        else:
            needs_reshape = False
            batch_size = actions.shape[0]

        # Normalize to [0, 1]
        action_min = self.action_min.to(actions.device)
        action_max = self.action_max.to(actions.device)
        actions_normalized = (actions - action_min) / (action_max - action_min + 1e-8)
        actions_normalized = torch.clamp(actions_normalized, 0.0, 1.0)

        # Discretize to [0, num_bins_actions]
        actions_discrete = torch.round(actions_normalized * self.config.num_bins_actions).long()

        # Convert to text
        text_actions = []
        for i in range(actions_discrete.shape[0]):
            action_str = " ".join(str(int(v)) for v in actions_discrete[i].cpu().tolist())
            text_actions.append(action_str)

        return text_actions

    def text_to_action(self, text_actions: list[str], device: torch.device) -> Tensor:
        """
        Convert discretized text representation back to continuous actions.

        Args:
            text_actions: List of action strings
            device: Target device for the output tensor

        Returns:
            Tensor of shape (batch_size, action_dim)
        """
        actions = []
        for text in text_actions:
            # Extract numbers from text
            numbers = re.findall(r"\d+", text)
            if len(numbers) == 0:
                # Fallback to zeros if parsing fails
                action = torch.zeros(self.config.action_dim, device=device)
            else:
                # Take only the first action_dim numbers
                numbers = numbers[: self.config.action_dim]
                # Pad with zeros if not enough numbers
                while len(numbers) < self.config.action_dim:
                    numbers.append("0")
                action_discrete = torch.tensor(
                    [int(n) for n in numbers], dtype=torch.float32, device=device
                )

                # Convert back to continuous values
                action_normalized = action_discrete / self.config.num_bins_actions
                action_normalized = torch.clamp(action_normalized, 0.0, 1.0)

                action = action_normalized * (
                    self.action_max.to(device) - self.action_min.to(device)
                ) + self.action_min.to(device)
            actions.append(action)

        return torch.stack(actions, dim=0)

    def prepare_inputs(
        self,
        images: list[Tensor],
        task: str | list[str],
        state: Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Prepare inputs for the VLM.

        Args:
            images: List of image tensors, each of shape (batch_size, C, H, W)
            task: Task description string or list of strings
            state: Optional state tensor of shape (batch_size, state_dim)

        Returns:
            Dictionary of model inputs
        """
        batch_size = images[0].shape[0]
        device = images[0].device

        # Handle task as list or single string
        if isinstance(task, str):
            tasks = [task] * batch_size
        else:
            tasks = task

        # Build prompts
        prompts = []
        for i in range(batch_size):
            # Format task prompt
            prompt = self.config.task_prompt_template.format(task=tasks[i])

            # Add state information if available
            if state is not None:
                state_str = " ".join(f"{v:.4f}" for v in state[i].cpu().tolist())
                prompt = f"Current state: [{state_str}]\n{prompt}"

            prompts.append(prompt)

        # Process images - convert from [0, 1] to PIL format expected by processor
        pil_images = []
        for img_batch in images:
            for i in range(batch_size):
                img = img_batch[i]  # (C, H, W)
                # Convert to PIL Image
                from PIL import Image

                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                pil_img = Image.fromarray(img_np)
                pil_images.append(pil_img)

        # Group images by batch
        images_per_sample = len(images)
        batch_images = []
        for i in range(batch_size):
            sample_images = [pil_images[j * batch_size + i] for j in range(images_per_sample)]
            batch_images.append(sample_images if len(sample_images) > 1 else sample_images[0])

        # Build conversation format for Qwen2.5-VL
        conversations = []
        for i, prompt in enumerate(prompts):
            # Build image content
            if images_per_sample == 1:
                image_content = [{"type": "image"}]
            else:
                image_content = [{"type": "image"} for _ in range(images_per_sample)]

            conversation = [
                {
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": prompt}],
                }
            ]
            conversations.append(conversation)

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]

        # Process with the processor
        # IMPORTANT: Do NOT use truncation with Qwen2.5-VL as it breaks image token alignment
        # The processor inserts special image tokens that must match the number of images provided
        # Truncation can remove these tokens causing a mismatch error
        inputs = self.processor(
            text=texts,
            images=batch_images,
            padding=True,
            truncation=False,  # MUST be False for Qwen2.5-VL
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}

        return inputs

    def forward(
        self,
        images: list[Tensor],
        task: str | list[str],
        state: Tensor | None = None,
        actions: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for training.

        Args:
            images: List of image tensors
            task: Task description
            state: Optional state tensor
            actions: Ground truth actions for computing loss

        Returns:
            Loss tensor
        """
        if actions is None:
            raise ValueError("Actions must be provided for training")

        # Prepare inputs
        inputs = self.prepare_inputs(images, task, state)
        batch_size = actions.shape[0]
        device = actions.device

        # Convert actions to text
        action_texts = self.action_to_text(actions)

        # Tokenize action texts (these are our labels)
        action_tokens = self.processor.tokenizer(
            action_texts,
            padding=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Concatenate input and action tokens for training
        # The model learns to predict action tokens given the input
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Create labels: -100 for input tokens (not used in loss), action tokens for output
        labels = torch.full_like(
            torch.cat([input_ids, action_tokens], dim=1),
            fill_value=-100,
        )
        labels[:, input_ids.shape[1] :] = action_tokens

        # Concatenate for forward pass
        full_input_ids = torch.cat([input_ids, action_tokens], dim=1)
        full_attention_mask = torch.cat(
            [attention_mask, torch.ones_like(action_tokens)], dim=1
        )

        # Forward pass
        outputs = self.vlm(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            labels=labels,
        )

        return outputs.loss

    @torch.no_grad()
    def generate_actions(
        self,
        images: list[Tensor],
        task: str | list[str],
        state: Tensor | None = None,
    ) -> Tensor:
        """
        Generate actions through text generation.

        Args:
            images: List of image tensors
            task: Task description
            state: Optional state tensor

        Returns:
            Action tensor of shape (batch_size, action_dim)
        """
        # Prepare inputs
        inputs = self.prepare_inputs(images, task, state)
        batch_size = images[0].shape[0]
        device = images[0].device

        # Generate with constraints
        generation_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature if self.config.do_sample else None,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "logits_processor": [self.number_processor],
        }

        # Generate
        outputs = self.vlm.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            **generation_config,
        )

        # Decode generated tokens (excluding input)
        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        generated_texts = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Convert text to actions
        actions = self.text_to_action(generated_texts, device)

        return actions


class VLA0Policy(PreTrainedPolicy):
    """
    VLA-0 Policy wrapper for LeRobot.

    This class wraps the VLA0Model to provide the standard LeRobot policy interface.
    """

    config_class = VLA0Config
    name = "vla0"

    def __init__(self, config: VLA0Config):
        super().__init__(config)
        self.config = config
        config.validate_features()

        self.model = VLA0Model(config)
        self.reset()

    def reset(self):
        """Reset the policy state."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        """Get parameters for optimization."""
        return self.model.parameters()

    def _prepare_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        """Extract and prepare images from batch."""
        images = []
        for key, feature in self.config.image_features.items():
            if key in batch:
                img = batch[key]
                # Handle observation sequence dimension
                if img.ndim == 5:  # (batch, seq, C, H, W)
                    img = img[:, -1]  # Take latest observation
                images.append(img)
        return images

    def _get_task(self, batch: dict[str, Tensor]) -> str | list[str]:
        """Extract task description from batch."""
        # Try to get task from batch
        if "task" in batch:
            return batch["task"]
        # Default task if not provided
        return "complete the task"

    def _get_state(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Extract state from batch."""
        if OBS_STATE in batch:
            state = batch[OBS_STATE]
            # Handle observation sequence dimension
            if state.ndim == 3:  # (batch, seq, state_dim)
                state = state[:, -1]  # Take latest observation
            return state
        return None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a chunk of actions."""
        self.eval()

        images = self._prepare_images(batch)
        task = self._get_task(batch)
        state = self._get_state(batch)

        actions = self.model.generate_actions(images, task, state)

        # VLA-0 generates single-step actions, expand to chunk if needed
        if self.config.chunk_size > 1:
            actions = actions.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
        else:
            actions = actions.unsqueeze(1)

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Select a single action."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], **kwargs) -> tuple[Tensor, dict]:
        """Forward pass for training."""
        images = self._prepare_images(batch)
        task = self._get_task(batch)
        state = self._get_state(batch)
        actions = batch[ACTION]

        # Handle action sequence dimension
        if actions.ndim == 3:  # (batch, chunk_size, action_dim)
            actions = actions[:, 0]  # Take first action for single-step prediction
        elif actions.ndim == 2:
            # Validate action shape for 2D input
            batch_size, action_last_dim = actions.shape
            if action_last_dim != self.config.action_dim:
                # Check if this looks like flattened chunk_size * action_dim
                if action_last_dim % self.config.action_dim == 0:
                    inferred_chunk = action_last_dim // self.config.action_dim
                    raise ValueError(
                        f"VLA-0 action shape mismatch! Got actions.shape={tuple(actions.shape)}, "
                        f"expected (batch, {self.config.action_dim}). "
                        f"Detected flattened shape: {action_last_dim} = {inferred_chunk} * {self.config.action_dim}. "
                        f"This suggests chunk_size={inferred_chunk} was used instead of chunk_size=1. "
                        f"VLA-0 requires chunk_size=1 (single-step prediction). "
                        f"Check your policy config: current config.chunk_size={self.config.chunk_size}"
                    )
                else:
                    raise ValueError(
                        f"VLA-0 action shape mismatch! Got actions.shape={tuple(actions.shape)}, "
                        f"expected (batch, {self.config.action_dim}). "
                        f"Check your dataset action_dim and policy.action_dim settings."
                    )

        loss = self.model.forward(images, task, state, actions)

        return loss, {"loss": loss.item()}
