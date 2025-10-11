#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
Octo: An Open-Source Generalist Robot Policy

[Paper](https://arxiv.org/pdf/2405.12213)
[Jax code](https://github.com/octo-models/octo)
[Original Pytorch code](https://github.com/emb-ai/octo-pytorch)
[lilkm Pytorch code](https://github.com/s1lent4gnt/octo-pytorch)

Example of using the octo pretrained model:
```python
policy = OctoPolicy.from_pretrained("lerobot/octo_base")
```

Example of training octo:
```bash
lerobot-train \
--policy.type=octo \
--dataset.repo_id=lilkm/panda_pick_octo_resized \
--batch_size=32 \
--steps=100000
```

"""

from collections import deque
from typing import Dict, Optional
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.constants import ACTION
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.octo.configuration_octo import OctoConfig
from lerobot.policies.octo.diffusion import DiffusionActionHead
from lerobot.policies.octo.tokenizers import TextProcessor
from lerobot.policies.octo.transformer import OctoWithoutHead
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import log_model_loading_keys, populate_queues

# TODO(lilkm): Be aware of normalization the image tokenizer (normalize_images function)


class OctoPolicy(PreTrainedPolicy):
    """Wrapper class around Octo model to train and run inference within LeRobot."""

    config_class = OctoConfig
    name = "octo"

    def __init__(
        self,
        config: OctoConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                    that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.text_processor = TextProcessor(
            tokenizer_name="t5-base",
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
            },
        )

        self.model = OctoDiffusion(self.config)

        # Apply selective freezing based on config
        self._apply_selective_freezing()

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _apply_selective_freezing(self):
        """Apply selective freezing based on configuration settings."""
        if hasattr(self.model.octo_transformer, "task_tokenizers"):
            for name, tokenizer in self.model.octo_transformer.task_tokenizers.items():
                if name == "language_instruction":
                    for param in tokenizer.parameters():
                        param.requires_grad = False
                    print("✓ T5 language encoder frozen (always frozen during finetuning)")

        # If train_action_head_only is True, freeze everything except the action head
        if self.config.train_action_head_only:
            # Freeze transformer
            for param in self.model.octo_transformer.parameters():
                param.requires_grad = False
            # Keep action head trainable
            for param in self.model.head.parameters():
                param.requires_grad = True
        else:
            # Apply more fine-grained freezing
            if self.config.freeze_transformer:
                for param in self.model.octo_transformer.parameters():
                    param.requires_grad = False

            if self.config.freeze_vision_encoder:
                # Freeze vision encoder components in the transformer
                if hasattr(self.model.octo_transformer, "observation_tokenizers"):
                    for tokenizer in self.model.octo_transformer.observation_tokenizers.values():
                        for param in tokenizer.parameters():
                            param.requires_grad = False

    def get_optim_params(self) -> dict:
        """Return only parameters that require gradients for optimization."""
        return filter(lambda p: p.requires_grad, self.parameters())

    @classmethod
    def _transform_state_dict_keys(cls, state_dict: dict) -> dict:
        """
        Transform state dict keys to match expected model structure.

        This handles the conversion from the new modular octo-pytorch checkpoint format
        to the LeRobot format.
        """
        transformed_dict = {}

        for key, value in state_dict.items():
            # Skip T5 encoder weights - they'll be loaded separately
            if "t5_encoder" in key or "language_tokenizer" in key:
                continue

            # Handle the new modular structure
            new_key = key

            # Add "model." prefix if not present
            if not new_key.startswith("model."):
                new_key = f"model.{new_key}"

            # 1. Replace "action_head." with "head."
            if "action_head." in new_key:
                new_key = new_key.replace("action_head.", "head.")

            # 2. Adjust the transformer nesting to match the LeRobot model.
            # The checkpoint has `transformer.transformer` but LeRobot expects
            # `transformer.transformer.transformer`.
            if "octo_transformer.transformer.transformer." in new_key:
                new_key = new_key.replace(
                    "octo_transformer.transformer.transformer.",
                    "octo_transformer.transformer.transformer.transformer.",
                )

            transformed_dict[new_key] = value

        return transformed_dict

    @classmethod
    def _load_as_safetensor(
        cls, model: "OctoPolicy", model_file: str, map_location: str, strict: bool
    ) -> "OctoPolicy":
        """Override to apply key transformations before loading."""
        from safetensors.torch import load_file

        from lerobot.utils.utils import init_logging

        init_logging()
        # Load the state dict from file safely
        state_dict = load_file(model_file, device=map_location)

        # Apply key transformations
        transformed_state_dict = cls._transform_state_dict_keys(state_dict)

        # Load the transformed state dict
        msg = model.load_state_dict(transformed_state_dict, strict=strict)

        # Log message
        log_model_loading_keys(msg.missing_keys, msg.unexpected_keys)
        return model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the from_pretrained method to display important information."""
        print(
            "📦 Loading Octo pretrained model...\n"
            "   This model was trained on diverse robot manipulation tasks.\n"
            "   Original implementation: https://github.com/octo-models/octo. \n"
            "   lilkm implementation: https://github.com/s1lent4gnt/octo-pytorch"
        )
        return super().from_pretrained(*args, **kwargs)

    def _prepare_batch(
        self, batch: dict[str, Tensor], raw_tasks: Sequence[str] | None = None
    ) -> dict[str, Tensor]:
        """
        Prepare batch for model input.
        Transforms a batch from the LeRobotDataset format to the format expected by the OctoModel.
        """
        # TODO(lilkm): check normalization
        # batch = self.normalize_inputs(batch)
        # Get device from any available tensor in the batch
        device = next(iter(batch.values())).device

        image_primary = batch["observation.images.front"].to(device)
        image_wrist = batch["observation.images.wrist"].to(device)
        proprio = batch["observation.state"].to(device)

        batch_size = image_primary.shape[0]

        if ACTION in batch:
            raw_actions = batch[ACTION].to(device)
        else:
            raw_actions = None

        if raw_tasks is None:
            raw_tasks = [""] * batch_size
        window_size = 1
        action_horizon = self.config.n_action_steps
        action_dim = self.config.action_dim

        image_primary = image_primary.permute(0, 2, 3, 1).unsqueeze(1)
        image_wrist = image_wrist.permute(0, 2, 3, 1).unsqueeze(1)

        proprio = proprio.unsqueeze(1)  # (B, W, D)

        # For window_size=1, timestep will be 0 for all samples
        timestep = torch.zeros((batch_size, window_size), dtype=torch.int32, device=device)

        # Create timestep_pad_mask - all True since we have real data (no padding)
        timestep_pad_mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=device)

        task_completed = torch.zeros(
            (batch_size, window_size, action_horizon), dtype=torch.bool, device=device
        )

        # Create pad_mask_dict for observations
        obs_pad_mask_dict = {
            "image_primary": torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
            "image_wrist": torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
            "proprio": torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
            "timestep": torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
        }

        observations = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "proprio": proprio,
            "timestep": timestep,
            "timestep_pad_mask": timestep_pad_mask,
            "task_completed": task_completed,
            "pad_mask_dict": obs_pad_mask_dict,
        }

        language_instruction = self.text_processor.encode(raw_tasks)
        language_instruction = {k: v.to(device) for k, v in language_instruction.items()}

        tasks = {
            "language_instruction": language_instruction,
            "pad_mask_dict": {
                "language_instruction": torch.ones(batch_size, dtype=torch.bool, device=device)
            },
        }

        # Handle actions only if they're present (during training)
        if raw_actions is not None:
            x_y_z = raw_actions[..., :3]  # x, y, z
            gripper = raw_actions[..., 3:4]  # gripper
            rx_ry_rz = torch.zeros_like(x_y_z, dtype=raw_actions.dtype, device=device)  # rx, ry, rz as zeros
            raw_actions = torch.cat([x_y_z, rx_ry_rz, gripper], dim=-1)  # x, y, z, rx, ry, rz, gripper

            # The dataloader provides a sequence of actions. We select the first `action_horizon`
            # actions to be the target for the diffusion model.
            # raw_actions has shape [batch_size, num_timestamps, action_dim]
            # We need shape [batch_size, window_size, action_horizon, action_dim]

            # Select the first `action_horizon` actions from the sequence.
            actions = raw_actions[:, :action_horizon]

            # Add the window_size dimension.
            actions = actions.unsqueeze(1)

            # Pad if the sequence is shorter than action_horizon.
            if actions.shape[2] < action_horizon:
                padding_shape = (
                    batch_size,
                    window_size,
                    action_horizon - actions.shape[2],
                    action_dim,
                )
                padding = torch.zeros(padding_shape, dtype=actions.dtype, device=actions.device)
                actions = torch.cat([actions, padding], dim=2)

            action_pad_mask = torch.ones_like(actions, dtype=torch.bool, device=device)
            if action_dim >= 7:
                # Mask out rotation dimensions (indices 3, 4, 5)
                action_pad_mask[:, :, :, 3:6] = False
        else:
            # During inference, we don't have actions
            actions = None
            action_pad_mask = None

        return observations, tasks, actions, action_pad_mask, timestep_pad_mask
        # return batch

    def create_tasks(
        self,
        goals: dict[str, torch.Tensor] | None = None,
        texts: Sequence[str] | None = None,
        device: torch.device | None = None,
    ):
        """Creates tasks dict from goals and texts."""
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}

        # Determine device
        if device is None:
            if goals is not None:
                device = next(iter(goals.values())).device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if goals is not None:
            tasks.update(goals)
            tasks["pad_mask_dict"].update(
                {k: torch.ones(v.shape[:1], dtype=torch.bool, device=device) for k, v in goals.items()}
            )
        else:
            batch_size = len(texts)
            # Create dummy goals if none are provided
            tasks.update(
                {"image_primary": torch.zeros((batch_size, 256, 256, 3), dtype=torch.uint8, device=device)}
            )
            tasks["pad_mask_dict"].update(
                {
                    k: torch.zeros(batch_size, dtype=torch.bool, device=device)
                    for k in tasks.keys()
                    if k != "pad_mask_dict"
                }
            )

        if texts is not None:
            assert self.text_processor is not None
            encoded = self.text_processor.encode(texts)
            # Move to the correct device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            tasks["language_instruction"] = encoded
            tasks["pad_mask_dict"]["language_instruction"] = torch.ones(
                len(texts), dtype=torch.bool, device=device
            )
        else:
            batch_size = next(iter(goals.values())).shape[0]
            dummy_texts = [""] * batch_size
            encoded = self.text_processor.encode(dummy_texts)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            tasks["language_instruction"] = encoded
            tasks["pad_mask_dict"]["language_instruction"] = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )

        return tasks

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Get a chunk of actions from the model."""
        # In this method, `batch` is the prepared_batch from `_prepare_batch`
        observations = batch[0]
        tasks = batch[1]
        timestep_pad_mask = batch[4]

        # Get actions from model
        actions = self.model(observations, tasks, timestep_pad_mask)

        # Unnormalize actions
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], tasks: Sequence[str] | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # First populate queues with the original batch
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        prepared_batch = self._prepare_batch(batch, raw_tasks=tasks)

        actions = self._get_action_chunk(prepared_batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # First, populate queues with the original, simple batch
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Then, prepare the complex batch for the model
        prepared_batch = self._prepare_batch(batch)

        # Action queue logic for n_action_steps > 1
        if len(self._queues[ACTION]) == 0:
            # Use the prepared_batch to get actions from the model
            actions = self._get_action_chunk(prepared_batch)

            # actions shape is [batch_size, n_action_steps, action_dim]
            # We need to queue up actions for each sample in the batch
            batch_size = actions.shape[0]

            # For now, just return the first action from the chunk
            # In a real implementation, you'd want to handle the queue per sample
            return actions[:, 0, :]

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        # batch = self.normalize_inputs(batch)
        # batch = self.normalize_targets(batch)

        # Prepare batch
        observations, tasks, actions, action_pad_mask, timestep_pad_mask = self._prepare_batch(batch)

        # Get transformer outputs for training
        transformer_outputs = self.model.octo_transformer(observations, tasks, timestep_pad_mask)

        # Compute diffusion loss
        loss, loss_dict = self.model.head.loss(
            transformer_outputs, actions, timestep_pad_mask, action_pad_mask
        )

        return loss, loss_dict


class OctoDiffusion(nn.Module):
    """
    Octo VLA with diffusion action head

    [Paper](https://arxiv.org/pdf/2405.12213)
    """

    def __init__(self, config: OctoConfig):
        super().__init__()
        self.config = config

        self.octo_transformer = OctoWithoutHead(
            model_name=self.config.model_name,
            repeat_task_tokens=self.config.model_name,
            freeze_language_encoder=self.config.freeze_language_encoder,
            token_embedding_size=self.config.token_embedding_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            mlp_dim=self.config.mlp_dim,
            chunk_size=self.config.chunk_size,
            dropout_rate=self.config.dropout_rate,
            attention_dropout_rate=self.config.attention_dropout_rate,
        )
        self.head = DiffusionActionHead(
            readout_key="readout_action",
            use_map=False,
            input_dim=self.config.token_embedding_size,
            action_dim=self.config.action_dim,
            action_horizon=self.config.n_action_steps,
            diffusion_steps=self.config.diffusion_steps,
            n_diffusion_samples=self.config.n_diffusion_samples,
            max_action=self.config.max_action,
            loss_type=self.config.loss_type,
            time_dim=self.config.time_dim,
            num_blocks=self.config.num_blocks,
            hidden_dim=self.config.hidden_dim,
            use_layer_norm=self.config.use_layer_norm,
            dropout_rate=self.config.dropout_rate,
        )

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        tasks: dict[str, torch.Tensor],
        timestep_pad_mask: torch.Tensor,
        embodiment_action_dim: int | None = None,
    ) -> torch.Tensor:
        transformer_outputs = self.octo_transformer(observations, tasks, timestep_pad_mask)
        actions = self.head.predict_action(
            transformer_outputs=transformer_outputs, embodiment_action_dim=embodiment_action_dim
        )

        return actions
