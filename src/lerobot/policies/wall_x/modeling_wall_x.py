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
Wall-X: Cross-embodiment robotic control using Qwen2.5-VL with flow matching.

[Paper](https://github.com/x2-robot/wall-x)

Install wall-x extra dependencies:
```bash
pip install -e ".[wall_x]"
```

Example of finetuning a wall-x model:
```bash
lerobot-train \
--policy.type=wall_x \
--dataset.repo_id=your/dataset \
--batch_size=32 \
--steps=100000
```
"""

import math
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype

# Add wall-x repo to path if available
WALL_X_PATH = Path("/x2robot_v2/vincent/workspace/lerobot_opensource/wall-x")
if WALL_X_PATH.exists():
    sys.path.insert(0, str(WALL_X_PATH))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionHead(nn.Module):
    """
    Action prediction head with flow matching.

    Implements Beta-distributed noise scheduling and temporal embeddings
    for action sequence prediction.
    """

    def __init__(self, config: WallXConfig):
        super().__init__()

        self.config = config
        self.action_dim = sum(config.dof_config.values())
        self.propri_dim = sum(config.agent_pos_config.values())
        self.hidden_size = config.hidden_size

        # Beta distribution for noise scheduling
        noise_config = config.noise_scheduler
        self.beta_alpha = noise_config.get("beta_alpha", 1.5)
        self.beta_beta = noise_config.get("beta_beta", 1.0)
        self.s = noise_config.get("s", 0.999)

        # Sinusoidal timestep embedding
        self.time_embed = SinusoidalPosEmb(config.hidden_size)

        # Action embedding network
        # *2 for action + DOF mask concatenation
        self.w1 = nn.Linear(self.action_dim * 2, self.hidden_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)  # *2 for action + time
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # Project back to action space
        self.action_proj_back = nn.Linear(self.hidden_size, self.action_dim, bias=False)

        # Proprioception projection
        self.propri_proj = nn.Linear(self.propri_dim * 2, self.hidden_size, bias=False)

    def sample_time(self, batch_size, device, dtype):
        """Sample timesteps using Beta distribution."""
        beta_dist = Beta(
            torch.tensor(self.beta_alpha, dtype=dtype, device=device),
            torch.tensor(self.beta_beta, dtype=dtype, device=device)
        )
        sample = beta_dist.sample([batch_size])
        time = (1 - sample) / self.s
        return time

    def forward(self, action_chunk, dof_mask=None):
        """
        Process action sequences with noise injection for training.

        Args:
            action_chunk: Action sequences [batch, seq_len, action_dim]
            dof_mask: DOF mask [batch, seq_len, action_dim]

        Returns:
            tuple: (action_embeddings, flow_target)
        """
        batch_size = action_chunk.shape[0]
        device = action_chunk.device
        dtype = action_chunk.dtype

        # Add noise using flow matching
        noise = torch.randn_like(action_chunk)
        time = self.sample_time(batch_size, device, dtype)
        t = time.unsqueeze(-1).unsqueeze(-1)

        # Linear interpolation
        noisy_action = (1 - t) * noise + t * action_chunk
        flow = action_chunk - noise

        # Generate time embeddings
        time_embed = self.time_embed(time)

        # Project noisy actions
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

        noisy_action = noisy_action.to(dtype=self.w1.weight.dtype)
        action_embed = self.w1(noisy_action)

        # Combine with time embeddings
        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(dtype=self.w2.weight.dtype)

        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed, flow

    def step(self, timestep, noisy_action, dof_mask=None):
        """Single denoising step for inference."""
        if dof_mask is not None:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

        time_embed = self.time_embed(timestep)
        action_embed = self.w1(noisy_action)

        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(device=noisy_action.device, dtype=noisy_action.dtype)

        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = self.w2(concat_embed)
        embed = self.w3(self.act_fn(concat_embed))

        return embed

    def flow_loss(self, action_hidden_states, flow, dof_mask=None):
        """Compute flow matching loss."""
        action_pred = self.action_proj_back(action_hidden_states)
        loss = F.mse_loss(action_pred, flow, reduction="none")

        if dof_mask is not None:
            dof_mask = dof_mask.reshape(-1, dof_mask.shape[-1])
            loss = loss * dof_mask

        return loss

    def project_proprioception(self, proprioception, dof_mask=None):
        """Project proprioceptive data to hidden space."""
        proprioception = proprioception.to(
            device=self.propri_proj.weight.device,
            dtype=self.propri_proj.weight.dtype
        )

        if dof_mask is not None:
            proprioception = torch.cat([proprioception, dof_mask], dim=-1)

        return self.propri_proj(proprioception)


class WallXVLMWrapper(nn.Module):
    """
    Wrapper around Qwen2.5-VL model from wall-x.

    This class attempts to load the wall-x model if available,
    otherwise provides a placeholder implementation.
    """

    def __init__(self, config: WallXConfig):
        super().__init__()
        self.config = config

        # Try to import wall-x model
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.vlm_model_name,
                torch_dtype=torch.bfloat16 if config.device != "cpu" else torch.float32,
                device_map=config.device if config.device != "cpu" else None,
            )

            self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)
            self.process_vision_info = process_vision_info
            self.available = True

            # Freeze vision encoder if requested
            if config.freeze_vision_encoder:
                for param in self.model.visual.parameters():
                    param.requires_grad = False

        except ImportError:
            print("Warning: Could not import wall-x dependencies. Using placeholder.")
            self.available = False
            self.model = None
            self.processor = None

    def forward(self, **kwargs):
        """Forward pass through VLM."""
        if not self.available:
            raise RuntimeError("Wall-X VLM not available. Install required dependencies.")
        return self.model(**kwargs)


class WallXPolicy(PreTrainedPolicy):
    """
    Wall-X policy for cross-embodiment robotic control.

    Integrates Qwen2.5-VL vision-language model with action prediction
    using flow matching for continuous action spaces.
    """

    config_class = WallXConfig
    name = "wall_x"

    def __init__(self, config: WallXConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize VLM wrapper
        self.vlm = WallXVLMWrapper(config)

        # Initialize action head
        self.action_head = ActionHead(config)

        self.reset()

    def reset(self):
        """Reset action queue."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self):
        """Get parameters for optimization."""
        params = []

        if self.vlm.available:
            # Add VLM parameters
            if not self.config.train_expert_only:
                params.extend(self.vlm.model.parameters())

        # Always add action head parameters
        if self.config.train_action_head:
            params.extend(self.action_head.parameters())

        return params

    def prepare_images(self, batch):
        """Prepare images for VLM processing."""
        images = []
        present_img_keys = [key for key in self.config.image_features if key in batch]

        if len(present_img_keys) == 0:
            raise ValueError("No image features found in batch")

        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            images.append(img)

        return images

    def prepare_state(self, batch):
        """Prepare proprioceptive state."""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        # Pad to expected dimension
        if state.shape[-1] < self.config.max_state_dim:
            padding = torch.zeros(
                *state.shape[:-1],
                self.config.max_state_dim - state.shape[-1],
                device=state.device,
                dtype=state.dtype
            )
            state = torch.cat([state, padding], dim=-1)
        return state

    def prepare_action(self, batch):
        """Prepare action chunk."""
        actions = batch[ACTION]
        # Pad to expected dimension
        if actions.shape[-1] < self.config.max_action_dim:
            padding = torch.zeros(
                *actions.shape[:-1],
                self.config.max_action_dim - actions.shape[-1],
                device=actions.device,
                dtype=actions.dtype
            )
            actions = torch.cat([actions, padding], dim=-1)
        return actions

    def _create_dof_mask(self, batch_size, device, dtype):
        """Create DOF mask for action dimensions."""
        # Create mask showing which dimensions are active
        mask = torch.ones(
            batch_size,
            self.config.chunk_size,
            sum(self.config.dof_config.values()),
            device=device,
            dtype=dtype
        )
        return mask

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Training forward pass.

        Args:
            batch: Dictionary containing observations and actions

        Returns:
            tuple: (loss, loss_dict)
        """
        # Prepare inputs
        images = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype

        # Create DOF mask
        dof_mask = self._create_dof_mask(batch_size, device, dtype)

        # Process actions through action head (adds noise, gets embeddings)
        action_embeds, flow_target = self.action_head(actions, dof_mask)

        # For now, use simplified loss computation
        # In full implementation, would pass through VLM transformer
        loss_dict = {}

        # Compute flow matching loss
        # Note: In full wall-x, action_embeds would go through VLM transformer first
        flow_loss = self.action_head.flow_loss(action_embeds, flow_target, dof_mask)
        loss = flow_loss.mean()

        loss_dict["loss"] = loss.item()
        loss_dict["flow_loss"] = loss.item()

        return loss, loss_dict

    def _sample_actions_flow(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Sample actions using flow matching / diffusion.

        Args:
            batch: Dictionary containing observations

        Returns:
            Predicted actions [batch, chunk_size, action_dim]
        """
        batch_size = 1  # Typically inference is single sample
        device = self.config.device
        dtype = torch.float32

        # Initialize with noise
        noisy_action = torch.randn(
            batch_size,
            self.config.chunk_size,
            sum(self.config.dof_config.values()),
            device=device,
            dtype=dtype
        )

        # Create DOF mask
        dof_mask = self._create_dof_mask(batch_size, device, dtype)

        # ODE integration for denoising
        num_steps = self.config.num_inference_timesteps
        dt = 1.0 / num_steps

        for step_idx in range(num_steps):
            t = torch.tensor(step_idx * dt, device=device, dtype=dtype)
            timestep = t.unsqueeze(0).repeat(batch_size)

            # Single denoising step
            action_embeds = self.action_head.step(timestep, noisy_action, dof_mask)

            # Predict flow (in full implementation, would go through VLM)
            flow_pred = self.action_head.action_proj_back(action_embeds)

            # Euler integration step
            noisy_action = noisy_action + dt * flow_pred

        return noisy_action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk for evaluation."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self.config.prediction_mode == "flow":
            actions = self._sample_actions_flow(batch)
        else:
            raise NotImplementedError(f"Prediction mode {self.config.prediction_mode} not implemented")

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action for environment execution."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Use action queue
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[:self.config.n_action_steps])

        return self._queues[ACTION].popleft()
