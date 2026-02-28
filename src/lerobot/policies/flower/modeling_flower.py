#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import functools
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from typing import Any, Dict, Optional, Tuple, Collection, List
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from timm.layers.mlp import Mlp
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.policies.flower.utils import generate_policy_prompt, ActionIndex
from lerobot.policies.flower.transformers_flower import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
    FlowBlock, 
    stateless_norm
)


dtype_map = {
    'bf16': torch.bfloat16,
    'no': torch.float32
}

class FlowerPolicy(PreTrainedPolicy):
    """
    Flower Policy as per "FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies"
    (paper: https://arxiv.org/pdf/2509.04996, code: https://intuitive-robots.github.io/flower_vla/).
    """

    config_class = FlowerConfig
    name = "flower" 

    def __init__(
        self,
        config: FlowerConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.flower = FlowerModel(config)
        self.reset()

        self.resize = torchvision.transforms.Resize((config.resize_h, config.resize_w))

    def get_optim_params(self) -> dict:
        """Get parameter groups for optimizer"""
        if self.config.training_stage == "pretrain":
            dit_optim_groups, vlm_optim_params = self.flower._configure_optimizers(self.config)
            return {"dit": dit_optim_groups, "vlm": vlm_optim_params}
        else:
            no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
            decay_group = []
            no_decay_group = []

            # Collect all parameters, excluding VLM if frozen
            for name, param in self.flower.named_parameters():
                if param.requires_grad:
                    if any(nd in name.lower() for nd in no_decay):
                        no_decay_group.append(param)
                    else:
                        decay_group.append(param)

            return [
                {"params": decay_group, "weight_decay": self.config.optimizer_weight_decay},
                {"params": no_decay_group, "weight_decay": 0.0}
            ]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        constructed_prompts, batch_action_index = self.flower.construct_prompts(batch['task'])
        text_inputs = self.flower._get_text_inputs(constructed_prompts)

        batch['text_input_ids'] = text_inputs['input_ids']
        batch['text_attention_mask'] = text_inputs.data["attention_mask"]
        batch['action_index'] = batch_action_index

        actions = self.flower.generate_actions(batch, noise=noise)

        return actions
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original     
            batch = self.preprocess_batch(batch)
        batch = self.process_padding(batch, self.flower.max_action_dim)
        
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def preprocess_batch(self, batch):
        if self.config.image_features:
            images = []
            for key in self.config.image_features:
                if key in batch:
                    image = batch[key] if len(batch[key].shape)==5 else batch[key].unsqueeze(1)
                    bs, obs, c, h, w = image.shape
                    image = image.view(bs*obs, c, h, w)
                    image = self.resize(image)
                    image = image.view(bs, obs, c, self.config.resize_h, self.config.resize_w)
                    images.append(image)
            batch[OBS_IMAGES] = torch.stack(images, dim=-4)  # (bs, obs, cam, c, h, w)

        return batch
    
    def process_padding(self, batch, max_action_dim):
        if ACTION in batch:
            if len(batch[ACTION].shape) == 2:
                batch[ACTION] = batch[ACTION].unsqueeze(1)
            bs, horizon, action_dim = batch[ACTION].shape
            if action_dim > max_action_dim:
                raise ValueError(f"The action dimension {action_dim} exceeds the maximum allowed dimension {max_action_dim}")
            action_pad = max_action_dim - action_dim
            batch[ACTION] = F.pad(
                batch[ACTION], 
                (0, action_pad) + (0, 0) * (batch[ACTION].ndim - 1), 
                mode='constant', 
                value=0.0
                )
            batch[f'{ACTION}_mask'] = torch.ones(
                bs, max_action_dim,
                device=batch[ACTION].device, dtype=torch.bool
                )
            if action_pad > 0:
                batch[f'{ACTION}_mask'][..., -action_pad:] = False
        if len(batch[OBS_STATE].shape) == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
        bs, horizon, state_dim = batch[OBS_STATE].shape
        if state_dim > max_action_dim:
            raise ValueError(f"The state dimension {state_dim} exceeds the maximum allowed dimension {max_action_dim}")
        state_pad = max_action_dim - state_dim
        batch[OBS_STATE] = F.pad(
            batch[OBS_STATE], 
            (0, state_pad) + (0, 0) * (batch[OBS_STATE].ndim - 1), 
            mode='constant', 
            value=0.0
            )
        batch[f'{OBS_STATE}_mask'] = torch.ones(
            bs, max_action_dim,
            device=batch[OBS_STATE].device, dtype=torch.bool
            )
        if state_pad>0:
            batch[f'{OBS_STATE}_mask'][..., -state_pad:] = False

        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch = self.preprocess_batch(batch)
        batch = self.process_padding(batch, self.flower.max_action_dim)
        loss = self.flower.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class FlowerModel(nn.Module):
    def __init__(self, config: FlowerConfig):
        super().__init__()
        # super().__init__(*args, **kwargs)
        self.config = config
        self.num_inference_steps = config.num_inference_steps
        self.device = config.device
        self.mixed_precision = config.mixed_precision
        
        # Setup VLM and core components
        self._setup_vlm(
            config.vlm_path, 
            config.freeze_vision_tower, 
            config.freeze_florence,
            config.freeze_embeddings_only
            )
        self.config.hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = self.config.hidden_dim

        # Setup DiT components
        self.action_space_index = ActionIndex()
        self._setup_dit_components()
        
        # Load pretrained weights if specified
        if config.load_pretrained and config.pretrained_model_path is not None:
            self._load_pretrained_weights(config.pretrained_model_path)
        
        self.max_action_dim = self.action_space_index.get_max_action_dim()

    # ========= init  ============
    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool, freeze_embeddings_only: bool):
        """Initialize and configure the Florence-2 VLM"""
        print(f"Loading Florence-2 from {vlm_path}")
        
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
        
        # Handle parameter freezing
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif freeze_embeddings_only:
            embedding_layer = self.vlm.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = False
            if hasattr(self.vlm.language_model, 'shared'):
                for param in self.vlm.language_model.shared.parameters():
                    param.requires_grad = False
        if not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # Create prompt embedding
        self.prompt_embeds = self._create_prompt_embed("<Flow>")
        
        # Remove unnecessary components
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(self.config.token_dropout)

    def _setup_dit_components(self):
        """Setup DiT model components"""

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.config.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
            
        self.adaln = nn.ModuleDict() if self.config.action_type_adaln else None

        # Core components
        self.cond_linear = nn.Linear(self.config.hidden_dim, self.config.dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.config.dit_dim)
        self.cond_norm = RmsNorm(self.config.hidden_dim)
        self.frequency_embedder = FreqEmbedder(self.config.dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            self.config.dit_dim, 
            max_actions=len(self.action_space_index.action_spaces)
            )

        # Positional encoding if not using ROPE/NOPE
        if not self.config.use_rope and not self.config.use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, self.config.horizon, self.config.dit_dim) * 0.1)

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock(
                self.config.dit_dim, 
                self.config.n_heads,
                attn_pdrop=self.config.attn_pdrop,
                resid_pdrop=self.config.resid_pdrop,
                mlp_pdrop=self.config.mlp_pdrop,
                use_cross_attn=self.config.use_cross_attn,
                use_rope=self.config.use_rope,
                query_seq_len=self.config.query_seq_len,
                rope_theta=self.config.rope_theta,

            ) for _ in range(self.config.n_layers)
        ])

        # Create components per action space
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            
            # Add encoder/decoder for this action
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim, 
                hidden_features=self.config.dit_dim, 
                out_features=self.config.dit_dim, 
                bias=True
                ).to(self.device) 
            self.action_decoders[action_name] = nn.Linear(self.config.dit_dim, input_dim).to(self.device) 
                
            if self.config.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    self.config.dit_dim, 
                    global_conddim=self.config.dit_dim, 
                    use_cross_attn=self.config.use_cross_attn
                    ).to(self.device) 

            if self.config.use_proprio:
                # Add proprio encoder if needed for bimanual nav variant otherwise use zero encoder
                if self.action_space_index.get_num_arms(action_idx) == 2:
                    self.proprio_encoders[action_name] = Mlp(
                    input_dim, 
                    self.config.dit_dim, 
                    out_features=self.config.dit_dim, 
                    drop=0.2
                    ).to(self.device)
                else:
                    self.proprio_encoders[action_name] = ZeroEncoder(
                        self.config.dit_dim,
                        device=self.device
                    ).to(self.device)
            
    def _load_pretrained_weights(self, pretrained_model_path: str, mean_resizing: bool = False):
        """Loads pretrained weights, handling key mismatches (e.g., different prefixes)."""


        print(f"Loading pretrained weights from {pretrained_model_path}...")
        # Determine file type and load accordingly
        if pretrained_model_path.suffix == ".safetensors":
            # Load safetensors file
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path, device=str(self.device))
            checkpoint = {"state_dict": state_dict}  # Create checkpoint-like structure for compatibility
            print("Loaded safetensors file")
        else:
            # Load PyTorch checkpoint (.pt, .pth, .ckpt)
            checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
            # Extract the state dict (handle PyTorch Lightning or plain models)
            state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract the state dict (handle PyTorch Lightning or plain models)
        state_dict = checkpoint.get("state_dict", checkpoint)

        if ("callbacks" in checkpoint and 
                "EMA" in checkpoint["callbacks"] and 
                "ema_weights" in checkpoint["callbacks"]["EMA"]):
                
                print("Found EMA weights in checkpoint, attempting to load them...")
                ema_weights_list = checkpoint['callbacks']['EMA']['ema_weights']
                
                # Get the original state dict to use as a reference for parameter names and shapes
                original_state_dict = checkpoint.get("state_dict", checkpoint)
                
                # Create a new state dict by matching EMA weights with original parameter names
                state_dict = {}
                ema_idx = 0
                
                for param_name, original_param in original_state_dict.items():
                    if ema_idx < len(ema_weights_list):
                        ema_weight = ema_weights_list[ema_idx]
                        
                        # Check if shapes match
                        if ema_weight.shape == original_param.shape:
                            state_dict[param_name] = ema_weight
                            ema_idx += 1
                        else:
                            # Shape mismatch - try to find the correct EMA weight by shape
                            found_match = False
                            for temp_idx in range(ema_idx, min(ema_idx + 20, len(ema_weights_list))):
                                if ema_weights_list[temp_idx].shape == original_param.shape:
                                    state_dict[param_name] = ema_weights_list[temp_idx]
                                    # Swap to maintain order
                                    ema_weights_list[temp_idx], ema_weights_list[ema_idx] = ema_weights_list[ema_idx], ema_weights_list[temp_idx]
                                    ema_idx += 1
                                    found_match = True
                                    break
                            
                            if not found_match:
                                # If no match found, use original parameter
                                print(f"Warning: No matching EMA weight found for {param_name}, using original")
                                state_dict[param_name] = original_param
                    else:
                        # No more EMA weights available, use original
                        print(f"Warning: Ran out of EMA weights at {param_name}, using original")
                        state_dict[param_name] = original_param
                
                print(f"Successfully matched {ema_idx} EMA weights out of {len(ema_weights_list)} total")

        # Fix key mismatches: remove 'agent.' prefix if it exists
        new_state_dict = {}
        # Handle language encoder/model naming mismatch
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")  # Remove 'agent.' if it exists
            new_key = new_key.replace("flower.", "")
            # Handle language encoder/model naming mismatch
            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")
            # Handle MLP naming mismatch
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state_dict[new_key] = value

        current_state_dict = self.state_dict()
        filtered_state_dict = {}
        for key, value in new_state_dict.items():
            if key in current_state_dict:
                if current_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
        
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        # Log mismatches for debugging
        print(f"Pretrained weights loaded with the following issues:")
        print(f"⚠️ Missing keys: {len(missing_keys)}")
        if missing_keys:
            print(f"  ⚠️ Missing keys (not found in checkpoint, using default init): {len(missing_keys)}")
            print(f"    {missing_keys[:30]} ...")  # Show first 30 for brevity
        print(f"🫥 Unexpected keys: {len(unexpected_keys)}")
        if unexpected_keys:
            print(f"  🫥 Unexpected keys (ignored): {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:30]} ...")  # Show first 30 for brevity
        if not missing_keys and not unexpected_keys:
            print("  ✅ All keys matched successfully!") 


        return missing_keys, unexpected_keys

    def _configure_optimizers(self, optimizer_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []
        vlm_params = set(p for p in self.vlm.parameters())
        for name, param in self.named_parameters():
            if param.requires_grad and param.is_leaf and param not in vlm_params:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)
        dit_optim_groups = [
            {"params": decay_group, "weight_decay": optimizer_config.weight_decay["transformer_weight_decay"]},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]
        vlm_optim_params = [p for p in self.vlm.parameters() if p.requires_grad]
        return dit_optim_groups, vlm_optim_params
    
    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        # Sample prior.
        z = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.max_action_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
         # Integration
        dt = 1.0 / self.num_inference_steps
        dt_tensor = torch.tensor([dt] * batch_size, device=device).view([batch_size] + [1]*(z.dim()-1))

        for i in range(self.num_inference_steps, 0, -1):
            t_val = i / self.num_inference_steps
            t_tensor = torch.full((batch_size,), t_val, device=device)

            # Predict velocity field
            vc, _ = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc
        
        sample = z.clamp(-1, 1)
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        cond = self.encode_observations(batch)

        # run sampling
        actions = self.conditional_sample(batch_size, cond=cond, noise=noise)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        # actions = actions[:, start:end]
        adim = self.action_space_index.get_action_dim(batch['action_index'][0])
        actions = actions[:, :, :adim]
        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps
        
        # cond:
        cond = self.encode_observations(batch)
        # Forward rectified flow.
        trajectory = batch[ACTION]
        b = trajectory.shape[0]
        device = trajectory.device
        default_dtype = trajectory.dtype

        # Sample time based on sampling strategy
        if self.config.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.config.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.config.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        # Interpolate between actions and noise
        texp = t.view([b] + [1] * (trajectory.dim() - 1))

        z1 = torch.zeros_like(trajectory)
        action_type = cond['action_type']
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn(
                    (mask.sum(), trajectory.size(1), adim), 
                    dtype=default_dtype, 
                    device=device
                    )
                z1[mask, :, :adim] = noise_slice
        zt = (1 - texp) * trajectory + texp * z1

        # Forward pass
        vtheta, _ = self.dit_forward(zt, t, cond)
        
        # valid_mask
        valid_mask = torch.zeros_like(trajectory, dtype=torch.bool).to(device)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_mask[mask, :, :adim] = True
        
        # Compute loss on valid dimensions only
        diff = (z1 - trajectory) - vtheta
        valid_diff = torch.where(
            valid_mask, 
            diff, 
            torch.tensor(0.0, device=diff.device)
            )
        loss = (valid_diff ** 2)  # l2
        # loss = torch.abs(valid_diff)  # l1
        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            in_episode_bound = in_episode_bound.unsqueeze(-1).expand(*in_episode_bound.shape, loss.size(-1))
            valid_mask = valid_mask & in_episode_bound
            loss = loss * in_episode_bound
        loss_mean = loss.sum() / (valid_mask.sum().float() + 1e-8)
        return loss_mean
    
    def encode_observations(self, batch):
        """Encode observations using Florence-2"""
        
        device = get_device_from_parameters(self)  # device = self.device
        default_dtype = get_dtype_from_parameters(self)  # next(self.parameters()).dtype
        
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        # Extract visual features
        images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
        img_features_list = torch.cat([
            self.vlm._encode_image(images) for images in images_per_camera
            ])
        img_features = einops.rearrange(
            img_features_list, "(n b s) c dim -> b (s n c) dim", b=batch_size, s=n_obs_steps
            )
        
        # Get text embeddings
        # Get text embeddings once to reuse
        batch_action_index = batch['action_index'].to(device)
        text_embeds = self._get_text_embeddings_new(batch['text_input_ids'], device)
        txt_attention_mask = batch['text_attention_mask'].to(device)
        # Add task prompt and aggregation tokens
        task_prompt = self.prompt_embeds.expand(batch_size, -1, -1)
        
        # Merge sequence
        merged_embeds = torch.cat([
            task_prompt.to(img_features.device),
            img_features,
            text_embeds.to(img_features.device)
        ], dim=1)

        # Create attention mask
        # attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)
        prompt_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device) # 
        txt_attention_mask = txt_attention_mask.to(device).squeeze(1)  # get attention mask from txt
        vis_attention_mask = torch.ones(img_features.shape[:2], device=device)  # define attention mask for image
        attention_mask = torch.cat([prompt_mask, vis_attention_mask, txt_attention_mask], dim=1)
        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Apply dropout 
        features = self.vlm_token_dropout(features)

        # Prepare frequency and action space embeddings
        frequency_embeds = self.frequency_embedder(
            torch.ones(batch_size, 1, 1).to(device) * self.config.data_frequency
        )
        
        # Get proprioception if enabled
        proprio = None
        if self.config.use_proprio:
            proprio = batch['observation.state'].to(device).to(default_dtype)

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': self.action_space_embedder(batch_action_index.to(device)),
            'action_type': batch_action_index.to(device),
            'proprio': proprio,
            'attention_mask': attention_mask,
        }
    
    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """
        Forward pass through the DiT blocks.
        """
        device = get_device_from_parameters(self)  # device = self.device
        default_dtype = get_dtype_from_parameters(self)  # next(self.parameters()).dtype

        # Get conditioning information
        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(device)
        
        # Handle proprioception
        if self.config.use_proprio and cond_dict['proprio'] is not None:
            proprio = cond_dict['proprio'].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)

        # Encode actions
        z, valid_dims = self.encode_actions(z, action_type)
        
        # Add positional encoding if not using ROPE/NOPE
        if not self.config.use_rope and not self.config.use_nope:
            z = z + self.positional_encoding

        # Process embeddings
        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1)
        
        cond = self.cond_linear(self.cond_norm(cond))
        
        # Set up conditioning
        if self.config.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.config.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb
        
        # Setup context
        cx = z
        context = cond if self.config.use_cross_attn else None
        
        # Get adaln signals
        if not self.config.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)

        # Process through DiT blocks
        for layer in self.dit:
            cx = layer(
                cx, 
                global_cond, 
                context=context, 
                is_causal=True, 
                global_adaln=global_adaln
            )
            
        # Decode and return
        return self.decode_actions(cx, action_type, valid_dims), cx

    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens"""
        # Add special token if not in vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        
        # Get token ID and create embedding
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)), 
            requires_grad=False
        )
    
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def construct_prompts(self, tasks):
        language_instruction = tasks
        text_prompts = []
        batch_action_index = []
        for idx, instruction in enumerate(language_instruction):
            robot_type = 'panda'
            action_index = self.action_space_index.robot_mapping[robot_type]
            batch_action_index.append(action_index)
            instruction = generate_policy_prompt(
                instruction,
                robot_name=robot_type,
                num_arms=self.action_space_index.get_num_arms(action_index),
                action_space=f"{self.action_space_index.get_action_dim(action_index)}D continuous",
                prompt_style="minimal",
                include_meta=True
                )
            text_prompts.append(instruction)
        batch_action_index = torch.tensor(batch_action_index)
        return text_prompts, batch_action_index
    
    def _get_text_inputs(self, constructed_prompts):
        text_inputs = self.tokenizer(
                constructed_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            )
        return text_inputs
    
    def _get_text_embeddings_new(self, text_inputs, device):
        """Get text embeddings to use with VLM"""
        text_inputs = text_inputs.to(device)
        text_embeds = self.vlm.get_input_embeddings()(text_inputs)
        return text_embeds
    
    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encode proprioception based on action type.
        """
        batch_size, _, _ = output_shape
        device = get_device_from_parameters(self)
        default_dtype = dtype_map[self.mixed_precision] #next(self.parameters()).dtype
        
        if not self.config.use_proprio:
            return torch.zeros(batch_size, self.config.dit_dim, device=device)

        proprio = proprio.mean(dim=1).to(device)
        encoded_proprio = torch.zeros(batch_size, self.config.dit_dim, device=device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                encoded_proprio[mask] = self.proprio_encoders[action_name](proprio[mask, :adim]).squeeze(1).to(default_dtype)
        return encoded_proprio
    
    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> torch.Tensor:
        """Encode actions using action-specific encoders."""
        batch_size, _, _ = z.shape
        device = get_device_from_parameters(self)
        default_dtype = dtype_map[self.mixed_precision] # next(self.parameters()).dtype
        
        encoded = torch.zeros(batch_size, z.shape[1], self.config.dit_dim, device=device, dtype=default_dtype)
        valid_dims = torch.zeros_like(z, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim])
        
        return encoded, valid_dims

    def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        """Decode actions using action-specific decoders."""
        device = get_device_from_parameters(self)
        default_dtype = dtype_map[self.mixed_precision] #next(self.parameters()).dtype
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                decoded = self.action_decoders[action_name](z)

        batch_size = z.shape[0]
        decoded = torch.zeros(batch_size, z.shape[1], self.max_action_dim, device=device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                pred = self.action_decoders[action_name](z[mask])
                decoded[mask, :, :adim] = pred[..., :adim] * valid_dims[mask, :, :adim]

        return decoded

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate action-specific AdaLN signals.
        """
        device = get_device_from_parameters(self)  # global_cond.device
        default_dtype = dtype_map[self.mixed_precision] # next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.config.use_cross_attn else 6
        
        mod_signals = [
            torch.zeros(batch_size, self.config.dit_dim, device=device, dtype=default_dtype) 
            for _ in range(num_chunks)
        ]

        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond[mask])
                for i, signal in enumerate(action_mod):
                    mod_signals[i][mask] = signal
        return mod_signals

