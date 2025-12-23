#!/usr/bin/env python

# Copyright 2025 Nvidia and The HuggingFace Inc. team. All rights reserved.
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
Groot N1.6 Policy Wrapper for LeRobot Integration

Minimal integration that delegates to the Gr00tN1d6 model components. The intent is to:

- Download and load the pretrained GR00T N1.6 model via Gr00tN1d6.__init__
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via Gr00tN1d6 model.get_action
- Provide a training forward that can call the Gr00tN1d6 model forward if batch
  structure matches.

Key differences from N1.5:
- Uses AlternateVLDiT instead of standard DiT
- 32 DiT layers instead of 16
- Unfrozen top 4 VLM layers instead of 4-layer post-VLM adapter
- State-relative action chunks instead of absolute joint angles
- Uses Cosmos-Reason-2B variant backbone (Eagle-Block2A-2B-v2)

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see the training scripts.
"""

import os
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.gr00t_n1d6.configuration_gr00t_n1d6 import Gr00tN1d6Config
from lerobot.policies.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from lerobot.policies.pretrained import PreTrainedPolicy


class Gr00tN1d6Policy(PreTrainedPolicy):
    """Wrapper around Groot N1.6 model for LeRobot integration.

    This policy wraps the Gr00tN1d6 model (a Vision-Language-Action model with
    Eagle backbone and flow matching action head) to expose the standard LeRobot
    policy interface: forward(), predict_action_chunk(), select_action(), and reset().

    Key architectural features of N1.6:
    - AlternateVLDiT with 32 layers (vs 16 in N1.5)
    - Unfrozen top 4 VLM layers (instead of 4-layer post-VLM adapter)
    - CategorySpecificMLP and MultiEmbodimentActionEncoder
    - State-relative action chunks
    """

    name = "gr00t_n1d6"
    config_class = Gr00tN1d6Config

    def __init__(self, config: Gr00tN1d6Config, **kwargs):
        """Initialize Groot N1.6 policy wrapper.

        Args:
            config: Configuration for the Groot N1.6 policy
            **kwargs: Additional arguments passed to the model
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the Gr00tN1d6 model
        self._groot_model = self._create_groot_model()

        self.reset()

    def _create_groot_model(self) -> Gr00tN1d6:
        """Create and initialize the Gr00tN1d6 model.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps:
        1) Handle Flash Attention compatibility issues
        2) Initialize Gr00tN1d6 model with config

        Returns:
            Gr00tN1d6: The initialized model
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        # Create model with transformers loading kwargs
        transformers_loading_kwargs = {
            "trust_remote_code": True,
        }

        model = Gr00tN1d6(
            config=self.config,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        return model

    def reset(self):
        """Reset policy state when environment resets.

        Clears the action queue used for temporal ensembling.
        """
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        """Return trainable parameters for the optimizer.

        Returns:
            Generator of trainable parameters
        """
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Gr00tN1d6 model.forward when inputs are compatible.

        Args:
            batch: Dictionary containing:
                - VLM inputs (pixel_values, input_ids, attention_mask, etc.)
                - Action inputs (state, action, embodiment_id, action_mask, etc.)

        Returns:
            tuple[Tensor, dict]: Loss tensor and dictionary with loss info
        """
        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("vlm_") or k.startswith("pixel_"))
            and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        # Gr00tN1d6 returns a dict with 'loss' key
        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Gr00tN1d6.

        Args:
            batch: Dictionary containing:
                - VLM inputs (pixel_values, input_ids, attention_mask, etc.)
                - State inputs (state, embodiment_id, etc.)

        Returns:
            Tensor: Predicted actions of shape (B, n_action_steps, action_dim)
        """
        self.eval()

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {"state", "state_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("vlm_") or k.startswith("pixel_"))
            and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        # Trim to original action dimension (model pads to max_action_dim)
        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue.

        Uses temporal ensembling: predicts an action chunk when the queue is empty,
        then pops one action at a time from the queue.

        Args:
            batch: Dictionary containing observation data

        Returns:
            Tensor: Single action of shape (B, action_dim)
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # Transpose to (n_action_steps, B, action_dim) for queue
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """
        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[GR00T N1.6] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GR00T N1.6] Flash Attention not available: {e}")
            print("[GR00T N1.6] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GR00T N1.6] Flash Attention compatibility issue detected: {e}")
                print("[GR00T N1.6] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GR00T N1.6] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GR00T N1.6] Continuing with fallback attention mechanism")
            else:
                print(f"[GR00T N1.6] Flash Attention error: {e}")
                print("[GR00T N1.6] Continuing with fallback attention mechanism")
