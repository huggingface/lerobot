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

# ruff: noqa: N806

import os
import warnings
from collections import deque

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ):
        """Load a finetuned Gr00tN1d6Policy from a pretrained checkpoint.

        Note: The processor is NOT loaded here. It should be set via set_processor()
        after the PolicyProcessorPipeline loads it from the safetensors checkpoint.
        This aligns with LeRobot's pattern where stats are saved/loaded via the
        processor pipeline's state_dict mechanism.

        Args:
            pretrained_name_or_path: Path or HuggingFace repo ID of the finetuned model
            config: Optional config override
            **kwargs: Additional arguments passed to parent's from_pretrained

        Returns:
            Gr00tN1d6Policy: The loaded policy
        """
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters."""
        return next(self.parameters()).dtype

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

        # Detect checkpoint's expected state dimension from loaded weights
        # The checkpoint may have been trained with a different max_state_dim
        # than our config, so we need to pad states to match the checkpoint
        self._checkpoint_max_state_dim = self._detect_checkpoint_state_dim()

        # Detect checkpoint's expected action dimension from loaded weights
        # The checkpoint may have been trained with a different max_action_dim
        # than our config, so we need to pad actions to match the checkpoint
        self._checkpoint_max_action_dim = self._detect_checkpoint_action_dim()
        self._checkpoint_action_horizon = self._detect_checkpoint_action_horizon()

        # Processor reference for decoding full action chunks with per-timestep stats
        self._gr00t_processor = None

        self.reset()

    def _create_groot_model(self) -> Gr00tN1d6:
        """Create and initialize the Gr00tN1d6 model.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps:
        1) Handle Flash Attention compatibility issues
        2) Load pretrained Gr00tN1d6 model via from_pretrained (like N1.5)

        Returns:
            Gr00tN1d6: The initialized model
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        # Use from_pretrained like N1.5 does
        model = Gr00tN1d6.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
            tune_vlln=self.config.tune_vlln,
            tune_top_llm_layers=self.config.tune_top_llm_layers,
            transformers_loading_kwargs={"trust_remote_code": True},
        )

        # IMPORTANT: Do NOT override the model's action_horizon!
        # The pretrained model was trained with a specific action_horizon (e.g., 50 for N1.6)
        # and the diffusion process depends on this. Overriding it changes the diffusion
        # dynamics and produces different (incorrect) outputs.
        # The processor's decode_action will trim to the desired number of timesteps.

        return model

    def _detect_checkpoint_state_dim(self) -> int:
        """Detect the checkpoint's expected state dimension from loaded weights.

        The pretrained checkpoint may have been trained with a different
        max_state_dim than our config. We detect this from the state encoder's
        weight matrix shape.

        Returns:
            int: The checkpoint's expected state dimension
        """
        state_encoder = self._groot_model.action_head.state_encoder
        # The first layer's weight matrix shape is [num_categories, input_dim, hidden_dim]
        # We need input_dim
        if hasattr(state_encoder, "layer1") and hasattr(state_encoder.layer1, "W"):
            checkpoint_state_dim = int(state_encoder.layer1.W.shape[1])
            if checkpoint_state_dim != self.config.max_state_dim:
                warnings.warn(
                    f"Checkpoint expects max_state_dim={checkpoint_state_dim}, "
                    f"but config has max_state_dim={self.config.max_state_dim}. "
                    f"States will be padded/truncated to {checkpoint_state_dim}.",
                    stacklevel=2,
                )
            return checkpoint_state_dim
            # Fallback to config value if detection fails
            return self.config.max_state_dim

    def _detect_checkpoint_action_dim(self) -> int:
        """Detect the checkpoint's expected action dimension from loaded weights.

        The pretrained checkpoint may have been trained with a different
        max_action_dim than our config. We detect this from the action encoder's
        weight matrix shape.

        Returns:
            int: The checkpoint's expected action dimension
        """
        action_encoder = self._groot_model.action_head.action_encoder
        # The W1 layer's weight matrix shape is [num_categories, input_dim, hidden_dim]
        # We need input_dim, which is the action dimension
        if hasattr(action_encoder, "W1") and hasattr(action_encoder.W1, "W"):
            checkpoint_action_dim = int(action_encoder.W1.W.shape[1])
            if checkpoint_action_dim != self.config.max_action_dim:
                warnings.warn(
                    f"Checkpoint expects max_action_dim={checkpoint_action_dim}, "
                    f"but config has max_action_dim={self.config.max_action_dim}. "
                    f"Actions will be padded/truncated to {checkpoint_action_dim}.",
                    stacklevel=2,
                )
            return checkpoint_action_dim
        # Fallback to config value if detection fails
        return self.config.max_action_dim

    def _detect_checkpoint_action_horizon(self) -> int:
        """Detect checkpoint's expected action horizon from model config.

        The pretrained model can use a diffusion horizon that differs from
        LeRobot's action_horizon. We keep action_horizon for decoding/execution
        and pad training actions to the checkpoint horizon expected by the model.

        Returns:
            int: The checkpoint's expected action horizon
        """
        model_cfg = getattr(self._groot_model, "config", None)
        checkpoint_horizon = getattr(model_cfg, "action_horizon", None)
        if checkpoint_horizon is None:
            return self.config.action_horizon

        checkpoint_horizon = int(checkpoint_horizon)
        if checkpoint_horizon != self.config.action_horizon:
            warnings.warn(
                f"Checkpoint expects action_horizon={checkpoint_horizon}, "
                f"but config has action_horizon={self.config.action_horizon}. "
                f"Actions will be padded/truncated to {checkpoint_horizon} for training.",
                stacklevel=2,
            )
        return checkpoint_horizon

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
        allowed_base = {
            "state",
            "state_mask",
            "action",
            "action_mask",
            "embodiment_id",
            "input_ids",
            "attention_mask",
        }
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("vlm_") or k.startswith("pixel_"))
            and not (k.startswith("next.") or k == "info")
        }

        # Debug: Check what state keys are in the batch
        state_keys_in_batch = [k for k in batch if "state" in k.lower()]
        if state_keys_in_batch:
            import logging

            logging.debug(f"State keys found in batch: {state_keys_in_batch}")

        # Validate and fix state dimensions if needed
        # The state encoder expects [B, T, max_state_dim] where max_state_dim=29
        # Check multiple possible state keys
        state_tensor = None
        if "state" in groot_inputs:
            state_tensor = groot_inputs["state"]
        elif "state" in batch:
            state_tensor = batch["state"]
        elif "observation.state" in batch:
            state_tensor = batch["observation.state"]

        if state_tensor is not None:
            state = state_tensor
            device = state.device
            # Use checkpoint's expected state dimension instead of config
            # The checkpoint may have been trained with a different max_state_dim
            max_state_dim = self._checkpoint_max_state_dim

            # Handle different state formats
            if state.ndim == 2:
                # [B, state_dim] -> need to check if it's padded correctly
                batch_size, state_dim = state.shape

                if state_dim != max_state_dim:
                    # State is not padded correctly - pad or truncate
                    if state_dim < max_state_dim:
                        # Pad with zeros
                        padding = torch.zeros(
                            batch_size,
                            max_state_dim - state_dim,
                            device=device,
                            dtype=state.dtype,
                        )
                        state = torch.cat([state, padding], dim=1)
                    else:
                        # Truncate if somehow larger (shouldn't happen, but be defensive)
                        # This suggests a preprocessing issue - log a warning
                        warnings.warn(
                            f"State dimension mismatch: expected {max_state_dim}, "
                            f"got {state_dim}. Truncating state tensor. "
                            "This may indicate a preprocessing issue.",
                            stacklevel=2,
                        )
                        state = state[:, :max_state_dim]

                # Always update groot_inputs with processed state
                groot_inputs["state"] = state
            elif state.ndim == 3:
                # [B, T, state_dim] - check last dimension
                batch_size, seq_len, state_dim = state.shape

                if state_dim != max_state_dim:
                    if state_dim < max_state_dim:
                        # Pad with zeros
                        padding = torch.zeros(
                            batch_size,
                            seq_len,
                            max_state_dim - state_dim,
                            device=device,
                            dtype=state.dtype,
                        )
                        state = torch.cat([state, padding], dim=2)
                    else:
                        # Truncate if somehow larger
                        warnings.warn(
                            f"State dimension mismatch: expected {max_state_dim}, "
                            f"got {state_dim}. Truncating state tensor. "
                            "This may indicate a preprocessing issue.",
                            stacklevel=2,
                        )
                        state = state[:, :, :max_state_dim]

                # Always update groot_inputs with processed state
                groot_inputs["state"] = state
            else:
                # Unexpected state format - try to handle gracefully
                warnings.warn(
                    f"Unexpected state tensor shape: {state.shape}. Expected 2D [B, D] or 3D [B, T, D].",
                    stacklevel=2,
                )
                groot_inputs["state"] = state
        else:
            # No state found - this is an error
            raise ValueError(
                "State tensor not found in batch. Expected keys: 'state' or "
                "'observation.state'. Found keys: " + str(list(batch.keys()))
            )

        # Get device from model parameters if state wasn't found
        if "state" not in groot_inputs:
            device = next(self.parameters()).device
        else:
            device = groot_inputs["state"].device

        # Process actions: reshape and pad/truncate to match checkpoint
        # The checkpoint may expect a different action dimension than our config
        action_tensor = None
        if "action" in groot_inputs:
            action_tensor = groot_inputs["action"]
        elif "action" in batch:
            action_tensor = batch["action"]

        if action_tensor is not None:
            actions = action_tensor
            max_action_dim = self._checkpoint_max_action_dim

            # Handle different action formats
            if actions.ndim == 2:
                # [B*T, action_dim] or [B, action_dim] - need to reshape
                batch_size_total, action_dim = actions.shape

                # The processor outputs action_horizon timesteps; pad/truncate to
                # the checkpoint horizon expected by the diffusion model.
                expected_T = self._checkpoint_action_horizon

                # Try to infer batch size from state if available
                if "state" in groot_inputs:
                    state_shape = groot_inputs["state"].shape
                    B = state_shape[0]  # noqa: N806 Batch size from state

                    if batch_size_total == B:
                        # Actions are [B, action_dim], need to add T dimension
                        # Pad to expected_T
                        actions = actions.unsqueeze(1)  # [B, action_dim] -> [B, 1, action_dim]
                        if actions.shape[1] < expected_T:
                            # Pad sequence dimension to expected_T
                            padding = torch.zeros(
                                B,
                                expected_T - actions.shape[1],
                                action_dim,
                                device=device,
                                dtype=actions.dtype,
                            )
                            actions = torch.cat([actions, padding], dim=1)
                    elif batch_size_total == B * expected_T:
                        # Actions are [B*T, action_dim], reshape to [B, T, action_dim]
                        actions = actions.view(B, expected_T, action_dim)
                    elif batch_size_total % B == 0:
                        # Actions are [B*T, action_dim] where T != expected_T
                        T = batch_size_total // B
                        actions = actions.view(B, T, action_dim)
                        # Pad or truncate to expected_T
                        if expected_T > T:
                            padding = torch.zeros(
                                B,
                                expected_T - T,
                                action_dim,
                                device=device,
                                dtype=actions.dtype,
                            )
                            actions = torch.cat([actions, padding], dim=1)
                        elif expected_T < T:
                            warnings.warn(
                                f"Action sequence length {T} exceeds expected {expected_T}. "
                                "Truncating to expected length.",
                                stacklevel=2,
                            )
                            actions = actions[:, :expected_T, :]
                    else:
                        # Fallback: assume single batch
                        B = 1
                        T = batch_size_total
                        actions = actions.unsqueeze(0)  # [T, action_dim] -> [1, T, action_dim]
                        if expected_T > T:
                            padding = torch.zeros(
                                1,
                                expected_T - T,
                                action_dim,
                                device=device,
                                dtype=actions.dtype,
                            )
                            actions = torch.cat([actions, padding], dim=1)
                        elif expected_T < T:
                            actions = actions[:, :expected_T, :]
                else:
                    # No state to infer from, use expected_T
                    B = batch_size_total // expected_T if batch_size_total >= expected_T else 1
                    T = expected_T if B > 0 else batch_size_total
                    if batch_size_total == B * T:
                        actions = actions.view(B, T, action_dim)
                    else:
                        # Can't reshape cleanly, pad or truncate
                        if batch_size_total < expected_T:
                            # Pad to [1, expected_T, action_dim]
                            padding = torch.zeros(
                                expected_T - batch_size_total,
                                action_dim,
                                device=device,
                                dtype=actions.dtype,
                            )
                            actions = torch.cat([actions, padding], dim=0)
                            actions = actions.unsqueeze(
                                0
                            )  # [expected_T, action_dim] -> [1, expected_T, action_dim]
                        else:
                            # Truncate and reshape
                            actions = actions[:expected_T, :]
                            actions = actions.unsqueeze(
                                0
                            )  # [expected_T, action_dim] -> [1, expected_T, action_dim]

                # Now pad/truncate action dimension
                B, T, action_dim = actions.shape
                if action_dim != max_action_dim:
                    if action_dim < max_action_dim:
                        # Pad with zeros
                        padding = torch.zeros(
                            B,
                            T,
                            max_action_dim - action_dim,
                            device=device,
                            dtype=actions.dtype,
                        )
                        actions = torch.cat([actions, padding], dim=2)
                    else:
                        # Truncate if larger
                        warnings.warn(
                            f"Action dimension mismatch: expected {max_action_dim}, "
                            f"got {action_dim}. Truncating action tensor.",
                            stacklevel=2,
                        )
                        actions = actions[:, :, :max_action_dim]

                groot_inputs["action"] = actions

            elif actions.ndim == 3:
                # [B, T, action_dim] - pad/truncate sequence length and action dimension
                B, T, action_dim = actions.shape
                expected_T = self._checkpoint_action_horizon

                # Pad or truncate sequence length to expected_T
                if expected_T > T:
                    padding = torch.zeros(
                        B,
                        expected_T - T,
                        action_dim,
                        device=device,
                        dtype=actions.dtype,
                    )
                    actions = torch.cat([actions, padding], dim=1)
                elif expected_T < T:
                    warnings.warn(
                        f"Action sequence length {T} exceeds expected {expected_T}. "
                        "Truncating to expected length.",
                        stacklevel=2,
                    )
                    actions = actions[:, :expected_T, :]

                # Pad or truncate action dimension
                if action_dim != max_action_dim:
                    if action_dim < max_action_dim:
                        # Pad with zeros
                        padding = torch.zeros(
                            B,
                            expected_T,
                            max_action_dim - action_dim,
                            device=device,
                            dtype=actions.dtype,
                        )
                        actions = torch.cat([actions, padding], dim=2)
                    else:
                        # Truncate if larger
                        warnings.warn(
                            f"Action dimension mismatch: expected {max_action_dim}, "
                            f"got {action_dim}. Truncating action tensor.",
                            stacklevel=2,
                        )
                        actions = actions[:, :, :max_action_dim]

                groot_inputs["action"] = actions
            else:
                warnings.warn(
                    f"Unexpected action tensor shape: {actions.shape}. "
                    "Expected 2D [B*T, D] or [B, D] or 3D [B, T, D].",
                    stacklevel=2,
                )
                groot_inputs["action"] = actions

        # Create action_mask if missing (should have shape [B, T, D] matching action)
        if "action" in groot_inputs and "action_mask" not in groot_inputs:
            action = groot_inputs["action"]
            if isinstance(action, torch.Tensor):
                # Create default action_mask (all valid) with proper batch shape
                groot_inputs["action_mask"] = torch.ones_like(action)
                logging.debug(
                    f"Created missing action_mask with shape {groot_inputs['action_mask'].shape} "
                    f"to match action shape {action.shape}"
                )

        # Pad action_mask to match padded actions [B, T, max_action_dim]
        # The mask must have 0s for padded dimensions so they don't contribute to loss
        if "action_mask" in groot_inputs and "action" in groot_inputs:
            action_mask = groot_inputs["action_mask"]
            padded_actions = groot_inputs["action"]
            B, T, max_action_dim = padded_actions.shape

            # Handle missing batch dimension
            if action_mask.ndim == 2:
                # Shape [T, action_dim] -> expand to [B, T, action_dim]
                action_mask = action_mask.unsqueeze(0).expand(B, -1, -1)

            # Pad action dimension if needed
            if action_mask.shape[-1] < max_action_dim:
                padding = torch.zeros(
                    B,
                    T,
                    max_action_dim - action_mask.shape[-1],
                    device=action_mask.device,
                    dtype=action_mask.dtype,
                )
                action_mask = torch.cat([action_mask, padding], dim=-1)
            elif action_mask.shape[-1] > max_action_dim:
                action_mask = action_mask[:, :, :max_action_dim]

            # Truncate action_mask TIME dimension to match truncated actions
            # This handles cases where processor action_horizon differs from checkpoint horizon
            if action_mask.shape[1] != T:
                action_mask = action_mask[:, :T, :]

            groot_inputs["action_mask"] = action_mask

        # DEBUG: Verify action_mask shape during training
        if "action" in groot_inputs:
            action_shape = (
                groot_inputs["action"].shape if isinstance(groot_inputs["action"], torch.Tensor) else "N/A"
            )
            action_mask_shape = (
                groot_inputs["action_mask"].shape
                if "action_mask" in groot_inputs and isinstance(groot_inputs["action_mask"], torch.Tensor)
                else "MISSING"
            )
            logging.debug(f"DEBUG: action shape: {action_shape}, action_mask shape: {action_mask_shape}")

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

        This method returns NORMALIZED actions. Unnormalization and relative->absolute
        conversion are handled by the postprocessor (Gr00tN1d6UnnormalizerStep),
        following the standard LeRobot pattern.

        Args:
            batch: Dictionary containing:
                - VLM inputs (pixel_values, input_ids, attention_mask, etc.)
                - State inputs (state, embodiment_id, etc.)

        Returns:
            Tensor: Predicted normalized actions of shape (B, action_horizon, action_dim)
        """
        self.eval()

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {"state", "state_mask", "embodiment_id", "input_ids", "attention_mask", "vlm_content"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("vlm_") or k.startswith("pixel_"))
            and not (k.startswith("next.") or k == "info")
        }

        # Pad state to checkpoint's expected dimension (same logic as forward)
        if "state" in groot_inputs:
            state = groot_inputs["state"]
            max_state_dim = self._checkpoint_max_state_dim
            device = state.device

            if state.ndim == 2:
                batch_size, state_dim = state.shape
                if state_dim < max_state_dim:
                    padding = torch.zeros(
                        batch_size, max_state_dim - state_dim, device=device, dtype=state.dtype
                    )
                    groot_inputs["state"] = torch.cat([state, padding], dim=1)
            elif state.ndim == 3:
                batch_size, seq_len, state_dim = state.shape
                if state_dim < max_state_dim:
                    padding = torch.zeros(
                        batch_size, seq_len, max_state_dim - state_dim, device=device, dtype=state.dtype
                    )
                    groot_inputs["state"] = torch.cat([state, padding], dim=2)

        # Get device from model parameters
        device = next(self.parameters()).device

        if not hasattr(self, "_pac_debug_step"):
            self._pac_debug_step = 0
        self._pac_debug_step += 1
        _debug = (self._pac_debug_step <= 3) or (self._pac_debug_step % 50 == 0)

        if _debug:
            print(f"\n  [PREDICT_ACTION_CHUNK step={self._pac_debug_step}]")
            if "state" in groot_inputs:
                s = groot_inputs["state"]
                print(f"    state to model: shape={s.shape}, values={s.flatten()[:6].cpu().numpy()}")
            if "embodiment_id" in groot_inputs:
                print(f"    embodiment_id: {groot_inputs['embodiment_id']}")
            print(f"    checkpoint max_state_dim={self._checkpoint_max_state_dim}, max_action_dim={self._checkpoint_max_action_dim}")

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        if _debug:
            print(f"    model output actions: shape={actions.shape}")
            print(f"    first timestep values: {actions[0, 0, :6].cpu().numpy()}")
            print(f"    last timestep values: {actions[0, -1, :6].cpu().numpy()}")
            print(f"    range: [{actions.min():.4f}, {actions.max():.4f}]")

        # Return normalized actions - unnormalization is handled by postprocessor
        # (Gr00tN1d6UnnormalizerStep) following the standard LeRobot pattern
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue.

        When a processor is set (via set_processor()), decodes the full action chunk
        at once using per-timestep statistics before queueing. This matches the
        original GR00T where get_action() returns already-decoded actions.

        When no processor is set, queues normalized actions (legacy behavior,
        unnormalization handled by postprocessor).

        Args:
            batch: Dictionary containing observation data

        Returns:
            Tensor: Single action of shape (B, action_dim)
        """
        self.eval()

        if not hasattr(self, "_sa_debug_step"):
            self._sa_debug_step = 0
        self._sa_debug_step += 1
        _debug = (self._sa_debug_step <= 3) or (self._sa_debug_step % 50 == 0)

        if _debug:
            print(f"\n  [SELECT_ACTION step={self._sa_debug_step}] Queue len before: {len(self._action_queue)}")

        # Pick up processor reference from batch (passed through by Gr00tN1d6ProcessStep)
        if self._gr00t_processor is None and "_gr00t_processor" in batch:
            self._gr00t_processor = batch.pop("_gr00t_processor")

        if len(self._action_queue) == 0:
            if _debug:
                print(f"  [SELECT_ACTION] Queue empty -> calling predict_action_chunk")
                if "state" in batch:
                    print(f"  [SELECT_ACTION] Input state: shape={batch['state'].shape}, values={batch['state'].flatten()[:6].cpu().numpy()}")

            actions = self.predict_action_chunk(batch)

            if _debug:
                print(f"  [SELECT_ACTION] predict_action_chunk output (full): shape={actions.shape}")
                print(f"    values (first timestep, t=0): {actions[0, 0, :6].cpu().numpy()}")
                print(f"    values (last timestep, t={actions.shape[1]-1}): {actions[0, -1, :6].cpu().numpy()}")
                print(f"    range: [{actions.min():.4f}, {actions.max():.4f}]")

            # Decode full chunk with per-timestep stats before queueing
            actions = self._decode_action_chunk(actions, _debug)

            # Only use the FIRST n_action_steps timesteps from the predicted chunk.
            # The model predicts a full action_horizon (e.g., 50 timesteps), but we
            # only want to execute the first few. Without slicing, deque(maxlen=n_action_steps)
            # would DISCARD the first timesteps and keep the LAST ones (far-future actions),
            # causing the robot to jump to the trajectory endpoint instead of taking
            # the next immediate actions.
            actions = actions[:, :self.config.n_action_steps, :]

            if _debug:
                print(f"  [SELECT_ACTION] after slicing to first {self.config.n_action_steps} timesteps: shape={actions.shape}")
                print(f"    values (t=0, immediate next): {actions[0, 0, :6].cpu().numpy()}")

            # Transpose to (n_action_steps, B, action_dim) for queue
            self._action_queue.extend(actions.transpose(0, 1))

        action = self._action_queue.popleft()

        if _debug:
            print(f"  [SELECT_ACTION] Popped action: shape={action.shape}, values={action.flatten()[:6].cpu().numpy()}")
            print(f"  [SELECT_ACTION] Queue len after: {len(self._action_queue)}")

        return action

    def _decode_action_chunk(self, actions: Tensor, _debug: bool = False) -> Tensor:
        """Decode a full action chunk using per-timestep statistics.

        Converts the normalized action chunk [B, horizon, dim] to decoded absolute
        actions using the processor's decode_action with per-timestep stats.

        Args:
            actions: Normalized actions of shape [B, horizon, action_dim]
            _debug: Whether to print debug info

        Returns:
            Tensor: Decoded absolute actions of shape [B, action_horizon, action_dim]
        """
        from lerobot.policies.gr00t_n1d6.utils import EmbodimentTag

        processor = self._gr00t_processor
        actions_np = actions.float().cpu().numpy()

        # Get raw_state from processor cache (set during preprocessing)
        raw_state = getattr(processor, "_cached_raw_state", None)

        if _debug:
            print(f"  [DECODE_CHUNK] Decoding full chunk: shape={actions_np.shape}")
            if raw_state is not None:
                for rk, rv in raw_state.items():
                    print(f"    raw_state[{rk}]: shape={rv.shape}, values={rv.flatten()[:6]}")
            else:
                print(f"    WARNING: raw_state is None!")

        # Get embodiment tag from processor
        embodiment_tag_mapping = getattr(processor, "embodiment_id_mapping", {})
        embodiment_tag_str = (
            list(embodiment_tag_mapping.keys())[0] if embodiment_tag_mapping else "new_embodiment"
        )
        embodiment_tag_enum = EmbodimentTag(embodiment_tag_str)

        # Decode: unnormalize with per-timestep stats + relative→absolute
        decoded_actions = processor.decode_action(
            actions_np,
            embodiment_tag_enum,
            state=raw_state,
        )

        if _debug:
            print(f"  [DECODE_CHUNK] decoded actions:")
            for dk, dv in decoded_actions.items():
                print(f"    {dk}: shape={dv.shape}, values={dv.flatten()[:6]}")

        # Concatenate joint groups back into single tensor
        modality_keys = processor.modality_configs[embodiment_tag_str]["action"].modality_keys
        action_tensors = []
        for key in modality_keys:
            action_arr = decoded_actions[key]
            action_tensor = torch.from_numpy(action_arr).to(actions.device)
            if action_tensor.ndim == 2:
                action_tensor = action_tensor.unsqueeze(1)  # [B, D] -> [B, 1, D]
            action_tensors.append(action_tensor)

        decoded_tensor = torch.cat(action_tensors, dim=-1)  # [B, action_horizon, total_action_dim]

        if _debug:
            print(f"  [DECODE_CHUNK] concatenated: shape={decoded_tensor.shape}, values={decoded_tensor.flatten()[:6]}")

        return decoded_tensor

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
