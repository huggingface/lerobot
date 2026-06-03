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
Groot Policy Wrapper for LeRobot Integration

Minimal integration that delegates to Isaac-GR00T components where possible
without porting their code. The intent is to:

- Download and load the pretrained GR00T model via GR00TN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via GR00T model.get_action
- Provide a training forward that can call the GR00T model forward if batch
  structure matches.

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see `GrootPolicy.finetune_with_groot_runner` below.
"""

import builtins
import os
from collections import deque
from pathlib import Path
from typing import TypeVar

import torch
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from .configuration_groot import (
    GROOT_N1_5,
    GROOT_N1_7,
    GrootConfig,
    infer_groot_model_version,
    infer_groot_n1_7_action_execution_horizon,
    infer_groot_n1_7_action_horizon,
    normalize_groot_model_version,
)
from .groot_n1 import GR00TN15

T = TypeVar("T", bound="GrootPolicy")


class GrootPolicy(PreTrainedPolicy):
    """Wrapper around external Groot model for LeRobot integration."""

    name = "groot"
    config_class = GrootConfig

    def __init__(self, config: GrootConfig, **kwargs):
        """Initialize Groot policy wrapper."""
        require_package("transformers", extra="groot")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize GR00T model using ported components
        self._groot_model = self._create_groot_model()
        self._action_queue_steps = self._resolve_action_queue_steps()

        self.reset()

    def _create_groot_model(self):
        """Create and initialize the GR00T model using Isaac-GR00T API.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps (delegating to Isaac-GR00T):
        1) Download and load pretrained model via GR00TN15.from_pretrained
        2) Align action horizon with data_config if provided
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.base_model_path,
            "tune_llm": self.config.tune_llm,
            "tune_visual": self.config.tune_visual,
            "tune_projector": self.config.tune_projector,
            "tune_diffusion_model": self.config.tune_diffusion_model,
        }
        if self.config.model_version == GROOT_N1_7:
            from .groot_n1_7 import GR00TN17

            model = GR00TN17.from_pretrained(
                **model_kwargs,
                tune_vlln=True,
                transformers_loading_kwargs={"trust_remote_code": True},
            )
        else:
            model = GR00TN15.from_pretrained(**model_kwargs)

        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        return model

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self._action_queue_steps)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: GrootConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load Groot policy from pretrained model.

        Handles two cases:
        1. Base GR00T models (e.g., 'nvidia/GR00T-N1.5-3B') - loads the raw model
        2. Fine-tuned LeRobot checkpoints - loads config and weights from safetensors

        Args:
            pretrained_name_or_path: Path to the GR00T model or fine-tuned checkpoint
            config: Optional GrootConfig. If None, loads from checkpoint or creates default
            force_download: Force download even if cached
            resume_download: Resume interrupted download
            proxies: Proxy settings
            token: HuggingFace authentication token
            cache_dir: Cache directory path
            local_files_only: Only use local files
            revision: Specific model revision
            strict: Strict state dict loading
            **kwargs: Additional arguments (passed to config)

        Returns:
            Initialized GrootPolicy instance with loaded model
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from huggingface_hub.errors import HfHubHTTPError

        requested_version = (
            normalize_groot_model_version(config.model_version)
            if config is not None
            else infer_groot_model_version(str(pretrained_name_or_path)) or GROOT_N1_5
        )
        print(
            f"The Groot policy is a wrapper around Nvidia's GR00T {requested_version} model.\n"
            f"Loading pretrained model from: {pretrained_name_or_path}"
        )

        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = False

        # Check if this is a fine-tuned LeRobot checkpoint (has model.safetensors)
        try:
            if os.path.isdir(model_id):
                is_finetuned_checkpoint = os.path.exists(os.path.join(model_id, SAFETENSORS_SINGLE_FILE))
            else:
                # Try to download the safetensors file to check if it exists
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=False,  # Just check, don't force download
                        proxies=proxies,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    is_finetuned_checkpoint = True
                except HfHubHTTPError:
                    is_finetuned_checkpoint = False
        except Exception:
            is_finetuned_checkpoint = False

        if is_finetuned_checkpoint:
            # This is a fine-tuned LeRobot checkpoint - use parent class loading
            print("Detected fine-tuned LeRobot checkpoint, loading with state dict...")
            return super().from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                config=config,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=strict,
                **kwargs,
            )

        # This is a base GR00T model - load it fresh
        print("Detected base GR00T model, loading from HuggingFace...")

        if config is None:
            model_version = infer_groot_model_version(str(pretrained_name_or_path)) or GROOT_N1_5
            # Create default config with the pretrained path
            config = GrootConfig(
                model_version=model_version,
                base_model_path=str(pretrained_name_or_path),
            )

            # Add minimal visual feature required for validation
            # validate_features() will automatically add state and action features
            # These are placeholders - actual robot features come from the preprocessor
            if not config.input_features:
                config.input_features = {
                    f"{OBS_IMAGES}.camera": PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, 224, 224),  # Default image size from config
                    ),
                }
        else:
            # Override the base_model_path with the provided path
            config.base_model_path = str(pretrained_name_or_path)

        # Pass through any additional config overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.model_version = normalize_groot_model_version(config.model_version)
        inferred_version = infer_groot_model_version(config.base_model_path)
        if inferred_version is not None and inferred_version != config.model_version:
            raise ValueError(
                f"GR00T model_version '{config.model_version}' does not match base_model_path "
                f"'{config.base_model_path}', which looks like '{inferred_version}'."
            )
        if config.model_version == GROOT_N1_7:
            if config.max_state_dim == 64:
                config.max_state_dim = 132
            if config.max_action_dim == 32:
                config.max_action_dim = 132
            if config.chunk_size == 50:
                config.chunk_size = 40
            if config.n_action_steps == 50:
                config.n_action_steps = 40
            if tuple(config.image_size) == (224, 224):
                config.image_size = (256, 256)

        # Create a fresh policy instance - this will automatically load the GR00T model
        # in __init__ via _create_groot_model()
        policy = cls(config)

        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _resolve_action_queue_steps(self) -> int:
        n_action_steps = int(self.config.n_action_steps)
        if self.config.model_version != GROOT_N1_7:
            return n_action_steps

        checkpoint_action_horizon = infer_groot_n1_7_action_horizon(
            self.config.base_model_path,
            self.config.embodiment_tag,
        )
        execution_horizon = infer_groot_n1_7_action_execution_horizon(
            self.config.base_model_path,
            self.config.embodiment_tag,
        )
        horizons = [n_action_steps]
        if checkpoint_action_horizon is not None:
            horizons.append(checkpoint_action_horizon)
        if execution_horizon is not None:
            horizons.append(execution_horizon)
        return min(horizons)

    def _resolve_prediction_horizon(self, actions: Tensor) -> int:
        """Return the policy-facing action horizon for a native GR00T prediction."""

        if self.config.model_version != GROOT_N1_7:
            return actions.shape[1]

        horizons = [actions.shape[1]]
        checkpoint_action_horizon = infer_groot_n1_7_action_horizon(
            self.config.base_model_path,
            self.config.embodiment_tag,
        )
        if checkpoint_action_horizon is not None:
            horizons.append(checkpoint_action_horizon)

        for horizon in (self.config.chunk_size, self.config.n_action_steps):
            horizon = int(horizon)
            if horizon > 0:
                horizons.append(horizon)

        return max(1, min(horizons))

    def _filter_groot_inputs(self, batch: dict[str, Tensor], *, include_action: bool) -> dict[str, Tensor]:
        allowed_base = {"state", "state_mask", "embodiment_id"}
        if include_action:
            allowed_base.update({"action", "action_mask"})

        if self.config.model_version == GROOT_N1_7:
            allowed_base.update(
                {
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "image_grid_thw",
                    "mm_token_type_ids",
                    "pixel_values_videos",
                    "video_grid_thw",
                }
            )
            allowed_base.add("action_mask")
        else:
            allowed_base.update({"action_mask"} if include_action else set())

        return {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

    def _prepare_n1_7_rtc_inputs(
        self,
        inputs: dict[str, Tensor],
        *,
        inference_delay: object,
        prev_chunk_left_over: object,
    ) -> tuple[dict[str, Tensor], dict[str, object] | None]:
        if self.config.model_version != GROOT_N1_7 or prev_chunk_left_over is None:
            return inputs, None
        if not isinstance(prev_chunk_left_over, torch.Tensor):
            raise TypeError("prev_chunk_left_over must be a torch.Tensor for GR00T N1.7 RTC.")
        if prev_chunk_left_over.numel() == 0:
            return inputs, None

        prev_actions = prev_chunk_left_over
        if prev_actions.ndim == 2:
            prev_actions = prev_actions.unsqueeze(0)
        elif prev_actions.ndim != 3:
            raise ValueError(
                "prev_chunk_left_over must have shape (T, A) or (B, T, A) for GR00T N1.7 RTC."
            )

        state = inputs.get("state")
        if state is None:
            raise ValueError("GR00T N1.7 RTC requires `state` in the preprocessed batch.")
        batch_size = state.shape[0]
        if prev_actions.shape[0] == 1 and batch_size > 1:
            prev_actions = prev_actions.expand(batch_size, -1, -1).clone()
        elif prev_actions.shape[0] != batch_size:
            raise ValueError(
                "prev_chunk_left_over batch size must match the current GR00T N1.7 batch size."
            )

        # The generic LeRobot RTC engine pads short leftovers with exact zero
        # rows for fixed-shape policy calls. Native GR00T N1.7 RTC treats every
        # provided prefix row as a real action constraint, so strip that padding
        # before constructing the native overlap options.
        valid_prefix_rows = prev_actions.detach().abs().sum(dim=(0, 2)) > 0
        if valid_prefix_rows.any():
            valid_prefix_steps = int(valid_prefix_rows.nonzero()[-1].item()) + 1
            prev_actions = prev_actions[:, :valid_prefix_steps, :]
        else:
            return inputs, None

        model_action_horizon = int(getattr(self._groot_model.config, "action_horizon", self.config.chunk_size))
        max_action_dim = int(getattr(self._groot_model.config, "max_action_dim", self.config.max_action_dim))
        if prev_actions.shape[1] > model_action_horizon:
            prev_actions = prev_actions[:, -model_action_horizon:, :]

        action_horizon = int(prev_actions.shape[1])
        if action_horizon <= 0:
            return inputs, None

        if prev_actions.shape[2] > max_action_dim:
            prev_actions = prev_actions[:, :, :max_action_dim]
        elif prev_actions.shape[2] < max_action_dim:
            pad = torch.zeros(
                prev_actions.shape[0],
                prev_actions.shape[1],
                max_action_dim - prev_actions.shape[2],
                dtype=prev_actions.dtype,
                device=prev_actions.device,
            )
            prev_actions = torch.cat([prev_actions, pad], dim=2)

        prev_actions = prev_actions.to(device=state.device, dtype=state.dtype)

        rtc_config = getattr(self.config, "rtc_config", None)
        execution_horizon = int(getattr(rtc_config, "execution_horizon", action_horizon))
        overlap_steps = max(0, min(action_horizon, execution_horizon))
        if overlap_steps == 0:
            return inputs, None

        try:
            frozen_steps = int(inference_delay or 0)
        except (TypeError, ValueError):
            frozen_steps = 0
        frozen_steps = max(0, min(frozen_steps, overlap_steps))

        options = {
            "action_horizon": action_horizon,
            "rtc_overlap_steps": overlap_steps,
            "rtc_frozen_steps": frozen_steps,
            "rtc_ramp_rate": float(getattr(self._groot_model.config, "rtc_ramp_rate", 6.0)),
        }

        inputs = dict(inputs)
        inputs["action"] = prev_actions
        return inputs, options

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
        groot_inputs = self._filter_groot_inputs(batch, include_action=True)

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Isaac-GR00T.

        Returns a tensor of shape (B, n_action_steps, action_dim).

        For N1.7, LeRobot's RTC leftovers are converted into the native GR00T
        action-overlap options before calling the underlying model.
        """
        self.eval()

        # Preprocessing is handled by the processor pipeline, so we just filter the batch.
        # During inference, we do not pass action because it is predicted.
        # N1.7 still carries a 2-D action horizon mask from its checkpoint processor.
        groot_inputs = self._filter_groot_inputs(batch, include_action=False)
        groot_options = None
        if self.config.model_version == GROOT_N1_7:
            groot_inputs, groot_options = self._prepare_n1_7_rtc_inputs(
                groot_inputs,
                inference_delay=kwargs.get("inference_delay"),
                prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            )

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            if groot_options is not None:
                outputs = self._groot_model.get_action(groot_inputs, options=groot_options)
            else:
                outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        prediction_horizon = self._resolve_prediction_horizon(actions)
        actions = actions[:, :prediction_horizon]

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions[:, : self._action_queue_steps].transpose(0, 1))
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

            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
