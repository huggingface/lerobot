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
from __future__ import annotations

import abc
import builtins
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, ClassVar, TypedDict, TypeVar, Unpack

import packaging
import safetensors
from huggingface_hub import hf_hub_download, save_torch_state_dict
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.hub import HubMixin

from .utils import log_model_loading_keys

T = TypeVar("T", bound="PreTrainedPolicy")

# Pinned far above any policy's total size so save_torch_state_dict always emits exactly one
# `model.safetensors` (no shards, no index) — a constant, not a computed byte count.
_SINGLE_FILE_SHARD_SIZE = "1TB"


class ActionSelectKwargs(TypedDict, total=False):
    noise: Tensor | None


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    # --- declarative parallelism/acceleration surface ----------------------------------------
    # Module CLASS names forming the FSDP2 wrap units (and, once wired, the activation-
    # checkpointing units). Resolved onto the accelerate plugin right before
    # `accelerator.prepare()` by `lerobot.distributed.set_fsdp_wrap_modules`; sharded training
    # with no wrap source anywhere fails loudly instead of silently wrapping only the root.
    _fsdp_wrap_modules: ClassVar[list[str] | None] = None
    # Non-`forward` entry points that must trigger FSDP2 unshard/reshard hooks when called on a
    # sharded policy (registered post-prepare via `torch.distributed.fsdp
    # .register_fsdp_forward_method`); calling them unregistered crashes on mixed Tensor/DTensor.
    _fsdp_forward_methods: ClassVar[tuple[str, ...]] = ("select_action", "predict_action_chunk")
    # Capability gate for the (future) activation-checkpointing wiring.
    supports_gradient_checkpointing: ClassVar[bool] = False
    # Declarative context-parallel plan (diffusers `ContextParallelModelPlan` semantics:
    # module FQN -> sequence split/gather spec). Reserved for the CP engine round.
    _cp_plan: ClassVar[dict[str, Any] | None] = None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        """Serialize this policy's parameters (and config) into `save_directory`.

        Sharding is handled internally: under FSDP2 the full state dict is gathered through a
        COLLECTIVE, so when the policy is sharded this method (via `save_pretrained`) must be
        called on EVERY rank — a rank-0-gated call deadlocks. File writes happen on the main
        process only, in all layouts (single, DDP, sharded).

        Args:
            save_directory (Path): Target directory for the policy config (`config.json`) and the
                safetensors weight file(s).
        """
        # Lazy imports: the persistence layer pulls in lerobot.distributed only when saving.
        from lerobot.distributed.checkpoint import full_model_state_dict, is_sharded_module
        from lerobot.distributed.utils import is_main_process

        model_to_save = self.module if hasattr(self, "module") else self
        if is_sharded_module(model_to_save):
            logging.info("Gathering the full state dict from all ranks (sharded policy).")
        state_dict = full_model_state_dict(model_to_save)  # collective when sharded; {} off-main
        if not state_dict or not is_main_process():
            # Sharded: the gather materializes on the main rank only (emptiness check).
            # Non-sharded multi-rank (DDP): every rank holds a full dict — the explicit rank
            # gate prevents N ranks racing on the same files. Single process: never taken.
            return
        self.config._save_pretrained(save_directory)
        save_torch_state_dict(state_dict, str(save_directory), max_shard_size=_SINGLE_FILE_SHARD_SIZE)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        # Create base kwargs
        kwargs = {"strict": strict}

        # Add device parameter for newer versions that support it
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location

        # Load the model with appropriate kwargs
        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
        log_model_loading_keys(missing_keys, unexpected_keys)

        # For older versions, manually move to device if needed
        if "device" not in kwargs and map_location != "cpu":
            logging.warning(
                "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                " This means that the model is loaded on 'cpu' first and then copied to the device."
                " This leads to a slower loading time."
                " Please update safetensors to version 0.4.3 or above for improved performance."
            )
            model.to(map_location)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Returns the action chunk (for action chunking policies) for a given observation, potentially in batch mode.

        Child classes using action chunking should use this method within `select_action` to form the action chunk
        cached for selection.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError

    def wrap_with_peft(
        self,
        peft_config=None,
        peft_cli_overrides: dict | None = None,
    ) -> PreTrainedPolicy:
        """
        Wrap this policy with PEFT adapters for parameter-efficient fine-tuning.

        This method is the single entry point for PEFT integration. Subclasses should
        override `_get_default_peft_targets()` to provide default target modules, and
        `_validate_peft_config()` for policy-specific validation.

        Args:
            peft_config: Optional PEFT adapter configuration (e.g., LoraConfig).
                If provided, used directly (with CLI overrides applied).
            peft_cli_overrides: Optional dict of CLI overrides (method_type, target_modules, r, etc.)
                These are merged with policy defaults to build the final config.
        """
        from peft import get_peft_model

        # If user provided a complete config, use it directly (with overrides)
        if peft_config is not None:
            final_config = peft_config
            if peft_cli_overrides:
                final_config = self._apply_peft_cli_overrides(final_config, peft_cli_overrides)
        else:
            # Build config from defaults + CLI overrides
            final_config = self._build_peft_config(peft_cli_overrides or {})

        # Validate the configuration
        self._validate_peft_config(final_config)

        # Freeze base parameters, only adapter params will be trained
        for p in self.parameters():
            p.requires_grad_(False)

        # Store pretrained path for PEFT's base_model_name_or_path
        if self.config.pretrained_path:
            self.name_or_path = str(self.config.pretrained_path)

        # Wrap with PEFT
        peft_model = get_peft_model(self, final_config)

        # Mark config as using PEFT for proper loading later
        peft_model.config.use_peft = True

        logging.info(f"Wrapped {self.name} with PEFT ({type(final_config).__name__})")
        return peft_model

    def _get_default_peft_targets(self) -> dict[str, any] | None:
        """
        Return default PEFT target modules for this policy.

        Override this in subclasses to provide policy-specific defaults. These defaults
        are PEFT-method agnostic - they only specify which modules to target.

        """
        return None

    def _validate_peft_config(self, peft_config) -> None:
        """
        Validate the PEFT configuration for this policy.

        Override this in subclasses to add policy-specific validation or warnings.
        The default implementation checks that a pretrained_path exists.

        Args:
            peft_config: The PEFT configuration to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.config.pretrained_path:
            raise ValueError(
                "Training from scratch using PEFT is unlikely to yield good results. "
                "Supply a `policy.pretrained_path` to fine-tune an existing model."
            )

    def _preprocess_peft_cli_overrides(self, cli_overrides: dict, peft_method_type) -> dict:
        """
        Preprocess CLI overrides: rename keys and handle method-specific init_type.

        Args:
            cli_overrides: Dict of CLI options (will be copied, not mutated).
            peft_method_type: The PeftType enum value for the PEFT method.

        Returns:
            Preprocessed dict with renamed keys and init_type mapped to method-specific key.
        """
        from peft import PeftType

        cli_overrides = cli_overrides.copy()

        # Handle the full_training_modules -> modules_to_save rename
        if "full_training_modules" in cli_overrides:
            cli_overrides["modules_to_save"] = cli_overrides.pop("full_training_modules")

        # Remove method_type as it's handled separately
        cli_overrides.pop("method_type", None)

        # Handle init_type specially based on PEFT method
        init_type = cli_overrides.pop("init_type", None)
        if init_type is not None:
            if peft_method_type == PeftType.LORA:
                cli_overrides["init_lora_weights"] = init_type
            elif peft_method_type == PeftType.MISS:
                cli_overrides["init_weights"] = init_type
            else:
                raise ValueError(f"Init type '{init_type}' unknown for PEFT method {peft_method_type}.")

        return cli_overrides

    def _build_peft_config(self, cli_overrides: dict):
        """Build a PEFT config from policy defaults and CLI overrides."""
        from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType

        # Determine PEFT method type (default to LORA)
        method_type_str = cli_overrides.get("method_type") or "lora"
        peft_method_type = PeftType[method_type_str.upper()]
        peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

        # Preprocess CLI overrides
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # Start with policy defaults, apply CLI overrides
        config_dict = dict(self._get_default_peft_targets() or {})
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        # Ensure we have target_modules
        if not config_dict.get("target_modules"):
            raise ValueError(
                f"Policy '{self.name}' does not define default target_modules. "
                "Please pass --peft.target_modules explicitly."
            )

        return peft_config_cls(**config_dict)

    def _apply_peft_cli_overrides(self, peft_config, cli_overrides: dict):
        """Apply CLI overrides to an existing PEFT config."""
        from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType

        # Get method type from existing config or CLI override
        method_type_str = cli_overrides.get("method_type")
        if method_type_str:
            peft_method_type = PeftType[method_type_str.upper()]
            peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]
        else:
            peft_method_type = PeftType(peft_config.peft_type)
            peft_config_cls = type(peft_config)

        # Preprocess CLI overrides
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # Start with existing config, apply CLI overrides
        config_dict = {k: v for k, v in dataclasses.asdict(peft_config).items() if not k.startswith("_")}
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        return peft_config_cls(**config_dict)
