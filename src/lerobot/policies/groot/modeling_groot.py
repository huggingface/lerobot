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

Minimal integration that delegates to Isaac-GR00T N1.7 components where
possible without porting their code. Dataset loading and training
orchestration are handled by LeRobot's standard training stack.
"""

import builtins
import dataclasses
import hashlib
import json
import logging
import os
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, Unpack

import torch
from huggingface_hub import hf_hub_download, is_offline_mode, snapshot_download
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError, HFValidationError
from huggingface_hub.utils import httpx
from safetensors import safe_open
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.import_utils import _transformers_available, require_package

from ..chunk import ChunkPolicySpec
from ..pretrained import PreTrainedPolicy, RTCActionSelectKwargs
from ..utils import get_device_from_parameters
from .configuration_groot import (
    _BASE_CONFIG_DIR,
    GROOT_N1_5,
    GROOT_N1_5_REMOVAL_GUIDANCE,
    GROOT_N1_7,
    GrootConfig,
    groot_n1_7_action_execution_horizon,
    infer_groot_model_version,
    is_raw_groot_n1_7_config,
)
from .groot_n1_7 import GR00TN17, GR00TN17Config, _tie_unused_qwen_lm_head

if TYPE_CHECKING or _transformers_available:
    from transformers.trainer_pt_utils import get_parameter_names
else:
    get_parameter_names = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="GrootPolicy")

# The LM head is tied to the input embedding, so a saved checkpoint holds only one of the two names.
_TIED_WEIGHT_NAMES = (
    "backbone.model.lm_head.weight",
    "backbone.model.model.language_model.embed_tokens.weight",
)


def _state_dict_error(
    mismatched: tuple[str, ...] = (), missing: tuple[str, ...] = (), unexpected: tuple[str, ...] = ()
) -> RuntimeError:
    """Return the error that the regular load raises for these differences, in the same words.

    There torch's load_state_dict reports the size mismatches, and safetensors' load_model the keys.
    """
    error = "Error(s) in loading state_dict for GrootPolicy:" + "".join(f"\n\t{line}" for line in mismatched)
    if missing:
        error += "\n    Missing key(s) in state_dict: " + ", ".join(f'"{key}"' for key in missing)
    if unexpected:
        error += "\n    Unexpected key(s) in state_dict: " + ", ".join(f'"{key}"' for key in unexpected)
    return RuntimeError(error)


def _is_groot_n1_7_config(data: bytes) -> bool:
    try:
        return is_raw_groot_n1_7_config(json.loads(data))
    except ValueError:
        return False


def _read_copy(checkpoint: str | Path | None, base_config_sha256: str | None) -> bytes | None:
    """Return the checkpoint's copy of the base model's config.json, if it is one to use.

    It is used only if its SHA-256 is base_config_sha256, from the config saved with it, as another writer of
    the checkpoint (for example lerobot-convert-dcp) can leave the copy of another base model, and only if it
    is a GR00T N1.7 config, as for base_model_path.
    """
    path = None if checkpoint is None else Path(checkpoint) / _BASE_CONFIG_DIR / CONFIG_NAME
    if path is None or not path.is_file():
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != base_config_sha256 or not _is_groot_n1_7_config(data):
        return None
    return data


def _read_base(
    base_model_path: str | None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> tuple[bytes | None, str | None]:
    """Return the config.json of base_model_path (None if there is none to read) and the commit to build from.

    A repo id is downloaded with the Hub options, and the build reads the base weights at that file's commit,
    as a push or a cache that is behind the Hub could otherwise pair them with another config. Raises for a
    file that is not a GR00T N1.7 config, as a checkpoint names base_model_path and a save copies it.
    """
    if base_model_path is None:
        return None, None
    local = Path(base_model_path).expanduser()
    revision = None
    if local.exists():
        path = local / CONFIG_NAME
    else:
        try:
            # snapshot_download, as the build uses: with the Hub unreachable and nothing cached, it fails
            # at once with the build's error, where hf_hub_download first retries.
            snapshot = snapshot_download(
                base_model_path,
                allow_patterns=[CONFIG_NAME],
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
            )
        except HFValidationError:
            # Neither a folder nor a repo id, for example a folder on another machine.
            return None, None
        path = Path(snapshot) / CONFIG_NAME
        # Offline, the build reads the commit that the cache's main names, as this download did; given a
        # commit, snapshot_download would ask the Hub for its file list unless that list was cached.
        if not is_offline_mode():
            revision = path.parent.name
    if not path.is_file():
        return None, None
    data = path.read_bytes()
    if not _is_groot_n1_7_config(data):
        raise ValueError(
            f"{base_model_path} is not a GR00T N1.7 model: its config.json does not name one "
            "(model_type Gr00tN1d7)."
        )
    return data, revision


@contextmanager
def _weight_source(config: GrootConfig, checkpoint: Path | None) -> Iterator[None]:
    """Point pretrained_path at the checkpoint a policy loads while it is built, as make_policy does."""
    parent = config.pretrained_path
    config.pretrained_path = checkpoint
    try:
        yield
    finally:
        config.pretrained_path = parent


class GrootPolicy(PreTrainedPolicy):
    """Wrapper around external Groot model for LeRobot integration."""

    config: GrootConfig
    name = "groot"
    config_class = GrootConfig

    def supports_rtc(self) -> bool:
        return True

    def __init__(
        self,
        config: GrootConfig,
        *,
        _groot_model: GR00TN17 | None = None,
        _base_config: bytes | None = None,
        _base_revision: str | None = None,
        **kwargs,
    ):
        """Initialize Groot policy wrapper."""
        require_package("transformers", extra="groot")
        super().__init__(config)
        config.validate_features()
        self.config = config
        if _groot_model is None:
            # Read before the build, so a base model that is not N1.7 is refused before its weights are
            # downloaded, and the model is built from the bytes that a save copies: those that from_pretrained
            # read with the caller's Hub options, else the copy in the local checkpoint that pretrained_path
            # names (from_pretrained points it at the checkpoint it loads), else the base model's.
            self._base_config = _base_config or _read_copy(config.pretrained_path, config.base_config_sha256)
            self._base_revision = _base_revision
            if self._base_config is None:
                # If the download fails (offline with nothing cached, for example), the build reads
                # base_model_path itself, so it succeeds or fails as it would without this read.
                with suppress(OSError, httpx.HTTPError):
                    self._base_config, self._base_revision = _read_base(config.base_model_path)
            self._groot_model = self._create_groot_model()
        else:
            # Private, as the model must be the one that from_pretrained built from these bytes, or a save
            # would copy the config of another model.
            self._groot_model, self._base_config = _groot_model, _base_config
        digest = None if self._base_config is None else hashlib.sha256(self._base_config).hexdigest()
        if (
            config.pretrained_path is not None
            and config.base_config_sha256
            and config.base_config_sha256 != digest
        ):
            logger.warning(
                "%s records a base config.json with SHA-256 %s but holds no copy of it, and base_model_path "
                "%s gives another one or none, so the model may differ from the one trained.",
                config.pretrained_path,
                config.base_config_sha256,
                config.base_model_path,
            )
        config.base_config_sha256 = digest
        # Saved with the config, so loading a checkpoint needs no base model files for them, for any tag.
        config.embodiment_action_horizons = config.resolve_embodiment_action_horizons()
        self._action_queue_steps = self._resolve_action_queue_steps()
        self._warned_native_relative_rtc_prefix_disabled = False

        self.reset()

    @staticmethod
    def _groot_tuning_kwargs(config: GrootConfig) -> dict[str, bool]:
        return {
            "tune_llm": config.tune_llm,
            "tune_visual": config.tune_visual,
            "tune_projector": config.tune_projector,
            "tune_diffusion_model": config.tune_diffusion_model,
            "tune_vlln": config.tune_vlln,
        }

    @staticmethod
    def _groot_config_overrides(config: GrootConfig) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            # Read back by set_trainable_parameters.
            "tune_top_llm_layers": config.tune_top_llm_layers,
            "use_flash_attention": config.use_flash_attention,
        }
        # Surface the inference-time knobs onto the model config only when the user set them; None
        # leaves the value baked into the checkpoint untouched.
        if config.num_inference_timesteps is not None:
            overrides["num_inference_timesteps"] = config.num_inference_timesteps
        if config.rtc_ramp_rate is not None:
            overrides["rtc_ramp_rate"] = config.rtc_ramp_rate
        return overrides

    @classmethod
    def _groot_model_config(cls, base_config: bytes, config: GrootConfig) -> GR00TN17Config:
        return GR00TN17Config.from_dict(json.loads(base_config), **cls._groot_config_overrides(config))

    def _create_groot_model(self) -> GR00TN17:
        """Create and initialize the GR00T N1.7 model using the ported components."""
        if self._base_config is None:
            # Nothing read (a folder without config.json, a path that is neither a folder nor a repo id, or a
            # failed download), so the build reads base_model_path itself.
            config_kwargs = self._groot_config_overrides(self.config)
        else:
            config_kwargs = {
                "config": self._groot_model_config(self._base_config, self.config),
                "revision": self._base_revision,
            }
        model = GR00TN17.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            **config_kwargs,
            **self._groot_tuning_kwargs(self.config),
        )
        backbone = getattr(model, "backbone", None)
        qwen_model = getattr(backbone, "model", None)
        if qwen_model is not None:
            _tie_unused_qwen_lm_head(qwen_model)
        if self.config.dtype == torch.float32:
            self._cast_model_parameters_to_fp32(model)
        return model

    @staticmethod
    def _cast_model_parameters_to_fp32(model: torch.nn.Module) -> None:
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(torch.float32)

    @classmethod
    def _build_groot_model_from_checkpoint(
        cls,
        config: GrootConfig,
        model_file: str,
        base_config: bytes | None,
        strict: bool,
    ) -> GR00TN17 | None:
        """Build GR00T from a fine-tuned checkpoint without downloading or loading NVIDIA's base weights.

        Reads only NVIDIA's config files. Raises the regular load's error for a checkpoint it rejects too, and
        returns None for one that only the regular load can load, so the caller loads it that way.
        """
        require_package("transformers", extra="groot")
        if config.device is None:
            raise ValueError(f"{type(config).__name__}.device is unset; cannot load the weights")
        # The constructor checks this too, but here it runs before the model is built.
        config.validate_features()
        prefix = "_groot_model."
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            lm_head, embedding = _TIED_WEIGHT_NAMES
            # Weights outside GR00T, and the second name of the tied pair, as a save keeps one of them.
            unexpected = [key for key in keys if not key.startswith(prefix)]
            if prefix + lm_head in keys and prefix + embedding in keys:
                unexpected.append(prefix + embedding)
            if unexpected and not strict:
                logger.warning("Checkpoint holds weights that GR00T lacks; loading the base weights first.")
                return None
            # Views of the memory-mapped file, so transformers converts them to each parameter's dtype one
            # tensor at a time, instead of the whole checkpoint being copied first.
            state_dict = {
                key.removeprefix(prefix): checkpoint.get_tensor(key) for key in keys if key.startswith(prefix)
            }
            # The regular load checks each weight's shape under the name the checkpoint holds it, so the other
            # name of the tied pair is only filled in, and not checked.
            held = set(state_dict)
            if lm_head in state_dict:
                state_dict.setdefault(embedding, state_dict[lm_head])
            elif embedding in state_dict:
                state_dict[lm_head] = state_dict[embedding]

            if base_config is None:
                logger.warning(
                    "No config.json to read at %s; loading it the regular way.", config.base_model_path
                )
                return None
            groot_config = cls._groot_model_config(base_config, config)
            if config.dtype != torch.float32 and groot_config.dtype is None:
                # "auto" then falls back to the dtype of the base weights, which only the regular path reads.
                logger.warning("Base config does not name a dtype; loading the base weights first.")
                return None
            model = GR00TN17.from_pretrained(
                pretrained_model_name_or_path=None,
                config=groot_config,
                state_dict=state_dict,
                # Weights take the dtype their parameters are built in, so build in fp32 for fp32 parameters.
                dtype=torch.float32 if config.dtype == torch.float32 else "auto",
                # Checked below, to give the regular load's error.
                ignore_mismatched_sizes=True,
                **cls._groot_tuning_kwargs(config),
            )
            shapes = {name: tensor.shape for name, tensor in model.state_dict().items()}
            # Base weights cannot fill a weight of another shape, so the regular load rejects it too, listing
            # such weights in the model's order.
            mismatched = tuple(
                f"size mismatch for {prefix}{name}: copying a param with shape {state_dict[name].shape} from "
                f"checkpoint, the shape in current model is {shape}."
                for name, shape in shapes.items()
                if name in held and shape != state_dict[name].shape
            )
            if mismatched:
                raise _state_dict_error(mismatched=mismatched)
            # The tied LM head and embedding are one tensor; the regular load reports it once, as the head.
            missing = tuple(prefix + name for name in sorted(shapes.keys() - state_dict.keys() - {embedding}))
            unexpected += [prefix + name for name in state_dict.keys() - shapes.keys()]
            if missing or unexpected:
                if strict:
                    raise _state_dict_error(missing=missing, unexpected=tuple(sorted(unexpected)))
                logger.warning("Checkpoint keys differ from GR00T; loading the base weights first.")
                return None
            # With `load_bf16`, the backbone still builds what its base config freezes in bfloat16, which
            # rounds a fp32 checkpoint. The regular path casts to fp32 before loading, so it stays exact.
            if config.dtype == torch.float32 and any(
                parameter.is_floating_point() and parameter.dtype != torch.float32
                for parameter in model.parameters()
            ):
                logger.warning("Base config builds some weights in bfloat16; loading the base weights first.")
                return None
            # Tie first so the shared weight is copied once. Weights that kept their dtype still point into
            # the file, so copy each one out before moving it: a device copy straight from the file is slow
            # and keeps its pages as private memory on Jetson. Weights converted to another dtype are copies.
            _tie_unused_qwen_lm_head(model.backbone.model)
            in_file = {tensor.untyped_storage().data_ptr() for tensor in state_dict.values()}
            for tensor in (*model.parameters(), *model.buffers()):
                data = tensor.data
                if data.untyped_storage().data_ptr() in in_file:
                    data = data.clone()
                tensor.data = data.to(config.device)
        return model

    @staticmethod
    def _build_weight_decay_parameter_groups(model: torch.nn.Module) -> list[dict[str, Any]]:
        forbidden_name_patterns = [
            r"bias",
            r"layernorm",
            r"rmsnorm",
            r"(?:^|\.)norm(?:$|\.)",
            r"_norm(?:$|\.)",
        ]
        decay_names = set(get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns))
        decay_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name in decay_names
        ]
        no_decay_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name not in decay_names
        ]
        return [
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self._action_queue_steps)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the policy, plus the base model config file that loading it reads."""
        from lerobot.distributed.utils import is_main_process

        target = save_directory / _BASE_CONFIG_DIR
        # Replaced, not written through, as the folder it links to may hold another checkpoint's copies.
        if is_main_process() and target.is_symlink():
            target.unlink()
        super()._save_pretrained(save_directory)
        if not is_main_process() or self._base_config is None:
            return
        target.mkdir(exist_ok=True)
        # Replaced, not written through, as a link may point outside the checkpoint. A copy that a save left
        # when it stopped halfway is read only if its SHA-256 is the one config.json records.
        (target / CONFIG_NAME).unlink(missing_ok=True)
        (target / CONFIG_NAME).write_bytes(self._base_config)

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
        1. Base GR00T N1.7 models - loads the raw model
        2. Fine-tuned LeRobot checkpoints - loads config and weights from safetensors

        While the policy is built, config.pretrained_path names the local folder of the checkpoint it loads
        (None for a base model), as in make_policy.

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
        requested_version = infer_groot_model_version(str(pretrained_name_or_path)) or GROOT_N1_7
        logger.info(
            "The Groot policy wraps NVIDIA's GR00T %s model. Loading pretrained model from: %s",
            requested_version,
            pretrained_name_or_path,
        )

        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = False
        model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)

        # Check if this is a fine-tuned LeRobot checkpoint (has model.safetensors)
        try:
            if os.path.isdir(model_id):
                is_finetuned_checkpoint = os.path.exists(model_file)
            else:
                # Try to download the safetensors file to check if it exists
                try:
                    model_file = hf_hub_download(
                        repo_id=model_id,
                        filename=SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=False,  # Just check, don't force download
                        token=token,
                        local_files_only=local_files_only,
                    )
                    is_finetuned_checkpoint = True
                except HfHubHTTPError:
                    is_finetuned_checkpoint = False
        except Exception:
            is_finetuned_checkpoint = False

        if is_finetuned_checkpoint:
            logger.info("Detected fine-tuned LeRobot checkpoint, loading with state dict...")
            if force_download and not os.path.isdir(model_id):
                # Outside the check above, as on main, so that a failed forced download reaches the caller.
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=True,
                    token=token,
                    local_files_only=local_files_only,
                )
            checkpoint = Path(model_file).parent
            if config is None:
                config = PreTrainedConfig.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    # The commit of the weights file found above, which the copies come from too, so a push in
                    # between cannot pair them with another config.json.
                    revision=revision if os.path.isdir(model_id) else checkpoint.name,
                    **kwargs,
                )
            if config.base_config_sha256 and not os.path.isdir(model_id):
                # A copy is used only with a recorded SHA-256, so an older checkpoint downloads none.
                # Into the snapshot folder that holds the weights file found above, from its commit. Offline,
                # snapshot_download fails for a commit whose file list it has not cached, even when the copies
                # are, so they are looked for below.
                with suppress(OSError, httpx.HTTPError):
                    snapshot_download(
                        repo_id=model_id,
                        allow_patterns=[f"{_BASE_CONFIG_DIR}/{CONFIG_NAME}"],
                        revision=checkpoint.name,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
            base_config, base_revision = _read_copy(checkpoint, config.base_config_sha256), None
            if base_config is None:
                if not os.path.isdir(model_id):
                    # An older checkpoint has no copies, and an offline cache may lack them.
                    logger.warning("No base config copies for %s; reading base_model_path.", model_id)
                base_config, base_revision = _read_base(
                    config.base_model_path,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    token=token,
                )
            # Every way of building the policy below, a subclass's too, reads the base config there. Each way
            # draws random numbers differently while it builds, so the global generator is left as it was.
            with _weight_source(config, checkpoint), torch.random.fork_rng(devices=[]):
                if cls is GrootPolicy:
                    groot_model = cls._build_groot_model_from_checkpoint(
                        config, model_file, base_config, strict
                    )
                    if groot_model is not None:
                        policy = cls(config, _groot_model=groot_model, _base_config=base_config, **kwargs)
                        policy.to(config.device)
                        policy.eval()
                        return policy
                    # The regular load builds from these bytes too, rather than reading them again without the
                    # caller's Hub options.
                    kwargs.update(_base_config=base_config, _base_revision=base_revision)
                else:
                    # A subclass may build its own model or add weights, so it keeps the regular load.
                    logger.warning(
                        "%s is a subclass, so it builds its own model from the base weights before loading.",
                        cls.__name__,
                    )
                return super().from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    config=config,
                    # The files were downloaded above, so a forced download would fetch them twice.
                    force_download=False,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    # The commit of the weights file found above, which the copies came from too.
                    revision=revision if os.path.isdir(model_id) else checkpoint.name,
                    strict=strict,
                    **kwargs,
                )

        # This is a base GR00T model - load it fresh
        logger.info("Detected base GR00T model, loading from HuggingFace...")

        # Config overrides from kwargs go through the constructor (or `dataclasses.replace`) rather than
        # setattr, so that `__post_init__` validates them (and warns about deprecated keys) exactly as it
        # does for config.json and the CLI.
        field_names = {f.name for f in dataclasses.fields(GrootConfig)}
        field_kwargs = {key: value for key, value in kwargs.items() if key in field_names}

        if config is None:
            # Create default config with the pretrained path
            config = GrootConfig(**{"base_model_path": str(pretrained_name_or_path), **field_kwargs})

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
            if field_kwargs:
                config = dataclasses.replace(config, **field_kwargs)

        inferred_version = infer_groot_model_version(config.base_model_path)
        if inferred_version is not None and inferred_version != GROOT_N1_7:
            message = (
                f"GR00T model_version '{GROOT_N1_7}' does not match base_model_path "
                f"'{config.base_model_path}', which looks like '{inferred_version}'."
            )
            if inferred_version == GROOT_N1_5:
                message = f"{message} {GROOT_N1_5_REMOVAL_GUIDANCE}"
            raise ValueError(message)
        # Create a fresh policy instance - this will automatically load the GR00T model
        # in __init__ via _create_groot_model(). It is built from the base model alone, not from the base
        # config copies of a checkpoint the config names.
        with _weight_source(config, None):
            policy = cls(config)

        policy.eval()
        return policy

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Isaac-GR00T excludes biases and normalization parameters from weight decay."""
        return self._build_weight_decay_parameter_groups(self)

    def _resolve_action_queue_steps(self) -> int:
        n_action_steps = int(self.config.n_action_steps)
        checkpoint_action_horizon = (self.config.embodiment_action_horizons or {}).get(
            self.config.embodiment_tag
        )
        execution_horizon = groot_n1_7_action_execution_horizon(
            checkpoint_action_horizon, self.config.embodiment_tag
        )
        horizons = [n_action_steps]
        if checkpoint_action_horizon is not None:
            horizons.append(checkpoint_action_horizon)
        if execution_horizon is not None:
            horizons.append(execution_horizon)
        return min(horizons)

    def chunk_inference_spec(self) -> ChunkPolicySpec:
        """Declare the native wrapper's cropped prediction and playback horizons."""
        spec = super().chunk_inference_spec()
        native_horizon = getattr(self._groot_model.config, "action_horizon", None)
        if not isinstance(native_horizon, int) or native_horizon <= 0:
            raise ValueError("GR00T chunk serving requires a positive native action_horizon.")
        prediction_steps = self._bound_prediction_horizon(native_horizon)
        return replace(
            spec,
            prediction_steps=prediction_steps,
            execution_steps=min(prediction_steps, self._resolve_action_queue_steps()),
        )

    def _bound_prediction_horizon(self, available_steps: int) -> int:
        """Apply the same checkpoint and playback bounds before and after inference."""

        horizons = [available_steps]
        checkpoint_action_horizon = (self.config.embodiment_action_horizons or {}).get(
            self.config.embodiment_tag
        )
        if checkpoint_action_horizon is not None:
            horizons.append(checkpoint_action_horizon)

        for horizon in (self.config.chunk_size, self.config.n_action_steps):
            horizon = int(horizon)
            if horizon > 0:
                horizons.append(horizon)

        return max(1, min(horizons))

    def _resolve_prediction_horizon(self, actions: Tensor) -> int:
        """Return the policy-facing action horizon for a native GR00T prediction."""
        return self._bound_prediction_horizon(actions.shape[1])

    def _filter_groot_inputs(self, batch: dict[str, Tensor], *, include_action: bool) -> dict[str, Tensor]:
        allowed_base = {"state", "state_mask", "action_mask", "embodiment_id"}
        if include_action:
            allowed_base.add("action")

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

        return {
            k: v for k, v in batch.items() if k in allowed_base and not (k.startswith("next.") or k == "info")
        }

    def _prepare_n1_7_rtc_inputs(
        self,
        inputs: dict[str, Tensor],
        *,
        inference_delay: int | None,
        prev_chunk_left_over: Tensor | None,
    ) -> tuple[dict[str, Tensor], dict[str, object] | None]:
        if prev_chunk_left_over is None:
            return inputs, None
        if getattr(self.config, "use_relative_actions", False):
            # Generic RTC only provides normalized leftovers from the previous chunk. For
            # native relative-action N1.7 checkpoints those rows are tied to the old
            # observation state and old per-horizon stats row, so using them as the next
            # prefix can push the policy in the wrong direction. Run without native RTC
            # overlap guidance until a GROOT-specific RTC path can pass re-anchored
            # absolute leftovers through.
            if not getattr(self, "_warned_native_relative_rtc_prefix_disabled", False):
                logger.info("Disabling native GR00T RTC prefix for relative-action policy")
                self._warned_native_relative_rtc_prefix_disabled = True
            return inputs, None
        if not isinstance(prev_chunk_left_over, torch.Tensor):
            raise TypeError("prev_chunk_left_over must be a torch.Tensor for GR00T N1.7 RTC.")
        if prev_chunk_left_over.numel() == 0:
            return inputs, None

        prev_actions = prev_chunk_left_over
        if prev_actions.ndim == 2:
            prev_actions = prev_actions.unsqueeze(0)
        elif prev_actions.ndim != 3:
            raise ValueError("prev_chunk_left_over must have shape (T, A) or (B, T, A) for GR00T N1.7 RTC.")

        state = inputs.get("state")
        if state is None:
            raise ValueError("GR00T N1.7 RTC requires `state` in the preprocessed batch.")
        batch_size = state.shape[0]
        if prev_actions.shape[0] == 1 and batch_size > 1:
            prev_actions = prev_actions.expand(batch_size, -1, -1).clone()
        elif prev_actions.shape[0] != batch_size:
            raise ValueError("prev_chunk_left_over batch size must match the current GR00T N1.7 batch size.")

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

        model_action_horizon = int(
            getattr(self._groot_model.config, "action_horizon", self.config.chunk_size)
        )
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

        options: dict[str, object] = {
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
        device = get_device_from_parameters(self)

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")
        if loss is None:
            raise RuntimeError(
                "GR00T model.forward did not return a 'loss'. Training batches must include "
                "'action' and 'action_mask'; check the preprocessor output."
            )

        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Isaac-GR00T.

        Returns a tensor of shape (B, n_action_steps, action_dim).

        For N1.7, LeRobot's RTC leftovers are converted into the native GR00T
        action-overlap options before calling the underlying model.

        `noise` is the starting sample of the flow-matching loop, shaped as the underlying
        model's (B, action_horizon, max_action_dim). When it is None, fresh noise is drawn.
        """
        self.eval()

        # Preprocessing is handled by the processor pipeline, so we just filter the batch.
        # During inference, we do not pass action because it is predicted.
        # N1.7 still carries a 2-D action horizon mask from its checkpoint processor.
        groot_inputs = self._filter_groot_inputs(batch, include_action=False)
        groot_inputs, groot_options = self._prepare_n1_7_rtc_inputs(
            groot_inputs,
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
        )

        # Get device from model parameters
        device = get_device_from_parameters(self)

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            if groot_options is not None:
                outputs = self._groot_model.get_action(groot_inputs, options=groot_options, noise=noise)
            else:
                outputs = self._groot_model.get_action(groot_inputs, noise=noise)

        actions = outputs.get("action_pred")

        prediction_horizon = self._resolve_prediction_horizon(actions)
        actions = actions[:, :prediction_horizon]

        if self.config.output_features is None:
            raise ValueError("`GrootConfig.output_features` must be resolved before predicting actions.")
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Select single action from action queue."""
        if getattr(self.config, "use_relative_actions", False):
            raise NotImplementedError(
                "GrootPolicy.select_action does not support relative-action policies because cached "
                "relative chunk actions can be decoded against newer observation states. Use "
                "predict_action_chunk and postprocess the full chunk before queuing actions, or use "
                "the RTC/chunked rollout inference path."
            )

        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._action_queue.extend(actions[:, : self._action_queue_steps].transpose(0, 1))
        return self._action_queue.popleft()
