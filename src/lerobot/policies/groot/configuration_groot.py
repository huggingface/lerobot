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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

GROOT_N1_7 = "n1.7"
# Legacy GR00T N1.5 identifier. N1.5 is NOT a supported model_version (it is
# intentionally absent from _GROOT_MODEL_VERSION_ALIASES so normalize_groot_model_version
# still rejects it). It is retained only so that infer_groot_model_version can recognise
# an N1.5 base path/checkpoint and the N1.7 config/loader can reject the mismatch.
GROOT_N1_5 = "n1.5"
# Canonical guidance appended to every error raised when an N1.5 checkpoint, config,
# or processor pipeline is detected. Keep this message in sync with docs/source/groot.mdx.
GROOT_N1_5_REMOVAL_GUIDANCE = (
    "GR00T N1.5 support was removed from LeRobot. "
    "To keep using an N1.5 checkpoint, pin the last release that supports it: "
    "`pip install 'lerobot==0.5.1'`. To use the current release, migrate to GR00T N1.7 "
    "(model_version='n1.7', base model nvidia/GR00T-N1.7-3B)."
)
GROOT_N1_7_BASE_MODEL = "nvidia/GR00T-N1.7-3B"
GROOT_N1_7_BACKBONE_MODEL = "nvidia/Cosmos-Reason2-2B"
GROOT_ACTION_DECODE_TRANSFORM_LIBERO = "libero"
# Sentinel meaning "the user did not pick an action decode transform": __post_init__ resolves it
# to the embodiment default ('libero' for 'libero_sim', otherwise None). It is distinct from an
# explicit 'none' (resolved to None) so an opt-out survives a draccus save/load round-trip.
GROOT_ACTION_DECODE_TRANSFORM_AUTO = "auto"

_GROOT_MODEL_VERSION_ALIASES = {
    "n1.7": GROOT_N1_7,
    "n1_7": GROOT_N1_7,
    "n1d7": GROOT_N1_7,
    "n17": GROOT_N1_7,
    "1.7": GROOT_N1_7,
}

# Legacy N1.5 spellings, kept ONLY so they can be detected and rejected with
# GROOT_N1_5_REMOVAL_GUIDANCE (see GROOT_N1_5 above). Never map these to a supported version.
_GROOT_N1_5_VERSION_ALIASES = {"n1.5", "n1_5", "n1d5", "n15", "1.5"}

_GROOT_ACTION_DECODE_TRANSFORM_ALIASES = {
    GROOT_ACTION_DECODE_TRANSFORM_AUTO: GROOT_ACTION_DECODE_TRANSFORM_AUTO,
    "none": None,
    "": None,
    GROOT_ACTION_DECODE_TRANSFORM_LIBERO: GROOT_ACTION_DECODE_TRANSFORM_LIBERO,
}


def normalize_groot_model_version(model_version: str) -> str:
    normalized = _GROOT_MODEL_VERSION_ALIASES.get(model_version.lower())
    if normalized is None:
        supported = GROOT_N1_7
        message = f"Unsupported GR00T model_version '{model_version}'. Supported versions: {supported}."
        if model_version.lower() in _GROOT_N1_5_VERSION_ALIASES:
            message = f"{message} {GROOT_N1_5_REMOVAL_GUIDANCE}"
        raise ValueError(message)
    return normalized


def normalize_groot_action_decode_transform(transform: str | None) -> str | None:
    if transform is None:
        return None
    normalized = _GROOT_ACTION_DECODE_TRANSFORM_ALIASES.get(transform.lower())
    if normalized is None and transform.lower() not in _GROOT_ACTION_DECODE_TRANSFORM_ALIASES:
        supported = ", ".join(
            sorted(key for key, value in _GROOT_ACTION_DECODE_TRANSFORM_ALIASES.items() if value is not None)
        )
        raise ValueError(
            f"Unsupported GR00T N1.7 action decode transform '{transform}'. "
            f"Supported transforms: none, {supported}."
        )
    return normalized


def infer_groot_model_version(model_path: str | None) -> str | None:
    if not model_path:
        return None
    model_path_lower = model_path.lower()
    if "gr00t-n1.7" in model_path_lower or "gr00t_n1.7" in model_path_lower:
        return GROOT_N1_7
    # Detect legacy N1.5 paths so the N1.7 config/loader can reject the mismatch.
    # N1.5 is unsupported, but it must still be recognised here to fail loudly
    # rather than silently treating an N1.5 checkpoint as N1.7.
    if "gr00t-n1.5" in model_path_lower or "gr00t_n1.5" in model_path_lower:
        return GROOT_N1_5
    config_version = _infer_groot_model_version_from_local_config(model_path)
    if config_version is not None:
        return config_version
    return None


def is_raw_groot_n1_7_checkpoint(model_path: str | Path | None) -> bool:
    if model_path is None:
        return False

    path = Path(model_path).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
    elif path.name == "config.json":
        config_path = path
    else:
        return False

    try:
        with config_path.open() as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return "type" not in config and _infer_groot_model_version_from_config(config) == GROOT_N1_7


def infer_groot_n1_7_embodiment_tag(model_path: str | Path | None) -> str | None:
    if model_path is None:
        return None

    processor_config_path = Path(model_path).expanduser() / "processor_config.json"
    try:
        with processor_config_path.open() as f:
            processor_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    modality_configs = processor_config.get("processor_kwargs", {}).get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    if "libero_sim" in modality_configs:
        return "libero_sim"
    if len(modality_configs) == 1:
        return next(iter(modality_configs))
    return None


def infer_groot_n1_7_action_horizon(
    model_path: str | Path | None, embodiment_tag: str | None = None
) -> int | None:
    if model_path is None:
        return None

    processor_config_path = Path(model_path).expanduser() / "processor_config.json"
    try:
        with processor_config_path.open() as f:
            processor_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    processor_kwargs = processor_config.get("processor_kwargs", {})
    if not isinstance(processor_kwargs, dict):
        return None
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None

    if embodiment_tag is None:
        embodiment_tag = infer_groot_n1_7_embodiment_tag(model_path)
    if embodiment_tag is None:
        return None

    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    action_config = embodiment_config.get("action", {})
    if not isinstance(action_config, dict):
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list):
        return None
    return len(delta_indices) or None


def infer_groot_n1_7_action_execution_horizon(
    model_path: str | Path | None, embodiment_tag: str | None = None
) -> int | None:
    action_horizon = infer_groot_n1_7_action_horizon(model_path, embodiment_tag)
    if action_horizon is None:
        return None

    if embodiment_tag is None:
        embodiment_tag = infer_groot_n1_7_embodiment_tag(model_path)
    if embodiment_tag == "libero_sim":
        # NVIDIA's N1.7 LIBERO rollout wrapper replans after 8 of the 16 decoded
        # actions. Keeping that execution cadence avoids stale open-loop chunks.
        return min(action_horizon, 8)
    return action_horizon


def resolve_groot_n1_7_backbone_model(model_name: str, cache_dir: str | Path | None = None) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    cached_snapshot = _find_cached_hf_snapshot(model_name, cache_dir=cache_dir)
    return str(cached_snapshot) if cached_snapshot is not None else model_name


def _find_cached_hf_snapshot(repo_id: str, cache_dir: str | Path | None = None) -> Path | None:
    repo_cache_name = f"models--{repo_id.replace('/', '--')}"
    required_files = (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )

    for hub_cache in _candidate_hf_hub_caches(cache_dir):
        repo_cache = hub_cache / repo_cache_name
        snapshots_dir = repo_cache / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        candidates: list[Path] = []
        ref_path = repo_cache / "refs" / "main"
        try:
            ref = ref_path.read_text().strip()
        except OSError:
            ref = ""
        if ref:
            candidates.append(snapshots_dir / ref)
        candidates.extend(
            sorted(
                (path for path in snapshots_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

        seen: set[Path] = set()
        for snapshot in candidates:
            if snapshot in seen:
                continue
            seen.add(snapshot)
            if all((snapshot / filename).exists() for filename in required_files):
                return snapshot
    return None


def _candidate_hf_hub_caches(cache_dir: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser()
        candidates.append(cache_path)
        candidates.append(cache_path / "hub")

    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        candidates.append(Path(hub_cache).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(candidate)
    return deduped


def _infer_groot_model_version_from_local_config(model_path: str) -> str | None:
    path = Path(model_path).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
    elif path.name == "config.json":
        config_path = path
    else:
        return None

    if not config_path.exists():
        return None

    try:
        with config_path.open() as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return _infer_groot_model_version_from_config(config)


def _infer_groot_model_version_from_config(config: dict) -> str | None:
    model_version = config.get("model_version")
    if isinstance(model_version, str):
        if model_version.lower() in _GROOT_N1_5_VERSION_ALIASES:
            return GROOT_N1_5
        try:
            return normalize_groot_model_version(model_version)
        except ValueError:
            return None

    candidates = [config.get("model_type"), *(config.get("architectures") or [])]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.lower().replace("-", "_")
        if normalized in {"gr00tn1d7", "gr00t_n1d7", "gr00t_n1_7"}:
            return GROOT_N1_7
        # nvidia/GR00T-N1.5-3B ships model_type 'gr00t_n1_5' and architectures ['GR00T_N1_5'].
        # Recognise them so N1.5 checkpoints at generic local paths are rejected loudly
        # instead of being silently treated as N1.7 (see infer_groot_model_version).
        if normalized in {"gr00t_n1_5", "gr00tn1_5", "gr00t_n15", "gr00t_n1d5", "gr00tn1d5"}:
            return GROOT_N1_5
    if config.get("model_name") == GROOT_N1_7_BACKBONE_MODEL:
        return GROOT_N1_7
    # The Eagle VLM backbone is specific to pre-N1.7 GR00T checkpoints (N1.7 uses Cosmos/Qwen3-VL).
    backbone_cfg = config.get("backbone_cfg")
    if isinstance(backbone_cfg, dict) and "eagle_path" in backbone_cfg:
        return GROOT_N1_5
    return None


@PreTrainedConfig.register_subclass("groot")
@dataclass
class GrootConfig(PreTrainedConfig):
    """Configuration for Groot policy wrapper."""

    # Basic policy settings
    n_obs_steps: int = 1
    chunk_size: int = 40
    n_action_steps: int = 40

    # Dimension settings (must match pretrained GR00T model expectations)
    # Maximum state dimension. Shorter states will be zero-padded.
    max_state_dim: int = 132

    # Maximum action dimension. Shorter actions will be zero-padded.
    max_action_dim: int = 132

    # Normalization (start with identity, adjust as needed)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Deprecated and unused: image sizing is handled by the backbone's image processor.
    # Kept only so config.json files saved with earlier versions still parse.
    image_size: tuple[int, int] = (256, 256)

    # Groot-specific model parameters (from groot_finetune_script.py)

    # Explicit GR00T model family selection. LeRobot supports GR00T N1.7 only.
    model_version: str = GROOT_N1_7

    # Path or HuggingFace model ID for the base Groot model
    base_model_path: str | None = None

    # HF repo ID (or local path) for the GR00T N1.7 Cosmos/Qwen3-VL backbone processor.
    n1_7_backbone_model: str = GROOT_N1_7_BACKBONE_MODEL

    # Optional named action transform applied after raw N1.7 checkpoint decoding and before env.step().
    # 'auto' (default) resolves to the embodiment default ('libero' for 'libero_sim', otherwise no
    # transform). Pass 'none' to explicitly disable the transform, including for 'libero_sim'.
    action_decode_transform: str | None = GROOT_ACTION_DECODE_TRANSFORM_AUTO

    # Deprecated, GR00T N1.5 only — do not set. Kept so config.json files saved by lerobot<=0.5.1
    # still parse (draccus rejects unknown fields) and can be rejected in __post_init__ with a
    # clear error pointing at GROOT_N1_5_REMOVAL_GUIDANCE instead of a cryptic DecodingError.
    tokenizer_assets_repo: str | None = None

    # Embodiment tag to use for training (e.g. 'new_embodiment', 'gr1')
    embodiment_tag: str = "new_embodiment"

    # Fine-tuning control arguments

    # Whether to fine-tune the llm backbone
    tune_llm: bool = False

    # Whether to fine-tune the vision tower
    tune_visual: bool = False

    # Whether to fine-tune the projector
    tune_projector: bool = True

    # Whether to fine-tune the diffusion model
    tune_diffusion_model: bool = True

    # LoRA parameters (from groot_finetune_script.py)
    # Rank for the LORA model. If 0, no LORA will be used.
    lora_rank: int = 0

    # Alpha value for the LORA model
    lora_alpha: int = 16

    # Dropout rate for the LORA model
    lora_dropout: float = 0.1

    # Whether to use the full model for LORA
    lora_full_model: bool = False

    # Training parameters (matching groot_finetune_script.py)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_bf16: bool = True

    # Deprecated Isaac-GR00T runner fields below — unused by the LeRobot N1.7 implementation
    # (nothing in src/lerobot reads them). They are kept only so config.json files saved by
    # earlier lerobot releases still parse: draccus rejects unknown fields, so removing them
    # would break every previously saved groot checkpoint at config-load time.
    video_backend: str = "decord"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    def __post_init__(self):
        # 'tokenizer_assets_repo' only ever existed for GR00T N1.5 (lerobot<=0.5.1) and was
        # serialized into every groot checkpoint config.json, so a value here means a legacy
        # N1.5 checkpoint or config is being loaded.
        if self.tokenizer_assets_repo is not None:
            raise ValueError(
                "Config sets 'tokenizer_assets_repo', which only existed for GR00T N1.5; this looks "
                f"like a legacy GR00T N1.5 checkpoint or config. {GROOT_N1_5_REMOVAL_GUIDANCE}"
            )

        self.model_version = normalize_groot_model_version(self.model_version)
        self.action_decode_transform = normalize_groot_action_decode_transform(self.action_decode_transform)
        if self.base_model_path is None:
            self.base_model_path = GROOT_N1_7_BASE_MODEL

        # The N1.7 LIBERO checkpoints emit a [0, 1] gripper action, but the LIBERO
        # simulator expects the OpenVLA/[-1, 1] sign convention. NVIDIA's rollout
        # wrapper applies this conversion; mirror it here so eval on the
        # 'libero_sim' embodiment grasps correctly instead of scoring 0% success.
        # This matches the embodiment-specific handling already done for the
        # action execution horizon (see infer_groot_n1_7_action_execution_horizon).
        # Only the 'auto' sentinel resolves to the embodiment default; an explicit
        # 'none' (normalized to None above) keeps the transform disabled.
        if self.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_AUTO:
            self.action_decode_transform = (
                GROOT_ACTION_DECODE_TRANSFORM_LIBERO if self.embodiment_tag == "libero_sim" else None
            )

        # GR00T N1.5-era default values (e.g. --policy.chunk_size=50 from old commands or
        # stale configs) are migrated to the values the N1.7 checkpoints expect, with a
        # warning. The dataclass defaults are already the N1.7 values, so a plain
        # GrootConfig() never triggers this.
        legacy_default_remaps = (
            ("max_state_dim", 64, 132),
            ("max_action_dim", 32, 132),
            ("chunk_size", 50, 40),
            ("n_action_steps", 50, 40),
            ("image_size", (224, 224), (256, 256)),
        )
        for field_name, legacy_value, n1_7_value in legacy_default_remaps:
            current_value = getattr(self, field_name)
            if isinstance(legacy_value, tuple):
                current_value = tuple(current_value)
            if current_value == legacy_value:
                logger.warning(
                    "GrootConfig.%s=%s matches a legacy GR00T N1.5-era default; remapping it to %s, "
                    "the value expected by GR00T N1.7 checkpoints. Set a different value explicitly "
                    "if this is not what you want.",
                    field_name,
                    legacy_value,
                    n1_7_value,
                )
                setattr(self, field_name, n1_7_value)

        inferred_version = infer_groot_model_version(self.base_model_path)
        if inferred_version is not None and inferred_version != self.model_version:
            message = (
                f"GR00T model_version '{self.model_version}' does not match base_model_path "
                f"'{self.base_model_path}', which looks like '{inferred_version}'."
            )
            if inferred_version == GROOT_N1_5:
                message = f"{message} {GROOT_N1_5_REMOVAL_GUIDANCE}"
            raise ValueError(message)

        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

        # groot_repo_path is now optional since we ported the components
        # No validation needed

    def validate_features(self) -> None:
        """Validate and set up input/output features for Groot."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Groot policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Return scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(10000 * self.warmup_ratio),  # 5% warmup by default
            num_decay_steps=10000,  # Adjust based on training steps
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Return indices for delta observations (None for Groot)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        model_action_horizon = (
            infer_groot_n1_7_action_horizon(self.base_model_path, self.embodiment_tag) or 40
        )
        return list(range(min(self.chunk_size, model_action_horizon)))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for Groot)."""
        return None
