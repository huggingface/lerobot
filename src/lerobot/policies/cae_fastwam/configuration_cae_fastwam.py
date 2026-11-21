# Copyright 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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

"""Configuration of CAE-FastWAM (`policy.type=cae_fastwam`), a FastWAM subclass."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.fastwam.configuration_fastwam import (
    _FASTWAM_ACTION_BASE_COMPAT_KEYS,
    _FASTWAM_VIDEO_BASE_COMPAT_KEYS,
    FastWAMConfig,
    default_action_dit_config,
    default_video_dit_config,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
# Converted LIBERO checkpoint. Like `lerobot/fastwam_base` for FastWAM, `--policy.type=cae_fastwam` initializes
# from it when the architecture fields keep their defaults (it holds the frozen action VAE training needs).
CAE_FASTWAM_BASE_MODEL_ID = "amd/CAE-FastWAM-lerobot"


def default_cae_action_dit_config(action_latent_dim: int) -> dict[str, Any]:
    """FastWAM action expert of CAE-FastWAM: a 32-wide expert (about 25M parameters) that denoises
    `action_latent_dim`-wide latent tokens. Attention width, depth and conditioning match the video
    expert, exactly as in FastWAM."""
    config = default_action_dit_config(action_latent_dim)
    config.update(hidden_dim=32, ffn_dim=256)
    return config


# Fields that set the shape of a weight of `base_model_id` outside the two DiT configs.
_BASE_MODEL_FIELDS = (
    "action_dim",
    "proprio_dim",
    "action_horizon",
    "action_latent_tokens",
    "action_latent_dim",
    "action_decoder_hidden_dim",
    "action_encoder_hidden_dim",
    "action_encoder_variational",
    "task_suites",
)


def is_cae_fastwam_base_compatible_config(config: CAEFastWAMConfig) -> bool:
    """Return whether every weight of `base_model_id` (the default architecture) fits `config`."""
    defaults = {
        f.name: f.default_factory() if f.default_factory is not MISSING else f.default
        for f in fields(CAEFastWAMConfig)
        if f.name in _BASE_MODEL_FIELDS
    }
    if any(getattr(config, name) != default for name, default in defaults.items()):
        return False
    video_config, action_config = config.video_dit_config or {}, config.action_dit_config or {}
    default_video = default_video_dit_config(config.action_dim)
    default_action = default_cae_action_dit_config(config.action_latent_dim)
    return all(video_config.get(k) == default_video.get(k) for k in _FASTWAM_VIDEO_BASE_COMPAT_KEYS) and all(
        action_config.get(k) == default_action.get(k)
        for k in (*_FASTWAM_ACTION_BASE_COMPAT_KEYS, "action_dim")
    )


@PreTrainedConfig.register_subclass("cae_fastwam")
@dataclass
class CAEFastWAMConfig(FastWAMConfig):
    """Configuration for CAE-FastWAM, FastWAM with a Compact Action Expert.

    CAE-FastWAM keeps the FastWAM architecture (Wan2.2-TI2V-5B video expert, mixture-of-transformers
    action expert, cached first-frame video K/V), but its action expert is small (hidden 32, FFN 256,
    about 25M parameters) and denoises `action_latent_tokens` latent tokens of width
    `action_latent_dim` instead of `action_horizon` raw actions. A suite-conditioned action decoder
    (the decoder of a temporal action VAE) maps the denoised latents to the
    `[action_horizon, action_dim]` chunk. State and action are min/max-normalized with the
    statistics of the selected `task_suite` (`task_suite_stats`).

    Training (`CAEFastWAMPolicy.forward`) follows the original recipe: the frozen action VAE encoder
    maps the normalized ground-truth chunk to latent tokens (a posterior sample), and FastWAM's joint
    flow-matching loss is applied to the video latents and to these action latents. The whole
    mixture-of-transformers (video and action experts) and the proprio encoder are trained; the Wan2.2
    VAE, the text encoder and the action VAE stay frozen.

    Every field of `FastWAMConfig` applies; the fields below are specific to CAE-FastWAM or change a
    FastWAM default.

    Args:
        action_latent_tokens (int): Number of latent tokens the action expert denoises per chunk.
        action_latent_dim (int): Width of each latent token (the action expert's input/output width).
        action_decoder_hidden_dim (int): Channel width of the action decoder.
        action_encoder_hidden_dim (int): Channel width of the action encoder (training only).
        action_encoder_variational (bool): The action encoder predicts a mean and a log-variance and
            training encodes each chunk to a posterior sample (as the released action VAE does).
            False: the encoder output is the latent.
        task_suites (list[str]): Suites the checkpoint was trained on, in the order of the action
            decoder's suite embedding.
        task_suite (str | None): Suite to run (one per run). Selects the action decoder's suite
            embedding and the statistics in `task_suite_stats`. Required when `task_suites` has more
            than one entry.
        task_suite_stats (dict | None): Per-suite min/max statistics of `observation.state` and
            `action`, `{suite: {key: {"min": [...], "max": [...]}}}`. FastWAM's (un)normalizer uses the
            selected suite's set (the action VAE was trained with them). None: the dataset statistics,
            as in FastWAM.
        n_action_steps (int): Actions executed per chunk. 10 is the replanning interval the
            checkpoint was evaluated with.
        base_model_id (str | None): Checkpoint that `--policy.type=cae_fastwam` initializes from (all
            weights, including the frozen action VAE) when the architecture fields keep their defaults,
            as FastWAM does with `lerobot/fastwam_base`. None starts from random weights.
        fp32_attention (bool): False: attention in the model dtype, as the original implementation.
        toggle_action_dimensions (list[int]): `[-1]` maps the LIBERO gripper from the dataset
            convention (0 close, 1 open) to the simulator's (+1 close, -1 open) and binarizes it.
    """

    action_latent_tokens: int = 8
    action_latent_dim: int = 16
    action_decoder_hidden_dim: int = 128
    action_encoder_hidden_dim: int = 128
    action_encoder_variational: bool = True
    task_suites: list[str] = field(default_factory=lambda: list(LIBERO_SUITES))
    task_suite: str | None = None
    task_suite_stats: dict[str, dict[str, dict[str, list[float]]]] | None = None

    n_action_steps: int = 10
    base_model_id: str | None = CAE_FASTWAM_BASE_MODEL_ID
    fp32_attention: bool = False
    toggle_action_dimensions: list[int] = field(default_factory=lambda: [-1])
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self) -> None:
        if self.action_dit_config is None:
            self.action_dit_config = default_cae_action_dit_config(self.action_latent_dim)
        if self.input_features is None:
            # The released checkpoint sees two LIBERO cameras side by side (FastWAM's own default is
            # a single full-width image) and the 8-D state.
            height, width = (int(size) for size in self.image_size)
            self.input_features = {
                f"{OBS_IMAGES}.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width // 2))
                for name in ("image", "image2")
            }
            if self.proprio_dim is not None:
                self.input_features[OBS_STATE] = PolicyFeature(
                    type=FeatureType.STATE, shape=(self.proprio_dim,)
                )
        super().__post_init__()
        self.task_suites = [str(suite) for suite in self.task_suites]
        self._validate_cae_options()
        # FastWAM's `fastwam_base` check does not apply to CAE-FastWAM; use the CAE-FastWAM one.
        if getattr(self, "_auto_pretrained_path", False):
            self.pretrained_path = None
            self._auto_pretrained_path = False
        if (
            not self.pretrained_path
            and not self.use_peft
            and self.base_model_id
            and is_cae_fastwam_base_compatible_config(self)
        ):
            self.pretrained_path = Path(self.base_model_id)
            self._auto_pretrained_path = True

    def _validate_cae_options(self) -> None:
        """Check the CAE-FastWAM fields against each other and against FastWAM's."""
        if self.action_dit_config is None:
            raise ValueError("`action_dit_config` must be resolved first.")
        if int(self.action_dit_config["action_dim"]) != self.action_latent_dim:
            raise ValueError(
                f"The action expert denoises tokens of width {self.action_dit_config['action_dim']}, but "
                f"`action_latent_dim` is {self.action_latent_dim}; they must match."
            )
        if self.action_latent_tokens <= 0 or self.action_horizon % self.action_latent_tokens != 0:
            raise ValueError(
                f"`action_horizon` ({self.action_horizon}) must be a positive multiple of "
                f"`action_latent_tokens` ({self.action_latent_tokens})."
            )
        if self.use_relative_actions:
            raise ValueError("CAE-FastWAM does not support `use_relative_actions`.")
        if not self.task_suites or len(set(self.task_suites)) != len(self.task_suites):
            raise ValueError(
                f"`task_suites` must be a non-empty list of unique names, got {self.task_suites}."
            )
        if self.task_suite is not None and self.task_suite not in self.task_suites:
            raise ValueError(f"Unknown `task_suite` {self.task_suite!r}; expected one of {self.task_suites}.")
        if self.task_suite_stats is not None and set(self.task_suite_stats) != set(self.task_suites):
            raise ValueError(
                f"`task_suite_stats` has suites {sorted(self.task_suite_stats)}; expected {self.task_suites}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        """AdamW with the original CAE-FastWAM recipe's betas and gradient clipping."""
        # The original CAE-FastWAM recipe: AdamW, betas (0.9, 0.95), gradient norm clipped to 1.0.
        # (Its learning-rate schedule, 5 % linear warmup then cosine decay to 1 % of the peak, has no
        # exact LeRobot preset; set `--scheduler.type` to approximate it.)
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=(0.9, 0.95),
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=1.0,
        )

    def resolve_task_suite(self, task_suite: str | None = None) -> str:
        """Return the suite to run: `task_suite`, else the configured one, else the only trained suite."""
        suite = task_suite or self.task_suite
        if suite is None:
            if len(self.task_suites) == 1:
                return self.task_suites[0]
            raise ValueError(
                "CAE-FastWAM is suite-conditioned: set `task_suite` (for example "
                f"`--policy.task_suite=libero_spatial`) to one of {self.task_suites}."
            )
        if suite not in self.task_suites:
            raise ValueError(f"Unknown task suite {suite!r}; expected one of {self.task_suites}.")
        return suite

    def suite_id(self, task_suite: str | None = None) -> int:
        """Index of the suite in the action decoder's suite embedding."""
        return self.task_suites.index(self.resolve_task_suite(task_suite))
