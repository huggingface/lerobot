#!/usr/bin/env python

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

import logging
from dataclasses import dataclass, field

from lerobot.transforms import ImageTransformsConfig
from lerobot.utils.import_utils import get_safe_default_video_backend

from .video import DEFAULT_DEPTH_UNIT, DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """A dataset to train on. `TrainPipelineConfig.dataset` may be a list of these, concatenated together.

    Only data keys common between multiple datasets are kept. Each dataset gets an additional transform
    that inserts the `"dataset_index"` into the returned item, with the index mapping made according to
    the order in which the datasets are provided.

    Args:
        repo_id (`str`): The Hub repo ID (or local dataset name, if `root` is set) to load.
        repo_type (`str`, *optional*, defaults to `"dataset"`): Hub repository type: `"dataset"` (the
            default) or `"bucket"` for an HF Storage Bucket streamed over `hf://buckets/`. Buckets are
            streaming-only, so `"bucket"` requires `streaming=True`.
        root (`str | None`, *optional*): Root directory for a concrete local dataset tree (e.g.
            `'dataset/path'`). If `None`, local datasets are looked up under `$HF_LEROBOT_HOME/repo_id` and
            Hub downloads use a revision-safe cache under `$HF_LEROBOT_HOME/hub`.
        episodes (`list[int] | None`, *optional*): Episode indices to include. If `None`, all episodes are
            used.
        exclude_episodes (`list[int] | None`, *optional*): Episode indices to drop (e.g. corrupt or
            heterogeneous ones). Applied on top of `episodes`.
        image_transforms (`ImageTransformsConfig`, *optional*): Image augmentation settings applied at load
            time.
        revision (`str | None`, *optional*): Hub revision (commit hash, branch, or tag) to load.
        use_imagenet_stats (`bool`, *optional*, defaults to `True`): Whether to use ImageNet normalization
            statistics for visual features instead of the dataset's own.
        video_backend (`str`, *optional*): The video decoding backend to use.
        return_uint8 (`bool`, *optional*, defaults to `False`): When `True`, RGB video frames are returned
            as `uint8` tensors (0-255) instead of `float32` (0.0-1.0). This reduces memory and speeds up
            DataLoader IPC. The training pipeline handles the conversion.
        depth_output_unit (`str`, *optional*, defaults to `"mm"`): Physical unit depth maps are dequantized
            to at load time: `"mm"` (millimeters) or `"m"` (metres). Has no effect on datasets without depth
            cameras.
        streaming (`bool`, *optional*, defaults to `False`): Stream the dataset instead of downloading it
            locally.
        eval_split (`float`, *optional*, defaults to 0.0): Fraction of episodes held out per task for
            offline evaluation (0.0 = disabled).
    """

    repo_id: str
    repo_type: str = "dataset"
    root: str | None = None
    episodes: list[int] | None = None
    exclude_episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_video_backend)
    return_uint8: bool = False
    depth_output_unit: str = DEFAULT_DEPTH_UNIT
    streaming: bool = False
    eval_split: float = 0.0

    def __post_init__(self) -> None:
        """Validate `repo_type`/`streaming`/`depth_output_unit`/`eval_split`/`episodes`/`exclude_episodes`.

        Raises:
            ValueError: If `repo_type` isn't `"dataset"` or `"bucket"`; if `repo_type="bucket"` is combined
                with `streaming=False` or a nonzero `eval_split`; if `depth_output_unit` isn't a recognized
                unit; if `eval_split` is outside `[0.0, 1.0)`; or if `episodes` contains negative or
                duplicate indices.
        """
        if self.repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {self.repo_type!r}")
        if self.repo_type == "bucket" and not self.streaming:
            raise ValueError(
                "repo_type='bucket' is streaming-only: set streaming=true to train from an HF Storage Bucket."
            )
        if self.repo_type == "bucket" and self.eval_split != 0.0:
            raise ValueError(
                "eval_split requires map-style datasets and is not supported with repo_type='bucket'."
            )
        if self.depth_output_unit not in (DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
            raise ValueError(
                f"depth_output_unit must be '{DEPTH_METER_UNIT}' or '{DEPTH_MILLIMETER_UNIT}', got {self.depth_output_unit!r}"
            )
        if not (0.0 <= self.eval_split < 1.0):
            raise ValueError(f"eval_split must be in [0.0, 1.0), got {self.eval_split}")
        if self.episodes is not None:
            if any(ep < 0 for ep in self.episodes):
                raise ValueError(
                    f"Episode indices must be non-negative, got: {[ep for ep in self.episodes if ep < 0]}"
                )
            if len(self.episodes) != len(set(self.episodes)):
                duplicates = sorted({ep for ep in self.episodes if self.episodes.count(ep) > 1})
                raise ValueError(f"Episode indices contain duplicates: {duplicates}")
        if self.exclude_episodes is not None:
            negative_episodes = [episode for episode in self.exclude_episodes if episode < 0]
            if negative_episodes:
                logger.warning(
                    "Ignoring negative exclude_episodes entries: %s",
                    negative_episodes,
                )
                self.exclude_episodes = [episode for episode in self.exclude_episodes if episode >= 0]


@dataclass
class WandBConfig:
    """Weights & Biases logging settings for `lerobot-train`.

    Args:
        enable (`bool`, *optional*, defaults to `False`): Whether to log this run to Weights & Biases.
        disable_artifact (`bool`, *optional*, defaults to `False`): Set to `True` to disable saving an
            artifact despite `save_checkpoint=True`.
        project (`str`, *optional*, defaults to `"lerobot"`): The WandB project to log to.
        entity (`str | None`, *optional*): The WandB entity (team or username) to log under.
        notes (`str | None`, *optional*): Notes attached to the WandB run.
        run_id (`str | None`, *optional*): An existing WandB run id to resume logging into.
        mode (`str | None`, *optional*): WandB mode: `"online"`, `"offline"`, or `"disabled"`. Defaults to
            `"online"`.
        add_tags (`bool`, *optional*, defaults to `True`): If `True`, save the training configuration as
            tags on the WandB run.
    """

    enable: bool = False
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None
    add_tags: bool = True


@dataclass
class EvalConfig:
    """Settings for the periodic in-training simulation-environment evaluation.

    Args:
        n_episodes (`int`, *optional*, defaults to 50): Number of episodes to run per evaluation.
        batch_size (`int`, *optional*, defaults to 0): The number of environments to use in a
            `gym.vector.VectorEnv`. `0` auto-tunes based on available CPU cores and `n_episodes`.
        use_async_envs (`bool`, *optional*, defaults to `True`): Whether to use asynchronous environments
            (multiprocessing). Automatically downgraded to a `SyncVectorEnv` when `batch_size` is 1.
        recording (`bool`, *optional*, defaults to `False`): Whether to record eval rollouts as a LeRobot
            dataset on disk.
        recording_repo_id (`str | None`, *optional*): If set, push recorded eval datasets to the Hub under
            this repo id (one repo per task, suffixed by task and env index). Requires `recording=True`.
        recording_private (`bool`, *optional*, defaults to `False`): Whether the pushed recording
            repositories should be private.
    """

    n_episodes: int = 50
    batch_size: int = 0
    use_async_envs: bool = True
    recording: bool = False
    recording_repo_id: str | None = None
    recording_private: bool = False

    def __post_init__(self) -> None:
        """Validate `recording_repo_id`/`recording`, and resolve/cap `batch_size`.

        Raises:
            ValueError: If `recording_repo_id` is set without `recording=True`.
        """
        if self.recording_repo_id is not None and not self.recording:
            raise ValueError("eval.recording_repo_id requires eval.recording=true.")
        if self.batch_size == 0:
            self.batch_size = self._auto_batch_size()
        if self.batch_size > self.n_episodes:
            self.batch_size = self.n_episodes

    def _auto_batch_size(self) -> int:
        """Pick batch_size based on CPU cores, capped by n_episodes."""
        import math
        import os

        cpu_cores = os.cpu_count() or 4
        # Each async env worker needs ~1 core; leave headroom for main process + inference.
        by_cpu = max(1, math.floor(cpu_cores * 0.7))
        return min(by_cpu, self.n_episodes, 64)


@dataclass
class EMAConfig:
    """Exponential moving average (EMA) of the policy weights.

    Standard practice for diffusion-style policies (Chi et al. 2023, "Diffusion Policy", section V.D):
    the reference implementation enables it in every config and evaluates the EMA weights. Off by
    default here because it keeps a second full copy of the parameters in memory.

    The decay follows the warmup schedule from diffusers' `EMAModel`:
    `decay_t = 1 - (1 + t / inv_gamma) ** -power`, clamped to `[min_decay, max_decay]`.
    The defaults mirror the reference implementation. Alternatively, set `decay` for a constant
    decay at every step, as used by openpi for pi0/pi05 (`ema_decay=0.99`).
    """

    enable: bool = False
    # Constant decay coefficient (openpi-style, e.g. 0.99 for pi0/pi05). When set, the warmup
    # schedule below is bypassed and the shadow uses this decay at every step.
    decay: float | None = None
    # Number of optimizer steps during which the shadow stays a hard copy of the live weights.
    update_after_step: int = 0
    # Warmup schedule parameters (see class docstring).
    inv_gamma: float = 1.0
    power: float = 0.75
    min_decay: float = 0.0
    max_decay: float = 0.9999
    # Evaluate the EMA weights (instead of the live ones) during periodic env eval.
    # Offline eval-loss (--eval_steps) always uses the live weights: it runs on every rank
    # while the EMA shadow only lives on the main process.
    use_for_eval: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_decay <= self.max_decay <= 1.0):
            raise ValueError(
                "Expected 0 <= ema.min_decay <= ema.max_decay <= 1, got "
                f"min_decay={self.min_decay} and max_decay={self.max_decay}."
            )
        if self.inv_gamma <= 0:
            raise ValueError(f"ema.inv_gamma must be positive, got {self.inv_gamma}.")
        if self.power <= 0:
            raise ValueError(f"ema.power must be positive, got {self.power}.")
        if self.update_after_step < 0:
            raise ValueError(f"ema.update_after_step must be >= 0, got {self.update_after_step}.")
        if self.decay is not None:
            if not 0.0 <= self.decay <= 1.0:
                raise ValueError(f"ema.decay must be in [0, 1], got {self.decay}.")
            # Keep the literals in sync with the field defaults above.
            if self.min_decay != 0.0 or self.max_decay != 0.9999:
                raise ValueError(
                    "ema.decay (constant decay) and ema.min_decay/ema.max_decay (schedule clamp) are "
                    "mutually exclusive: set one or the other."
                )


@dataclass
class PeftConfig:
    """PEFT (parameter-efficient fine-tuning) settings, e.g. LoRA adapters.

    PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the
    most effective methods so we'll focus on those in this high-level config interface.

    Args:
        target_modules (`list[str] | str | None`, *optional*): Either a string (module name suffix or
            `'all-linear'`), a list of module name suffixes, or a regular expression describing module
            names to target with the configured PEFT method. Some policies have a default value for this
            so that you don't *have* to choose which layers to adapt, but it might still be worthwhile
            depending on your case.
        full_training_modules (`list[str] | None`, *optional*): Names/suffixes of modules to fully
            fine-tune and store alongside adapter weights. Useful for layers that are not part of a
            pre-trained model (e.g., action state projections). Depending on the policy this defaults to
            layers that are newly created in pre-trained policies. If you're fine-tuning an already trained
            policy you might want to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
        method_type (`str`, *optional*, defaults to `"LORA"`): The PEFT (adapter) method to apply to the
            policy. Needs to be a valid PEFT type.
        init_type (`str | None`, *optional*): Adapter initialization method. Look at the specific PEFT
            adapter documentation for defaults.
        r (`int`, *optional*, defaults to 16): We expect that all PEFT adapters are in some way doing
            rank-decomposition, therefore this parameter specifies the rank used for the adapter. In
            general a higher rank means more trainable parameters and closer to full fine-tuning.
        lora_alpha (`int | None`, *optional*): Alpha parameter for LoRA scaling (`scaling = lora_alpha /
            r`). In general, a higher alpha means stronger adaptation signal. If `None`, the PEFT library
            defaults to `alpha=8`, which may dampen high-rank adapters. Common values are `r` (`alpha ==
            rank`) or `2*r`.
    """

    target_modules: list[str] | str | None = None
    full_training_modules: list[str] | None = None
    method_type: str = "LORA"
    init_type: str | None = None
    r: int = 16
    lora_alpha: int | None = None


@dataclass
class JobConfig:
    """Where and how a training run executes: locally, or dispatched to an HF Jobs flavor.

    Args:
        target (`str | None`, *optional*): Where training runs. `None` (omitted) or `"local"` runs on this
            machine. Any other value is an HF Jobs flavor and submits the run to HF Jobs. List available
            flavors and pricing with the `hf jobs hardware` command.
        image (`str`, *optional*, defaults to `"huggingface/lerobot-gpu:latest"`): Runtime image for the
            remote job (ignored for local runs).
        timeout (`str | None`, *optional*, defaults to `"2d"`): Max wall-clock for the remote job as an HF
            Jobs duration string (e.g. `"2h"`). HF Jobs itself defaults to `"2d"`; we pass an explicit,
            generous cap instead. Set a smaller value to fail fast, or a larger one for long runs.
        detach (`bool`, *optional*, defaults to `False`): Submit and exit instead of streaming the job logs
            in the foreground.
        tags (`list[str]`, *optional*): Extra tags attached to the HF job and to any dataset this run
            pushes to the Hub. A `"lerobot"` tag is always added; e.g. `--job.tags '["lelab"]'` adds more.
    """

    target: str | None = None
    image: str = "huggingface/lerobot-gpu:latest"
    timeout: str | None = "2d"
    detach: bool = False
    tags: list[str] = field(default_factory=list)

    # Two entry points to the same predicate: the staticmethod tests a raw target string
    # straight from argv (before any JobConfig exists, to decide dispatch early), while the
    # property is the ergonomic accessor for code that already holds a config instance.
    @staticmethod
    def is_remote_target(target: str | None) -> bool:
        """True when `target` names an HF Jobs flavor rather than a local run."""
        return target not in (None, "local")

    @property
    def is_remote(self) -> bool:
        """True when training should run on HF Jobs rather than this machine."""
        return self.is_remote_target(self.target)
