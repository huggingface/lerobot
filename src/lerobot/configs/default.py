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

from dataclasses import dataclass, field

from lerobot.transforms import ImageTransformsConfig
from lerobot.utils.import_utils import get_safe_default_video_backend

from .video import DEFAULT_DEPTH_UNIT, DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory for a concrete local dataset tree (e.g. 'dataset/path'). If None, local datasets are
    # looked up under $HF_LEROBOT_HOME/repo_id and Hub downloads use a revision-safe cache under $HF_LEROBOT_HOME/hub.
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_video_backend)
    # When True, RGB video frames are returned as uint8 tensors (0-255) instead of float32 (0.0-1.0).
    # This reduces memory and speeds up DataLoader IPC. The training pipeline handles the conversion.
    return_uint8: bool = False
    # Physical unit depth maps are dequantized to at load time: "mm" (millimeters) or "m" (metres).
    # Has no effect on datasets without depth cameras.
    depth_output_unit: str = DEFAULT_DEPTH_UNIT
    streaming: bool = False
    # Fraction of episodes held out per task for offline evaluation (0.0 = disabled).
    eval_split: float = 0.0

    def __post_init__(self) -> None:
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


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'
    add_tags: bool = True  # If True, save configuration as tags in the WandB run.


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    # Set to 0 for auto-tuning based on available CPU cores and n_episodes.
    batch_size: int = 0
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    # Defaults to True; automatically downgraded to SyncVectorEnv when batch_size=1.
    use_async_envs: bool = True
    # Whether to record eval rollouts as a LeRobot dataset on disk.
    recording: bool = False
    # If set, push recorded eval datasets to the Hub under this repo id (one repo per task,
    # suffixed by task and env index). Requires recording=true.
    recording_repo_id: str | None = None
    # Whether the pushed recording repositories should be private.
    recording_private: bool = False

    def __post_init__(self) -> None:
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
class PeftConfig:
    # PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    # effective methods so we'll focus on those in this high-level config interface.

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16

    # Alpha parameter for LoRA scaling (scaling = lora_alpha / r).
    # In general, a higher alpha means stronger adaptation signal.
    # If None, the PEFT library defaults to alpha=8, which may dampen high-rank adapters.
    # Common values are r (alpha == rank) or 2*r.
    lora_alpha: int | None = None


@dataclass
class JobConfig:
    # Where training runs. None (omitted) or "local" runs on this machine.
    # Any other value is an HF Jobs flavor and submits the run to HF Jobs.
    # List available flavors + pricing with `hf jobs hardware` command.
    target: str | None = None
    # Runtime image for the remote job (ignored for local runs).
    image: str = "huggingface/lerobot-gpu:latest"
    # Max wall-clock for the remote job as an HF Jobs duration string (e.g. "2h").
    # Defaults to "2d": We pass an explicit, generous cap instead. Set a smaller
    # value to fail fast, or a larger one for long runs.
    timeout: str | None = "2d"
    # Submit and exit instead of streaming the job logs in the foreground.
    detach: bool = False
    # Extra tags attached to the HF job and to any dataset this run pushes to the
    # Hub. A "lerobot" tag is always added; e.g. --job.tags '["lelab"]' adds more.
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
