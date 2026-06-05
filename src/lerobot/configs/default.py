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
    # When True, video frames are returned as uint8 tensors (0-255) instead of float32 (0.0-1.0).
    # This reduces memory and speeds up DataLoader IPC. The training pipeline handles the conversion.
    return_uint8: bool = False
    streaming: bool = False

    def __post_init__(self) -> None:
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
    # Periodic training-example dump (independent of ``log_freq``). When > 0,
    # every ``log_examples_freq`` steps the trainer pushes a ``wandb.Table``
    # with one row per sampled batch element containing each camera view
    # (rendered as ``wandb.Image``), any text fields present in the batch
    # (``task`` / ``subtask`` / ``memory`` / ``instruction``), and the
    # ground-truth action chunk's first + last frames. Defaults to 5000 — set
    # to 0 to disable. Only fires when ``enable=True``, so runs without wandb
    # are unaffected.
    log_examples_freq: int = 5000
    # Number of batch elements to include in each example dump.
    log_examples_n: int = 4
    # If True (default), also run ``policy.predict_action_chunk`` on the logged
    # samples (in eval mode, no_grad) and add predicted vs ground-truth action
    # columns to the table. Costs one extra forward pass per dump — negligible
    # at the 5k-step default cadence. Set to ``False`` if your policy doesn't
    # implement ``predict_action_chunk`` or you want to skip the extra forward.
    log_examples_predict_actions: bool = True


@dataclass
class EMAConfig:
    """Exponential Moving Average of trainable policy parameters.

    Diffusion / flow-matching policies (Diffusion Policy, π0/π0.5,
    pi052) benefit substantially from averaging late-training
    parameter oscillations — see Chi et al. 2023 §V.D. The official
    JAX openpi trainer ships EMA with ``ema_decay=0.99`` (default) and
    ``0.999`` for its pi05_libero config; the openpi PyTorch port
    explicitly lists EMA as unsupported, and LeRobot main inherited
    that gap. Enabling this flag plugs ema-pytorch
    (https://github.com/lucidrains/ema-pytorch) into the LeRobot
    training loop with a shadow ``nn.Module`` clone of the policy.

    Cost: 1× model params in fp32 shadow (~13 GB for pi052's 3.3B
    params) + one elementwise update per training step (~1% step time).

    Off by default (opt-in): EMA is only beneficial for flow-matching /
    diffusion policies (pi0/pi05/pi052), and the fp32 shadow copy is pure
    overhead for other policies (e.g. VLA-JEPA). Set ``--ema.enable=true``
    to turn it on (the pi05/pi052 training recipes do this). openpi (JAX)
    ships EMA on for every config; enable it explicitly to match that.
    """

    enable: bool = False
    # Target EMA decay β in θ_ema ← β·θ_ema + (1-β)·θ_live (passed to
    # ema-pytorch as ``beta``).
    #   0.999  — last ~1000 steps; pi05_libero default in openpi
    #   0.99   — last ~100 steps; openpi top-level default
    #   0.75   — very fast EMA (Diffusion Policy original setting)
    #   0.9999 — very slow EMA (long classification runs)
    decay: float = 0.999
    # Skip the first N calls to ``ema.update()``; during this window
    # the shadow is just a hard copy of the live weights (no averaging).
    # Lets early-training rapid changes settle before averaging begins.
    # Maps to ema-pytorch's ``update_after_step`` (NOT a smooth decay
    # ramp like older lerobot EMA implementations).
    warmup_steps: int = 0
    # When True, the periodic eval block uses the EMA shadow model
    # directly (``ema.ema_model``) instead of the live policy. Standard
    # practice for diffusion-style policies — eval scores are usually
    # 1–3% higher than the live policy at the same step.
    use_for_eval: bool = True
    # When True, the periodic wandb training-example dump uses the EMA
    # shadow for the optional predicted-action columns (so what you see
    # in W&B matches eval behavior).
    use_for_wandb_examples: bool = True


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    # Set to 0 for auto-tuning based on available CPU cores and n_episodes.
    batch_size: int = 0
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    # Defaults to True; automatically downgraded to SyncVectorEnv when batch_size=1.
    use_async_envs: bool = True

    def __post_init__(self) -> None:
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
