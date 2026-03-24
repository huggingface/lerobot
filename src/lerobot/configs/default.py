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

from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False


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


@dataclass
class EvalDockerConfig:
    # Docker image to use for evaluation (e.g., "ghcr.io/org/lerobot-eval-libero:latest").
    # Takes precedence over eval.envhub_ref.
    image: str | None = None
    # Optional EnvHub reference to resolve an image, e.g. "envhub://lerobot/libero_plus@v1".
    envhub_ref: str | None = None
    # If true, mount the local repository and prefer local source code in the container.
    use_local_code: bool = True
    # Pull the image before running.
    pull: bool = True
    # Docker --gpus value. Set to None to disable GPU flags and run CPU-only.
    gpus: str | None = "all"
    # Docker --shm-size value (increase when using larger eval.batch_size values).
    shm_size: str = "8g"
    # Port on which the host HTTP policy inference server listens.
    port: int = 50051


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # Number of sub-envs per task inside one VectorEnv. Increase to improve per-task
    # inference throughput until GPU or simulator memory saturates.
    batch_size: int = 50
    # Use AsyncVectorEnv (multiprocessing). Prefer SyncVectorEnv unless your environment
    # spends significant time in Python-side stepping and can benefit from process parallelism.
    use_async_envs: bool = False
    # Runtime where evaluation executes: "local", "docker", or "multiprocess".
    # "multiprocess" spawns local worker processes + policy servers.
    runtime: str = "local"
    docker: EvalDockerConfig = field(default_factory=EvalDockerConfig)
    # Number of parallel eval script instances to launch for one run.
    # instance_count > 1 enables multi-instance task sharding.
    instance_count: int = 1
    # 0-indexed shard id for this process. Users usually leave this at 0.
    # Additional shards are launched automatically by `lerobot-eval` when instance_count > 1.
    instance_id: int = 0
    # Number of policy inference servers to run in parallel (docker/multiprocess runtimes).
    # Each server loads a copy of the model and listens on consecutive ports
    # starting from eval.docker.port. Workers are round-robin assigned.
    policy_servers: int = 1
    # Base port for policy servers in multiprocess mode.
    port: int = 50051

    def __post_init__(self) -> None:
        if self.runtime not in {"local", "docker", "multiprocess"}:
            raise ValueError(
                f"Unsupported eval.runtime '{self.runtime}'. Expected one of: local, docker, multiprocess."
            )
        if self.instance_count < 1:
            raise ValueError("eval.instance_count must be >= 1.")
        if self.instance_id < 0 or self.instance_id >= self.instance_count:
            raise ValueError(
                f"eval.instance_id must be in [0, {self.instance_count - 1}] (got {self.instance_id})."
            )
        if self.policy_servers < 1:
            raise ValueError("eval.policy_servers must be >= 1.")
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


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
