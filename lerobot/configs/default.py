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

from lerobot.common import (
    policies,  # noqa: F401
)
from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.common.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)


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
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self):
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
    # PEFT offers many methods, layer adapters are the most common and currently also the most effective methods so
    # we'll focus on those in this high-level config interface.

    # `target_modules` can be set by the user but default to specific values depending on the used policy. See
    # `get_peft_configuration` in `scripts/train.py`.
    #
    target_modules: list[str] | str | None = None

    # Similarly to `target_modules` this will have policy-dependent defaults which the user can override.
    modules_to_save: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific adapter method documentation for defaults.
    init_type: str | None = None

    # We expect that all adapters are in some way doing rank-decomposition. This is not true, there are several
    # methods that don't but we're focussing on these methods for now.
    r: int = 16
