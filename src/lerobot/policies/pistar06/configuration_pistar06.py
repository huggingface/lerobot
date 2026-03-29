# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pistar06")
@dataclass
class PiStar06Config(PI05Config):
    """Configuration for PiStar06: advantage-conditioned Pi0.5 policy.

    Extends PI05Config with RECAP-style advantage conditioning parameters.
    A frozen SmolVLA value network (trained separately) labels training data
    with per-sample advantages; a learned embedding injects the binarized
    advantage directly into the action expert's input pathway (embed_suffix).
    """

    # Frozen value network checkpoint (path to .pt file from RECAP training)
    value_network_checkpoint: str | None = None

    # Advantage conditioning
    advantage_threshold: float = 0.0
    advantage_dropout: float = 0.3

    # Return computation (must match the c_fail used to train the value network)
    c_fail: float = 500.0

    # Classifier-free guidance scale (1.0 = no CFG, >1.0 = sharpen with CFG)
    cfg_beta: float = 1.0

    # Episode labels CSV for on-the-fly return computation
    episode_labels_path: str | None = None
