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
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolstar06")
@dataclass
class SmolStar06Config(SmolVLAConfig):
    """Configuration for SmolStar06: advantage-conditioned SmolVLA policy.

    Extends SmolVLAConfig with RECAP-style advantage conditioning parameters.
    The frozen value network labels training data with per-sample advantages;
    at inference the model simply conditions on "Advantage: positive".
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

    # Increase tokenizer_max_length to accommodate advantage tokens (~6 extra tokens)
    tokenizer_max_length: int = 64
