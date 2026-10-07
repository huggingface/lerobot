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

"""
LeRobot -- PyTorch library for real-world robotics.

Provides datasets, pretrained policies, and tools for training, evaluation,
data collection, and robot control. Integrates with Hugging Face Hub for
model and dataset sharing.

The base install is intentionally lightweight. Feature-specific dependencies
are gated behind optional extras::

    pip install 'lerobot[dataset]'       # dataset loading & creation
    pip install 'lerobot[training]'      # training loop + wandb
    pip install 'lerobot[hardware]'      # real robot control
    pip install 'lerobot[core_scripts]'  # dataset + hardware + viz (record, replay, calibrate, etc.)
    pip install 'lerobot[all]'           # everything
"""

from lerobot.__version__ import __version__

# Maps optional extras to the CLI entry-points they unlock.
available_extras: dict[str, list[str]] = {
    "dataset": ["lerobot-dataset-viz", "lerobot-imgtransform-viz", "lerobot-edit-dataset"],
    "training": ["lerobot-train"],
    "hardware": [
        "lerobot-calibrate",
        "lerobot-find-port",
        "lerobot-find-cameras",
        "lerobot-find-joint-limits",
        "lerobot-setup-motors",
    ],
    "core_scripts": ["lerobot-record", "lerobot-replay", "lerobot-teleoperate"],
    "evaluation": ["lerobot-eval"],
}

__all__ = ["__version__", "available_extras"]
