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

"""Use this script to get a quick summary of your system config.
It should be able to run without any of LeRobot's dependencies or LeRobot itself installed.
"""

import platform

HAS_HF_HUB = True
HAS_HF_DATASETS = True
HAS_NP = True
HAS_TORCH = True
HAS_LEROBOT = True

try:
    import huggingface_hub
except ImportError:
    HAS_HF_HUB = False

try:
    import datasets
except ImportError:
    HAS_HF_DATASETS = False

try:
    import numpy as np
except ImportError:
    HAS_NP = False

try:
    import torch
except ImportError:
    HAS_TORCH = False

try:
    import lerobot
except ImportError:
    HAS_LEROBOT = False


lerobot_version = lerobot.__version__ if HAS_LEROBOT else "N/A"
hf_hub_version = huggingface_hub.__version__ if HAS_HF_HUB else "N/A"
hf_datasets_version = datasets.__version__ if HAS_HF_DATASETS else "N/A"
np_version = np.__version__ if HAS_NP else "N/A"

torch_version = torch.__version__ if HAS_TORCH else "N/A"
torch_cuda_available = torch.cuda.is_available() if HAS_TORCH else "N/A"
cuda_version = torch._C._cuda_getCompiledVersion() if HAS_TORCH and torch.version.cuda is not None else "N/A"


# TODO(aliberts): refactor into an actual command `lerobot env`
def display_sys_info() -> dict:
    """Run this to get basic system info to help for tracking issues & bugs."""
    info = {
        "`lerobot` version": lerobot_version,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Huggingface_hub version": hf_hub_version,
        "Dataset version": hf_datasets_version,
        "Numpy version": np_version,
        "PyTorch version (GPU?)": f"{torch_version} ({torch_cuda_available})",
        "Cuda version": cuda_version,
        "Using GPU in script?": "<fill in>",
        # "Using distributed or parallel set-up in script?": "<fill in>",
    }
    print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the last point.\n")
    print(format_dict(info))
    return info


def format_dict(d: dict) -> str:
    return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"


if __name__ == "__main__":
    display_sys_info()
