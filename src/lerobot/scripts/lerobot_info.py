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
Use this script to get a quick summary of your system config.
It should be able to run without any of LeRobot's dependencies or LeRobot itself installed.

Example:

```shell
lerobot-info
```
"""

import importlib
import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, distribution

PACKAGE_NAME = "lerobot"


def get_ffmpeg_version() -> str:
    """Get the ffmpeg version if installed, otherwise return 'N/A'."""
    command_path = shutil.which("ffmpeg")
    if command_path is None:
        return "N/A"
    try:
        result = subprocess.run([command_path, "-version"], capture_output=True, text=True, check=True)
        first_line = result.stdout.splitlines()[0]
        version_info = first_line.split(" ")[2]
        return version_info
    except (subprocess.SubprocessError, IndexError):
        return "Installed (version parsing failed)"


def get_package_version(package_name: str) -> str:
    """Get the version of a package if it exists, otherwise return 'N/A'."""
    try:
        module = importlib.import_module(package_name)
        return getattr(module, "__version__", "Installed (version not found)")
    except ImportError:
        return "N/A"


def get_sys_info() -> dict[str, str]:
    """Run this to get basic system info to help for tracking issues & bugs."""
    # General package versions
    info = {
        "LeRobot version": get_package_version(PACKAGE_NAME),
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Huggingface Hub version": get_package_version("huggingface_hub"),
        "Datasets version": get_package_version("datasets"),
        "Numpy version": get_package_version("numpy"),
        "FFmpeg version": get_ffmpeg_version(),
    }

    # PyTorch and GPU specific information
    torch_version = "N/A"
    torch_cuda_available = "N/A"
    cuda_version = "N/A"
    gpu_model = "N/A"
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda_available = torch.cuda.is_available()
        if torch_cuda_available:
            cuda_version = str(torch.version.cuda)
            # Gets the name of the first available GPU
            gpu_model = torch.cuda.get_device_name(0)
    except ImportError:
        # If torch is not installed, the default "N/A" values will be used.
        pass

    info.update(
        {
            "PyTorch version": torch_version,
            "Is PyTorch built with CUDA support?": str(torch_cuda_available),
            "Cuda version": cuda_version,
            "GPU model": gpu_model,
            "Using GPU in script?": "<fill in>",
        }
    )
    scripts = "N/A"
    try:
        dist = distribution(PACKAGE_NAME)
        scripts = [ep.name for ep in dist.entry_points if ep.group == "console_scripts"]
    except PackageNotFoundError:
        pass

    info.update({f"{PACKAGE_NAME} scripts": str(scripts)})

    return info


def format_dict_for_markdown(d: dict[str, str]) -> str:
    """Formats a dictionary into a markdown-friendly bulleted list."""
    return "\n".join([f"- {prop}: {val}" for prop, val in d.items()])


def main():
    """
    Main function to print system info in markdown format.
    """
    system_info = get_sys_info()
    print(format_dict_for_markdown(system_info))


if __name__ == "__main__":
    main()
