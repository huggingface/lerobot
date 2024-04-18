import platform

import huggingface_hub

# import dataset
import numpy as np
import torch

from lerobot import __version__ as version

pt_version = torch.__version__
pt_cuda_available = torch.cuda.is_available()
pt_cuda_available = torch.cuda.is_available()
cuda_version = torch._C._cuda_getCompiledVersion() if torch.version.cuda is not None else "N/A"


# TODO(aliberts): refactor into an actual command `lerobot env`
def display_sys_info() -> dict:
    """Run this to get basic system info to help for tracking issues & bugs."""
    info = {
        "`lerobot` version": version,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Huggingface_hub version": huggingface_hub.__version__,
        # TODO(aliberts): Add dataset when https://github.com/huggingface/lerobot/pull/73 is merged
        # "Dataset version": dataset.__version__,
        "Numpy version": np.__version__,
        "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
        "Cuda version": cuda_version,
        "Using GPU in script?": "<fill in>",
        "Using distributed or parallel set-up in script?": "<fill in>",
    }
    print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
    print(format_dict(info))
    return info


def format_dict(d: dict) -> str:
    return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"


if __name__ == "__main__":
    display_sys_info()
