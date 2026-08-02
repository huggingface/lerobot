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

import logging
from contextlib import nullcontext

import torch


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    elif torch.xpu.is_available():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available.

    Raises:
        ValueError: If the requested device family is known but not available on
            this machine (``AssertionError`` was previously used and is easy to
            mistake for a programmer bug under ``python -O`` where asserts vanish).
    """
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device {try_device!r} but CUDA is not available.")
        device = torch.device(try_device)
    elif try_device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps' but MPS is not available.")
        device = torch.device("mps")
    elif try_device == "xpu":
        if not torch.xpu.is_available():
            raise ValueError("Requested device 'xpu' but XPU is not available.")
        device = torch.device("xpu")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logging.warning("Using CPU, this will be slow.")
    else:
        device = torch.device(try_device)
        if log:
            logging.warning(f"Using custom {try_device} device.")
    return device


def resolve_safetensors_device(map_location: str | torch.device) -> str:
    """Resolve a device string for a safetensors load, working around a device-mapping quirk.

    safetensors' load maps the bare string "cuda" to cuda:0 regardless of the current device
    (unlike torch's .to("cuda"), which honors torch.cuda.current_device()). Under multi-GPU
    accelerate/FSDP every rank would then load its weights onto GPU 0, OOMing it before sharding.
    Resolve "cuda" to the concrete current-device index so each rank loads onto its own GPU.
    """
    map_location = str(map_location)
    if map_location == "cuda" and torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return map_location


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            # NOTE: Some Intel XPU devices do not support double precision (FP64).
            # The `has_fp64` flag is returned by `torch.xpu.get_device_capability()`
            # when available; if False, we fall back to float32 for compatibility.
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "xpu":
        return torch.xpu.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu or cpu.")


def is_amp_available(device: str):
    if device in ["cuda", "xpu", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


def get_autocast_context(device_type: str, dtype: torch.dtype = torch.bfloat16):
    """Return a device-safe autocast context manager.

    Hardcoding `torch.autocast(dtype=torch.bfloat16)` breaks on backends without AMP
    (MPS) and silently misbehaves on pre-Ampere CUDA GPUs that lack bf16 support. This
    picks a safe context per device:
      - no AMP support (e.g. mps): `nullcontext()` (run in the tensors' native dtype)
      - CPU asked for a dtype its autocast does not implement (notably float32, used to force
        a block back to full precision): `nullcontext()`. `torch.autocast` would accept it,
        then warn and disable itself on *every* call, so this keeps the same behavior without
        the per-forward log spam.
      - CUDA requesting bf16 on compute capability < 8.0 (pre-Ampere): fall back to fp16
      - otherwise: `torch.autocast(device_type, dtype)`
    """
    if not is_amp_available(device_type):
        return nullcontext()
    if device_type == "cpu" and dtype not in (torch.bfloat16, torch.float16):
        return nullcontext()
    if (
        device_type == "cuda"
        and dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] < 8
    ):
        dtype = torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)
