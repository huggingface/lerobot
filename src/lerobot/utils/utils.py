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
import os
import platform
import select
import subprocess
import sys
import time
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from accelerate import Accelerator
from datasets.utils.logging import disable_progress_bar, enable_progress_bar


def inside_slurm():
    """Check whether the python process was launched through slurm"""
    # TODO(rcadene): return False for interactive mode `--pty bash`
    return "SLURM_JOB_ID" in os.environ


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
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        assert torch.cuda.is_available()
        device = torch.device(try_device)
    elif try_device == "mps":
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
    elif try_device == "xpu":
        assert torch.xpu.is_available()
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


def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    accelerator: Accelerator | None = None,
):
    """Initialize logging configuration for LeRobot.

    In multi-GPU training, only the main process logs to console to avoid duplicate output.
    Non-main processes have console logging suppressed but can still log to file.

    Args:
        log_file: Optional file path to write logs to
        display_pid: Include process ID in log messages (useful for debugging multi-process)
        console_level: Logging level for console output
        file_level: Logging level for file output
        accelerator: Optional Accelerator instance (for multi-GPU detection)
    """

    def custom_format(record: logging.LogRecord) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        pid_str = f"[PID: {os.getpid()}] " if display_pid else ""
        return f"{record.levelname} {pid_str}{dt} {fnameline[-15:]:>15} {record.getMessage()}"

    formatter = logging.Formatter()
    formatter.format = custom_format

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # Clear any existing handlers
    logger.handlers.clear()

    # Determine if this is a non-main process in distributed training
    is_main_process = accelerator.is_main_process if accelerator is not None else True

    # Console logging (main process only)
    if is_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level.upper())
        logger.addHandler(console_handler)
    else:
        # Suppress console output for non-main processes
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def say(text: str, blocking: bool = False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0)


def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:  # (h, w, c) -> (c, h, w)
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    Return True if a given string can be converted to a numpy dtype.
    """
    try:
        # Attempt to convert the string to a numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # If a TypeError is raised, the string is not a valid dtype
        return False


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")  # enter key
        return False
    else:
        return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == ""


def move_cursor_up(lines):
    """Move the cursor up by a specified number of lines."""
    print(f"\033[{lines}A", end="")


def get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time_s: float):
    days = int(elapsed_time_s // (24 * 3600))
    elapsed_time_s %= 24 * 3600
    hours = int(elapsed_time_s // 3600)
    elapsed_time_s %= 3600
    minutes = int(elapsed_time_s // 60)
    seconds = elapsed_time_s % 60
    return days, hours, minutes, seconds


class SuppressProgressBars:
    """
    Context manager to suppress progress bars.

    Example
    --------
    ```python
    with SuppressProgressBars():
        # Code that would normally show progress bars
    ```
    """

    def __enter__(self):
        disable_progress_bar()

    def __exit__(self, exc_type, exc_val, exc_tb):
        enable_progress_bar()


class TimerManager:
    """
    Lightweight utility to measure elapsed time.

    Examples
    --------
    ```python
    # Example 1: Using context manager
    timer = TimerManager("Policy", log=False)
    for _ in range(3):
        with timer:
            time.sleep(0.01)
    print(timer.last, timer.fps_avg, timer.percentile(90))  # Prints: 0.01 100.0 0.01
    ```

    ```python
    # Example 2: Using start/stop methods
    timer = TimerManager("Policy", log=False)
    timer.start()
    time.sleep(0.01)
    timer.stop()
    print(timer.last, timer.fps_avg, timer.percentile(90))  # Prints: 0.01 100.0 0.01
    ```
    """

    def __init__(
        self,
        label: str = "Elapsed-time",
        log: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.label = label
        self.log = log
        self.logger = logger
        self._start: float | None = None
        self._history: list[float] = []

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started.")
        elapsed = time.perf_counter() - self._start
        self._history.append(elapsed)
        self._start = None
        if self.log:
            if self.logger is not None:
                self.logger.info(f"{self.label}: {elapsed:.6f} s")
            else:
                logging.info(f"{self.label}: {elapsed:.6f} s")
        return elapsed

    def reset(self):
        self._history.clear()

    @property
    def last(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def avg(self) -> float:
        return mean(self._history) if self._history else 0.0

    @property
    def total(self) -> float:
        return sum(self._history)

    @property
    def count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[float]:
        return deepcopy(self._history)

    @property
    def fps_last(self) -> float:
        return 0.0 if self.last == 0 else 1.0 / self.last

    @property
    def fps_avg(self) -> float:
        return 0.0 if self.avg == 0 else 1.0 / self.avg

    def percentile(self, p: float) -> float:
        """
        Return the p-th percentile of recorded times.
        """
        if not self._history:
            return 0.0
        return float(np.percentile(self._history, p))

    def fps_percentile(self, p: float) -> float:
        """
        FPS corresponding to the p-th percentile time.
        """
        val = self.percentile(p)
        return 0.0 if val == 0 else 1.0 / val
