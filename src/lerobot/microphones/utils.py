# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Helpers to create microphones from configurations and to drive several microphones together."""

import time
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Thread
from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .configs import MicrophoneConfig
from .microphone import Microphone


def make_microphones_from_configs(microphone_configs: dict[str, MicrophoneConfig]) -> dict[str, Microphone]:
    """Instantiate microphones from their configurations.

    Third-party microphones (installed as `lerobot_microphone_*` packages) are created from their
    configuration class name, like third-party cameras.

    Args:
        microphone_configs (`dict[str, MicrophoneConfig]`):
            Microphone configurations keyed by name.

    Returns:
        `dict[str, Microphone]`: The (not yet connected) microphones, keyed by the same names.

    Raises:
        ValueError: If a microphone cannot be created from its configuration.
    """
    microphones: dict[str, Microphone] = {}

    for key, cfg in microphone_configs.items():
        if cfg.type == "portaudio":
            from .portaudio import PortAudioMicrophone

            microphones[key] = PortAudioMicrophone(cfg)

        else:
            try:
                microphones[key] = cast(Microphone, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating microphone {key} with config {cfg}: {e}") from e

    return microphones


def async_microphones_start_recording(
    microphones: dict[str, Microphone],
    output_files: list[str | Path | None] | None = None,
    multiprocessing: bool = False,
    overwrite: bool = True,
) -> None:
    """Start recording on several microphones at the same time.

    Each microphone is started from its own thread, and a shared barrier makes them all start capturing
    together. If one of them fails to start, the others are stopped and the error is raised.

    Args:
        microphones (`dict[str, Microphone]`):
            Connected microphones keyed by name.
        output_files (`list[str | Path | None]`, *optional*):
            One output file per microphone, in the order of `microphones`. Nothing is saved if `None`.
        multiprocessing (`bool`, *optional*, defaults to `False`):
            Write the files from separate processes instead of threads.
        overwrite (`bool`, *optional*, defaults to `True`):
            Overwrite the output files if they already exist.

    Raises:
        ValueError: If the number of output files does not match the number of microphones.
        RuntimeError: If a microphone fails to start recording.
    """
    if output_files is None:
        output_files = [None] * len(microphones)
    elif len(output_files) != len(microphones):
        raise ValueError(
            f"Expected one output file per microphone ({len(microphones)}), got {len(output_files)}."
        )

    barrier = Barrier(len(microphones))
    errors: dict[str, BaseException] = {}

    def start_recording(key: str, microphone: Microphone, output_file: str | Path | None) -> None:
        try:
            microphone.start_recording(output_file, multiprocessing, overwrite, barrier)
        except BaseException as e:
            # Release the microphones waiting on the barrier instead of letting them hang.
            barrier.abort()
            errors[key] = e

    _run_in_threads(
        [
            Thread(target=start_recording, args=(key, microphone, output_file))
            for (key, microphone), output_file in zip(microphones.items(), output_files, strict=True)
        ]
    )
    _raise_first_error(errors, "start recording")


def async_microphones_stop_recording(microphones: dict[str, Microphone]) -> None:
    """Stop recording on several microphones in parallel.

    Args:
        microphones (`dict[str, Microphone]`):
            Recording microphones keyed by name.

    Raises:
        RuntimeError: If a microphone fails to stop recording.
    """
    errors: dict[str, BaseException] = {}

    def stop_recording(key: str, microphone: Microphone) -> None:
        try:
            microphone.stop_recording()
        except BaseException as e:
            errors[key] = e

    _run_in_threads(
        [Thread(target=stop_recording, args=(key, microphone)) for key, microphone in microphones.items()]
    )
    _raise_first_error(errors, "stop recording")


def wait_for_first_sample_time(microphone: Microphone, timeout_s: float = 1.0) -> float:
    """Return the capture time of the first sample of the current recording.

    Call it right after `start_recording`: it reads the samples captured so far, which does not affect the
    file being written, and dates the first one from the capture time of the last one.

    Args:
        microphone (`Microphone`):
            A recording microphone.
        timeout_s (`float`, *optional*, defaults to `1.0`):
            Maximum time to wait for the first samples.

    Returns:
        `float`: The capture time of the first sample, in the `time.perf_counter()` clock.

    Raises:
        TimeoutError: If no sample is captured within `timeout_s`.
    """
    if microphone.sample_rate is None:
        raise ValueError(f"{microphone} has no sample rate.")
    num_samples = 0
    deadline = time.perf_counter() + timeout_s
    while True:
        num_samples += len(microphone.read())
        if num_samples > 0 and microphone.last_read_timestamp is not None:
            return microphone.last_read_timestamp - (num_samples - 1) / microphone.sample_rate
        if time.perf_counter() > deadline:
            raise TimeoutError(f"{microphone} captured no audio within {timeout_s} s.")
        time.sleep(0.005)


def _run_in_threads(threads: list[Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _raise_first_error(errors: dict[str, BaseException], action: str) -> None:
    """Re-raises in the calling thread an error raised by one of the per-microphone threads."""
    # A BrokenBarrierError only means that another microphone failed first.
    root_errors = {key: e for key, e in errors.items() if not isinstance(e, BrokenBarrierError)} or errors
    if root_errors:
        key, error = next(iter(root_errors.items()))
        raise RuntimeError(
            f"Failed to {action} on microphone(s) {sorted(errors)}: {key}: {error!r}"
        ) from error
