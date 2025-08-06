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

from multiprocessing import Barrier
from threading import Thread

from .configs import MicrophoneConfig
from .microphone import Microphone


def make_microphones_from_configs(microphone_configs: dict[str, MicrophoneConfig]) -> dict[str, Microphone]:
    microphones = {}

    for key, cfg in microphone_configs.items():
        if cfg.type == "portaudio":
            from .portaudio import PortAudioMicrophone

            microphones[key] = PortAudioMicrophone(cfg)
        else:
            raise ValueError(f"The microphone type '{cfg.type}' is not valid.")

    return microphones


def async_microphones_start_recording(
    microphones: dict[str, Microphone],
    output_files: list[str | None] | None = None,
    multiprocessing: bool = False,
    overwrite: bool = True,
) -> None:
    """
    Starts recording on multiple microphones asynchronously to avoid delays.

    Args:
        microphones: A dictionary of microphones.
        output_files: A list of output files.
        multiprocessing: If True, enables multiprocessing for recording.
        overwrite: If True, overwrites existing files at output_file path.
    """

    start_recording_threads = []
    if output_files is None:
        output_files = [None] * len(microphones)

    barrier = Barrier(len(microphones))

    for microphone, output_file in zip(microphones.values(), output_files, strict=False):
        start_recording_threads.append(
            Thread(target=microphone.start_recording, args=(output_file, multiprocessing, overwrite, barrier))
        )

    for thread in start_recording_threads:
        thread.start()
    for thread in start_recording_threads:
        thread.join()


def async_microphones_stop_recording(microphones: dict[str, Microphone]) -> None:
    """
    Stops recording on multiple microphones asynchronously to avoid delays.

    Args:
        microphones: A dictionary of microphones.
    """

    stop_recording_threads = []

    for microphone in microphones.values():
        stop_recording_threads.append(Thread(target=microphone.stop_recording))

    for thread in stop_recording_threads:
        thread.start()
    for thread in stop_recording_threads:
        thread.join()
