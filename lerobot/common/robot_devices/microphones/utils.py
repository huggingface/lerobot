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

from queue import Queue
from threading import Thread
from typing import Protocol

from lerobot.common.robot_devices.microphones.configs import MicrophoneConfig, PortAudioMicrophoneConfig


# Defines a microphone type
class Microphone(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def start_recording(
        self,
        output_file: str | None = None,
        multiprocessing: bool | None = False,
        overwrite: bool | None = True,
    ): ...
    def stop_recording(self): ...
    def read(self): ...


def make_microphones_from_configs(microphone_configs: dict[str, MicrophoneConfig]) -> list[Microphone]:
    microphones = {}

    for key, cfg in microphone_configs.items():
        if cfg.type == "portaudio":
            from lerobot.common.robot_devices.microphones.portaudio import PortAudioMicrophone

            microphones[key] = PortAudioMicrophone(cfg)
        else:
            raise ValueError(f"The microphone type '{cfg.type}' is not valid.")

    return microphones


def make_microphone(microphone_type, **kwargs) -> Microphone:
    if microphone_type == "portaudio":
        from lerobot.common.robot_devices.microphones.portaudio import PortAudioMicrophone

        config = PortAudioMicrophoneConfig(**kwargs)
        return PortAudioMicrophone(config)
    else:
        raise ValueError(f"The microphone type '{microphone_type}' is not valid.")


def async_microphones_start_recording(
    microphones: dict[str, Microphone],
    output_files: list[str | None] | None,
    multiprocessing: bool = False,
    overwrite: bool = True,
):
    """
    Starts recording on multiple microphones asynchronously to avoid delays
    """

    start_recording_threads = []
    if output_files is None:
        output_files = [None] * len(microphones)

    for microphone, output_file in zip(microphones.values(), output_files, strict=False):
        start_recording_threads.append(
            Thread(target=microphone.start_recording, args=(output_file, multiprocessing, overwrite))
        )

    for thread in start_recording_threads:
        thread.start()
    for thread in start_recording_threads:
        thread.join()


def async_microphones_stop_recording(microphones: dict[str, Microphone]):
    """
    Stops recording on multiple microphones asynchronously to avoid delays
    """

    stop_recording_threads = []

    for microphone in microphones.values():
        stop_recording_threads.append(Thread(target=microphone.stop_recording))

    for thread in stop_recording_threads:
        thread.start()
    for thread in stop_recording_threads:
        thread.join()


def async_microphones_read(microphones: dict[str, Microphone]):
    """
    Reads from multiple microphones asynchronously to avoid delays
    """

    read_threads = []
    read_queue = Queue()

    for microphone_key, microphone in microphones.items():
        read_threads.append(
            Thread(
                target=lambda microphone, output, microphone_key: output.put_nowait(
                    {microphone_key: microphone.read()}
                ),
                args=(microphone, read_queue, microphone_key),
            )
        )

    for thread in read_threads:
        thread.start()
    for thread in read_threads:
        thread.join()

    return dict(kv for d in read_queue.queue for kv in d.items())
