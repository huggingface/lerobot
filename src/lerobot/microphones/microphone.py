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

import abc
from pathlib import Path
from typing import Any

import numpy as np

from .configs import MicrophoneConfig


class Microphone(abc.ABC):
    """Base class for microphone implementations.

    Defines a standard interface for microphone operations across different backends.
    Subclasses must implement all abstract methods.

    Manages basic microphone properties (sample rate, channels) and core operations:
    - Connection/disconnection
    - Start/stop recording
    - Audio chunk reading

    Attributes:
        sample_rate (int | None): Configured sample rate in Hz
        channels (list[int] | None): List of channel numbers to record

    Example:
        class MyMicrophone(Microphone):
            def __init__(self, config): ...
            @property
            def is_connected(self) -> bool: ...
            def connect(self): ...
            # Plus other required methods
    """

    def __init__(self, config: MicrophoneConfig):
        """Initialize the microphone with the given configuration.

        Args:
            config: Microphone configuration containing sample rate and channels.
        """
        self.sample_rate: int | None = config.sample_rate
        self.channels: list[int] | None = config.channels

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if the microphone is currently connected.

        Returns:
            bool: True if the microphone is connected and ready to start recording,
                  False otherwise.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_microphones() -> list[dict[str, Any]]:
        """Detects available microphones connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains information about a detected microphone.
        """
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish connection to the microphone."""
        pass

    @abc.abstractmethod
    def start_recording(
        self,
        output_file: str | Path | None = None,
        multiprocessing: bool | None = False,
        overwrtie: bool | None = True,
    ) -> None:
        """Start recording audio from the microphone.

        Args:
            output_file: Optional path to save the recorded audio.
            multiprocessing: If True, enables multiprocessing for recording.
            overwrite: If True, overwrites existing files at output_file path.
        """
        pass

    @abc.abstractmethod
    def read(self) -> np.ndarray:
        """Capture and return a single audio chunk from the microphone.

        Returns:
            np.ndarray: Captured audio chunk as a numpy array.
        """
        pass

    @abc.abstractmethod
    def stop_recording(self) -> None:
        """Stop recording audio from the microphone."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect the microphone and release any resources."""
        pass
