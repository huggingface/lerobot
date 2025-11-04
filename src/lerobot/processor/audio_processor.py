#!/usr/bin/env python

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
from dataclasses import dataclass, field

from torch import Tensor
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import Compose, Lambda, Resize

from lerobot.datasets.utils import DEFAULT_AUDIO_CHUNK_DURATION
from lerobot.utils.constants import OBS_AUDIO

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="audio_processor")
class AudioProcessorStep(ObservationProcessorStep):
    """
    Processes audio waveform data into a mel-spectrogram image representation.

    **Audio Processing:**
    -   Averages waveform data over all channels.
    -   Resamples the waveform to 16kHz.
    -   Converts the waveform to a mel-spectrogram.
    -   Converts the mel-spectrogram to decibels.
    -   Resizes the mel-spectrogram to 224×224.
    -   Converts the mel-spectrogram to a channel-first, normalized tensor.

    Attributes:
        output_height: Height of the output mel-spectrogram image in pixels.
        output_width: Width of the output mel-spectrogram image in pixels.
        output_channels: Number of channels in the output image (3 for RGB-like format).
        input_audio_chunk_duration: Duration of the input audio chunk in seconds.
        input_sample_rate: Original sample rate of the input audio in Hz.

        intermediate_sample_rate: Reduced intermediate sample rate in Hz.
                                  Downsampling improves the temporal resolution but reduces the frequency range.
        n_fft: Size of the FFT window for spectrogram computation.
               Increasing the window size increases the frequency resolution but decreases the temporal resolution.

        hop_length: Number of samples between successive frames, computed automatically to match the output_width.
                    Decreasing the hop length increases the temporal resolution but decreases the frequency resolution.
        n_mels: Number of mel filter banks, computed automatically to match the output_height.
                Increasing the number of banks increases the number of rows in the spectrogram and the frequency resolution.
        mel_spectrogram_transform: The complete audio processing pipeline.
    """

    output_height: int = 224
    output_width: int = 224
    output_channels: int = 3
    input_audio_chunk_duration: float = DEFAULT_AUDIO_CHUNK_DURATION

    input_sample_rate: int = 48000
    intermediate_sample_rate: int = 16000

    n_fft: int = 1024

    # Parameters computed from other parameters at initialization
    hop_length: int = field(init=False)
    n_mels: int = field(init=False)
    mel_spectrogram_transform: Compose = field(init=False, repr=False)

    def __post_init__(self):
        self.hop_length = int(
            self.intermediate_sample_rate * self.input_audio_chunk_duration
            - self.n_fft // self.output_width
            - 1
        )
        self.n_mels = self.output_height

        self.mel_spectrogram_transform = Compose(
            [
                Lambda(lambda x: x.mean(dim=1)),  # Average over all channels (second dimension after batch)
                Resample(orig_freq=self.input_sample_rate, new_freq=self.intermediate_sample_rate),
                MelSpectrogram(
                    sample_rate=self.intermediate_sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    power=2,  # Power spectrum
                ),
                Lambda(
                    lambda x: amplitude_to_DB(x, multiplier=10, amin=1e-10, db_multiplier=0)
                ),  # Convert to decibels
                Resize(
                    (self.output_height, self.output_width)
                ),  # Resize spectrogram to output_height×output_width
                Lambda(
                    lambda x: x.unsqueeze(1).expand(-1, self.output_channels, -1, -1)
                ),  # Duplicate across 3 channels to mimic RGB images. Dimensions are [batch, rgb, height, width].
            ]
        )

    def _process_observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Processes audio data contained in the provided observation.
        """
        processed_obs = observation.copy()

        # Process single audio observation
        if OBS_AUDIO in processed_obs:
            audio_data = processed_obs[OBS_AUDIO]
            if isinstance(audio_data, Tensor) and audio_data.dim() == 3:  # Batch, Channels, Samples
                processed_obs[OBS_AUDIO] = self.mel_spectrogram_transform(audio_data)

        # Process multiple audio observations
        for key, value in processed_obs.items():
            if (
                key.startswith(f"{OBS_AUDIO}.") and isinstance(value, Tensor) and value.dim() == 3
            ):  # Batch, Channels, Samples
                processed_obs[key] = self.mel_spectrogram_transform(value)

        return processed_obs

    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        return self._process_observation(observation)
