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
from dataclasses import dataclass

from torch import Tensor
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import Compose, Lambda, Resize

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
    """

    # TODO(CarolinePascal) : add variable parametrization
    mel_spectrogram_transform = Compose(
        [
            Lambda(lambda x: x.mean(dim=1)),  # Average over all channels (second dimension after batch)
            Resample(
                orig_freq=48000, new_freq=16000
            ),  # Subsampling (less samples, reduced temporal resolution, lower frequency range)
            MelSpectrogram(
                sample_rate=16000,  # Subsampling (less samples, reduced temporal resolution, lower frequency range)
                n_fft=1024,  # FFT window size (the bigger the window, the more frequency information, the lower the temporal resolution)
                hop_length=36,  # Number of samples between frames (the smaller the hop, the higher the temporal resolution) - Value picked to match ResNet18 input and a 0.5s input
                n_mels=224,  # Number of Mel bands (the more bands, the more rows in the spectrogram, the higher the frequency resolution)
                power=2,  # Power spectrum
            ),
            Lambda(
                lambda x: amplitude_to_DB(x, multiplier=10, amin=1e-10, db_multiplier=0)
            ),  # Convert to decibels
            Resize((224, 224)),  # Resize spectrogram to 224×224
            Lambda(
                lambda x: x.unsqueeze(1).expand(-1, 3, -1, -1)
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
