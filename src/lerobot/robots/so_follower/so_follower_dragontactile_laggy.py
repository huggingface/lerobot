#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property
import time

import cv2
import numpy as np
import scipy.signal

from lerobot.types import RobotObservation

from .config_so_follower import SO101FollowerDragontactileConfig
from .so_follower import SOFollower

logger = logging.getLogger(__name__)


class SO101FollowerDragontactile(SOFollower):
    """SO101 follower with an additional tactile spectrogram observation."""

    config_class = SO101FollowerDragontactileConfig
    name = "so101_follower_dragontactile"

    def __init__(self, config: SO101FollowerDragontactileConfig):
        super().__init__(config)
        self._tactile_obs_key = "left_tactile_spectrogram"

        self._sampling_rate = 20_000 # fs = 20 kHz

        self._crop_data = 10 # crop the data with this factor from 0 Hz to 10/self._crop_data kHz
        self._nfft = 1024
        self._width, self._height = 224, 224 # For ResNet
        self._target_size = (self._width, self._height)

        self._df=self._sampling_rate/2/self._height
        dt=1/(2*self._df) # *2 because of 50% overlap
        self._window_duration = self._width*dt
        
        # Fixed color scale for stable spectrogram visualization across frames.
        self._spectrogram_min_db = -70.0
        self._spectrogram_max_db = 40.0

        display_buffer_size = int(self._sampling_rate * self._window_duration)
        self._display_buffer = np.zeros(display_buffer_size, dtype=np.float32)
        self._last_spectrogram_frame: np.ndarray | None = None

        self._instance = None
        self._reader = None
        self._init_tactile_reader()



    def _init_tactile_reader(self) -> None:
        try:
            import opendaq
        except ImportError:
            logger.warning("opendaq is not installed. Tactile spectrogram stream is disabled.")
            return
        try:
            self._instance = opendaq.Instance()
            available_devices = list(self._instance.available_devices)
            if not available_devices:
                logger.warning("No openDAQ devices found. Tactile spectrogram stream is disabled.")
                return

            target = next((d for d in available_devices if "IOLITE-X" in d.name), available_devices[0])
            device = self._instance.add_device(target.connection_string)

            channel = device.channels[0]  # first channel of IOLITE-X
            signal = channel.signals[0]  # main data stream
            amplifier = channel.get_function_blocks()[0]  # contains the amplifier settings
            self._configure_iepe_amplifier(amplifier)

            try:
                device.set_property_value("SampleRate", self._sampling_rate)
            except Exception:
                logger.warning("Could not set SampleRate to %s on tactile device.", self._sampling_rate)

            self._reader = opendaq.StreamReader(signal)
            logger.info("Connected tactile stream from %s.", target.name)
        except Exception as exc:  # nosec B110
            logger.warning("Failed to initialize tactile stream: %s", exc)
            self._reader = None


    @staticmethod
    def _configure_iepe_amplifier(amplifier) -> None:   # Fnction to upgrade : give sensor and it returns settings for gain 1 10 100
        try:
            amplifier.set_property_value("Measurement", 1)  # IEPE
            amplifier.set_property_value("Range", 0)  # 10V
            amplifier.set_property_value("HPFilter", 0)  # 0.1Hz
            amplifier.set_property_value("Excitation", 1)  # 4mA
        except Exception:
            logger.debug("Could not configure IEPE amplifier with default settings.")

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = dict(super().observation_features)
        # expose the original single spectrogram key for backward compatibility
        # and two suffixed keys for separate views (full-band and low-band).
        features[self._tactile_obs_key] = (self._target_size[1], self._target_size[0], 3)
        features[self._tactile_obs_key + "0-10kHz"] = (self._target_size[1], self._target_size[0], 3)
        features[self._tactile_obs_key + "0-1kHz"] = (self._target_size[1], self._target_size[0], 3)

        # downsample_factor = 1000  # Downsample the time-series data for the tactile display to reduce dimensionality -> 20 Fps
        # lite_size = len(self._display_buffer[::downsample_factor]) 
        # features["tactile_display_lite"] = (lite_size,)
        return features

    def _read_tactile_spectrogram(self) -> np.ndarray | None:
        if self._reader is None:
            return self._last_spectrogram_frame

        available = self._reader.available_count
        if available <= 0:
            return self._last_spectrogram_frame

        chunk = np.asarray(self._reader.read(available), dtype=np.float32)
        if chunk.size == 0:
            return self._last_spectrogram_frame

        if chunk.size < self._display_buffer.size:
            self._display_buffer = np.roll(self._display_buffer, -chunk.size)
            self._display_buffer[-chunk.size :] = chunk
        else:
            self._display_buffer[:] = chunk[-self._display_buffer.size :]

        nperseg = min(self._nfft, self._display_buffer.size)
        noverlap = nperseg // 2
        if nperseg < 16:
            return self._last_spectrogram_frame

        _, _, sxx = scipy.signal.spectrogram(
            self._display_buffer,
            fs=self._sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        # original spectrogram (frequency x time)
        original_height = sxx.shape[0]

        # create full-band spectrogram image
        sxx_db_full = 10.0 * np.log10(sxx + 1e-12)

        # create low-band spectrogram by cropping low frequencies and then upscaling
        cropped_height = max(1, int(original_height / self._crop_data))
        sxx_low = sxx[:cropped_height, :]
        # upscale low-band to match original_height (freq axis) using linear interpolation
        # cv2.resize expects (width, height) -> (time, freq)
        sxx_low_up = cv2.resize(
            sxx_low.astype(np.float32),
            (sxx_low.shape[1], original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        sxx_db_low = 10.0 * np.log10(sxx_low_up + 1e-12)

        spectrograms: list[np.ndarray] = []
        for sxx_db in (sxx_db_full, sxx_db_low):
            normalized = (
                np.clip(sxx_db, self._spectrogram_min_db, self._spectrogram_max_db) - self._spectrogram_min_db
            ) / max(self._spectrogram_max_db - self._spectrogram_min_db, 1e-6)
            image_32bit = np.uint8(np.flipud(normalized) * 255.0)
            spectro_bgr = cv2.cvtColor(image_32bit, cv2.COLOR_GRAY2BGR)
            spectro_bgr = cv2.resize(spectro_bgr, self._target_size, interpolation=cv2.INTER_LINEAR)
            spectro_rgb = cv2.cvtColor(spectro_bgr, cv2.COLOR_BGR2RGB)
            spectrograms.append(spectro_rgb)

        # cache last frame as the full-band image (useful for fallbacks)
        if spectrograms:
            self._last_spectrogram_frame = spectrograms[0]
        return spectrograms

    def get_observation(self) -> RobotObservation:
        tick_start = time.perf_counter()

        # sensor_sensitivity_voltage_per_unit=10.8
        # tactile_value_mum_m = np.array([self._display_buffer[-1]/sensor_sensitivity_voltage_per_unit], dtype=np.float32)
        
        spectrograms = self._read_tactile_spectrogram()
        obs = super().get_observation()

        # 4. Assemble the synchronized dictionary
        # obs["tactile_display_"] = tactile_value_mum_m
        
        if spectrograms is not None:
            # Keep the original single-key observation for backward compatibility
            # (used by dataset writers / display). Use the full-band image for it.
            obs[self._tactile_obs_key] = spectrograms[0]
            obs[self._tactile_obs_key + "0-10kHz"] = spectrograms[0]
            # Provide low-band view under a separate key for optional visualization
            if len(spectrograms) > 1:
                obs[self._tactile_obs_key + "0-1kHz"] = spectrograms[1]

        total_latency = (time.perf_counter() - tick_start) * 1000
        logger.debug(f"Observation tick sync completed in {total_latency:.2f}ms")
        return obs