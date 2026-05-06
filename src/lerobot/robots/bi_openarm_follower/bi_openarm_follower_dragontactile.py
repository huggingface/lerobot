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

from ..openarm_follower import OpenFollowerDragonTacile
from .bi_openarm_follower import BiOpenArmFollower
from .config_bi_openarm_follower import BiOpenArmFollowerConfig

logger = logging.getLogger(__name__)


class BiOpenFollowerDragonTacile(BiOpenArmFollower) :

    config_class = BiOpenArmFollowerConfig
    name = "bi_openarm_follower_dragontactile"

    def __init__(self, config: BiOpenArmFollowerConfig):

        super().__init__(config)



        # 2. Re-initialize the Right Arm with your custom class
        right_arm_config = self.right_arm.config 
        
        # Replace the standard follower with your Tactile-enabled follower class
        self.right_arm = OpenFollowerDragonTacile(right_arm_config)

        # 3. Initialize the hardware reader
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
        features[self._tactile_obs_key] = (self._target_size[1], self._target_size[0], 3)

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
        sxx_db = 10.0 * np.log10(sxx + 1e-12)
        normalized = (
            np.clip(sxx_db, self._spectrogram_min_db, self._spectrogram_max_db) - self._spectrogram_min_db
        ) / self._diff_max_min_db
        image_32bit = np.uint8(np.flipud(normalized) * 255.0)
        spectro_bgr = cv2.cvtColor(image_32bit, cv2.COLOR_GRAY2BGR)
        spectro_bgr = cv2.resize(spectro_bgr, self._target_size, interpolation=cv2.INTER_LINEAR)
        spectro_rgb = cv2.cvtColor(spectro_bgr, cv2.COLOR_BGR2RGB)
        self._last_spectrogram_frame = spectro_rgb
        return spectro_rgb

    def get_observation(self) -> RobotObservation:

        sensor_sensitivity_voltage_per_unit=10.8


        spectrogram = self._read_tactile_spectrogram()
        tactile_value_mum_m = np.array([self._display_buffer[-1]/sensor_sensitivity_voltage_per_unit], dtype=np.float32)

        obs = super().get_observation()

        # 4. Assemble the synchronized dictionary
        # obs["tactile_display_"] = tactile_value_mum_m
        
        if spectrogram is not None:
            start = time.perf_counter()
            obs[self._tactile_obs_key] = spectrogram
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {self._tactile_obs_key}: {dt_ms:.1f}ms")

        return obs