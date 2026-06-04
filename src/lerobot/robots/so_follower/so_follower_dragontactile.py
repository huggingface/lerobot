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

from .config_so_follower import SO101FollowerConfig
from .so_follower import SOFollower

logger = logging.getLogger(__name__)


class SO101FollowerDragontactile(SOFollower):
    """SO101 follower with an additional tactile spectrogram observation."""

    config_class = SO101FollowerConfig
    name = "so101_follower_dragontactile"

    def __init__(self, config: SO101FollowerConfig):
        super().__init__(config)
        self._tactile_obs_key = "left_tactile_spectrogram"

        self._sampling_rate_hz = 20_000 # fs = 20 kHz

        self._crop_data = 10 # crop the data with this factor from 0 Hz to 10/self._crop_data kHz
        self._begin_crop_freq = 0 # crop from this frequency (Hz)
        self._nfft = 1024
        self._width, self._height = 224, 224 # For ResNet
        self._target_size = (self._width, self._height)

        self._df=self._sampling_rate_hz/2/self._height
        dt=1/(2*self._df) # *2 because of 50% overlap
        self._window_duration = self._width*dt
        
        # Fixed color scale for stable spectrogram visualization across frames.
        self._spectrogram_min_db = -70.0
        self._spectrogram_max_db = 40.0

        display_buffer_size = int(self._sampling_rate_hz * self._window_duration)
        self._display_buffer = np.zeros(display_buffer_size, dtype=np.float32)
        self._last_spectrogram_frame: np.ndarray | None = None

        self._instance = None
        self._reader = None
        self._init_tactile_reader()

    def _infer_target_size_from_camera_config(self) -> tuple[int, int]:
        """Returns target (width, height) from the first configured camera, with safe fallback."""
        if not self.config.cameras:
            return (400, 300)

        first_camera_cfg = next(iter(self.config.cameras.values()))
        width = int(first_camera_cfg.width) if first_camera_cfg.width is not None else 400
        height = int(first_camera_cfg.height) if first_camera_cfg.height is not None else 300
        return (width, height)

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

            channel = device.channels[0]
            signal = channel.signals[0]
            amplifier = channel.get_function_blocks()[0]
            self._configure_iepe_amplifier(amplifier)

            try:
                device.set_property_value("SampleRate", self._sampling_rate_hz)
            except Exception:
                logger.warning("Could not set SampleRate to %s on tactile device.", self._sampling_rate_hz)

            self._reader = opendaq.StreamReader(signal)
            logger.info("Connected tactile stream from %s.", target.name)
        except Exception as exc:  # nosec B110
            logger.warning("Failed to initialize tactile stream: %s", exc)
            self._reader = None

    @staticmethod
    def _configure_iepe_amplifier(amplifier) -> None:
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

        frequencies, _, sxx = scipy.signal.spectrogram(
            self._display_buffer,
            fs=self._sampling_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        nyquist = self._sampling_rate / 2.0
        upper_crop_freq = min(nyquist, self._begin_crop_freq + nyquist / max(self._crop_data, 1e-6))
        start_bin = int(np.searchsorted(frequencies, self._begin_crop_freq, side="left"))
        end_bin = int(np.searchsorted(frequencies, upper_crop_freq, side="right"))
        if end_bin <= start_bin:
            return self._last_spectrogram_frame

        cropped = np.zeros_like(sxx)
        cropped[start_bin:end_bin, :] = sxx[start_bin:end_bin, :]
        sxx = cropped

        sxx_db = 10.0 * np.log10(sxx + 1e-12)
        normalized = np.clip(sxx_db, self._spectrogram_min_db, self._spectrogram_max_db)
        normalized = (normalized - self._spectrogram_min_db) / max(self._spectrogram_max_db - self._spectrogram_min_db, 1e-6)
        image_32bit = np.uint8(np.flipud(normalized) * 255.0)
        spectro_bgr = cv2.cvtColor(image_32bit, cv2.COLOR_GRAY2BGR)
        spectro_bgr = cv2.resize(spectro_bgr, self._target_size, interpolation=cv2.INTER_LINEAR)
        spectro_rgb = cv2.cvtColor(spectro_bgr, cv2.COLOR_BGR2RGB)
        self._last_spectrogram_frame = spectro_rgb
        return spectro_rgb

    def get_observation(self) -> RobotObservation:
        tick_start = time.perf_counter()

        # sensor_sensitivity_voltage_per_unit=10.8
        # tactile_value_mum_m = np.array([self._display_buffer[-1]/sensor_sensitivity_voltage_per_unit], dtype=np.float32)
        
        spectrogram = self._read_tactile_spectrogram()
        obs = super().get_observation()

        # 4. Assemble the synchronized dictionary
        # obs["tactile_display_"] = tactile_value_mum_m
        
        if spectrogram is not None:
            obs[self._tactile_obs_key] = spectrogram

        total_latency = (time.perf_counter() - tick_start) * 1000
        logger.debug(f"Observation tick sync completed in {total_latency:.2f}ms")
        return obs