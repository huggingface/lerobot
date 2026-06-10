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


class SO101FollowerDragonMultitactile(SOFollower):
    """SO101 follower with an additional tactile spectrogram observation."""

    config_class = SO101FollowerConfig
    name = "so101_follower_dragon_multitactile"

    def __init__(self, config: SO101FollowerConfig):
        super().__init__(config)
        self._200k_obs_key = "100kHz_tactile_spectrogram"
        self._20k_obs_key = "10kHz_tactile_spectrogram"
        self._01k_obs_key = "0_1kHz_tactile_spectrogram"
        self._56k_obs_key = "5_6kHz_tactile_spectrogram"

        self._sampling_rate_hz = 200_000  # 200 kHz stream from openDAQ.
        self._sampling_rate_hz_20 = 20_000
        self._downsample_factor_20 = self._sampling_rate_hz // self._sampling_rate_hz_20


        self._nfft = 1024
        self._width, self._height = 224, 224 # For ResNet
        self._target_size = (self._width, self._height)

        self._df_200=self._sampling_rate_hz/2/self._height
        self._df_20=self._sampling_rate_hz_20/2/self._height
        dt_200=1/(2*self._df_200) # *2 because of 50% overlap
        dt_20=1/(2*self._df_20)
        self._window_duration_200 = self._width*dt_200
        self._window_duration_20 = self._width*dt_20

        # Fixed color scale for stable spectrogram visualization across frames.
        self._spectrogram_min_db = -70.0
        self._spectrogram_max_db = 40.0

        display_buffer_size_200 = int(self._sampling_rate_hz * self._window_duration_200)
        self._display_buffer_200 = np.zeros(display_buffer_size_200, dtype=np.float32)
        self._last_spectrogram_frame_200: np.ndarray | None = None

        display_buffer_size_20 = int(self._sampling_rate_hz_20 * self._window_duration_20)
        self._display_buffer_20 = np.zeros(display_buffer_size_20, dtype=np.float32)
        self._last_spectrogram_frame_20: np.ndarray | None = None
        self._last_spectrogram_frame_01k: np.ndarray | None = None
        self._last_spectrogram_frame_56k: np.ndarray | None = None
        self._downsample_remainder_20 = np.zeros(0, dtype=np.float32)

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
        features[self._200k_obs_key] = (self._target_size[1], self._target_size[0], 3)
        features[self._20k_obs_key] = (self._target_size[1], self._target_size[0], 3)
        features[self._01k_obs_key] = (self._target_size[1], self._target_size[0], 3)
        features[self._56k_obs_key] = (self._target_size[1], self._target_size[0], 3)

        # downsample_factor = 1000  # Downsample the time-series data for the tactile display to reduce dimensionality -> 20 Fps
        # lite_size = len(self._display_buffer[::downsample_factor]) 
        # features["tactile_display_lite"] = (lite_size,)
        return features

    @staticmethod
    def _push_to_ring_buffer(buffer: np.ndarray, chunk: np.ndarray) -> None:
        if chunk.size >= buffer.size:
            buffer[:] = chunk[-buffer.size :]
            return

        buffer[:] = np.roll(buffer, -chunk.size)
        buffer[-chunk.size :] = chunk

    def _mean_downsample_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if self._downsample_remainder_20.size:
            chunk = np.concatenate((self._downsample_remainder_20, chunk))

        usable_size = (chunk.size // self._downsample_factor_20) * self._downsample_factor_20
        if usable_size == 0:
            self._downsample_remainder_20 = chunk
            return np.zeros(0, dtype=np.float32)

        downsampled = chunk[:usable_size].reshape(-1, self._downsample_factor_20).mean(axis=1).astype(np.float32)
        self._downsample_remainder_20 = chunk[usable_size:]
        return downsampled

    def _render_spectrogram_frame(self, buffer: np.ndarray, fs_hz: int) -> np.ndarray | None:
        nperseg = min(self._nfft, buffer.size)
        if nperseg < 16:
            return None

        noverlap = nperseg // 2
        _, _, sxx = scipy.signal.spectrogram(buffer, fs=fs_hz, nperseg=nperseg, noverlap=noverlap)

        return self._render_sxx_frame(sxx)

    def _render_sxx_frame(self, sxx: np.ndarray) -> np.ndarray | None:
        if sxx.size == 0:
            return None

        sxx_db = 10.0 * np.log10(sxx + 1e-12)
        normalized = np.clip(sxx_db, self._spectrogram_min_db, self._spectrogram_max_db)
        normalized = (normalized - self._spectrogram_min_db) / max(
            self._spectrogram_max_db - self._spectrogram_min_db,
            1e-6,
        )
        image_8bit = np.uint8(np.flipud(normalized) * 255.0)
        spectro_bgr = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        spectro_bgr = cv2.resize(spectro_bgr, self._target_size, interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(spectro_bgr, cv2.COLOR_BGR2RGB)

    def _crop_sxx_frequency_band(
        self,
        frequencies: np.ndarray,
        sxx: np.ndarray,
        begin_crop_freq: float,
        end_crop_freq: float,
    ) -> np.ndarray | None:
        start_bin = int(np.searchsorted(frequencies, begin_crop_freq, side="left"))
        end_bin = int(np.searchsorted(frequencies, end_crop_freq, side="right"))
        if end_bin <= start_bin:
            return None

        cropped = np.zeros_like(sxx)
        cropped[start_bin:end_bin, :] = sxx[start_bin:end_bin, :]
        return cropped

    def _read_tactile_spectrogram(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self._reader is None:
            if (
                self._last_spectrogram_frame_200 is None
                or self._last_spectrogram_frame_20 is None
                or self._last_spectrogram_frame_01k is None
                or self._last_spectrogram_frame_56k is None
            ):
                return None
            return (
                self._last_spectrogram_frame_200,
                self._last_spectrogram_frame_20,
                self._last_spectrogram_frame_01k,
                self._last_spectrogram_frame_56k,
            )

        available = self._reader.available_count
        if available <= 0:
            if (
                self._last_spectrogram_frame_200 is None
                or self._last_spectrogram_frame_20 is None
                or self._last_spectrogram_frame_01k is None
                or self._last_spectrogram_frame_56k is None
            ):
                return None
            return (
                self._last_spectrogram_frame_200,
                self._last_spectrogram_frame_20,
                self._last_spectrogram_frame_01k,
                self._last_spectrogram_frame_56k,
            )

        chunk = np.asarray(self._reader.read(available), dtype=np.float32)
        if chunk.size == 0:
            if (
                self._last_spectrogram_frame_200 is None
                or self._last_spectrogram_frame_20 is None
                or self._last_spectrogram_frame_01k is None
                or self._last_spectrogram_frame_56k is None
            ):
                return None
            return (
                self._last_spectrogram_frame_200,
                self._last_spectrogram_frame_20,
                self._last_spectrogram_frame_01k,
                self._last_spectrogram_frame_56k,
            )

        self._push_to_ring_buffer(self._display_buffer_200, chunk)
        chunk_20 = self._mean_downsample_chunk(chunk)
        if chunk_20.size:
            self._push_to_ring_buffer(self._display_buffer_20, chunk_20)

        spectrogram_200 = self._render_spectrogram_frame(self._display_buffer_200, self._sampling_rate_hz)
        nperseg_20 = min(self._nfft, self._display_buffer_20.size)
        if nperseg_20 < 16:
            spectrogram_20 = None
            spectrogram_01k = None
            spectrogram_56k = None
        else:
            noverlap_20 = nperseg_20 // 2
            frequencies_20, _, sxx_20 = scipy.signal.spectrogram(
                self._display_buffer_20,
                fs=self._sampling_rate_hz_20,
                nperseg=nperseg_20,
                noverlap=noverlap_20,
            )
            spectrogram_20 = self._render_sxx_frame(sxx_20)
            cropped_01k = self._crop_sxx_frequency_band(frequencies_20, sxx_20, 0.0, 1_000.0)
            cropped_56k = self._crop_sxx_frequency_band(frequencies_20, sxx_20, 5_000.0, 6_000.0)
            spectrogram_01k = self._render_sxx_frame(cropped_01k) if cropped_01k is not None else None
            spectrogram_56k = self._render_sxx_frame(cropped_56k) if cropped_56k is not None else None

        if spectrogram_200 is not None:
            self._last_spectrogram_frame_200 = spectrogram_200
        if spectrogram_20 is not None:
            self._last_spectrogram_frame_20 = spectrogram_20
        if spectrogram_01k is not None:
            self._last_spectrogram_frame_01k = spectrogram_01k
        if spectrogram_56k is not None:
            self._last_spectrogram_frame_56k = spectrogram_56k

        if (
            self._last_spectrogram_frame_200 is None
            or self._last_spectrogram_frame_20 is None
            or self._last_spectrogram_frame_01k is None
            or self._last_spectrogram_frame_56k is None
        ):
            return None

        return (
            self._last_spectrogram_frame_200,
            self._last_spectrogram_frame_20,
            self._last_spectrogram_frame_01k,
            self._last_spectrogram_frame_56k,
        )

    def get_observation(self) -> RobotObservation:
        tick_start = time.perf_counter()

        spectrograms = self._read_tactile_spectrogram()
        obs = super().get_observation()

        if spectrograms is not None:
            obs[self._200k_obs_key] = spectrograms[0]
            obs[self._20k_obs_key] = spectrograms[1]
            obs[self._01k_obs_key] = spectrograms[2]
            obs[self._56k_obs_key] = spectrograms[3]

        total_latency = (time.perf_counter() - tick_start) * 1000
        logger.debug(f"Observation tick sync completed in {total_latency:.2f}ms")
        return obs