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
import time
import threading
from functools import cached_property

import torch
import torch.nn.functional as F
import numpy as np

from lerobot.types import RobotObservation

from .config_so_follower import SpectrobotTrifoldConfig
from .so_follower import SOFollower

logger = logging.getLogger(__name__)


def _make_iepe_profile(
    display_name: str,
    sensor_ref: str,
    sensor_sensitivity_V_per_unit: float,
    sensor_sensitivity_unit: str,
    range_choice: int = 10000,
    range_unit: str = "mV",
    hpf_choice: float = 0.1,
    excitation_choice: int = 4,
) -> dict[str, object]:
    return {
        "display_name": display_name,
        "sensor_name": display_name,
        "sensor_ref": sensor_ref,
        "measurement_choice": "IEPE",
        "range_choice": range_choice,
        "range_unit": range_unit,
        "hpf_choice": hpf_choice,
        "excitation_choice": excitation_choice,
        "excitation_unit": "mA",
        "sensor_sensitivity_V_per_unit": sensor_sensitivity_V_per_unit,
        "sensor_sensitivity_unit": sensor_sensitivity_unit,
    }


class SpectrobotTrifold(SOFollower):
    """SO101 follower computing tactile spectrograms on GPU from 3 IEPE Dragonfly sensors."""

    config_class = SpectrobotTrifoldConfig
    name = "spectrobot_trifold"

    _width = 224
    _height = 224
    _target_size = (_width, _height)

    # Rates & Downsampling
    _sampling_rate_hz = 200_000
    _sampling_rate_hz_10k = 20_000
    _downsample_factor_10k = _sampling_rate_hz // _sampling_rate_hz_10k

    # NFFT Configurations
    _nfft_10k_512 = 512

    _num_dragonflies = 3
    _dragonfly_profile = _make_iepe_profile("Dragonfly", "DGF-UNI-W220405-10", 10.8, "mV/(um/m)")

    # Force vector KPI, in the finger frame: z along the finger towards the tip, x and y across it.
    # All 3 sensors are considered at the same point (origin), each measuring a rotation (bending / torsion)
    # about one axis: ch0 -> y, ch1 -> x, ch2 -> z (see `sensor_rotation_axes` in the config).
    _rotation_axis_keys = ("tactile_rotation_x", "tactile_rotation_y", "tactile_rotation_z")
    _force_axis_keys = ("tactile_force_x", "tactile_force_y", "tactile_force_z")
    _force_norm_key = "tactile_force_norm"

    _measurement_map = {"Voltage": 0, "IEPE": 1}
    _range_map = {10000: 0, 5000: 1, 1000: 2, 200: 3}
    _hpf_map = {0: 0, 0.1: 0, 1: 1}
    _excitation_map = {2: 0, 4: 1, 6: 2}

    def __init__(self, config: SpectrobotTrifoldConfig):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Spectrogram processing will run on: {self.device}")

        channels = list(config.dragonfly_channels)
        if len(channels) != self._num_dragonflies or len(set(channels)) != len(channels):
            raise ValueError(
                f"dragonfly_channels must contain {self._num_dragonflies} distinct channel indices, got {channels}."
            )

        self._selected_sensor_keys = [f"dragonfly_{i + 1}" for i in range(self._num_dragonflies)]
        self._selected_sensor_profiles = [self._dragonfly_profile] * self._num_dragonflies
        self._selected_channel_indices = channels

        self._size_10k_512 = (self._width - 1) * (self._nfft_10k_512 // 2) + self._nfft_10k_512

        # Independent buffers for each sensor
        self._buffers_10k_512 = [np.zeros(self._size_10k_512, dtype=np.float32) for _ in self._selected_sensor_keys]

        # Downsample remainders per sensor
        self._downsample_remainders = [np.zeros(0, dtype=np.float32) for _ in self._selected_sensor_keys]

        self._readers = []
        self._instance = None
        self._device = None

        # Force vector: 20 kS/s stream block-averaged down to force_rate_hz, smoothed over force_window_s
        self._force_downsample_factor = max(1, self._sampling_rate_hz_10k // config.force_rate_hz)
        force_rate_hz = self._sampling_rate_hz_10k / self._force_downsample_factor
        self._force_window_size = max(1, round(config.force_window_s * force_rate_hz))
        self._force_buffers = [np.zeros(self._force_window_size, dtype=np.float32) for _ in self._selected_sensor_keys]
        self._force_remainders = [np.zeros(0, dtype=np.float32) for _ in self._selected_sensor_keys]
        # Signals -> rotation vector: rotation = sum_i signal_i * axis_i, i.e. rotation = axes.T @ signals
        self._sensor_rotation_axes = np.asarray(config.sensor_rotation_axes, dtype=np.float64)
        if self._sensor_rotation_axes.shape != (self._num_dragonflies, 3):
            raise ValueError(
                f"sensor_rotation_axes must be a 3x3 matrix, got shape {self._sensor_rotation_axes.shape}."
            )
        self._lever_arm = np.asarray(config.force_lever_arm_m, dtype=np.float64)
        # Force from the 3 sensor signals (s1, s2, s3 = dragonfly_1, dragonfly_2, dragonfly_3), F = matrix @ s:
        #   Fx = s3 + s2
        #   Fy = s2 - s3
        #   Fz = 2*s1 - s2 + s3
        # Applied in `_compute_force`.
        # self._sensor_force_matrix = np.array(
        #     [
        #         [1.0, -3.0, -2.0],
        #         [0.0, -1.0, 1.0],
        #         [-12.0/5, 1.0, 1.0],
        #     ]
        # )

        self._sensor_force_matrix = np.array(
            [
                [-0.0721,  0.0087,  0.0007],
                [ 0.0279, -0.0088, -0.0589],
                [ 0.0268, -0.0086,  0.0598],
            ]
        )
        # Slow moving baseline (EMA) removed from each signal: cancels the IEPE settling / drift so that only
        # the strain produced by a contact remains. No output until the baseline has settled (warm-up).
        self._force_baseline_alpha = 1.0 / max(config.force_baseline_tau_s * force_rate_hz, 1.0)
        self._force_warmup_samples = round(config.force_warmup_s * force_rate_hz)

        # Last seconds of the baseline-removed signals, block-averaged down to timeseries_rate_hz (for display)
        self._timeseries_downsample_factor = max(1, round(force_rate_hz / config.timeseries_rate_hz))
        self._timeseries_rate_hz = force_rate_hz / self._timeseries_downsample_factor
        timeseries_size = max(1, round(config.timeseries_duration_s * self._timeseries_rate_hz))
        self._timeseries_buffers = [np.zeros(timeseries_size, dtype=np.float32) for _ in self._selected_sensor_keys]
        self._timeseries_remainders = [np.zeros(0, dtype=np.float32) for _ in self._selected_sensor_keys]
        # Total samples produced and wall-clock time (s since epoch) of the latest sample, per sensor
        self._timeseries_counts = np.zeros(self._num_dragonflies, dtype=np.int64)
        self._timeseries_last_time = np.zeros(self._num_dragonflies, dtype=np.float64)
        self._force_sample_counts = np.zeros(self._num_dragonflies, dtype=np.int64)
        self._force_baseline = np.full(self._num_dragonflies, np.nan, dtype=np.float64)
        self._last_force = dict.fromkeys(
            (*self._rotation_axis_keys, *self._force_axis_keys, self._force_norm_key), 0.0
        )

        # Build Frame Keys Dictionary
        self._frame_keys_by_channel = [
            {"10k_512": f"tactile_spectrogram_{sensor_key}_10kHz_nfft_512"}
            for sensor_key in self._selected_sensor_keys
        ]

        # Composite spectrogram: R, G, B = dragonfly_1, dragonfly_2, dragonfly_3
        self._rgb_frame_key = "tactile_spectrogram_rgb_10kHz_nfft_512"
        self._per_sensor_spectrograms = config.per_sensor_spectrograms
        self._rgb_spectrogram = config.rgb_spectrogram

        frame_keys = []
        if self._per_sensor_spectrograms:
            frame_keys += [key for channel in self._frame_keys_by_channel for key in channel.values()]
        if self._rgb_spectrogram:
            frame_keys.append(self._rgb_frame_key)
        self._last_frames = {
            key: np.zeros((self._target_size[1], self._target_size[0], 3), dtype=np.uint8) for key in frame_keys
        }

        self._init_tactile_reader()

        # Threading setup targeting 30 FPS computations
        self._target_fps = 30.0
        self._lock = threading.Lock()
        self._running = True
        self._worker_thread = threading.Thread(target=self._spectrogram_worker, daemon=True)
        self._worker_thread.start()

    def _infer_target_size_from_camera_config(self) -> tuple[int, int]:
        if not self.config.cameras:
            return (224, 224)

        first_camera_cfg = next(iter(self.config.cameras.values()))
        width = int(first_camera_cfg.width) if first_camera_cfg.width is not None else 224
        height = int(first_camera_cfg.height) if first_camera_cfg.height is not None else 224
        return (width, height)

    def _init_tactile_reader(self) -> None:
        try:
            import opendaq
        except ImportError:
            logger.warning("opendaq is not installed. Tactile spectrogram stream is disabled.")
            return

        
        self._instance = opendaq.Instance()
        available_devices = list(self._instance.available_devices)
        if not available_devices:
            logger.warning("No openDAQ devices found. Tactile spectrogram stream is disabled.")
            return

        target = next((device for device in available_devices if "IOLITE-X" in device.name), available_devices[0])
        self._device = self._instance.add_device(target.connection_string)

        max_channel_required = max(self._selected_channel_indices)
        if len(self._device.channels) <= max_channel_required:
            logger.warning(
                "Spectrobot trifold requires hardware channel index %s, but the device only exposes %s channels.",
                max_channel_required,
                len(self._device.channels),
            )
            return

        try:
            self._device.set_property_value("SampleRate", self._sampling_rate_hz)
        except Exception:
            logger.warning("Could not set SampleRate to %s on tactile device.", self._sampling_rate_hz)

        for idx, profile in enumerate(self._selected_sensor_profiles):
            hw_channel_idx = self._selected_channel_indices[idx]
            sensor_name = self._selected_sensor_keys[idx]

            channel = self._device.channels[hw_channel_idx]
            channel.active = True
            amplifier = channel.get_function_blocks()[0]
            self._configure_amplifier(amplifier, profile)
            self._readers.append(opendaq.StreamReader(channel.signals[0]))
            
            logger.info("Bound sensor '%s' to IOLITE-X physical channel %s", sensor_name, hw_channel_idx)

        logger.info("Connected 3-Dragonfly tactile stream from %s on channels %s.", target.name, self._selected_channel_indices)
        

    def _configure_amplifier(self, amplifier, profile: dict[str, object]) -> None:
        
        measurement_choice = str(profile["measurement_choice"])
        range_choice = int(profile["range_choice"])
        hpf_choice = float(profile["hpf_choice"])
        if measurement_choice == "IEPE":
            excitation_choice = int(profile["excitation_choice"])

        amplifier.set_property_value("Measurement", self._measurement_map[measurement_choice])
        amplifier.set_property_value("Range", self._range_map[range_choice])
        amplifier.set_property_value("HPFilter", self._hpf_map[hpf_choice])
        if measurement_choice == "IEPE":
            amplifier.set_property_value("Excitation", self._excitation_map[excitation_choice])
        
        

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = dict(super().observation_features)
        for key in self._last_force:
            features[key] = float
        for frame_key in self._last_frames:
            features[frame_key] = (self._target_size[1], self._target_size[0], 3)
        return features

    @property
    def tactile_layout(self) -> dict[str, object]:
        """Sensor axes, lever arm and observation keys needed to draw the rotation and force vectors in 3D."""
        return {
            "rotation_keys": self._rotation_axis_keys,
            "force_keys": self._force_axis_keys,
            "sensor_axes": self._sensor_rotation_axes.tolist(),
            "sensor_labels": tuple(
                f"ch{ch} {key}" for ch, key in zip(self._selected_channel_indices, self._selected_sensor_keys, strict=True)
            ),
            "lever_arm": self._lever_arm.tolist(),
            "sensor_force_matrix": self._sensor_force_matrix.tolist(),
            "arrow_scale": self.config.force_arrow_scale,
            "rgb_spectrogram_key": self._rgb_frame_key if self._rgb_spectrogram else None,
            "timeseries_fn": self.get_tactile_samples_since,
            "timeseries_duration_s": self.config.timeseries_duration_s,
        }

    @staticmethod
    def _push_to_ring_buffer(buffer: np.ndarray, chunk: np.ndarray) -> None:
        if chunk.size >= buffer.size:
            buffer[:] = chunk[-buffer.size :]
            return
        buffer[:] = np.roll(buffer, -chunk.size)
        buffer[-chunk.size :] = chunk

    @staticmethod
    def _block_mean(chunk: np.ndarray, remainder: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
        """Downsamples by averaging blocks of `factor` samples; returns (downsampled, new remainder)."""
        if remainder.size:
            chunk = np.concatenate((remainder, chunk))

        usable_size = (chunk.size // factor) * factor
        downsampled = chunk[:usable_size].reshape(-1, factor).mean(axis=1).astype(np.float32)
        return downsampled, chunk[usable_size:]

    def _mean_downsample_chunk(self, chunk: np.ndarray, channel_index: int) -> np.ndarray:
        downsampled, self._downsample_remainders[channel_index] = self._block_mean(
            chunk, self._downsample_remainders[channel_index], self._downsample_factor_10k
        )
        return downsampled

    def _push_force_samples(self, chunk_10k: np.ndarray, channel_index: int) -> None:
        chunk_force, self._force_remainders[channel_index] = self._block_mean(
            chunk_10k, self._force_remainders[channel_index], self._force_downsample_factor
        )
        if chunk_force.size == 0:
            return

        baseline = self._force_baseline[channel_index]
        if np.isnan(baseline):
            baseline = float(chunk_force[0])
        alpha = self._force_baseline_alpha
        detrended = np.empty_like(chunk_force)
        for k, x in enumerate(chunk_force.tolist()):
            baseline += alpha * (x - baseline)
            detrended[k] = x - baseline
        self._force_baseline[channel_index] = baseline
        self._force_sample_counts[channel_index] += chunk_force.size

        self._push_to_ring_buffer(self._force_buffers[channel_index], detrended)

        chunk_ts, self._timeseries_remainders[channel_index] = self._block_mean(
            detrended, self._timeseries_remainders[channel_index], self._timeseries_downsample_factor
        )
        if chunk_ts.size:
            with self._lock:
                self._push_to_ring_buffer(self._timeseries_buffers[channel_index], chunk_ts)
                self._timeseries_counts[channel_index] += chunk_ts.size
                self._timeseries_last_time[channel_index] = time.time()

    def get_tactile_samples_since(
        self, counts: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Baseline-removed sensor samples produced since `counts` (as returned by the previous call).

        Returns (new counts, per sensor (timestamps in s since epoch, values)). At most the last
        `timeseries_duration_s` are returned.
        """
        if counts is None:
            counts = np.zeros(self._num_dragonflies, dtype=np.int64)
        samples = []
        with self._lock:
            new_counts = self._timeseries_counts.copy()
            for i, buffer in enumerate(self._timeseries_buffers):
                n_new = int(min(new_counts[i] - counts[i], buffer.size))
                values = buffer[buffer.size - n_new :].copy()
                times = self._timeseries_last_time[i] - np.arange(n_new - 1, -1, -1) / self._timeseries_rate_hz
                samples.append((times, values))
        return new_counts, samples

    def _compute_force(self) -> dict[str, float]:
        """Rotation vector and force (Fx, Fy, Fz) from the 3 sensor signals."""
        signals = np.array([buf.mean(dtype=np.float64) for buf in self._force_buffers])
        if (self._force_sample_counts < self._force_warmup_samples).any():
            signals[:] = 0.0  # baseline still settling
        rotation = self._sensor_rotation_axes.T @ signals
        # Fx = s3 + s2, Fy = s2 - s3, Fz = 2*s1 - s2 + s3 (see `_sensor_force_matrix`)
        force = self._sensor_force_matrix @ signals

        values = dict(zip(self._rotation_axis_keys, rotation.tolist(), strict=True))
        values.update(zip(self._force_axis_keys, force.tolist(), strict=True))
        values[self._force_norm_key] = float(np.linalg.norm(force))
        return values

    def _render_spectrogram_frame(
        self, 
        buffer: np.ndarray, 
        fs_hz: int, 
        nfft: int, 
        min_db: float, 
        max_db: float
    ) -> np.ndarray | None:
        """PyTorch GPU accelerated spectrogram rendering."""
        nperseg = min(nfft, buffer.size)
        if nperseg < 16:
            return None

        tensor_buf = torch.from_numpy(buffer).to(self.device)
        noverlap = nperseg // 2
        window = torch.hann_window(nperseg, device=self.device)
        
        stft_out = torch.stft(
            tensor_buf,
            n_fft=nfft,
            hop_length=nperseg - noverlap,
            win_length=nperseg,
            window=window,
            return_complex=True,
            center=False
        )

        sxx = stft_out.abs().pow(2)
        scale_factor = 2.0 / (fs_hz * (window ** 2).sum())
        sxx = sxx * scale_factor

        sxx_db = 10.0 * torch.log10(sxx + 1e-12)
        
        normalized = torch.clamp(sxx_db, min_db, max_db)
        normalized = (normalized - min_db) / max(max_db - min_db, 1e-6)
        normalized = torch.flip(normalized, dims=[0])

        normalized = normalized.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(normalized, size=self._target_size, mode='bilinear', align_corners=False).squeeze()

        image_8bit = (resized * 255.0).to(torch.uint8)
        image_rgb = image_8bit.unsqueeze(0).expand(3, -1, -1)

        return image_rgb.permute(1, 2, 0).cpu().numpy()

    def _drain_reader_to_buffers(self, channel_index: int) -> None:
        """Reads data and correctly routes to all multi-resolution buffers."""
        if channel_index >= len(self._readers):
            return

        reader = self._readers[channel_index]
        available = reader.available_count
        if available <= 0:
            return

        chunk = np.asarray(reader.read(available), dtype=np.float32)
        if chunk.size == 0:
            return

        # Process and Push to 10kHz Buffers
        chunk_10k = self._mean_downsample_chunk(chunk, channel_index)
        if chunk_10k.size > 0:
            self._push_to_ring_buffer(self._buffers_10k_512[channel_index], chunk_10k)
            self._push_force_samples(chunk_10k, channel_index)

    def _spectrogram_worker(self):
        """Background thread ensuring GPU renders consistently at a smooth 30 FPS."""
        interval = 1.0 / self._target_fps
        last_compute_time = time.perf_counter()

        while self._running:
            now = time.perf_counter()
            
            for i in range(len(self._readers)):
                self._drain_reader_to_buffers(i)

            if len(self._readers) == self._num_dragonflies:
                force = self._compute_force()
                with self._lock:
                    self._last_force.update(force)

            if now - last_compute_time >= interval:
                new_frames = {}
                gray_frames = []

                for ch_idx in range(len(self._readers)):
                    keys = self._frame_keys_by_channel[ch_idx]

                    f_10k_512 = self._render_spectrogram_frame(
                        self._buffers_10k_512[ch_idx],
                        self._sampling_rate_hz_10k,
                        self._nfft_10k_512,
                        self.config.spectrogram_min_db,
                        self.config.spectrogram_max_db,
                    )
                    if f_10k_512 is None:
                        continue
                    gray_frames.append(f_10k_512[..., 0])
                    if self._per_sensor_spectrograms:
                        new_frames[keys["10k_512"]] = f_10k_512

                if self._rgb_spectrogram and len(gray_frames) == self._num_dragonflies:
                    new_frames[self._rgb_frame_key] = np.stack(gray_frames, axis=-1)

                if new_frames:
                    with self._lock:
                        self._last_frames.update(new_frames)
                        
                last_compute_time = now
            
            time.sleep(0.001)

    def get_observation(self) -> RobotObservation:
        tick_start = time.perf_counter()
        
        obs = super().get_observation()

        with self._lock:
            obs.update(self._last_force)
            obs.update(self._last_frames)

        total_latency = (time.perf_counter() - tick_start) * 1000
        logger.debug("Observation tick sync completed in %.2fms", total_latency)
        return obs

    def disconnect(self):
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        super().disconnect()
