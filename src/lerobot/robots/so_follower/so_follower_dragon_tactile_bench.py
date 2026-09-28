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

from .config_so_follower import SO101FollowerDragonTactileBenchConfig
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

def _make_voltage_profile(
    display_name: str,
    sensor_ref: str,
    sensor_sensitivity_V_per_unit: float,
    sensor_sensitivity_unit: str,
    range_choice: int = 10000,
    range_unit: str = "V",
    hpf_choice: float = 0,
) -> dict[str, object]:
    return {
        "display_name": display_name,
        "sensor_name": display_name,
        "sensor_ref": sensor_ref,
        "measurement_choice": "Voltage",
        "range_choice": range_choice,
        "range_unit": range_unit,
        "hpf_choice": hpf_choice,
        "sensor_sensitivity_V_per_unit": sensor_sensitivity_V_per_unit,
        "sensor_sensitivity_unit": sensor_sensitivity_unit,
    }


class SO101FollowerDragonTactileBench(SOFollower):
    """SO101 follower calculating multi-resolution tactile spectrograms for a two-sensor bench on GPU."""

    config_class = SO101FollowerDragonTactileBenchConfig
    name = "so101_follower_dragon_tactile_bench"

    _width = 224
    _height = 224
    _target_size = (_width, _height)

    _spectrogram_min_db_dgf = -72.0
    _spectrogram_max_db_dgf = 40.0

    _spectrogram_min_db_dgf_passif = -120.0
    _spectrogram_max_db_dgf_passif = -50.0

    _spectrogram_min_db_load_cell = -80.0
    _spectrogram_max_db_load_cell = -10.0

    _spectrogram_min_db_strain_gauge = -117.0
    _spectrogram_max_db_strain_gauge = -40.0

    _spectrogram_min_db_pastille_pzt = -120.0
    _spectrogram_max_db_pastille_pzt = -20.0

    _spectrogram_min_db_mems_acc = -120.0
    _spectrogram_max_db_mems_acc = -30.0

    # Rates & Downsampling
    _sampling_rate_hz = 200_000
    _sampling_rate_hz_10k = 20_000
    _downsample_factor_10k = _sampling_rate_hz // _sampling_rate_hz_10k

    # NFFT Configurations
    _nfft_100k_512 = 512
    _nfft_100k_4096 = 4096
    _nfft_10k_64 = 64
    _nfft_10k_512 = 512

    _sensor_profiles = {
        "dragonfly": _make_iepe_profile("Dragonfly", "DGF-UNI-W220405-10", 10.8, "mV/(um/m)"),
        "accelero": _make_iepe_profile("Accelero_PCB_piezo", "TLD352A56", 100.0, "mV/g"),
        "load_cell": _make_iepe_profile("Cell force", "SNLW56997", 10.64, "mV/N"),
        "dgf_passif": _make_voltage_profile("Dragonfly", "DGF-UNI-AA20405-10", -2.70e-3, "mV/(um/m)"),
        "pastille_pzt": _make_voltage_profile("Pastille_PZT", "", 0.1, "mV/(um/m)"),
        "strain_gauge": _make_voltage_profile("Strain_Gauge", "SG-1234", 0.002, "mV/(um/m)"),
        "acc_mems": _make_voltage_profile("MEMS_ADXL356", "ADXL356CZ", 0.5, "mV/g"),
    }

    _bench_sensor_pairs = {
        "dragonfly_accelero": ("dragonfly", "accelero"),
        "dragonfly_load_cell": ("dragonfly", "load_cell"),
        "5sensors": ("dragonfly","dgf_passif", "acc_mems", "strain_gauge", "pastille_pzt"), 
    }

    _sensor_channels = {
        "dragonfly": 0,   # Choix du canal pour chaque capteur dans le banc tactile
        "accelero": 1,    
        "load_cell": 2,   
        "dgf_passif": 1, 
        "pastille_pzt": 2,  
        "strain_gauge": 3, 
        "acc_mems": 4,    
    }

    _measurement_map = {"Voltage": 0, "IEPE": 1}
    _range_map = {10000: 0, 5000: 1, 1000: 2, 200: 3}
    _hpf_map = {0: 0, 0.1: 0, 1: 1}
    _excitation_map = {2: 0, 4: 1, 6: 2}

    def __init__(self, config: SO101FollowerDragonTactileBenchConfig):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Spectrogram processing will run on: {self.device}")

        self._bench_setup = getattr(config, "bench_setup", "5sensors")
        if self._bench_setup not in self._bench_sensor_pairs:
            raise ValueError(
                f"Unsupported bench_setup '{self._bench_setup}'. "
                f"Expected one of {sorted(self._bench_sensor_pairs)}."
            )


        self._selected_sensor_keys = self._bench_sensor_pairs[self._bench_setup]
        self._selected_sensor_profiles = [self._sensor_profiles[key] for key in self._selected_sensor_keys]

        self._selected_channel_indices = [self._sensor_channels[key] for key in self._selected_sensor_keys]

        self._sensor_labels = [profile["display_name"] for profile in self._selected_sensor_profiles]

        # Calculate buffer sizes based on NFFTs
        # self._size_100k_512 = (self._width - 1) * (self._nfft_100k_512 // 2) + self._nfft_100k_512
        # self._size_100k_4096 = (self._width - 1) * (self._nfft_100k_4096 // 2) + self._nfft_100k_4096
        # self._size_10k_64 = (self._width - 1) * (self._nfft_10k_64 // 2) + self._nfft_10k_64
        self._size_10k_512 = (self._width - 1) * (self._nfft_10k_512 // 2) + self._nfft_10k_512

        # Independent buffers for each sensor and resolution
        # self._buffers_100k_512 = [np.zeros(self._size_100k_512, dtype=np.float32) for _ in self._selected_sensor_keys]
        # self._buffers_100k_4096 = [np.zeros(self._size_100k_4096, dtype=np.float32) for _ in self._selected_sensor_keys]
        # self._buffers_10k_64 = [np.zeros(self._size_10k_64, dtype=np.float32) for _ in self._selected_sensor_keys]
        self._buffers_10k_512 = [np.zeros(self._size_10k_512, dtype=np.float32) for _ in self._selected_sensor_keys]

        # Downsample remainders per sensor
        self._downsample_remainders = [np.zeros(0, dtype=np.float32) for _ in self._selected_sensor_keys]

        self._readers = []
        self._instance = None
        self._device = None

        # Build Frame Keys Dictionary
        self._frame_keys_by_channel = []
        for sensor_key in self._selected_sensor_keys:
            self._frame_keys_by_channel.append({
                
                "10k_512": f"tactile_spectrogram_{sensor_key}_10kHz_nfft_512",
            })

            # "100k_512": f"tactile_spectrogram_{sensor_key}_100kHz_nfft_512",
            # "100k_4096": f"tactile_spectrogram_{sensor_key}_100kHz_nfft_4096",
            # "10k_64": f"tactile_spectrogram_{sensor_key}_10kHz_nfft_64",

        self._last_frames = {
            key: np.zeros((self._target_size[1], self._target_size[0], 3), dtype=np.uint8)
            for channel in self._frame_keys_by_channel
            for key in channel.values()
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
                "The tactile bench requires hardware channel index %s, but the device only exposes %s channels.",
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

        logger.info("Connected multi-resolution tactile bench stream from %s with setup %s.", target.name, self._bench_setup)
        

    def _configure_amplifier(self, amplifier, profile: dict[str, object]) -> None:
        
        measurement_choice = str(profile["measurement_choice"])
        print(measurement_choice)
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
        for frame_key in self._last_frames:
            features[frame_key] = (self._target_size[1], self._target_size[0], 3)
        return features

    @staticmethod
    def _push_to_ring_buffer(buffer: np.ndarray, chunk: np.ndarray) -> None:
        if chunk.size >= buffer.size:
            buffer[:] = chunk[-buffer.size :]
            return
        buffer[:] = np.roll(buffer, -chunk.size)
        buffer[-chunk.size :] = chunk

    def _mean_downsample_chunk(self, chunk: np.ndarray, channel_index: int) -> np.ndarray:
        remainder = self._downsample_remainders[channel_index]
        
        if remainder.size:
            chunk = np.concatenate((remainder, chunk))

        usable_size = (chunk.size // self._downsample_factor_10k) * self._downsample_factor_10k
        if usable_size == 0:
            self._downsample_remainders[channel_index] = chunk
            return np.zeros(0, dtype=np.float32)

        downsampled = chunk[:usable_size].reshape(-1, self._downsample_factor_10k).mean(axis=1).astype(np.float32)
        self._downsample_remainders[channel_index] = chunk[usable_size:]
        return downsampled

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

        # Push to 100kHz Buffers
        # self._push_to_ring_buffer(self._buffers_100k_512[channel_index], chunk)
        # self._push_to_ring_buffer(self._buffers_100k_4096[channel_index], chunk)

        # Process and Push to 10kHz Buffers
        chunk_10k = self._mean_downsample_chunk(chunk, channel_index)
        if chunk_10k.size > 0:
            # self._push_to_ring_buffer(self._buffers_10k_64[channel_index], chunk_10k)
            self._push_to_ring_buffer(self._buffers_10k_512[channel_index], chunk_10k)

    def _spectrogram_worker(self):
        """Background thread ensuring GPU renders consistently at a smooth 30 FPS."""
        interval = 1.0 / self._target_fps
        last_compute_time = time.perf_counter()

        while self._running:
            now = time.perf_counter()
            
            for i in range(len(self._readers)):
                self._drain_reader_to_buffers(i)

            if now - last_compute_time >= interval:
                new_frames = {}
                
                for ch_idx in range(len(self._readers)):
                    keys = self._frame_keys_by_channel[ch_idx]
                    sensor_key = self._selected_sensor_keys[ch_idx]

                    # Pick sensor-specific dB limits
                    if "load_cell" in sensor_key:
                        min_db = self._spectrogram_min_db_load_cell
                        max_db = self._spectrogram_max_db_load_cell
                    elif "acc_mems" in sensor_key:
                        min_db = self._spectrogram_min_db_mems_acc
                        max_db = self._spectrogram_max_db_mems_acc
                    elif "dgf_passif" in sensor_key:
                        min_db = self._spectrogram_min_db_dgf_passif
                        max_db = self._spectrogram_max_db_dgf_passif
                    elif "pastille_pzt" in sensor_key:
                        min_db = self._spectrogram_min_db_pastille_pzt
                        max_db = self._spectrogram_max_db_pastille_pzt
                    elif "strain_gauge" in sensor_key:
                        min_db = self._spectrogram_min_db_strain_gauge
                        max_db = self._spectrogram_max_db_strain_gauge
                    else:
                        min_db = self._spectrogram_min_db_dgf
                        max_db = self._spectrogram_max_db_dgf
                    
                    # 100kHz equivalent rendering
                    # f_100k_512 = self._render_spectrogram_frame(
                    #     self._buffers_100k_512[ch_idx], self._sampling_rate_hz, self._nfft_100k_512, min_db, max_db
                    # )
                    # if f_100k_512 is not None: new_frames[keys["100k_512"]] = f_100k_512
                        
                    # f_100k_4096 = self._render_spectrogram_frame(
                    #     self._buffers_100k_4096[ch_idx], self._sampling_rate_hz, self._nfft_100k_4096, min_db, max_db
                    # )
                    # if f_100k_4096 is not None: new_frames[keys["100k_4096"]] = f_100k_4096

                    # 10kHz equivalent rendering
                    # f_10k_64 = self._render_spectrogram_frame(
                    #     self._buffers_10k_64[ch_idx], self._sampling_rate_hz_10k, self._nfft_10k_64, min_db, max_db
                    # )
                    # if f_10k_64 is not None: new_frames[keys["10k_64"]] = f_10k_64
                        
                    f_10k_512 = self._render_spectrogram_frame(
                        self._buffers_10k_512[ch_idx], self._sampling_rate_hz_10k, self._nfft_10k_512, min_db, max_db
                    )
                    if f_10k_512 is not None: new_frames[keys["10k_512"]] = f_10k_512

                if new_frames:
                    with self._lock:
                        self._last_frames.update(new_frames)
                        
                last_compute_time = now
            
            time.sleep(0.001)

    def get_observation(self) -> RobotObservation:
        tick_start = time.perf_counter()
        
        obs = super().get_observation()

        with self._lock:
            obs.update(self._last_frames)

        total_latency = (time.perf_counter() - tick_start) * 1000
        logger.debug("Observation tick sync completed in %.2fms", total_latency)
        return obs

    def close(self):
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        super().close()