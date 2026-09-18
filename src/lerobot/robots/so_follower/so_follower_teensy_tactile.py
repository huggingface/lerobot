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
import queue
import struct
import threading
import time
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.types import RobotObservation

# Adjust import paths depending on your folder layout
from .config_so_follower import SO101FollowerTeensyTactileConfig
from .so_follower import SOFollower

try:
    import serial
except ImportError:
    serial = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware & Packet Parsing Helpers
# ---------------------------------------------------------------------------

MAGIC = b"A868"
# 4 bytes magic, uint32 sequence, uint16 count, uint32 lost = 14 bytes
HEADER = struct.Struct("<4sIHI")

def raw_to_volts(raw: np.ndarray, v_min: float = -10.24, v_max: float = 10.24) -> np.ndarray:
    return v_min + raw.astype(np.float32) * ((v_max - v_min) / 65536.0)

MAX_PACKET_SAMPLES = 4096


class SerialReader(threading.Thread):
    """Background thread reading packets from the Teensy so serial I/O never blocks."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        mode_command: bytes = b"A",
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.mode_command = mode_command
        self.error = None

    def run(self):
        if serial is None:
            self.error = "pyserial is not installed."
            logger.error(self.error)
            return

        ser = None
        buffer = bytearray()

        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            # Send hardware channel selection to Teensy
            time.sleep(0.05)
            ser.write(self.mode_command)
            ser.flush()
            ser.reset_input_buffer()
            logger.info("Connected to Teensy on %s at %d baud (Mode: %s)", self.port, self.baudrate, self.mode_command.decode())

            while not self.stop_event.is_set():
                n = ser.in_waiting
                chunk = ser.read(n if n > 0 else 64)

                if chunk:
                    buffer.extend(chunk)

                while True:
                    i = buffer.find(MAGIC)
                    if i < 0:
                        if len(buffer) >= len(MAGIC):
                            del buffer[: -(len(MAGIC) - 1)]
                        break

                    if i > 0:
                        del buffer[:i]

                    if len(buffer) < HEADER.size:
                        break

                    magic, sequence, count, lost = HEADER.unpack_from(buffer, 0)

                    if magic != MAGIC or count == 0 or count > MAX_PACKET_SAMPLES:
                        del buffer[1]
                        continue

                    packet_size = HEADER.size + 2 * count
                    if len(buffer) < packet_size:
                        break

                    raw = np.frombuffer(buffer[HEADER.size : packet_size], dtype="<u2").copy()
                    del buffer[:packet_size]

                    try:
                        self.output_queue.put_nowait((sequence, lost, raw))
                    except queue.Full:
                        try:
                            _ = self.output_queue.get_nowait()
                            self.output_queue.put_nowait((sequence, lost, raw))
                        except queue.Empty:
                            pass

        except Exception as exc:
            self.error = str(exc)
            logger.error("SerialReader encountered error: %s", exc)
        finally:
            if ser is not None and ser.is_open:
                ser.close()
                logger.info("Closed Teensy serial port %s", self.port)


# ---------------------------------------------------------------------------
# LeRobot Follower Robot Class (CPU Spectrograms)
# ---------------------------------------------------------------------------

class SO101FollowerTeensyTactile(SOFollower):
    """SO101 follower reading tactile data from Teensy/ADS8688 and rendering CPU spectrograms."""

    config_class = SO101FollowerTeensyTactileConfig
    name = "so101_follower_teensy_tactile"

    _width = 224
    _height = 224
    _target_size = (_width, _height)

    # NFFT configuration
    _nfft = 512

    # Sensor channel configurations: label and dB normalization limits
    _all_channel_definitions = {
        0: {"key": "mems_acc", "name": "mems_acc", "min_db": -110.0, "max_db": -40.0},
        1: {"key": "dgf_iepe", "name": "dgf_iepe", "min_db": -107.0, "max_db": -37.0},
        2: {"key": "pzt_disk", "name": "pzt_disk", "min_db": -110.0, "max_db": -40.0},
    }

    def __init__(self, config: SO101FollowerTeensyTactileConfig):
        super().__init__(config)

        # Force CPU device execution
        self.device = torch.device("cpu")
        logger.info("Spectrogram processing running exclusively on CPU")

        # Parse requested channel selection from config: "all", 1, 2, 3, etc.
        raw_choice = getattr(config, "num_channels", "all")
        if isinstance(raw_choice, str):
            raw_choice = raw_choice.strip().lower()

        # Determine which channels we should build spectrograms for
        if raw_choice in ("all", "1,2,3", [1, 2, 3], (1, 2, 3)):
            self._target_channels = [0, 1, 2]
        elif raw_choice in (1, "1", [1]):
            self._target_channels = [0]
        elif raw_choice in (2, "2", [2]):
            self._target_channels = [1]
        elif raw_choice in (3, "3", [3]):
            self._target_channels = [2]
        else:
            logger.warning("Unrecognized num_channels '%s', defaulting to 'all'", raw_choice)
            self._target_channels = [0, 1, 2]

        # Force hardware to ALWAYS read all 3 channels to maintain timing and format expectations
        mode_cmd = b"A"
        self._num_hw_channels = 3

        self._channel_definitions = {
            ch: self._all_channel_definitions[ch]
            for ch in self._target_channels
        }

        # Sampling rates: each channel is physically sampled at 20,000 Hz
        self._fs_per_channel = 20_000.0

        self._v_min = getattr(config, "v_min", -10.24)
        self._v_max = getattr(config, "v_max", 10.24)

        # Buffer size: calculates required samples to fill 224 spectrogram time columns
        hop_length = self._nfft // 2
        self._buffer_size = (self._width - 1) * hop_length + self._nfft

        # Ring buffers (one per HARDWARE channel, even if we only process 1 for spectrogram)
        self._buffers = [np.zeros(self._buffer_size, dtype=np.float32) for _ in range(self._num_hw_channels)]

        # Precompute STFT Hann window on CPU
        self._window = torch.hann_window(self._nfft, device=self.device)

        # Setup observation frame keys (only for the requested config targets)
        self._frame_keys = {
            ch: f"tactile_spectrogram_{self._channel_definitions[ch]['key']}_nfft_{self._nfft}"
            for ch in self._target_channels
        }

        self._last_frames = {
            key: np.zeros((self._target_size[1], self._target_size[0], 3), dtype=np.uint8)
            for key in self._frame_keys.values()
        }

        # Threading and communication
        self._lock = threading.Lock()
        self._running = True
        self._stop_event = threading.Event()
        self._data_queue = queue.Queue(maxsize=100)

        # Serial reader thread with channel mode command
        port = getattr(config, "serial_port", "/dev/ttyACM0")
        baudrate = getattr(config, "baudrate", 2_000_000)
        self._serial_thread = SerialReader(
            port, baudrate, self._data_queue, self._stop_event, mode_command=mode_cmd
        )
        self._serial_thread.start()

        # Spectrogram computation worker thread (targeting 30 FPS)
        self._target_fps = 30.0
        self._worker_thread = threading.Thread(target=self._spectrogram_worker, daemon=True)
        self._worker_thread.start()

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = dict(super().observation_features)
        for frame_key in self._last_frames:
            features[frame_key] = (self._target_size[1], self._target_size[0], 3)
        return features

    @staticmethod
    def _push_to_ring_buffer(buffer: np.ndarray, chunk: np.ndarray) -> None:
        """Roll and append new sample chunk into 1D buffer."""
        if chunk.size >= buffer.size:
            buffer[:] = chunk[-buffer.size :]
            return
        buffer[:] = np.roll(buffer, -chunk.size)
        buffer[-chunk.size :] = chunk

    def _render_spectrogram_frame(
        self,
        buffer: np.ndarray,
        fs_hz: float,
        nfft: int,
        min_db: float,
        max_db: float,
    ) -> np.ndarray | None:
        """CPU STFT and spectrogram rendering matching signal.spectrogram(mode='psd')."""
        nperseg = min(nfft, buffer.size)
        if nperseg < 16:
            return None

        tensor_buf = torch.from_numpy(buffer)
        hop_length = nperseg // 2

        with torch.no_grad():
            # 1. Compute STFT
            stft_out = torch.stft(
                tensor_buf,
                n_fft=nfft,
                hop_length=hop_length,
                win_length=nperseg,
                window=self._window,
                return_complex=True,
                center=False,
            )

            # 2. Power Spectral Density (PSD)
            sxx = stft_out.abs().pow(2)
            scale_factor = 2.0 / (fs_hz * (self._window**2).sum())
            sxx = sxx * scale_factor

            # 3. Convert to dB
            sxx_db = 10.0 * torch.log10(sxx + 1e-12)

            # 4. Normalize to [0, 1] range based on dB limits
            normalized = torch.clamp(sxx_db, min_db, max_db)
            normalized = (normalized - min_db) / max(max_db - min_db, 1e-6)

            # Flip vertically so 0 Hz is at the bottom of the image
            normalized = torch.flip(normalized, dims=[0])

            # 5. Interpolate to target camera size (224, 224)
            normalized = normalized.unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(normalized, size=self._target_size, mode="bilinear", align_corners=False).squeeze()

            # 6. Convert to uint8 3-channel RGB
            image_8bit = (resized * 255.0).to(torch.uint8)
            image_rgb = image_8bit.unsqueeze(0).expand(3, -1, -1)

            return image_rgb.permute(1, 2, 0).numpy()

    def _drain_queue(self) -> None:
        """Drains incoming raw ADC packets and demultiplexes into per-channel buffers."""
        packets_read = 0

        while True:
            try:
                sequence, lost, raw = self._data_queue.get_nowait()
                packets_read += 1
            except queue.Empty:
                break

            if raw.size == 0:
                continue

            # 3-channel interleaved scan (Hardware is always forced to send 3 channels): [CH0, CH1, CH2, CH0, CH1, CH2...]
            for ch in range(self._num_hw_channels):
                ch_raw = raw[ch :: self._num_hw_channels]
                if ch_raw.size:
                    volts = raw_to_volts(ch_raw, self._v_min, self._v_max)
                    self._push_to_ring_buffer(self._buffers[ch], volts)

    def _spectrogram_worker(self):
        """Worker thread executing at 30 FPS to update observations."""
        interval = 1.0 / self._target_fps
        last_compute_time = time.perf_counter()

        while self._running:
            now = time.perf_counter()

            # Drain serial queue into numpy buffers
            self._drain_queue()

            if now - last_compute_time >= interval:
                new_frames = {}

                # Only compute spectrograms for the channel(s) designated in configuration
                for ch in self._target_channels:
                    cfg = self._channel_definitions[ch]
                    frame_key = self._frame_keys[ch]

                    frame = self._render_spectrogram_frame(
                        self._buffers[ch],
                        fs_hz=self._fs_per_channel,
                        nfft=self._nfft,
                        min_db=cfg["min_db"],
                        max_db=cfg["max_db"],
                    )
                    if frame is not None:
                        new_frames[frame_key] = frame

                if new_frames:
                    with self._lock:
                        self._last_frames.update(new_frames)

                last_compute_time = now

            time.sleep(0.001)

    def get_observation(self) -> RobotObservation:
        """Returns standard SOFollower observation merged with tactile spectrograms."""
        tick_start = time.perf_counter()

        obs = super().get_observation()

        with self._lock:
            obs.update(self._last_frames)

        total_latency = (time.perf_counter() - tick_start) * 1000
        logger.debug("Tactile observation sync completed in %.2fms", total_latency)
        return obs

    def close(self):
        self._running = False
        self._stop_event.set()

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self._serial_thread.is_alive():
            self._serial_thread.join(timeout=1.0)

        super().close()