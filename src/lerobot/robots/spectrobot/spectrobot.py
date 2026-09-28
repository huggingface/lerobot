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

from __future__ import annotations

import abc
import logging
import queue
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
import scipy.signal

from lerobot.types import RobotObservation

from ..so_follower.so_follower import SOFollower
from .config_spectrobot import SpectRoFollowerConfig, TactileSensorConfig

try:
    import serial
except ImportError:
    serial = None

logger = logging.getLogger(__name__)

MAGIC = b"A868"
HEADER = struct.Struct("<4sIHI")
MAX_PACKET_SAMPLES = 4096
TEENSY_NUM_CHANNELS = 8  # the firmware always streams all 8 ADS8688 inputs, interleaved


def raw_to_volts(raw: np.ndarray, v_min: float = -10.24, v_max: float = 10.24) -> np.ndarray:
    return v_min + raw.astype(np.float32) * ((v_max - v_min) / 65536.0)


def _normalize_tactile_data_stream(value: str) -> str:
    stream = value.strip().lower()
    if stream in {"opendaq", "open_daq", "open-daq"}:
        return "OpenDAQ"
    if stream in {"teensy", "teensy_ads8688", "ads8688", "serial"}:
        return "teensy_ADS8688"
    raise ValueError(
        f"Unsupported tactile_data_stream '{value}'. Expected 'teensy_ADS8688' or 'OpenDAQ'."
    )


def _make_teensy_default_sensors() -> dict[str, TactileSensorConfig]:
    return {
        "mems_acc": TactileSensorConfig(
            channel=0,
            observation_key="tactile_spectrogram_mems_acc_nfft_512",
            sample_rate_hz=20_000,
            nfft=512,
            min_db=-110.0,
            max_db=-40.0,
            measurement="Voltage",
            range_choice=10_000,
            hpf_choice=0.0,
        ),
        "dgf_iepe": TactileSensorConfig(
            channel=1,
            observation_key="tactile_spectrogram_dgf_iepe_nfft_512",
            sample_rate_hz=20_000,
            nfft=512,
            min_db=-107.0,
            max_db=-37.0,
            measurement="IEPE",
            range_choice=10_000,
            hpf_choice=0.1,
            excitation_choice=4,
        ),
        "pzt_disk": TactileSensorConfig(
            channel=2,
            observation_key="tactile_spectrogram_pzt_disk_nfft_512",
            sample_rate_hz=20_000,
            nfft=512,
            min_db=-110.0,
            max_db=-40.0,
            measurement="Voltage",
            range_choice=10_000,
            hpf_choice=0.0,
        ),
    }


def _make_opendaq_default_sensors() -> dict[str, TactileSensorConfig]:
    return {
        "dragonfly": TactileSensorConfig(
            channel=0,
            observation_key="tactile_spectrogram_dragonfly_10kHz_nfft_512",
            sample_rate_hz=200_000,
            nfft=512,
            min_db=-70.0,
            max_db=40.0,
            measurement="IEPE",
            range_choice=10_000,
            hpf_choice=0.1,
            excitation_choice=4,
        ),
        "dgf_passif": TactileSensorConfig(
            channel=1,
            observation_key="tactile_spectrogram_dgf_passif_10kHz_nfft_512",
            sample_rate_hz=200_000,
            nfft=512,
            min_db=-120.0,
            max_db=-50.0,
            measurement="Voltage",
            range_choice=10_000,
            hpf_choice=0.0,
        ),
        "pastille_pzt": TactileSensorConfig(
            channel=2,
            observation_key="tactile_spectrogram_pastille_pzt_10kHz_nfft_512",
            sample_rate_hz=200_000,
            nfft=512,
            min_db=-120.0,
            max_db=-20.0,
            measurement="Voltage",
            range_choice=10_000,
            hpf_choice=0.0,
        ),
        "strain_gauge": TactileSensorConfig(
            channel=3,
            observation_key="tactile_spectrogram_strain_gauge_10kHz_nfft_512",
            sample_rate_hz=200_000,
            nfft=512,
            min_db=-110.0,
            max_db=-40.0,
            measurement="Voltage",
            range_choice=10_000,
            hpf_choice=0.0,
        ),
        "acc_mems": TactileSensorConfig(
            channel=4,
            observation_key="tactile_spectrogram_acc_mems_10kHz_nfft_512",
            sample_rate_hz=200_000,
            nfft=512,
            min_db=-120.0,
            max_db=-20.0,
            measurement="Voltage",
            range_choice=10_000,
            hpf_choice=0.0,
        ),
    }


@dataclass
class _SensorState:
    key: str
    spec: TactileSensorConfig
    observation_key: str
    buffer: np.ndarray
    frame: np.ndarray


class _TactileBackend(abc.ABC):
    def __init__(self, sensors: OrderedDict[str, TactileSensorConfig]):
        self.sensors = sensors
        self.available_keys = tuple(sensors.keys())

    @abc.abstractmethod
    def drain(self) -> dict[str, np.ndarray]:
        pass

    def close(self) -> None:
        pass


class _TeensySerialReader(threading.Thread):
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
        self.error: str | None = None

    def run(self) -> None:
        if serial is None:
            self.error = "pyserial is not installed."
            logger.error(self.error)
            return

        ser = None
        buffer = bytearray()

        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            time.sleep(0.05)
            ser.write(self.mode_command)
            ser.flush()
            ser.reset_input_buffer()
            logger.info(
                "Connected to Teensy on %s at %d baud (Mode: %s)",
                self.port,
                self.baudrate,
                self.mode_command.decode(errors="ignore"),
            )

            while not self.stop_event.is_set():
                n = ser.in_waiting
                chunk = ser.read(n if n > 0 else 64)

                if chunk:
                    buffer.extend(chunk)

                while True:
                    start = buffer.find(MAGIC)
                    if start < 0:
                        if len(buffer) >= len(MAGIC):
                            del buffer[: -(len(MAGIC) - 1)]
                        break

                    if start > 0:
                        del buffer[:start]

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

        except Exception as exc:  # nosec B110
            self.error = str(exc)
            logger.error("SerialReader encountered error: %s", exc)
        finally:
            if ser is not None and ser.is_open:
                ser.close()
                logger.info("Closed Teensy serial port %s", self.port)


class _TeensyBackend(_TactileBackend):
    def __init__(
        self,
        sensors: OrderedDict[str, TactileSensorConfig],
        port: str,
        baudrate: int,
        mode_command: bytes,
        v_min: float,
        v_max: float,
    ):
        if serial is None:
            raise RuntimeError("pyserial is not installed.")

        super().__init__(sensors)
        self._v_min = v_min
        self._v_max = v_max
        if any(not 0 <= spec.channel < TEENSY_NUM_CHANNELS for spec in sensors.values()):
            raise ValueError(f"Teensy tactile channels must be in [0, {TEENSY_NUM_CHANNELS - 1}].")
        self._channel_count = TEENSY_NUM_CHANNELS
        self._stop_event = threading.Event()
        self._queue: queue.Queue[tuple[int, int, np.ndarray]] = queue.Queue(maxsize=100)
        self._reader = _TeensySerialReader(port, baudrate, self._queue, self._stop_event, mode_command)
        self._reader.start()

    def drain(self) -> dict[str, np.ndarray]:
        chunks: dict[str, list[np.ndarray]] = {key: [] for key in self.sensors}

        while True:
            try:
                _, _, raw = self._queue.get_nowait()
            except queue.Empty:
                break

            if raw.size == 0:
                continue

            for key, spec in self.sensors.items():
                channel_raw = raw[spec.channel :: self._channel_count]
                if channel_raw.size:
                    chunks[key].append(raw_to_volts(channel_raw, self._v_min, self._v_max))

        return {key: np.concatenate(parts) for key, parts in chunks.items() if parts}

    def close(self) -> None:
        self._stop_event.set()
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)


class _OpenDaqBackend(_TactileBackend):
    def __init__(self, sensors: OrderedDict[str, TactileSensorConfig]):
        super().__init__(sensors)

        try:
            import opendaq
        except ImportError as exc:
            raise RuntimeError("opendaq is not installed.") from exc

        self._opendaq = opendaq
        self._instance = opendaq.Instance()
        available_devices = list(self._instance.available_devices)
        if not available_devices:
            raise RuntimeError("No openDAQ devices found.")

        target = next((device for device in available_devices if "IOLITE-X" in device.name), available_devices[0])
        self._device = self._instance.add_device(target.connection_string)
        self._readers: dict[str, object] = {}

        max_channel_required = max(spec.channel for spec in sensors.values())
        if len(self._device.channels) <= max_channel_required:
            logger.warning(
                "The tactile bench requires hardware channel index %s, but the device only exposes %s channels.",
                max_channel_required,
                len(self._device.channels),
            )

        sample_rates = {spec.sample_rate_hz for spec in sensors.values()}
        if len(sample_rates) == 1:
            try:
                self._device.set_property_value("SampleRate", sample_rates.pop())
            except Exception:
                logger.warning("Could not set SampleRate on tactile device.")
        else:
            logger.warning("Mixed tactile sample rates detected; using each sensor's declared rate for rendering.")

        for key, spec in sensors.items():
            if spec.channel >= len(self._device.channels):
                logger.warning("Skipping tactile sensor '%s' because channel %s is unavailable.", key, spec.channel)
                continue

            channel = self._device.channels[spec.channel]
            channel.active = True
            amplifier = channel.get_function_blocks()[0]
            self._configure_amplifier(amplifier, spec)
            self._readers[key] = opendaq.StreamReader(channel.signals[0])
            logger.info("Bound tactile sensor '%s' to openDAQ channel %s", key, spec.channel)

        self.available_keys = tuple(self._readers.keys())

        logger.info("Connected tactile stream from %s.", target.name)

    @staticmethod
    def _configure_amplifier(amplifier, spec: TactileSensorConfig) -> None:
        measurement_map = {"Voltage": 0, "IEPE": 1}
        range_map = {10_000: 0, 5_000: 1, 1_000: 2, 200: 3}
        hpf_map = {0.0: 0, 0.1: 0, 1.0: 1}
        excitation_map = {2: 0, 4: 1, 6: 2}

        amplifier.set_property_value("Measurement", measurement_map[spec.measurement])
        amplifier.set_property_value("Range", range_map.get(spec.range_choice, 0))
        amplifier.set_property_value("HPFilter", hpf_map.get(spec.hpf_choice, 0))
        if spec.measurement == "IEPE":
            amplifier.set_property_value("Excitation", excitation_map.get(spec.excitation_choice, 1))

    def drain(self) -> dict[str, np.ndarray]:
        chunks: dict[str, np.ndarray] = {}
        for key, reader in self._readers.items():
            available = reader.available_count
            if available <= 0:
                continue

            chunk = np.asarray(reader.read(available), dtype=np.float32)
            if chunk.size:
                chunks[key] = chunk
        return chunks


def _build_sensors(config: SpectRoFollowerConfig) -> OrderedDict[str, TactileSensorConfig]:
    if config.tactile_sensors:
        return OrderedDict(config.tactile_sensors.items())

    if _normalize_tactile_data_stream(config.tactile_data_stream) == "OpenDAQ":
        return OrderedDict(_make_opendaq_default_sensors().items())

    return OrderedDict(_make_teensy_default_sensors().items())


class SpectRoFollower(SOFollower):
    """SO follower with a tactile spectrogram stream from either Teensy or openDAQ."""

    config_class = SpectRoFollowerConfig
    name = "spectrobot"

    def __init__(self, config: SpectRoFollowerConfig):
        super().__init__(config)
        self.config = config
        self._stream_name = _normalize_tactile_data_stream(config.tactile_data_stream)
        self._sensors = _build_sensors(config)
        self._target_size = (224,224)  # width, height
        self._target_fps = float(config.tactile_fps)

        self._sensor_states: OrderedDict[str, _SensorState] = OrderedDict()
        for key, spec in self._sensors.items():
            observation_key = spec.observation_key or f"tactile_spectrogram_{key}_nfft_{spec.nfft}"
            buffer_size = (self._target_size[0] - 1) * (spec.nfft // 2) + spec.nfft
            self._sensor_states[key] = _SensorState(
                key=key,
                spec=spec,
                observation_key=observation_key,
                buffer=np.zeros(buffer_size, dtype=np.float32),
                frame=np.zeros((self._target_size[1], self._target_size[0], 3), dtype=np.uint8),
            )

        self._lock = threading.Lock()
        self._running = True
        self._backend: _TactileBackend | None = None

        try:
            if self._stream_name == "OpenDAQ":
                self._backend = _OpenDaqBackend(self._sensors)
            else:
                TACTILE_V_MIN = -10.24
                TACTILE_V_MAX = 10.24
                TACTILE_BAUDRATE = 2_000_000
                MODE_COMMAND = "A".encode("ascii", errors="ignore")[:1] or b"A"

                self._backend = _TeensyBackend(
                    self._sensors,
                    port=config.tactile_serial_port,
                    baudrate=TACTILE_BAUDRATE,
                    mode_command=MODE_COMMAND,
                    v_min=TACTILE_V_MIN,
                    v_max=TACTILE_V_MAX,
                )
        except Exception as exc:  # nosec B110
            logger.warning("Failed to initialize tactile backend: %s", exc)

        if self._backend is not None:
            available_keys = set(self._backend.available_keys)
            self._sensor_states = OrderedDict(
                (key, state) for key, state in self._sensor_states.items() if key in available_keys
            )

        self._worker_thread = threading.Thread(target=self._spectrogram_worker, daemon=True)
        self._worker_thread.start()

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = dict(super().observation_features)
        for state in self._sensor_states.values():
            features[state.observation_key] = (self._target_size[1], self._target_size[0], 3)
        return features

    @staticmethod
    def _push_to_ring_buffer(buffer: np.ndarray, chunk: np.ndarray) -> None:
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
        nperseg = min(nfft, buffer.size)
        if nperseg < 16:
            return None

        frequencies, _, sxx = scipy.signal.spectrogram(
            buffer,
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=nperseg // 2,
        )

        if sxx.size == 0 or frequencies.size == 0:
            return None

        sxx_db = 10.0 * np.log10(sxx + 1e-12)
        normalized = np.clip(sxx_db, min_db, max_db)
        normalized = (normalized - min_db) / max(max_db - min_db, 1e-6)
        image_gray = np.uint8(np.flipud(normalized) * 255.0)
        spectro_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        spectro_bgr = cv2.resize(spectro_bgr, self._target_size, interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(spectro_bgr, cv2.COLOR_BGR2RGB)

    def _spectrogram_worker(self) -> None:
        interval = 1.0 / max(self._target_fps, 1e-6)
        last_compute_time = time.perf_counter()

        while self._running:
            now = time.perf_counter()

            if self._backend is not None:
                for key, chunk in self._backend.drain().items():
                    state = self._sensor_states.get(key)
                    if state is not None and chunk.size:
                        self._push_to_ring_buffer(state.buffer, chunk)

            if now - last_compute_time >= interval:
                new_frames: dict[str, np.ndarray] = {}
                for state in self._sensor_states.values():
                    frame = self._render_spectrogram_frame(
                        state.buffer,
                        fs_hz=state.spec.sample_rate_hz,
                        nfft=state.spec.nfft,
                        min_db=state.spec.min_db,
                        max_db=state.spec.max_db,
                    )
                    if frame is not None:
                        new_frames[state.observation_key] = frame

                if new_frames:
                    with self._lock:
                        for key, frame in new_frames.items():
                            for state in self._sensor_states.values():
                                if state.observation_key == key:
                                    state.frame = frame
                                    break

                last_compute_time = now

            time.sleep(0.001)

    def get_observation(self) -> RobotObservation:
        tick_start = time.perf_counter()
        obs = super().get_observation()

        with self._lock:
            for state in self._sensor_states.values():
                obs[state.observation_key] = state.frame.copy()

        total_latency = (time.perf_counter() - tick_start) * 1000
        logger.debug("Tactile observation sync completed in %.2fms", total_latency)
        return obs

    def close(self):
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        if self._backend is not None:
            self._backend.close()

        super().close()