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
TEENSY_SAMPLE_RATE_HZ = 20_000  # fixed per-channel rate of the firmware


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


class _StreamingDecimator:
    """Anti-aliased integer decimation that keeps filter state across chunks (no edge artifacts)."""

    def __init__(self, factor: int):
        self.factor = factor
        # Same FIR design as scipy.signal.decimate(ftype="fir").
        self._taps = scipy.signal.firwin(20 * factor + 1, 1.0 / factor, window="hamming")
        self._zi = np.zeros(self._taps.size - 1)
        self._phase = 0  # index of the next sample to keep in the upcoming chunk

    def process(self, chunk: np.ndarray) -> np.ndarray:
        filtered, self._zi = scipy.signal.lfilter(self._taps, 1.0, chunk, zi=self._zi)
        out = filtered[self._phase :: self.factor]
        self._phase = (self._phase - chunk.size) % self.factor
        return out.astype(np.float32, copy=False)


@dataclass
class _SensorState:
    key: str
    spec: TactileSensorConfig
    observation_key: str
    buffer: np.ndarray
    frame: np.ndarray
    decimator: _StreamingDecimator | None = None


class _TactileBackend(abc.ABC):
    def __init__(self, sensors: OrderedDict[str, TactileSensorConfig]):
        self.sensors = sensors
        self.available_keys = tuple(sensors.keys())
        # Rate at which the hardware actually delivers samples; sensors declaring a lower
        # sample_rate_hz are decimated from it.
        self.acquisition_rate_hz: int = max(spec.sample_rate_hz for spec in sensors.values())

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
        self.acquisition_rate_hz = TEENSY_SAMPLE_RATE_HZ
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
        # One reader per physical channel, fanned out to every sensor key bound to it, so the
        # same channel can be exposed several times (e.g. full band + low-frequency zoom).
        self._readers: dict[int, object] = {}
        self._channel_keys: dict[int, list[str]] = {}

        max_channel_required = max(spec.channel for spec in sensors.values())
        if len(self._device.channels) <= max_channel_required:
            logger.warning(
                "The tactile bench requires hardware channel index %s, but the device only exposes %s channels.",
                max_channel_required,
                len(self._device.channels),
            )

        # Acquire at the highest requested rate; lower-rate sensors are decimated in software.
        try:
            self._device.set_property_value("SampleRate", self.acquisition_rate_hz)
            self.acquisition_rate_hz = int(self._device.get_property_value("SampleRate"))
        except Exception:
            logger.warning("Could not set SampleRate=%s on tactile device.", self.acquisition_rate_hz)

        for key, spec in sensors.items():
            if spec.channel >= len(self._device.channels):
                logger.warning("Skipping tactile sensor '%s' because channel %s is unavailable.", key, spec.channel)
                continue
            self._channel_keys.setdefault(spec.channel, []).append(key)

        for channel_index, keys in self._channel_keys.items():
            spec = sensors[keys[0]]
            analog = (spec.measurement, spec.range_choice, spec.hpf_choice, spec.excitation_choice)
            for other in keys[1:]:
                o = sensors[other]
                if (o.measurement, o.range_choice, o.hpf_choice, o.excitation_choice) != analog:
                    logger.warning(
                        "Sensors %s share openDAQ channel %s but have different amplifier settings; using '%s'.",
                        keys,
                        channel_index,
                        keys[0],
                    )
                    break

            channel = self._device.channels[channel_index]
            channel.active = True
            amplifier = channel.get_function_blocks()[0]
            self._configure_amplifier(amplifier, spec)
            self._readers[channel_index] = opendaq.StreamReader(channel.signals[0])
            logger.info("Bound tactile sensors %s to openDAQ channel %s", keys, channel_index)

            # The SampleRate property may be ignored or rounded by the device; the signal's
            # time domain is the ground truth for the rate the samples actually arrive at.
            signal_rate = self._signal_rate_hz(channel.signals[0])
            if signal_rate is not None and signal_rate != self.acquisition_rate_hz:
                logger.warning(
                    "openDAQ channel %s delivers %d Hz (requested %d Hz); decimating from %d Hz.",
                    channel_index,
                    signal_rate,
                    self.acquisition_rate_hz,
                    signal_rate,
                )
                self.acquisition_rate_hz = signal_rate

        self.available_keys = tuple(key for keys in self._channel_keys.values() for key in keys)

        logger.info("Connected tactile stream from %s.", target.name)

    @staticmethod
    def _signal_rate_hz(signal) -> int | None:
        try:
            descriptor = signal.domain_signal.descriptor
            resolution = descriptor.tick_resolution
            delta = descriptor.rule.parameters["delta"]
            return round(resolution.denominator / (resolution.numerator * delta))
        except Exception:
            logger.warning("Could not read the sample rate from the openDAQ signal domain.")
            return None

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
        for channel_index, reader in self._readers.items():
            available = reader.available_count
            if available <= 0:
                continue

            chunk = np.asarray(reader.read(available), dtype=np.float32)
            if chunk.size:
                for key in self._channel_keys[channel_index]:
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
            self._setup_decimators(self._backend.acquisition_rate_hz)

        self._worker_thread = threading.Thread(target=self._spectrogram_worker, daemon=True)
        self._worker_thread.start()

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = dict(super().observation_features)
        for state in self._sensor_states.values():
            features[state.observation_key] = (self._target_size[1], self._target_size[0], 3)
        return features

    def _setup_decimators(self, acquisition_rate_hz: int) -> None:
        for state in self._sensor_states.values():
            target_hz = state.spec.sample_rate_hz
            factor, remainder = divmod(acquisition_rate_hz, target_hz)
            if factor < 1 or remainder:
                raise ValueError(
                    f"Tactile sensor '{state.key}': sample_rate_hz={target_hz} must divide the acquisition "
                    f"rate {acquisition_rate_hz} Hz by an integer factor."
                )
            if factor > 1:
                state.decimator = _StreamingDecimator(factor)

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

    def _log_spectrogram_setup(self) -> None:
        acquisition_hz = self._backend.acquisition_rate_hz if self._backend is not None else None
        for state in self._sensor_states.values():
            fs = state.spec.sample_rate_hz
            nfft = state.spec.nfft
            factor = state.decimator.factor if state.decimator is not None else 1
            logger.info(
                "[tactile] %s: acquisition=%s Hz, decimation=x%d -> fs=%d Hz, band=0-%.0f Hz, nfft=%d "
                "(window %.1f ms, df=%.2f Hz), image spans %.2f s",
                state.observation_key,
                acquisition_hz,
                factor,
                fs,
                fs / 2,
                nfft,
                1000.0 * nfft / fs,
                fs / nfft,
                state.buffer.size / fs,
            )

    def _log_measured_rates(self, raw_counts: dict[str, int], out_counts: dict[str, int], elapsed: float) -> None:
        for state in self._sensor_states.values():
            raw_hz = raw_counts.get(state.key, 0) / elapsed
            out_hz = out_counts.get(state.key, 0) / elapsed
            expected = state.spec.sample_rate_hz
            if abs(out_hz - expected) > 0.1 * expected:
                logger.warning(
                    "[tactile] %s: measured input=%.0f Hz, output fs=%.0f Hz but configured fs=%d Hz",
                    state.observation_key,
                    raw_hz,
                    out_hz,
                    expected,
                )

    def _spectrogram_worker(self) -> None:
        interval = 1.0 / max(self._target_fps, 1e-6)
        last_compute_time = time.perf_counter()
        # Measure the real per-spectrogram sample rates once, a few seconds after start.
        self._log_spectrogram_setup()
        rate_check_start: float | None = None
        raw_counts: dict[str, int] = {}
        out_counts: dict[str, int] = {}
        rate_check_done = False

        while self._running:
            now = time.perf_counter()

            if self._backend is not None:
                drained = self._backend.drain()
                # The first batch holds the backlog since connect, so measuring starts after it.
                measuring = not rate_check_done and rate_check_start is not None
                if drained and rate_check_start is None:
                    rate_check_start = now
                for key, chunk in drained.items():
                    state = self._sensor_states.get(key)
                    if state is None or not chunk.size:
                        continue
                    if measuring:
                        raw_counts[key] = raw_counts.get(key, 0) + chunk.size
                    if state.decimator is not None:
                        chunk = state.decimator.process(chunk)
                    if measuring:
                        out_counts[key] = out_counts.get(key, 0) + chunk.size
                    if chunk.size:
                        self._push_to_ring_buffer(state.buffer, chunk)
                if measuring and now - rate_check_start >= 3.0:
                    self._log_measured_rates(raw_counts, out_counts, now - rate_check_start)
                    rate_check_done = True

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