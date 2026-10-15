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

"""Tests for tactile sensor module."""

import numpy as np
import pytest

from lerobot.tactile.configs import TactileDataType
from lerobot.tactile.paxini.tactile_paxini import (
    HEADER_AUTO,
    build_read_request,
    build_write_request,
    checksum,
    decode_resultant_payload,
    parse_connected_slots,
)
from lerobot.tactile.simulated import SimulatedTactile, SimulatedTactileConfig


class TestTactileSensorConfig:
    """Test tactile sensor configuration classes."""

    def test_simulated_config_defaults(self):
        config = SimulatedTactileConfig()
        assert config.fps == 30
        assert config.num_points == 400
        assert config.data_type == TactileDataType.FULL
        assert config.data_dim == 6
        assert config.expected_shape == (400, 6)

    def test_data_type_dimension(self):
        assert SimulatedTactileConfig(data_type=TactileDataType.DISPLACEMENT).data_dim == 3
        assert SimulatedTactileConfig(data_type=TactileDataType.FORCE).data_dim == 3
        assert SimulatedTactileConfig(data_type=TactileDataType.FULL).data_dim == 6
        assert SimulatedTactileConfig(data_type=TactileDataType.WRENCH).data_dim == 6

    def test_expected_shape(self):
        config = SimulatedTactileConfig(num_points=100, data_type=TactileDataType.FULL)
        assert config.expected_shape == (100, 6)

    def test_wrench_shape(self):
        config = SimulatedTactileConfig(num_points=5, data_type=TactileDataType.WRENCH)
        assert config.expected_shape == (5, 6)

    def test_config_type(self):
        assert SimulatedTactileConfig().type == "simulated"

    def test_invalid_data_type(self):
        with pytest.raises(ValueError, match="expected to be in"):
            SimulatedTactileConfig(data_type="invalid")

    def test_invalid_fps(self):
        with pytest.raises(ValueError, match="fps.*must be positive"):
            SimulatedTactileConfig(fps=0)

    def test_invalid_num_points(self):
        with pytest.raises(ValueError, match="num_points.*must be positive"):
            SimulatedTactileConfig(num_points=-1)

    def test_data_type_string_coercion(self):
        assert SimulatedTactileConfig(data_type="full").data_type == TactileDataType.FULL
        assert SimulatedTactileConfig(data_type="wrench").data_type == TactileDataType.WRENCH


class TestSimulatedTactile:
    """Test simulated tactile sensor."""

    def test_connect_disconnect(self):
        sensor = SimulatedTactile(SimulatedTactileConfig())
        assert not sensor.is_connected
        sensor.connect(warmup=False)
        assert sensor.is_connected
        sensor.disconnect()
        assert not sensor.is_connected

    def test_context_manager(self):
        with SimulatedTactile(SimulatedTactileConfig()) as sensor:
            assert sensor.is_connected
        assert not sensor.is_connected

    def test_read_shape(self):
        with SimulatedTactile(SimulatedTactileConfig()) as sensor:
            data = sensor.read()
            assert data.shape == (400, 6)
            assert data.dtype == np.float64

    def test_read_without_connect_raises(self):
        sensor = SimulatedTactile(SimulatedTactileConfig())
        with pytest.raises(ConnectionError):
            sensor.read()

    def test_find_sensors(self):
        sensors = SimulatedTactile.find_sensors()
        assert len(sensors) >= 1
        assert sensors[0]["type"] == "simulated"

    def test_reproducible_with_seed(self):
        with (
            SimulatedTactile(SimulatedTactileConfig(seed=42)) as s1,
            SimulatedTactile(SimulatedTactileConfig(seed=42)) as s2,
        ):
            np.testing.assert_array_almost_equal(s1.read(), s2.read())

    def test_different_data_types(self):
        for data_type, expected_dim in [
            (TactileDataType.DISPLACEMENT, 3),
            (TactileDataType.FORCE, 3),
            (TactileDataType.FULL, 6),
            (TactileDataType.WRENCH, 6),
        ]:
            with SimulatedTactile(SimulatedTactileConfig(data_type=data_type)) as sensor:
                assert sensor.read().shape == (400, expected_dim)

    def test_no_delay_by_default(self):
        import time

        with SimulatedTactile(SimulatedTactileConfig(simulate_delay=False)) as sensor:
            start = time.perf_counter()
            for _ in range(10):
                sensor.read()
            assert time.perf_counter() - start < 1.0


class TestPaxiniCodec:
    """Test the PaXini wire-format codec (pure functions, no hardware)."""

    def test_checksum_roundtrip(self):
        frame = bytes([0xAA, 0x56, 0x00, 0x06, 0x00, 1, 2, 3, 4, 5, 6])
        lrc = checksum(frame)
        assert (sum(frame) + lrc) & 0xFF == 0

    def test_build_read_request_layout(self):
        req = build_read_request(0x0010, 4)
        assert req[:2] == bytes([0x55, 0xAA])
        assert req[3] == 0x03
        assert int.from_bytes(req[4:6], "little") == 0x0010
        assert int.from_bytes(req[6:8], "little") == 4
        assert req[-1] == checksum(req[:-1])

    def test_build_write_request_layout(self):
        req = build_write_request(0x0017, bytes([0x01]))
        assert req[:2] == bytes([0x55, 0xAA])
        assert req[3] == 0x10
        assert int.from_bytes(req[4:6], "little") == 0x0017
        assert req[8] == 0x01
        assert req[-1] == checksum(req[:-1])

    def test_decode_resultant_payload(self):
        # Two sensors: (+1.2N, -0.5N, 2.0N) and (0, 0, 0.1N)
        payload = bytes([12, 0, 251, 0xFF, 20, 0]) + bytes([0, 0, 0, 0, 1, 0])
        out = decode_resultant_payload(payload, 2)
        np.testing.assert_allclose(out[0], [1.2, -0.5, 2.0], atol=1e-9)
        np.testing.assert_allclose(out[1], [0.0, 0.0, 0.1], atol=1e-9)

    def test_decode_resultant_size_mismatch(self):
        with pytest.raises(ValueError, match="size mismatch"):
            decode_resultant_payload(bytes(5), 1)

    def test_parse_connected_slots(self):
        # Slots at (0,2),(0,6),(1,2),(1,6),(2,2): set slot0, slot2, slot4
        register = bytes([0b0000_0100, 0b0000_0100, 0b0000_0100, 0x00])
        assert parse_connected_slots(register) == [0, 2, 4]

    def test_auto_frame_lrc(self):
        payload = bytes(6)
        meta = bytes([0x00]) + len(payload).to_bytes(2, "little")
        frame_wo_lrc = HEADER_AUTO + meta + payload
        assert checksum(frame_wo_lrc) == (0x100 - (sum(frame_wo_lrc) & 0xFF)) & 0xFF


class TestPaxiniConfig:
    """Test PaXini configuration validation."""

    def test_defaults(self):
        from lerobot.tactile.paxini import PaxiniTactileConfig

        config = PaxiniTactileConfig()
        assert config.num_points == 5
        assert config.data_type == TactileDataType.FORCE
        assert config.expected_shape == (5, 3)
        assert config.type == "paxini"

    def test_rejects_non_force(self):
        from lerobot.tactile.paxini import PaxiniTactileConfig

        with pytest.raises(ValueError, match="only support data_type=FORCE"):
            PaxiniTactileConfig(data_type=TactileDataType.FULL)

    def test_rejects_bad_num_points(self):
        from lerobot.tactile.paxini import PaxiniTactileConfig

        with pytest.raises(ValueError, match="1-5 fingertip"):
            PaxiniTactileConfig(num_points=6)
