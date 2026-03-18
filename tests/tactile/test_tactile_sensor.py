#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from lerobot.tactile.simulated import SimulatedTactile, SimulatedTactileConfig


class TestTactileSensorConfig:
    """Test tactile sensor configuration classes."""

    def test_simulated_config_defaults(self):
        """Test simulated config has correct defaults."""
        config = SimulatedTactileConfig()
        assert config.fps == 30
        assert config.num_points == 400
        assert config.data_type == TactileDataType.FULL
        assert config.data_dim == 6
        assert config.expected_shape == (400, 6)

    def test_data_type_dimension(self):
        """Test data_dim property for different data types."""
        config_disp = SimulatedTactileConfig(data_type=TactileDataType.DISPLACEMENT)
        assert config_disp.data_dim == 3

        config_force = SimulatedTactileConfig(data_type=TactileDataType.FORCE)
        assert config_force.data_dim == 3

        config_full = SimulatedTactileConfig(data_type=TactileDataType.FULL)
        assert config_full.data_dim == 6

    def test_expected_shape(self):
        """Test expected_shape property."""
        config = SimulatedTactileConfig(num_points=100, data_type=TactileDataType.FULL)
        assert config.expected_shape == (100, 6)

    def test_config_type(self):
        """Test type property returns registered name."""
        config = SimulatedTactileConfig()
        assert config.type == "simulated"

    def test_invalid_data_type(self):
        """Test invalid data type raises ValueError."""
        with pytest.raises(ValueError, match="expected to be in"):
            SimulatedTactileConfig(data_type="invalid")

    def test_invalid_fps(self):
        """Test invalid fps raises ValueError."""
        with pytest.raises(ValueError, match="fps.*must be positive"):
            SimulatedTactileConfig(fps=0)

    def test_invalid_num_points(self):
        """Test invalid num_points raises ValueError."""
        with pytest.raises(ValueError, match="num_points.*must be positive"):
            SimulatedTactileConfig(num_points=-1)

    def test_data_type_string_coercion(self):
        """Test that string data_type is coerced to enum."""
        config = SimulatedTactileConfig(data_type="full")
        assert config.data_type == TactileDataType.FULL


class TestSimulatedTactile:
    """Test simulated tactile sensor."""

    def test_connect_disconnect(self):
        """Test sensor connection lifecycle."""
        config = SimulatedTactileConfig()
        sensor = SimulatedTactile(config)

        assert not sensor.is_connected
        sensor.connect(warmup=False)
        assert sensor.is_connected
        sensor.disconnect()
        assert not sensor.is_connected

    def test_context_manager(self):
        """Test sensor as context manager."""
        config = SimulatedTactileConfig()
        with SimulatedTactile(config) as sensor:
            assert sensor.is_connected
        assert not sensor.is_connected

    def test_read_shape(self):
        """Test read returns correct shape."""
        config = SimulatedTactileConfig()
        with SimulatedTactile(config) as sensor:
            data = sensor.read()
            assert data.shape == (400, 6)
            assert data.dtype == np.float64

    def test_read_without_connect_raises(self):
        """Test read without connect raises ConnectionError."""
        config = SimulatedTactileConfig()
        sensor = SimulatedTactile(config)
        with pytest.raises(ConnectionError):
            sensor.read()

    def test_find_sensors(self):
        """Test find_sensors returns at least simulated sensor."""
        sensors = SimulatedTactile.find_sensors()
        assert len(sensors) >= 1
        assert sensors[0]["type"] == "simulated"

    def test_reproducible_with_seed(self):
        """Test same seed produces same data."""
        config1 = SimulatedTactileConfig(seed=42)
        config2 = SimulatedTactileConfig(seed=42)

        with SimulatedTactile(config1) as s1, SimulatedTactile(config2) as s2:
            data1 = s1.read()
            data2 = s2.read()
            np.testing.assert_array_almost_equal(data1, data2)

    def test_different_data_types(self):
        """Test different data types produce correct shapes."""
        for data_type, expected_dim in [
            (TactileDataType.DISPLACEMENT, 3),
            (TactileDataType.FORCE, 3),
            (TactileDataType.FULL, 6),
        ]:
            config = SimulatedTactileConfig(data_type=data_type)
            with SimulatedTactile(config) as sensor:
                data = sensor.read()
                assert data.shape == (400, expected_dim)

    def test_no_delay_by_default(self):
        """Test simulated sensor doesn't delay by default."""
        import time

        config = SimulatedTactileConfig(simulate_delay=False)
        with SimulatedTactile(config) as sensor:
            start = time.perf_counter()
            for _ in range(10):
                sensor.read()
            elapsed = time.perf_counter() - start
            # Should complete quickly without delays
            assert elapsed < 1.0
