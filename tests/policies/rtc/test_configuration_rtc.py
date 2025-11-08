#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for RTC configuration module."""

import pytest

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

# ====================== Initialization Tests ======================


def test_rtc_config_default_initialization():
    """Test RTCConfig initializes with default values."""
    config = RTCConfig()

    assert config.enabled is False
    assert config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR
    assert config.max_guidance_weight == 10.0
    assert config.execution_horizon == 10
    assert config.debug is False
    assert config.debug_maxlen == 100


def test_rtc_config_custom_initialization():
    """Test RTCConfig initializes with custom values."""
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=5.0,
        execution_horizon=20,
        debug=True,
        debug_maxlen=200,
    )

    assert config.enabled is True
    assert config.prefix_attention_schedule == RTCAttentionSchedule.EXP
    assert config.max_guidance_weight == 5.0
    assert config.execution_horizon == 20
    assert config.debug is True
    assert config.debug_maxlen == 200


def test_rtc_config_partial_initialization():
    """Test RTCConfig with partial custom values."""
    config = RTCConfig(enabled=True, max_guidance_weight=15.0)

    assert config.enabled is True
    assert config.max_guidance_weight == 15.0
    # Other values should be defaults
    assert config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR
    assert config.execution_horizon == 10
    assert config.debug is False


# ====================== Validation Tests ======================


def test_rtc_config_validates_positive_max_guidance_weight():
    """Test RTCConfig validates max_guidance_weight is positive."""
    with pytest.raises(ValueError, match="max_guidance_weight must be positive"):
        RTCConfig(max_guidance_weight=0.0)

    with pytest.raises(ValueError, match="max_guidance_weight must be positive"):
        RTCConfig(max_guidance_weight=-1.0)


def test_rtc_config_validates_positive_debug_maxlen():
    """Test RTCConfig validates debug_maxlen is positive."""
    with pytest.raises(ValueError, match="debug_maxlen must be positive"):
        RTCConfig(debug_maxlen=0)

    with pytest.raises(ValueError, match="debug_maxlen must be positive"):
        RTCConfig(debug_maxlen=-10)


def test_rtc_config_accepts_valid_max_guidance_weight():
    """Test RTCConfig accepts valid positive max_guidance_weight."""
    config1 = RTCConfig(max_guidance_weight=0.1)
    assert config1.max_guidance_weight == 0.1

    config2 = RTCConfig(max_guidance_weight=100.0)
    assert config2.max_guidance_weight == 100.0


def test_rtc_config_accepts_valid_debug_maxlen():
    """Test RTCConfig accepts valid positive debug_maxlen."""
    config1 = RTCConfig(debug_maxlen=1)
    assert config1.debug_maxlen == 1

    config2 = RTCConfig(debug_maxlen=10000)
    assert config2.debug_maxlen == 10000


# ====================== Attention Schedule Tests ======================


def test_rtc_config_with_linear_schedule():
    """Test RTCConfig with LINEAR attention schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
    assert config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR


def test_rtc_config_with_exp_schedule():
    """Test RTCConfig with EXP attention schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP)
    assert config.prefix_attention_schedule == RTCAttentionSchedule.EXP


def test_rtc_config_with_zeros_schedule():
    """Test RTCConfig with ZEROS attention schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.ZEROS)
    assert config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS


def test_rtc_config_with_ones_schedule():
    """Test RTCConfig with ONES attention schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.ONES)
    assert config.prefix_attention_schedule == RTCAttentionSchedule.ONES


# ====================== Enabled/Disabled Tests ======================


def test_rtc_config_enabled_true():
    """Test RTCConfig with enabled=True."""
    config = RTCConfig(enabled=True)
    assert config.enabled is True


def test_rtc_config_enabled_false():
    """Test RTCConfig with enabled=False."""
    config = RTCConfig(enabled=False)
    assert config.enabled is False


# ====================== Debug Tests ======================


def test_rtc_config_debug_enabled():
    """Test RTCConfig with debug enabled."""
    config = RTCConfig(debug=True, debug_maxlen=500)
    assert config.debug is True
    assert config.debug_maxlen == 500


def test_rtc_config_debug_disabled():
    """Test RTCConfig with debug disabled."""
    config = RTCConfig(debug=False)
    assert config.debug is False


# ====================== Execution Horizon Tests ======================


def test_rtc_config_with_small_execution_horizon():
    """Test RTCConfig with small execution horizon."""
    config = RTCConfig(execution_horizon=1)
    assert config.execution_horizon == 1


def test_rtc_config_with_large_execution_horizon():
    """Test RTCConfig with large execution horizon."""
    config = RTCConfig(execution_horizon=100)
    assert config.execution_horizon == 100


def test_rtc_config_with_zero_execution_horizon():
    """Test RTCConfig accepts zero execution horizon."""
    # No validation on execution_horizon, so this should work
    config = RTCConfig(execution_horizon=0)
    assert config.execution_horizon == 0


def test_rtc_config_with_negative_execution_horizon():
    """Test RTCConfig accepts negative execution horizon."""
    # No validation on execution_horizon, so this should work
    config = RTCConfig(execution_horizon=-1)
    assert config.execution_horizon == -1


# ====================== Integration Tests ======================


def test_rtc_config_typical_production_settings():
    """Test RTCConfig with typical production settings."""
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=10.0,
        execution_horizon=8,
        debug=False,
    )

    assert config.enabled is True
    assert config.prefix_attention_schedule == RTCAttentionSchedule.EXP
    assert config.max_guidance_weight == 10.0
    assert config.execution_horizon == 8
    assert config.debug is False


def test_rtc_config_typical_debug_settings():
    """Test RTCConfig with typical debug settings."""
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=5.0,
        execution_horizon=10,
        debug=True,
        debug_maxlen=1000,
    )

    assert config.enabled is True
    assert config.debug is True
    assert config.debug_maxlen == 1000


def test_rtc_config_disabled_mode():
    """Test RTCConfig in disabled mode."""
    config = RTCConfig(enabled=False)

    assert config.enabled is False
    # Other settings still accessible even when disabled
    assert config.max_guidance_weight == 10.0
    assert config.execution_horizon == 10


# ====================== Dataclass Tests ======================


def test_rtc_config_is_dataclass():
    """Test that RTCConfig is a dataclass."""
    from dataclasses import is_dataclass

    assert is_dataclass(RTCConfig)


def test_rtc_config_equality():
    """Test RTCConfig equality comparison."""
    config1 = RTCConfig(enabled=True, max_guidance_weight=5.0)
    config2 = RTCConfig(enabled=True, max_guidance_weight=5.0)
    config3 = RTCConfig(enabled=False, max_guidance_weight=5.0)

    assert config1 == config2
    assert config1 != config3


def test_rtc_config_repr():
    """Test RTCConfig string representation."""
    config = RTCConfig(enabled=True, execution_horizon=20)
    repr_str = repr(config)

    assert "RTCConfig" in repr_str
    assert "enabled=True" in repr_str
    assert "execution_horizon=20" in repr_str


# ====================== Edge Cases Tests ======================


def test_rtc_config_very_small_max_guidance_weight():
    """Test RTCConfig with very small positive max_guidance_weight."""
    config = RTCConfig(max_guidance_weight=1e-10)
    assert config.max_guidance_weight == pytest.approx(1e-10)


def test_rtc_config_very_large_max_guidance_weight():
    """Test RTCConfig with very large max_guidance_weight."""
    config = RTCConfig(max_guidance_weight=1e10)
    assert config.max_guidance_weight == pytest.approx(1e10)


def test_rtc_config_minimum_debug_maxlen():
    """Test RTCConfig with minimum valid debug_maxlen."""
    config = RTCConfig(debug_maxlen=1)
    assert config.debug_maxlen == 1


def test_rtc_config_float_max_guidance_weight():
    """Test RTCConfig with float max_guidance_weight."""
    config = RTCConfig(max_guidance_weight=3.14159)
    assert config.max_guidance_weight == pytest.approx(3.14159)


# ====================== Type Tests ======================


def test_rtc_config_enabled_type():
    """Test RTCConfig enabled field accepts boolean."""
    config = RTCConfig(enabled=True)
    assert isinstance(config.enabled, bool)


def test_rtc_config_execution_horizon_type():
    """Test RTCConfig execution_horizon field accepts integer."""
    config = RTCConfig(execution_horizon=15)
    assert isinstance(config.execution_horizon, int)


def test_rtc_config_max_guidance_weight_type():
    """Test RTCConfig max_guidance_weight field accepts float."""
    config = RTCConfig(max_guidance_weight=7.5)
    assert isinstance(config.max_guidance_weight, float)


def test_rtc_config_debug_maxlen_type():
    """Test RTCConfig debug_maxlen field accepts integer."""
    config = RTCConfig(debug_maxlen=200)
    assert isinstance(config.debug_maxlen, int)
