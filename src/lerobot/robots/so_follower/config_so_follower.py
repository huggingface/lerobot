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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True



@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass

@RobotConfig.register_subclass("so101_follower_dragontactile")
@dataclass
class SO101FollowerDragontactileConfig(SOFollowerRobotConfig):
    pass

@RobotConfig.register_subclass("so101_follower_dragon_multitactile")
class SO101FollowerDragonMultitactileConfig(SOFollowerRobotConfig):
    pass

@RobotConfig.register_subclass("so101_follower_dragon_multiduration")
@dataclass
class SO101FollowerDragonMultidurationConfig(SOFollowerRobotConfig):
    pass


@RobotConfig.register_subclass("so101_follower_dragon_tactile_bench")
@dataclass
class SO101FollowerDragonTactileBenchConfig(SOFollowerRobotConfig):
    bench_setup: str = "5sensors"
    tactile_obs_key: str = "all" #"tactile_spectrogram_accelero_10kHz_nfft_512"
    
@RobotConfig.register_subclass("so101_follower_teensy_tactile")
@dataclass
class SO101FollowerTeensyTactileConfig(SOFollowerRobotConfig):
    num_channels: str = "1"


@RobotConfig.register_subclass("spectrobot_trifold")
@dataclass
class SpectrobotTrifoldConfig(SOFollowerRobotConfig):
    # IOLITE-X hardware channel index of each of the 3 IEPE Dragonfly sensors
    dragonfly_channels: list[int] = field(default_factory=lambda: [0, 1, 2])

    # Spectrograms added to the observation: one grayscale image per sensor and/or one RGB image
    # with R, G, B = dragonfly_1, dragonfly_2, dragonfly_3
    per_sensor_spectrograms: bool = True
    rgb_spectrogram: bool = True
    # dB range mapped to black..white in the spectrogram images
    spectrogram_min_db: float = -72.0
    spectrogram_max_db: float = 40.0

    # Force vector KPI, in the finger frame (z along the finger towards the tip, x and y across it).
    # The 20 kS/s stream is block-averaged down to this rate before computing the force
    force_rate_hz: int = 1000
    # Moving-average window applied on the downsampled signals
    force_window_s: float = 0.02
    # Time constant of the slow baseline removed from each signal (drift / IEPE settling)
    force_baseline_tau_s: float = 0.5
    # Force is reported as 0 during the first seconds, while the baseline settles
    force_warmup_s: float = 2.0
    # Rotation axis measured by each sensor (row i = sensor i, flip a sign to invert a sensor).
    # Default: ch0 -> rotation around y, ch1 -> rotation around x, ch2 -> rotation around z
    sensor_rotation_axes: list[list[float]] = field(
        default_factory=lambda: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # Moment per unit of rotation signal (N.m / signal unit). 1.0 until calibrated
    rotation_stiffness: float = 1.0
    # Point where the force is applied, from the sensors (m). The force is recovered from M = r x F
    force_lever_arm_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.075])
    # Rolling window of the 3 sensor signals kept for display (baseline removed, downsampled)
    timeseries_duration_s: float = 2.0
    timeseries_rate_hz: int = 200
    # Rerun 3D view: arrow length in meters per unit. None auto-scales each vector on its recent peak
    force_arrow_scale: float | None = None



SO100FollowerConfig = SOFollowerRobotConfig 
SO101FollowerConfig = SOFollowerRobotConfig