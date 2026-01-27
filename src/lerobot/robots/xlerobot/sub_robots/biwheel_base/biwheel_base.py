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

import abc
from functools import cached_property

import numpy as np

from ....robot import Robot
from .config_biwheel_base import BiwheelBaseConfig


class BiwheelBase(Robot, abc.ABC):
    """Base class for biwheel robots with shared kinematics."""

    config_class = BiwheelBaseConfig
    name = "biwheel_base"

    @property
    def _state_ft(self) -> dict[str, type]:
        return {
            "x.vel": float,
            "theta.vel": float,
        }

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return self._state_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    def _body_to_wheel_linear(self, x: float, theta: float) -> tuple[float, float]:
        """Convert body-frame velocities into wheel linear velocities (m/s)."""
        theta_rad = np.deg2rad(theta)
        half_wheelbase = self.config.wheel_base / 2.0
        left_linear = x - theta_rad * half_wheelbase
        right_linear = x + theta_rad * half_wheelbase
        return left_linear, right_linear

    def _wheel_linear_to_body(self, left_linear: float, right_linear: float) -> dict[str, float]:
        """Convert wheel linear velocities (m/s) into body-frame velocities."""
        x_vel = (left_linear + right_linear) / 2.0
        theta_rad = (right_linear - left_linear) / self.config.wheel_base
        theta_vel = np.rad2deg(theta_rad)
        return {
            "x.vel": x_vel,
            "theta.vel": theta_vel,
        }

    def _apply_inversion(self, left: float, right: float) -> tuple[float, float]:
        if self.config.invert_left_motor:
            left = -left
        if self.config.invert_right_motor:
            right = -right
        return left, right

    def _remove_inversion(self, left: float, right: float) -> tuple[float, float]:
        if self.config.invert_left_motor:
            left = -left
        if self.config.invert_right_motor:
            right = -right
        return left, right
