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
"""Configuration for keyboard teleoperators."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """KeyboardTeleopConfig"""

    # TODO(Steven): Consider setting in here the keys that we want to capture/listen


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    """Configuration for keyboard end-effector teleoperator.

    Used for controlling robot end-effectors with keyboard inputs.

    Attributes:
        use_gripper: Whether to include gripper control in actions
        use_orientation: Whether to expose rotational end-effector controls
        require_deadman: If True, motion commands are enabled only while deadman key is held
        linear_step: Per-cycle linear step command magnitude
        angular_step: Per-cycle angular step command magnitude
        gripper_step: Per-cycle gripper velocity command magnitude
    """

    use_gripper: bool = True
    use_orientation: bool = True
    require_deadman: bool = True
    linear_step: float = 0.004
    angular_step: float = 0.06
    gripper_step: float = 1.0


@TeleoperatorConfig.register_subclass("keyboard_joint")
@dataclass
class KeyboardJointTeleopConfig(KeyboardTeleopConfig):
    """Configuration for keyboard joint-level teleoperator.

    键盘按键直接控制各关节增量，无需运动学求解。

    按键映射:
        关节控制:
            Q/A: joint1 (底座旋转)
            W/S: joint2 (肩部)
            E/D: joint3 (肘部)
            R/F: joint4 (腕1)
            T/G: joint5 (腕2)
            Y/H: joint6 (腕3)
            U/J: joint7 (腕4/法兰)
            1/2: 夹爪 开/合
        安全:
            Space: 死人开关（按住才生效）
            ESC: 断开连接

    Attributes:
        joint_step: 每周期关节增量（弧度）
        gripper_step: 每周期夹爪增量（0-100 映射度数）
        require_deadman: 是否需要按住空格才允许运动
        num_joints: 关节数量（默认 7，适配 NERO）
    """

    joint_step: float = 0.05
    gripper_step: float = 2.0
    require_deadman: bool = False
    num_joints: int = 7


@TeleoperatorConfig.register_subclass("keyboard_rover")
@dataclass
class KeyboardRoverTeleopConfig(TeleoperatorConfig):
    """Configuration for keyboard rover teleoperator.

    Used for controlling mobile robots like EarthRover Mini Plus with WASD controls.

    Attributes:
        linear_speed: Default linear velocity magnitude (-1 to 1 range for SDK robots)
        angular_speed: Default angular velocity magnitude (-1 to 1 range for SDK robots)
        speed_increment: Amount to increase/decrease speed with +/- keys
        turn_assist_ratio: Forward motion multiplier when turning with A/D keys (0.0-1.0)
        angular_speed_ratio: Ratio of angular to linear speed for synchronized adjustments
        min_linear_speed: Minimum linear speed when decreasing (prevents zero speed)
        min_angular_speed: Minimum angular speed when decreasing (prevents zero speed)
    """

    linear_speed: float = 1.0
    angular_speed: float = 1.0
    speed_increment: float = 0.1
    turn_assist_ratio: float = 0.3
    angular_speed_ratio: float = 0.6
    min_linear_speed: float = 0.1
    min_angular_speed: float = 0.05
