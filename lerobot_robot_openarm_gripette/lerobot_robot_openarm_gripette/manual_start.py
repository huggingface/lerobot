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

import time
from collections.abc import Callable
from threading import Event, Thread

import numpy as np

from lerobot.robots.openarm_follower import OpenArmFollower


def prepare_manual_start(
    robot: OpenArmFollower,
    signs: np.ndarray,
    gripper_action: dict[str, float] | None = None,
    input_fn: Callable[[str], object] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
    engage_duration_s: float = 1.0,
    engage_frequency_hz: float = 20.0,
    disable_frequency_hz: float = 10.0,
    active_zero_torque: bool = False,
    zero_torque_frequency_hz: float = 50.0,
) -> np.ndarray:
    signs = np.asarray(signs, dtype=np.float64)
    if signs.shape != (7,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("signs must contain seven values, each +1 or -1")
    if (
        engage_duration_s <= 0
        or engage_frequency_hz <= 0
        or disable_frequency_hz <= 0
        or zero_torque_frequency_hz <= 0
    ):
        raise ValueError("engagement, disable, and zero-torque frequencies and duration must be positive")

    robot.bus.disable_torque()
    sleep_fn(0.05)
    robot.bus.disable_torque()
    if gripper_action is not None:
        robot.send_action(gripper_action)
    stop_positioning = Event()
    positioning_errors: list[BaseException] = []
    zero_torque_action: dict[str, float] = {}
    zero_gains: dict[str, float] = {}
    if active_zero_torque:
        robot.bus.configure_motors()
        observation = robot.get_observation()
        zero_torque_action = {f"joint_{i}.pos": float(observation[f"joint_{i}.pos"]) for i in range(1, 8)}
        zero_gains = {name.removesuffix(".pos"): 0.0 for name in zero_torque_action}
        robot.send_action(
            zero_torque_action,
            custom_kp=zero_gains,
            custom_kd=zero_gains,
        )
        robot.bus.enable_torque()

    def maintain_positioning_mode() -> None:
        frequency = zero_torque_frequency_hz if active_zero_torque else disable_frequency_hz
        while not stop_positioning.wait(1 / frequency):
            try:
                if active_zero_torque:
                    robot.send_action(
                        zero_torque_action,
                        custom_kp=zero_gains,
                        custom_kd=zero_gains,
                    )
                else:
                    robot.bus.disable_torque()
            except BaseException as error:
                positioning_errors.append(error)
                stop_positioning.set()

    positioning_thread = Thread(target=maintain_positioning_mode, daemon=True)
    positioning_thread.start()
    mode = "ACTIVE ZERO TORQUE" if active_zero_torque else "torque is OFF"
    try:
        input_fn(f"Arm {mode}. Position and keep supporting the arm, then press ENTER...")
    finally:
        stop_positioning.set()
        positioning_thread.join()
    if positioning_errors:
        raise positioning_errors[0]
    robot.bus.disable_torque()

    robot.bus.configure_motors()
    observation = robot.get_observation()
    measured = np.array([observation[f"joint_{i}.pos"] for i in range(1, 8)], dtype=np.float64)
    hold_action = {f"joint_{i}.pos": float(value) for i, value in enumerate(measured, start=1)}
    joint_names = [name.removesuffix(".pos") for name in hold_action]
    position_kp = dict(zip(joint_names, robot.config.position_kp[:7], strict=True))
    position_kd = dict(zip(joint_names, robot.config.position_kd[:7], strict=True))

    robot.send_action(
        hold_action,
        custom_kp=dict.fromkeys(joint_names, 0.0),
        custom_kd=position_kd,
    )
    robot.bus.enable_torque()
    ramp_steps = max(1, round(engage_duration_s * engage_frequency_hz))
    for scale in np.linspace(1 / ramp_steps, 1.0, ramp_steps):
        robot.send_action(
            hold_action,
            custom_kp={name: gain * scale for name, gain in position_kp.items()},
            custom_kd=position_kd,
        )
        sleep_fn(1 / engage_frequency_hz)
    return signs * measured
