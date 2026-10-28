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

from .openarm_gripette_follower import OpenArmGripetteFollower


def prepare_manual_start(
    robot: OpenArmGripetteFollower,
    signs: np.ndarray,
    gripper_action: dict[str, float] | None = None,
    input_fn: Callable[[str], object] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
    engage_duration_s: float = 2.0,
    engage_frequency_hz: float = 20.0,
    disable_frequency_hz: float = 10.0,
) -> np.ndarray:
    signs = np.asarray(signs, dtype=np.float64)
    if signs.shape != (7,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("signs must contain seven values, each +1 or -1")
    if engage_duration_s <= 0 or engage_frequency_hz <= 0 or disable_frequency_hz <= 0:
        raise ValueError("engagement and disable frequencies and duration must be positive")

    robot.arm.bus.disable_torque()
    sleep_fn(0.05)
    robot.arm.bus.disable_torque()
    if gripper_action is not None:
        robot.send_action(gripper_action)
    stop_disabling = Event()
    disable_errors: list[BaseException] = []

    def keep_torque_disabled() -> None:
        while not stop_disabling.wait(1 / disable_frequency_hz):
            try:
                robot.arm.bus.disable_torque()
            except BaseException as error:
                disable_errors.append(error)
                stop_disabling.set()

    disable_thread = Thread(target=keep_torque_disabled, daemon=True)
    disable_thread.start()
    try:
        input_fn("Arm torque is OFF. Position and keep supporting the arm, then press ENTER...")
    finally:
        stop_disabling.set()
        disable_thread.join()
    if disable_errors:
        raise disable_errors[0]
    robot.arm.bus.disable_torque()

    robot.arm.bus.configure_motors()
    observation = robot.arm.get_observation()
    measured = np.array([observation[f"joint_{i}.pos"] for i in range(1, 8)], dtype=np.float64)
    hold_action = {f"joint_{i}.pos": float(value) for i, value in enumerate(measured, start=1)}
    joint_names = list(hold_action)
    position_kp = dict(zip(joint_names, robot.arm.config.position_kp[:7], strict=True))
    position_kd = dict(zip(joint_names, robot.arm.config.position_kd[:7], strict=True))

    robot.arm.send_action(
        hold_action,
        custom_kp=dict.fromkeys(joint_names, 0.0),
        custom_kd=position_kd,
    )
    robot.arm.bus.enable_torque()
    ramp_steps = max(1, round(engage_duration_s * engage_frequency_hz))
    for scale in np.linspace(1 / ramp_steps, 1.0, ramp_steps):
        robot.arm.send_action(
            hold_action,
            custom_kp={name: gain * scale for name, gain in position_kp.items()},
            custom_kd=position_kd,
        )
        sleep_fn(1 / engage_frequency_hz)
    return signs * measured
