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

import platform
import time


def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def safe_disconnect(func):
    # TODO(aliberts): Allow to pass custom exceptions
    # (e.g. ThreadServiceExit, KeyboardInterrupt, SystemExit, UnpluggedError, DynamixelCommError)
    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as e:
            if robot.is_connected:
                robot.disconnect()
            raise e

    return wrapper


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)
