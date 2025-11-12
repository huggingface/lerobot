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
import keyboard

def busy_wait(seconds):
    if platform.system() == "Darwin" or platform.system() == "Windows":
        # On Mac and Windows, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def busy_wait_with_interrupt(seconds, interrupt_key="right"):
    """
    Wait for specified duration, but early terminate if interrupt key is pressed.

    Args:
        seconds: Time to wait in seconds
        interrupt_key: Key that interrupts waiting

    Returns:
        bool: True if interrupted by key, False if completed normally
    """
    if platform.system() in ["Darwin", "Windows"]:
        # Busy wait for Mac/Windows with interrupt check
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            if keyboard.is_pressed(interrupt_key):
                return True
            time.sleep(0.001)  # Reduce CPU usage
        return False
    else:
        # Linux: time.sleep with periodic interrupt checks
        check_interval = 0.1  # Check every 100ms
        remaining = seconds

        while remaining > 0:
            sleep_time = min(check_interval, remaining)
            time.sleep(sleep_time)
            remaining -= sleep_time

            if keyboard.is_pressed(interrupt_key):
                return True
        return False
