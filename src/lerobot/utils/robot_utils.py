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


def precise_sleep(seconds: float, spin_threshold: float = 0.010, sleep_margin: float = 0.003):
    """
    Wait for `seconds` with better precision than time.sleep alone at the expense of more CPU usage.

    Parameters:
      - seconds: duration to wait
      - spin_threshold: if remaining <= spin_threshold -> spin; otherwise sleep (seconds). Default 10ms
      - sleep_margin: when sleeping leave this much time before deadline to avoid oversleep. Default 3ms

    Note:
        The default parameters are chosen to prioritize timing accuracy over CPU usage for the common 30 FPS use case.
    """
    if seconds <= 0:
        return

    system = platform.system()
    # On macOS and Windows the scheduler / sleep granularity can make
    # short sleeps inaccurate. Instead of burning CPU for the whole
    # duration, sleep for most of the time and spin for the final few
    # milliseconds to achieve good accuracy with much lower CPU usage.
    if system in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while True:
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            # If there's more than a couple milliseconds left, sleep most
            # of the remaining time and leave a small margin for the final spin.
            if remaining > spin_threshold:
                # Sleep but avoid sleeping past the end by leaving a small margin.
                time.sleep(max(remaining - sleep_margin, 0))
            else:
                # Final short spin to hit precise timing without long sleeps.
                pass
    else:
        # On Linux time.sleep is accurate enough for most uses
        time.sleep(seconds)
