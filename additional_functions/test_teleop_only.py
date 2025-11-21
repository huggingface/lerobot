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

"""
Test script to measure teleoperator read performance in isolation.
This script ONLY reads from the teleoperator without initializing or communicating with a robot.

Example:

```shell
python test_teleop_only.py \
    --teleop.type=custom \
    --fps=60 \
    --duration=10.0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    custom
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up


@dataclass
class TestTeleopConfig:
    teleop: TeleoperatorConfig
    # Limit the maximum frames per second.
    fps: int = 60
    # Test duration in seconds
    duration: float = 10.0
    # Display detailed timing stats
    display_stats: bool = True


def test_teleop_loop(
    teleop: Teleoperator,
    fps: int,
    duration: float,
    display_stats: bool = True,
):
    """
    Test loop that only reads from teleoperator and measures performance.

    Args:
        teleop: The teleoperator device instance providing control actions.
        fps: The target frequency for the control loop in frames per second.
        duration: The duration of the test in seconds.
        display_stats: If True, displays detailed timing statistics.
    """

    start = time.perf_counter()
    iteration = 0
    total_read_time = 0
    min_read_time = float('inf')
    max_read_time = 0
    
    print(f"\nStarting teleoperator read test for {duration}s at {fps} FPS target...")
    print(f"Teleoperator: {teleop.__class__.__name__}")
    print(f"Action features: {list(teleop.action_features.keys())}")
    print(f"Number of actions: {len(teleop.action_features)}")
    print("-" * 80)

    while True:
        loop_start = time.perf_counter()

        # Measure just the get_action call
        read_start = time.perf_counter()
        action = teleop.get_action()
        read_time = time.perf_counter() - read_start
        
        # Track statistics
        total_read_time += read_time
        min_read_time = min(min_read_time, read_time)
        max_read_time = max(max_read_time, read_time)
        iteration += 1

        if display_stats:
            # Display timing information
            avg_read_time = total_read_time / iteration
            print(f"\nIteration: {iteration}")
            print(f"Read time:     {read_time * 1e3:>7.2f}ms")
            print(f"Avg read time: {avg_read_time * 1e3:>7.2f}ms")
            print(f"Min read time: {min_read_time * 1e3:>7.2f}ms")
            print(f"Max read time: {max_read_time * 1e3:>7.2f}ms")
            
            # Show only the wrist roll actions
            print("\nWrist Roll Actions:")
            print(f"  kLeftWristRoll.pos:  {action.get('kLeftWristRoll.pos', 'N/A'):.4f}")
            print(f"  kRightWristRoll.pos: {action.get('kRightWristRoll.pos', 'N/A'):.4f}")
            
            move_cursor_up(9)

        # Wait to maintain target FPS
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        
        loop_time = time.perf_counter() - loop_start
        actual_fps = 1 / loop_time if loop_time > 0 else 0
        
        # Check if duration reached
        elapsed = time.perf_counter() - start
        if elapsed >= duration:
            break

    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total iterations: {iteration}")
    print(f"Total time:       {elapsed:.2f}s")
    print(f"Actual FPS:       {iteration / elapsed:.2f}")
    print(f"Target FPS:       {fps}")
    print(f"\nget_action() timing:")
    print(f"  Average:        {total_read_time / iteration * 1e3:.2f}ms")
    print(f"  Min:            {min_read_time * 1e3:.2f}ms")
    print(f"  Max:            {max_read_time * 1e3:.2f}ms")
    print(f"  Total:          {total_read_time:.2f}s ({total_read_time/elapsed*100:.1f}% of loop time)")
    print("=" * 80)


@parser.wrap()
def test_teleop(cfg: TestTeleopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    print("\nInitializing teleoperator...")
    teleop = make_teleoperator_from_config(cfg.teleop)
    
    print("Connecting teleoperator...")
    teleop.connect()
    
    print(f"Connected: {teleop.is_connected}")
    print(f"Calibrated: {teleop.is_calibrated}")

    try:
        test_teleop_loop(
            teleop=teleop,
            fps=cfg.fps,
            duration=cfg.duration,
            display_stats=cfg.display_stats,
        )
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("\nDisconnecting teleoperator...")
        teleop.disconnect()


def main():
    register_third_party_devices()
    test_teleop()


if __name__ == "__main__":
    main()

