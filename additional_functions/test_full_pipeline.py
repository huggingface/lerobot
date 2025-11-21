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
Test script to measure each component of the teleoperation pipeline separately.

Example:

```shell
python test_full_pipeline.py \
    --robot.type=unitree_g1 \
    --teleop.type=custom \
    --fps=60 \
    --duration=10.0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    unitree_g1,
)
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
class TestPipelineConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    # Test duration in seconds
    duration: float = 10.0


def test_pipeline_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    duration: float,
):
    """
    Test loop that measures timing of each pipeline component separately.

    Args:
        teleop: The teleoperator device instance.
        robot: The robot instance.
        fps: The target frequency for the control loop in frames per second.
        duration: The duration of the test in seconds.
    """

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    start = time.perf_counter()
    iteration = 0
    
    # Timing accumulators
    total_obs_time = 0
    total_teleop_read_time = 0
    total_teleop_process_time = 0
    total_robot_process_time = 0
    total_robot_send_time = 0
    
    min_obs_time = float('inf')
    max_obs_time = 0
    min_teleop_read_time = float('inf')
    max_teleop_read_time = 0
    min_teleop_process_time = float('inf')
    max_teleop_process_time = 0
    min_robot_process_time = float('inf')
    max_robot_process_time = 0
    min_robot_send_time = float('inf')
    max_robot_send_time = 0
    
    print(f"\nStarting full pipeline test for {duration}s at {fps} FPS target...")
    print(f"Robot: {robot.__class__.__name__}")
    print(f"Teleoperator: {teleop.__class__.__name__}")
    print("-" * 80)

    while True:
        loop_start = time.perf_counter()

        # 1. Get robot observation
        t1 = time.perf_counter()
        obs = robot.get_observation()
        obs_time = time.perf_counter() - t1
        
        # 2. Get teleop action
        t2 = time.perf_counter()
        raw_action = teleop.get_action()
        teleop_read_time = time.perf_counter() - t2
        
        # 3. Process teleop action
        t3 = time.perf_counter()
        teleop_action = teleop_action_processor((raw_action, obs))
        teleop_process_time = time.perf_counter() - t3
        
        # 4. Process action for robot
        t4 = time.perf_counter()
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        robot_process_time = time.perf_counter() - t4
        
        # 5. Send action to robot
        t5 = time.perf_counter()
        _ = robot.send_action(robot_action_to_send)
        robot_send_time = time.perf_counter() - t5
        
        # Track statistics
        total_obs_time += obs_time
        total_teleop_read_time += teleop_read_time
        total_teleop_process_time += teleop_process_time
        total_robot_process_time += robot_process_time
        total_robot_send_time += robot_send_time
        
        min_obs_time = min(min_obs_time, obs_time)
        max_obs_time = max(max_obs_time, obs_time)
        min_teleop_read_time = min(min_teleop_read_time, teleop_read_time)
        max_teleop_read_time = max(max_teleop_read_time, teleop_read_time)
        min_teleop_process_time = min(min_teleop_process_time, teleop_process_time)
        max_teleop_process_time = max(max_teleop_process_time, teleop_process_time)
        min_robot_process_time = min(min_robot_process_time, robot_process_time)
        max_robot_process_time = max(max_robot_process_time, robot_process_time)
        min_robot_send_time = min(min_robot_send_time, robot_send_time)
        max_robot_send_time = max(max_robot_send_time, robot_send_time)
        
        iteration += 1

        # Display timing breakdown
        total_component_time = obs_time + teleop_read_time + teleop_process_time + robot_process_time + robot_send_time
        
        print(f"\nIteration: {iteration}")
        print(f"1. robot.get_observation():     {obs_time * 1e3:>7.2f}ms")
        print(f"2. teleop.get_action():         {teleop_read_time * 1e3:>7.2f}ms")
        print(f"3. teleop_action_processor:     {teleop_process_time * 1e3:>7.2f}ms")
        print(f"4. robot_action_processor:      {robot_process_time * 1e3:>7.2f}ms")
        print(f"5. robot.send_action():         {robot_send_time * 1e3:>7.2f}ms")
        print(f"   Total component time:        {total_component_time * 1e3:>7.2f}ms")
        
        # Show wrist roll values
        print(f"\nkLeftWristRoll.pos:  {robot_action_to_send.get('kLeftWristRoll.pos', 'N/A')}")
        print(f"kRightWristRoll.pos: {robot_action_to_send.get('kRightWristRoll.pos', 'N/A')}")
        
        move_cursor_up(11)

        # Wait to maintain target FPS
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        
        loop_time = time.perf_counter() - loop_start
        
        # Check if duration reached
        elapsed = time.perf_counter() - start
        if elapsed >= duration:
            break

    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL TIMING BREAKDOWN")
    print("=" * 80)
    print(f"Total iterations: {iteration}")
    print(f"Total time:       {elapsed:.2f}s")
    print(f"Actual FPS:       {iteration / elapsed:.2f}")
    print(f"Target FPS:       {fps}")
    print("\n" + "-" * 80)
    print(f"{'Component':<35} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'% of time':>10}")
    print("-" * 80)
    
    avg_obs = total_obs_time / iteration * 1e3
    avg_teleop_read = total_teleop_read_time / iteration * 1e3
    avg_teleop_proc = total_teleop_process_time / iteration * 1e3
    avg_robot_proc = total_robot_process_time / iteration * 1e3
    avg_robot_send = total_robot_send_time / iteration * 1e3
    
    total_avg = avg_obs + avg_teleop_read + avg_teleop_proc + avg_robot_proc + avg_robot_send
    
    print(f"{'1. robot.get_observation()':<35} {avg_obs:>10.2f} {min_obs_time*1e3:>10.2f} {max_obs_time*1e3:>10.2f} {avg_obs/total_avg*100:>9.1f}%")
    print(f"{'2. teleop.get_action()':<35} {avg_teleop_read:>10.2f} {min_teleop_read_time*1e3:>10.2f} {max_teleop_read_time*1e3:>10.2f} {avg_teleop_read/total_avg*100:>9.1f}%")
    print(f"{'3. teleop_action_processor':<35} {avg_teleop_proc:>10.2f} {min_teleop_process_time*1e3:>10.2f} {max_teleop_process_time*1e3:>10.2f} {avg_teleop_proc/total_avg*100:>9.1f}%")
    print(f"{'4. robot_action_processor':<35} {avg_robot_proc:>10.2f} {min_robot_process_time*1e3:>10.2f} {max_robot_process_time*1e3:>10.2f} {avg_robot_proc/total_avg*100:>9.1f}%")
    print(f"{'5. robot.send_action()':<35} {avg_robot_send:>10.2f} {min_robot_send_time*1e3:>10.2f} {max_robot_send_time*1e3:>10.2f} {avg_robot_send/total_avg*100:>9.1f}%")
    print("-" * 80)
    print(f"{'TOTAL':<35} {total_avg:>10.2f}")
    print("=" * 80)


@parser.wrap()
def test_pipeline(cfg: TestPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    print("\nInitializing teleoperator...")
    teleop = make_teleoperator_from_config(cfg.teleop)
    
    print("Initializing robot...")
    robot = make_robot_from_config(cfg.robot)
    
    print("Connecting teleoperator...")
    teleop.connect()
    
    print("Connecting robot...")
    robot.connect()
    
    print(f"\nTeleoperator connected: {teleop.is_connected}")
    print(f"Teleoperator calibrated: {teleop.is_calibrated}")
    print(f"Robot connected: {robot.is_connected}")

    try:
        test_pipeline_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            duration=cfg.duration,
        )
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("\nDisconnecting...")
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_devices()
    test_pipeline()


if __name__ == "__main__":
    main()

