#!/usr/bin/env python
"""Minimal test script for dance_102 motion imitation."""

import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

config = UnitreeG1Config(
    motion_imitation_control=True,  # Enable motion imitation (dance)
    simulation_mode=False
)

robot = UnitreeG1(config)

# Keep alive
try:
    print("\n" + "="*60)
    print("Dance_102 Motion Imitation Running!")
    print(f"Motion duration: {robot.motion_loader.duration:.2f}s")
    print("The robot will loop the dance motion")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    while True:
        time.sleep(1.0)
        # Print progress every second
        if robot.motion_loader:
            progress = (robot.motion_loader.current_time / robot.motion_loader.duration) * 100
            print(f"Progress: {progress:.1f}% - Time: {robot.motion_loader.current_time:.2f}s / {robot.motion_loader.duration:.2f}s")
except KeyboardInterrupt:
    print("\nStopping dance...")
    robot.stop_motion_imitation_thread()
    print("Dance stopped!")

