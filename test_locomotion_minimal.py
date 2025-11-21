#!/usr/bin/env python
"""Minimal test script - just initialize robot with locomotion."""

import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

config = UnitreeG1Config(locomotion_control=True, simulation_mode=True
, policy_path="src/lerobot/robots/unitree_g1/assets/g1/locomotion/GR00T-WholeBodyControl-Walk-converted.onnx")
#, policy_path="src/lerobot/robots/unitree_g1/assets/g1/locomotion/motion.pt")
# dance 102
robot = UnitreeG1(config)
# Keep alive
try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    robot.stop_locomotion_thread()


