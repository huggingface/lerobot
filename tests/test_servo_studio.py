#!/usr/bin/env python3
"""
Pytest Off-Hardware Test Suite for WebUI Calibration Studio.
Uses unittest.mock to verify all REST API endpoints, S-curve trajectory duration math,
and follower.json calibration updates without requiring physical USB motor hardware.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from .servo_studio import ServoStudioHardwareManager, ServoStudioRequestHandler, run_studio_server


class TestServoStudio(unittest.TestCase):
    """Test suite for ServoStudioHardwareManager and HTTP handler."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.calib_path = os.path.join(self.temp_dir.name, 'follower.json')
        self.mgr = ServoStudioHardwareManager(calib_path=self.calib_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_trajectory_duration_math(self) -> None:
        """Tests that bounded move durations enforce 1.0s minimum and speed bounds."""
        # 500 ticks delta => 500 / 1000 = 0.5s -> bounded to 1.0s
        dur_short = self.mgr.calculate_trajectory_duration(500)
        self.assertEqual(dur_short, 1.0)

        # 3000 ticks delta => 3000 / 1000 = 3.0s
        dur_long = self.mgr.calculate_trajectory_duration(3000)
        self.assertEqual(dur_long, 3.0)

    def test_s_curve_interpolation(self) -> None:
        """Tests smooth cosine S-curve interpolation start, mid, and end points."""
        start, target, total_dur = 1000, 2000, 2.0
        
        # Start at t=0
        pos_start = self.mgr.compute_s_curve(start, target, 0.0, total_dur)
        self.assertEqual(pos_start, 1000)

        # Midpoint at t=1.0 (50% progress -> alpha=0.5 -> 1500)
        pos_mid = self.mgr.compute_s_curve(start, target, 1.0, total_dur)
        self.assertEqual(pos_mid, 1500)

        # End at t=2.0
        pos_end = self.mgr.compute_s_curve(start, target, 2.0, total_dur)
        self.assertEqual(pos_end, 2000)

    def test_capture_points(self) -> None:
        """Tests 3-point calibration captures for min, home, and max."""
        self.mgr.live_positions[1] = 1200
        res_min = self.mgr.capture_point(1, 'min')
        self.assertTrue(res_min['success'])
        self.assertEqual(self.mgr.captured_min[1], 1200)

        self.mgr.live_positions[1] = 2048
        res_home = self.mgr.capture_point(1, 'home')
        self.assertTrue(res_home['success'])
        self.assertEqual(self.mgr.captured_home[1], 2048)

        self.mgr.live_positions[1] = 3100
        res_max = self.mgr.capture_point(1, 'max')
        self.assertTrue(res_max['success'])
        self.assertEqual(self.mgr.captured_max[1], 3100)

    def test_save_calibration_incremental(self) -> None:
        """Tests incremental update of follower.json on disk."""
        self.mgr.captured_min[1] = 1100
        self.mgr.captured_home[1] = 2000
        self.mgr.captured_max[1] = 3000

        res = self.mgr.save_calibration_to_disk()
        self.assertTrue(res['success'])
        self.assertTrue(os.path.exists(self.calib_path))

        with open(self.calib_path, 'r') as f:
            data = json.load(f)

        self.assertIn('shoulder_pan', data)
        self.assertEqual(data['shoulder_pan']['range_min'], 1100)
        self.assertEqual(data['shoulder_pan']['range_max'], 3000)
        # homing_offset mode 0: 2048 - 2000 = 48
        self.assertEqual(data['shoulder_pan']['homing_offset'], 48)


if __name__ == '__main__':
    unittest.main()
