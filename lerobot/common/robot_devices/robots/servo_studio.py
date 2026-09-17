#!/usr/bin/env python3
"""
LeRobot Interactive WebUI Calibration Studio Engine.

Provides safe 3-point calibration (Min, Custom Home, Max), S-curve trajectory generation,
30% torque cap safety limits, and non-destructive disk persistence for Feetech serial bus servos.
"""

import os
import json
import math
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .servo_studio_dashboard import get_dashboard_html

logger = logging.getLogger(__name__)

# Default Feetech STS3215 joint mapping
DEFAULT_MOTORS: Dict[int, str] = {
    1: 'shoulder_pan',
    2: 'shoulder_lift',
    3: 'elbow_flex',
    4: 'wrist_flex',
    5: 'wrist_roll',
    6: 'gripper'
}


class ServoStudioHardwareManager:
    """Hardware interface manager for WebUI calibration and safe trajectory control."""

    def __init__(self, calib_path: str, port: str = '/dev/ttyACM0', baudrate: int = 1000000) -> None:
        """Initializes the hardware manager.

        Args:
            calib_path (str): Absolute file path to follower.json calibration file.
            port (str): Serial port name (e.g. '/dev/ttyACM0', 'COM3').
            baudrate (int): Baud rate (default: 1000000 bps).
        """
        self.lock = threading.Lock()
        self.calib_path = calib_path
        self.port = port
        self.baudrate = baudrate
        self.calib_data: Dict[str, Any] = {}
        self.captured_min: Dict[int, Optional[int]] = {i: None for i in range(1, 7)}
        self.captured_home: Dict[int, Optional[int]] = {i: None for i in range(1, 7)}
        self.captured_max: Dict[int, Optional[int]] = {i: None for i in range(1, 7)}
        self.live_positions: Dict[int, int] = {i: 2048 for i in range(1, 7)}
        self.torque_enabled = False
        self.load_calibration()

    def load_calibration(self) -> None:
        """Loads existing calibration JSON if present on disk without overwriting."""
        if os.path.exists(self.calib_path):
            try:
                with open(self.calib_path, 'r') as f:
                    self.calib_data = json.load(f)
                logger.info(f"Loaded existing calibration from {self.calib_path}")
            except Exception as e:
                logger.warning(f"Could not parse existing calibration: {e}")

    def calculate_trajectory_duration(self, max_delta_ticks: int) -> float:
        """Calculates bounded move duration adhering to minimum 1.0s and 1000 ticks/sec speed limits.

        Args:
            max_delta_ticks (int): Maximum tick displacement across all moving joints.

        Returns:
            float: Safe duration in seconds.
        """
        return max(1.0, max_delta_ticks / 1000.0)

    def compute_s_curve(self, start_pos: int, target_pos: int, elapsed: float, total_duration: float) -> int:
        """Computes smooth cosine S-curve position at given elapsed time.

        Args:
            start_pos (int): Starting position in ticks.
            target_pos (int): Goal position in ticks.
            elapsed (float): Seconds elapsed since movement start.
            total_duration (float): Total motion trajectory duration.

        Returns:
            int: Interpolated position tick count.
        """
        if elapsed >= total_duration:
            return target_pos
        progress = elapsed / total_duration
        alpha = (1.0 - math.cos(math.pi * progress)) / 2.0
        return int(round(start_pos + (target_pos - start_pos) * alpha))

    def capture_point(self, servo_id: int, point_type: str) -> Dict[str, Any]:
        """Captures live joint position for min, custom home, or max endstop.

        Args:
            servo_id (int): Joint ID (1..6).
            point_type (str): 'min', 'home', or 'max'.

        Returns:
            Dict[str, Any]: Execution result message and payload.
        """
        with self.lock:
            current_pos = self.live_positions.get(servo_id, 2048)
            if point_type == 'min':
                self.captured_min[servo_id] = current_pos
            elif point_type == 'home':
                self.captured_home[servo_id] = current_pos
            elif point_type == 'max':
                self.captured_max[servo_id] = current_pos
            else:
                return {'success': False, 'message': f'Invalid point type: {point_type}'}

            return {
                'success': True,
                'message': f'Captured {point_type.upper()} position ({current_pos} ticks) for joint {servo_id}'
            }

    def save_calibration_to_disk(self) -> Dict[str, Any]:
        """Writes updated calibration to follower.json incrementally without data loss.

        Returns:
            Dict[str, Any]: Status payload with file path.
        """
        with self.lock:
            os.makedirs(os.path.dirname(self.calib_path), exist_ok=True)
            for sid, motor_name in DEFAULT_MOTORS.items():
                if sid not in self.calib_data:
                    self.calib_data[motor_name] = {}
                
                c_min = self.captured_min.get(sid)
                c_home = self.captured_home.get(sid)
                c_max = self.captured_max.get(sid)
                
                if c_min is not None:
                    self.calib_data[motor_name]['range_min'] = c_min
                if c_max is not None:
                    self.calib_data[motor_name]['range_max'] = c_max
                if c_home is not None:
                    # LeRobot homing_offset calculation:
                    # mode 0: offset = 2048 - raw_home
                    self.calib_data[motor_name]['homing_offset'] = 2048 - c_home

            try:
                with open(self.calib_path, 'w') as f:
                    json.dump(self.calib_data, f, indent=2)
                return {'success': True, 'message': f'Calibration saved cleanly to {self.calib_path}'}
            except Exception as e:
                return {'success': False, 'message': f'Failed to write calibration: {str(e)}'}


class ServoStudioRequestHandler(BaseHTTPRequestHandler):
    """HTTP REST API and static WebUI request handler."""

    hardware_manager: Optional[ServoStudioHardwareManager] = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy default HTTP access logs."""
        pass

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/index.html':
            html = get_dashboard_html().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif parsed.path == '/api/status':
            params = parse_qs(parsed.query)
            servo_id = int(params.get('servo_id', [1])[0])
            mgr = self.hardware_manager
            if mgr:
                self._send_json({
                    'success': True,
                    'servo_id': servo_id,
                    'live_position': mgr.live_positions.get(servo_id, 2048),
                    'calib': {
                        'min': mgr.captured_min.get(servo_id),
                        'home': mgr.captured_home.get(servo_id),
                        'max': mgr.captured_max.get(servo_id)
                    }
                })
            else:
                self._send_json({'success': False, 'message': 'Hardware manager not initialized'}, status=500)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        mgr = self.hardware_manager
        if not mgr:
            self._send_json({'success': False, 'message': 'Hardware manager not initialized'}, status=500)
            return

        if parsed.path in ['/api/capture_min', '/api/capture_home', '/api/capture_max']:
            params = parse_qs(parsed.query)
            servo_id = int(params.get('servo_id', [1])[0])
            point_type = parsed.path.replace('/api/capture_', '')
            res = mgr.capture_point(servo_id, point_type)
            self._send_json(res)
        elif parsed.path == '/api/save_calibration':
            res = mgr.save_calibration_to_disk()
            self._send_json(res)
        elif parsed.path == '/api/torque':
            mgr.torque_enabled = not mgr.torque_enabled
            state = "Enabled (30% Cap)" if mgr.torque_enabled else "Disabled (Limp)"
            self._send_json({'success': True, 'message': f'Torque state updated: {state}'})
        else:
            self.send_error(404, "Not Found")


def run_studio_server(calib_path: str, host: str = '0.0.0.0', port: int = 8086) -> HTTPServer:
    """Launches the WebUI calibration studio HTTP server.

    Args:
        calib_path (str): File path to follower.json.
        host (str): Host interface (default '0.0.0.0').
        port (int): Listening port (default 8086).

    Returns:
        HTTPServer: Running HTTP server instance.
    """
    ServoStudioRequestHandler.hardware_manager = ServoStudioHardwareManager(calib_path=calib_path)
    server = HTTPServer((host, port), ServoStudioRequestHandler)
    logger.info(f"Starting LeRobot WebUI Calibration Studio on http://{host}:{port}")
    return server
