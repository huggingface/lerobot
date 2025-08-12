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

import logging
import time
from collections import deque
from threading import Event, Thread

import numpy as np
from pynput import keyboard

from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.utils import TeleopEvents

logger = logging.getLogger(__name__)


class SO101LeaderFollower(SO101Leader):
    """
    Extended SO101 Leader that can both lead (human control) and follow (mimic follower).

    This class adds leader-follower functionality where:
    - In follow mode: The leader arm mimics the follower's position (torque enabled)
    - In lead mode: Human controls the leader (torque disabled) and provides actions
    """

    def __init__(self, config):
        super().__init__(config)

        # Leader-follower state
        self.is_intervening = False
        self.leader_torque_enabled = True

        # Tracking error for automatic intervention detection
        self.leader_tracking_error_queue = deque(maxlen=4)

        # Keyboard event handling
        self.keyboard_events = {
            "intervention": False,
            "success": False,
            "failure": False,
            "rerecord": False,
        }
        self.keyboard_thread = None
        self.stop_event = Event()

        # Store last follower position for action computation
        self.last_follower_pos = None

    def connect(self, calibrate: bool = True) -> None:
        """Connect and configure for leader-follower mode."""
        super().connect(calibrate)

        # Configure for leader-follower mode with lower gains
        # Lower gains allow manual intervention without injury risk
        self.bus.sync_write("Torque_Enable", 1)
        for motor in self.bus.motors:
            self.bus.write("P_Coefficient", motor, 16)
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 16)

        # Start keyboard listener
        self._start_keyboard_listener()

        print("- Leader-Follower Mode:")
        print("  - Press SPACE to toggle intervention (leader control)")
        print("  - When not intervening, leader follows follower position")
        print("  - When intervening, follower follows leader in end-effector space")
        print("  - Press 's' to mark episode as success")
        print("  - Press ESC to end episode as failure")
        print("  - Press 'r' to re-record episode")

    def _start_keyboard_listener(self):
        """Start keyboard listener thread for intervention control."""

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.keyboard_events["intervention"] = not self.keyboard_events["intervention"]
                    self.is_intervening = self.keyboard_events["intervention"]
                    state = "INTERVENTION MODE" if self.is_intervening else "FOLLOWING MODE"
                    logger.info(f"Toggled to {state}")
                elif key == keyboard.Key.esc:
                    self.keyboard_events["failure"] = True
                elif hasattr(key, "char"):
                    if key.char == "s":
                        self.keyboard_events["success"] = True
                    elif key.char == "r":
                        self.keyboard_events["rerecord"] = True
            except Exception as e:
                logger.error(f"Error handling key press: {e}")

        def listen():
            with keyboard.Listener(on_press=on_press) as listener:
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                listener.stop()

        self.keyboard_thread = Thread(target=listen, daemon=True)
        self.keyboard_thread.start()

    def send_action(self, action: dict[str, float]) -> None:
        """
        Send position commands to leader arm (follow mode).

        Args:
            action: Dictionary of motor positions to command
        """
        # Store follower position for later use
        self.last_follower_pos = np.array([action.get(f"{motor}.pos", 0) for motor in self.bus.motors])

        if not self.is_intervening:
            # Follow mode: enable torque and track follower
            if not self.leader_torque_enabled:
                self.bus.sync_write("Torque_Enable", 1)
                self.leader_torque_enabled = True

            # Send follower positions to leader
            goal_pos = {motor: action[f"{motor}.pos"] for motor in self.bus.motors}
            self.bus.sync_write("Goal_Position", goal_pos)

            # Track error for automatic intervention detection
            current_pos = self.bus.sync_read("Present_Position")
            current_array = np.array([current_pos[motor] for motor in self.bus.motors])
            error = np.linalg.norm(self.last_follower_pos[:-1] - current_array[:-1])
            self.leader_tracking_error_queue.append(error)

    def get_action(self) -> dict[str, float]:
        """
        Get action from leader arm.

        In follow mode: Returns neutral/current positions
        In lead mode: Returns actual leader positions for follower to track
        """
        start = time.perf_counter()

        if self.is_intervening:
            # Lead mode: disable torque if needed and return leader positions
            if self.leader_torque_enabled:
                self.bus.sync_write("Torque_Enable", 0)
                self.leader_torque_enabled = False

            # Get current leader position
            action = self.bus.sync_read("Present_Position")
            action = {f"{motor}.pos": val for motor, val in action.items()}

            # Track error
            if self.last_follower_pos is not None:
                current_array = np.array([action[f"{motor}.pos"] for motor in self.bus.motors])
                error = np.linalg.norm(self.last_follower_pos[:-1] - current_array[:-1])
                self.leader_tracking_error_queue.append(error)
        else:
            # Follow mode: return current/neutral positions
            action = self.bus.sync_read("Present_Position")
            action = {f"{motor}.pos": val for motor, val in action.items()}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Get current keyboard events."""
        events = {}

        # Map keyboard events to TeleopEvents
        if self.keyboard_events["success"]:
            events[TeleopEvents.SUCCESS] = True
            self.keyboard_events["success"] = False
        if self.keyboard_events["failure"]:
            events[TeleopEvents.FAILURE] = True
            events[TeleopEvents.TERMINATE_EPISODE] = True
            self.keyboard_events["failure"] = False
        if self.keyboard_events["rerecord"]:
            events[TeleopEvents.RERECORD_EPISODE] = True
            events[TeleopEvents.TERMINATE_EPISODE] = True
            self.keyboard_events["rerecord"] = False

        # Always report intervention state
        events[TeleopEvents.IS_INTERVENTION] = self.is_intervening

        return events

    def disconnect(self) -> None:
        """Disconnect and cleanup."""
        self.stop_event.set()
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1.0)
        super().disconnect()

    def reset(self) -> None:
        """Reset leader-follower state."""
        self.is_intervening = False
        self.leader_torque_enabled = True
        self.leader_tracking_error_queue.clear()
        self.keyboard_events = {"intervention": False, "success": False, "failure": False, "rerecord": False}
