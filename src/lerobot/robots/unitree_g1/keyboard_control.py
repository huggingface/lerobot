#!/usr/bin/env python
"""
Standalone keyboard control script for Unitree G1 robot.

This script provides keyboard-based velocity control for the G1 robot's
locomotion system. It can be run alongside the main robot control to
provide manual movement commands.

Usage:
    python keyboard_control.py [--robot-ip IP] [--simulation]

Controls:
    W/S: Forward/Backward
    A/D: Strafe Left/Right  
    Q/E: Rotate Left/Right
    R/F: Raise/Lower Height (GR00T policies only)
    Z: Stop (zero all velocity commands)
    ESC/Ctrl+C: Exit
"""

import argparse
import sys
import select
import time
import numpy as np

# Terminal handling for non-blocking keyboard input
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False
    print("Warning: termios not available. Keyboard controls require Linux/macOS.")


class KeyboardController:
    """Handles keyboard input and converts to locomotion commands."""
    
    def __init__(self, callback=None):
        """
        Initialize keyboard controller.
        
        Args:
            callback: Optional function called when commands change.
                      Signature: callback(vx, vy, yaw, height)
        """
        self.callback = callback
        self.running = False
        
        # Locomotion commands
        self.vx = 0.0      # Forward/backward velocity
        self.vy = 0.0      # Left/right velocity (strafe)
        self.yaw = 0.0     # Rotation rate
        self.height = 0.74  # Base height (for GR00T policies)
        
        # Command limits
        self.vx_limit = (-0.8, 0.8)
        self.vy_limit = (-0.5, 0.5)
        self.yaw_limit = (-1.0, 1.0)
        self.height_limit = (0.50, 1.00)
        
        # Increments per keypress
        self.vx_increment = 0.4
        self.vy_increment = 0.25
        self.yaw_increment = 0.5
        self.height_increment = 0.05
        
        self._old_terminal_settings = None
    
    def get_commands(self) -> tuple[float, float, float, float]:
        """Get current command values as tuple (vx, vy, yaw, height)."""
        return (self.vx, self.vy, self.yaw, self.height)
    
    def get_commands_array(self) -> np.ndarray:
        """Get velocity commands as numpy array [vx, vy, yaw]."""
        return np.array([self.vx, self.vy, self.yaw], dtype=np.float32)
    
    def reset_commands(self):
        """Reset all commands to zero (stop)."""
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self._notify_callback()
    
    def _clamp(self, value: float, limits: tuple[float, float]) -> float:
        """Clamp value to limits."""
        return max(limits[0], min(limits[1], value))
    
    def _notify_callback(self):
        """Call callback with current commands if set."""
        if self.callback:
            self.callback(self.vx, self.vy, self.yaw, self.height)
    
    def process_key(self, key: str) -> bool:
        """
        Process a single key press and update commands.
        
        Args:
            key: Single character key that was pressed.
            
        Returns:
            True if key was handled, False otherwise.
        """
        key = key.lower()
        handled = True
        
        if key == 'w':
            self.vx = self._clamp(self.vx + self.vx_increment, self.vx_limit)
        elif key == 's':
            self.vx = self._clamp(self.vx - self.vx_increment, self.vx_limit)
        elif key == 'a':
            self.vy = self._clamp(self.vy + self.vy_increment, self.vy_limit)
        elif key == 'd':
            self.vy = self._clamp(self.vy - self.vy_increment, self.vy_limit)
        elif key == 'q':
            self.yaw = self._clamp(self.yaw + self.yaw_increment, self.yaw_limit)
        elif key == 'e':
            self.yaw = self._clamp(self.yaw - self.yaw_increment, self.yaw_limit)
        elif key == 'r':
            self.height = self._clamp(self.height + self.height_increment, self.height_limit)
        elif key == 'f':
            self.height = self._clamp(self.height - self.height_increment, self.height_limit)
        elif key == 'z':
            self.reset_commands()
            return True  # Already notified in reset_commands
        else:
            handled = False
        
        if handled:
            self._notify_callback()
        
        return handled
    
    def _setup_terminal(self):
        """Set terminal to raw mode for single character input."""
        if HAS_TERMIOS:
            self._old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
    
    def _restore_terminal(self):
        """Restore terminal to original settings."""
        if HAS_TERMIOS and self._old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)
            self._old_terminal_settings = None
    
    def run(self):
        """Run the keyboard listener loop (blocking)."""
        if not HAS_TERMIOS:
            print("Error: Keyboard controls require termios (Linux/macOS)")
            return
        
        self.running = True
        self._print_controls()
        
        try:
            self._setup_terminal()
            
            while self.running:
                # Check for keyboard input with timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    # Handle escape sequences (arrow keys, etc.)
                    if key == '\x1b':  # ESC
                        self.running = False
                        break
                    
                    if self.process_key(key):
                        self._print_status()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self._restore_terminal()
            print("\nKeyboard controls stopped")
    
    def stop(self):
        """Stop the keyboard listener."""
        self.running = False
    
    def _print_controls(self):
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("KEYBOARD CONTROLS ACTIVE")
        print("=" * 60)
        print("  W/S: Forward/Backward")
        print("  A/D: Strafe Left/Right")
        print("  Q/E: Rotate Left/Right")
        print("  R/F: Raise/Lower Height (±5cm)")
        print("  Z: Stop (zero all commands)")
        print("  ESC: Exit")
        print("=" * 60 + "\n")
    
    def _print_status(self):
        """Print current command status."""
        print(f"[CMD] vx={self.vx:+.2f}, vy={self.vy:+.2f}, yaw={self.yaw:+.2f} | height={self.height:.3f}m")


class RobotKeyboardController(KeyboardController):
    """Keyboard controller that directly updates a robot's locomotion commands."""
    
    def __init__(self, robot):
        """
        Initialize with a UnitreeG1 robot instance.
        
        Args:
            robot: UnitreeG1 robot instance with locomotion_cmd attribute.
        """
        super().__init__()
        self.robot = robot
        
        # Initialize from robot's current state if available
        if hasattr(robot, 'locomotion_cmd'):
            self.vx = robot.locomotion_cmd[0]
            self.vy = robot.locomotion_cmd[1]
            self.yaw = robot.locomotion_cmd[2]
        
        if hasattr(robot, 'groot_height_cmd'):
            self.height = robot.groot_height_cmd
    
    def _notify_callback(self):
        """Update robot's locomotion commands directly."""
        if hasattr(self.robot, 'locomotion_cmd'):
            self.robot.locomotion_cmd[0] = self.vx
            self.robot.locomotion_cmd[1] = self.vy
            self.robot.locomotion_cmd[2] = self.yaw
        
        if hasattr(self.robot, 'groot_height_cmd'):
            self.robot.groot_height_cmd = self.height


def start_keyboard_control_thread(robot) -> tuple:
    """
    Start keyboard controls for a robot in a background thread.
    
    Args:
        robot: UnitreeG1 robot instance.
        
    Returns:
        Tuple of (controller, thread) for later stopping.
    """
    import threading
    
    controller = RobotKeyboardController(robot)
    thread = threading.Thread(target=controller.run, daemon=True)
    thread.start()
    
    return controller, thread


def stop_keyboard_control_thread(controller, thread, timeout: float = 2.0):
    """
    Stop the keyboard control thread.
    
    Args:
        controller: KeyboardController instance.
        thread: Thread running the controller.
        timeout: Max time to wait for thread to stop.
    """
    controller.stop()
    thread.join(timeout=timeout)


def main():
    """Standalone keyboard control with optional robot connection."""
    parser = argparse.ArgumentParser(description="Keyboard control for Unitree G1")
    parser.add_argument("--standalone", action="store_true",
                        help="Run in standalone mode (just print commands, no robot)")
    args = parser.parse_args()
    
    if args.standalone:
        # Standalone mode - just demonstrate keyboard input
        def print_callback(vx, vy, yaw, height):
            print(f"  → Would send: vx={vx:+.2f}, vy={vy:+.2f}, yaw={yaw:+.2f}, height={height:.3f}")
        
        controller = KeyboardController(callback=print_callback)
        print("Running in STANDALONE mode (no robot connection)")
        controller.run()
    else:
        print("To use with a robot, import and use RobotKeyboardController:")
        print("")
        print("  from lerobot.robots.unitree_g1.keyboard_control import (")
        print("      RobotKeyboardController,")
        print("      start_keyboard_control_thread,")
        print("      stop_keyboard_control_thread")
        print("  )")
        print("")
        print("  # Start keyboard controls")
        print("  controller, thread = start_keyboard_control_thread(robot)")
        print("")
        print("  # ... robot runs ...")
        print("")
        print("  # Stop keyboard controls")
        print("  stop_keyboard_control_thread(controller, thread)")
        print("")
        print("Or run with --standalone to test keyboard input without a robot.")


if __name__ == "__main__":
    main()

