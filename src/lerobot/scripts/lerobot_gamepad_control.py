#!/usr/bin/env python3
"""
SO-101 Xbox Controller Control Script
Controls SO-101 robotic arm using an Xbox controller (or any compatible gamepad)

Button Mapping:
    Left Stick          Shoulder pan & lift  (joints 0-1)  [ARM mode]
    Right Stick Y       Elbow flex           (joint 2)     [ARM mode]
    D-Pad Up/Down       Wrist flex           (joint 3)     [ARM mode]
    D-Pad Left/Right    Wrist roll           (joint 4)     [ARM mode]
    LT / RT             Gripper open/close                 [ARM mode]
    RT                  Drive forward                      [MOTOR mode]
    LT                  Drive backward                     [MOTOR mode]
    Left Stick X        Steer left/right                   [MOTOR mode]
    RB                  Switch to ARM mode  (tap to toggle)
    LB                  Switch to MOTOR mode (tap to toggle)
    RB + LB             IDLE (stop everything)
    A                   Preset → HOME
    B                   Preset → MOVEMENT
    X                   Preset → DROP
    Y                   Preset → GRAB
    Start               Exit

Safety Features:
- Tap RB to enter ARM mode, tap again to go IDLE
- Tap LB to enter MOTOR mode, tap again to go IDLE
- Hold RB + LB together to force IDLE immediately
- Max movement speed limit
- Automatic disconnect on exit
"""

import os
import time
import threading
import numpy as np
import torch
import pygame
from pathlib import Path

# Import LeRobot components
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

try:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    LEROBOT_CAMERA_AVAILABLE = True
except Exception:
    LEROBOT_CAMERA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Camera helpers (mirrors server CameraBuffer)
# ──────────────────────────────────────────────────────────────

def connect_cameras(camera_spec: str) -> dict:
    """
    Parse camera spec and connect each OpenCVCamera.
    Accepts the same format as the server's --cameras arg:
        "{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
           base:  {type: opencv, index_or_path: /dev/video2} }"
    Also accepts simple shorthand: "front:/dev/video0,base:/dev/video2"
    """
    cameras = {}
    if not camera_spec or not LEROBOT_CAMERA_AVAILABLE:
        if not LEROBOT_CAMERA_AVAILABLE:
            print("  ⚠️  lerobot cameras not available — skipping camera init")
        return cameras

    # Detect format: lerobot-style dict vs simple shorthand
    spec = camera_spec.strip()
    if spec.startswith("{"):
        # lerobot-style: "{ name: {index_or_path: /dev/videoN, ...}, ... }"
        entries = {}
        spec_inner = spec.strip("{}")
        depth, current, blocks = 0, "", []
        for ch in spec_inner:
            if ch == "{": depth += 1
            elif ch == "}": depth -= 1
            if ch == "," and depth == 0:
                blocks.append(current.strip()); current = ""
            else:
                current += ch
        if current.strip():
            blocks.append(current.strip())
        for block in blocks:
            colon = block.index(":")
            name = block[:colon].strip().strip("\"'")
            inner = block[colon+1:].strip().strip("{}")
            cfg = {}
            pairs, depth2, cur2 = [], 0, ""
            for ch in inner:
                if ch == "{": depth2 += 1
                elif ch == "}": depth2 -= 1
                if ch == "," and depth2 == 0:
                    pairs.append(cur2.strip()); cur2 = ""
                else:
                    cur2 += ch
            if cur2.strip():
                pairs.append(cur2.strip())
            for pair in pairs:
                if ":" not in pair: continue
                k, v = pair.split(":", 1)
                cfg[k.strip().strip("\"'")] = v.strip().strip("\"'")
            entries[name] = cfg
    else:
        # Simple shorthand: "front:/dev/video0,base:/dev/video2"
        entries = {}
        for part in spec.split(","):
            part = part.strip()
            if ":" not in part: continue
            name, path = part.split(":", 1)
            entries[name.strip()] = {"index_or_path": path.strip()}

    for name, cfg_dict in entries.items():
        try:
            raw_path = cfg_dict.get("index_or_path", "0")
            if str(raw_path).lstrip("-").isdigit():
                index = int(raw_path)
            elif str(raw_path).startswith("/dev/video"):
                index = int(raw_path.replace("/dev/video", ""))
            else:
                index = raw_path
            kwargs = {"index_or_path": index}
            if "width"  in cfg_dict: kwargs["width"]  = int(cfg_dict["width"])
            if "height" in cfg_dict: kwargs["height"] = int(cfg_dict["height"])
            if "fps"    in cfg_dict: kwargs["fps"]    = int(cfg_dict["fps"])
            cam = OpenCVCamera(OpenCVCameraConfig(**kwargs))
            cam.connect()
            cameras[name] = cam
            print(f"  ✓ Camera '{name}' connected ({raw_path})")
        except Exception as e:
            print(f"  ⚠️  Camera '{name}' failed: {e}")
    return cameras


class CameraBuffer:
    """
    Runs each camera in its own background thread, always holding the
    latest frame. Identical to the server's CameraBuffer.
    """

    def __init__(self, cameras: dict):
        self._cameras  = cameras
        self._frames   = {n: None for n in cameras}
        self._lock     = threading.Lock()
        self._stop_evt = threading.Event()
        self._threads  = []

    def start(self) -> None:
        for name, cam in self._cameras.items():
            t = threading.Thread(
                target=self._capture_loop, args=(name, cam),
                daemon=True, name=f"cam-{name}"
            )
            t.start()
            self._threads.append(t)

    def _capture_loop(self, name: str, cam) -> None:
        while not self._stop_evt.is_set():
            try:
                img = cam.async_read(timeout_ms=500)
                if img is not None:
                    with self._lock:
                        self._frames[name] = img
            except Exception:
                pass

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._frames)

    def stop(self) -> None:
        self._stop_evt.set()


class SO101GamepadController:
    """Xbox controller interface for SO-101 robot arm"""
    
    def __init__(
        self,
        robot_port="/dev/ttyACM0",
        robot_id="so101_follower",
        max_speed=2.0,  # Maximum change per control loop (in degrees for joints)
        control_frequency=30,  # Hz
        cameras="",     # Camera spec string, e.g. "front:/dev/video0,base:/dev/video2"
        show_images=False,
        image_hz=5,
    ):
        self.robot_port = robot_port
        self.robot_id = robot_id
        self.max_speed = max_speed
        self.control_dt = 1.0 / control_frequency

        # Initialize pygame for gamepad
        pygame.init()
        if show_images and CV2_AVAILABLE:
            self._pygame_screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("SockBot Camera")
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.init()
        pygame.joystick.init()
        
        # Check for gamepad
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected! Please connect an Xbox controller.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"✓ Gamepad connected: {self.joystick.get_name()}")
        print(f"  Axes: {self.joystick.get_numaxes()}")
        print(f"  Buttons: {self.joystick.get_numbuttons()}")
        
        # Initialize robot
        print(f"\n✓ Connecting to SO-101 on {robot_port}...")
        config = SO101FollowerConfig(
            port=robot_port,
            id=robot_id,
        )
        self.robot = SO101Follower(config)
        self.robot.connect()
        
        # Get initial state
        self.current_position = None
        self.num_joints = None
        
        # Initialize preset positions dictionary (before calling _initialize_presets)
        self.preset_positions = {}
        
        self._initialize_robot_state()
        
        # Initialize preset positions
        self._initialize_presets()
        
        # Control state
        self.running = True
        self.gripper_state = 0.0  # -1.0 = closed, 1.0 = open

        # Mode toggle: None | "arm" | "motor"
        # Tap RB → arm, tap LB → motor, RB+LB together → idle (None)
        self.mode = None
        self._prev_rb = False
        self._prev_lb = False
        self._prev_a  = False
        self._prev_b  = False
        self._prev_x  = False
        self._prev_y  = False

        # ── Camera display ─────────────────────────────────────
        self._show_images = show_images and CV2_AVAILABLE
        if show_images and not CV2_AVAILABLE:
            print("  ⚠️  --show-images requested but opencv-python is not installed — skipping")
        self._cam_buffer  = None
        self._img_interval = max(1, int(1.0 / self.control_dt) // max(1, image_hz))
        self._img_frame    = 0
        if cameras:
            print("\n  Connecting cameras...")
            cam_dict = connect_cameras(cameras)
            if cam_dict:
                self._cam_buffer = CameraBuffer(cam_dict)
                self._cam_buffer.start()
                print(f"  ✓ Camera buffer started ({len(cam_dict)} camera(s))")
            else:
                print("  ⚠️  No cameras connected")
        
        # Button/axis indices vary by PLATFORM (OS), not controller model!
        # The same controller has different mappings on Windows vs Linux
        import platform
        
        system = platform.system()
        print(f"  Platform: {system}")
        print(f"  Controller: {self.joystick.get_name()}")
        
        if system == "Linux":
            # Linux (including Raspberry Pi) - Xbox controllers
            print("  ✓ Using Linux/Raspberry Pi button mapping")
            self.AXIS_LEFT_X = 0
            self.AXIS_LEFT_Y = 1
            self.AXIS_RIGHT_X = 3      # Different on Linux!
            self.AXIS_RIGHT_Y = 4      # Different on Linux!
            self.AXIS_LT = 2           # Different on Linux!
            self.AXIS_RT = 5
            
            self.BTN_A = 0
            self.BTN_B = 1
            self.BTN_X = 2
            self.BTN_Y = 3
            self.BTN_LB = 4
            self.BTN_RB = 5
            self.BTN_BACK = 6
            self.BTN_START = 7
            self.BTN_LSTICK = 9
            self.BTN_RSTICK = 10
            
        else:
            # Windows - Xbox controllers
            print("  ✓ Using Windows or MacOS button mapping")
            self.AXIS_LEFT_X = 0
            self.AXIS_LEFT_Y = 1
            self.AXIS_RIGHT_X = 2
            self.AXIS_RIGHT_Y = 3
            self.AXIS_LT = 4
            self.AXIS_RT = 5
            
            self.BTN_A = 0
            self.BTN_B = 1
            self.BTN_X = 2
            self.BTN_Y = 3
            self.BTN_LB = 4
            self.BTN_RB = 5
            self.BTN_BACK = 6
            self.BTN_START = 7
            self.BTN_LSTICK = 8
            self.BTN_RSTICK = 9
        
        # Dead zone for analog sticks
        self.DEAD_ZONE = 0.15
        
    def _initialize_robot_state(self):
        """Get initial robot state"""
        obs = self.robot.get_observation()
        
        # SO-101 returns individual keys for each joint position
        joint_keys = [
            'shoulder_pan.pos',
            'shoulder_lift.pos', 
            'elbow_flex.pos',
            'wrist_flex.pos',
            'wrist_roll.pos',
            'gripper.pos'
        ]
        
        # Extract positions into numpy array
        positions = []
        for key in joint_keys:
            if key in obs:
                positions.append(obs[key])
            else:
                print(f"⚠️  Warning: Key '{key}' not found in observation")
        
        self.current_position = np.array(positions, dtype=np.float32)
        self.num_joints = len(self.current_position)
        self.joint_keys = joint_keys
        
        print(f"✓ Robot initialized with {self.num_joints} joints")
        print(f"  Current position (degrees): {np.round(self.current_position, 2)}")
        
        # Load calibrated joint limits
        self._load_joint_limits()
    
    def _load_joint_limits(self):
        """Load joint limits from calibration file"""
        import json
        
        calib_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / f"{self.robot_id}.json"
        
        # Default limits (conservative ±180°)
        self.joint_limits_lower = np.array([-180.0] * self.num_joints, dtype=np.float32)
        self.joint_limits_upper = np.array([180.0] * self.num_joints, dtype=np.float32)
        
        # Try to load calibration
        if calib_path.exists():
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
                
                print(f"  📄 Calibration file loaded from: {calib_path.name}")
                
                print(f"  ✓ Reading range_min/range_max from calibration motors")
                
                # STS3215 servo conversion: servo units to degrees
                # Servo range: 0-4095 maps to 0-360 degrees (approx)
                # But calibration uses a centered system where 2048 ≈ 0°
                SERVO_CENTER = 2048
                SERVO_UNITS_PER_DEGREE = 4096 / 360.0  # ~11.38 units per degree
                
                limits_found = False
                for i, key in enumerate(self.joint_keys):
                    joint_name = key.removesuffix('.pos')
                    
                    if joint_name in calib_data:
                        motor_config = calib_data[joint_name]
                        
                        if 'range_min' in motor_config and 'range_max' in motor_config:
                            range_min = motor_config['range_min']
                            range_max = motor_config['range_max']
                            
                            # Convert servo units to degrees (centered at 2048)
                            # Degrees = (servo_units - 2048) / (4096/360)
                            min_degrees = (range_min - SERVO_CENTER) / SERVO_UNITS_PER_DEGREE
                            max_degrees = (range_max - SERVO_CENTER) / SERVO_UNITS_PER_DEGREE
                            
                            self.joint_limits_lower[i] = min_degrees
                            self.joint_limits_upper[i] = max_degrees
                            limits_found = True
                
                if limits_found:
                    print(f"  ✓ Converted servo ranges to degrees:")
                    for i, key in enumerate(self.joint_keys):
                        print(f"    {key}: [{self.joint_limits_lower[i]:.1f}°, {self.joint_limits_upper[i]:.1f}°]")
                else:
                    print(f"  ⚠️  No range_min/range_max found in motor configs")
                    print(f"  Using default ±180° limits")
                
            except Exception as e:
                print(f"  ⚠️  Could not load calibration limits: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Using default ±180° limits")
        else:
            print(f"  ⚠️  Calibration file not found at {calib_path}")
            print(f"  Using default ±180° limits")
    
    def _initialize_presets(self):
        """Initialize preset positions"""
        # Home position
        self.preset_positions['home'] = np.array([
            0.0,    # shoulder_pan: centered
            -108.0, # shoulder_lift: back
            95.0,   # elbow_flex: 90° bend
            55.0,   # wrist_flex: back
            -90.0,  # wrist_roll: centered
            0.0     # gripper: neutral
        ], dtype=np.float32)

        # Movement position: ready for driving/navigation
        self.preset_positions['movement'] = np.array([
            0.3,    # shoulder_pan
            -85.7,  # shoulder_lift
            95.0,   # elbow_flex
            -23.0,  # wrist_flex
            -90.0,  # wrist_roll
            0.0     # gripper: neutral
        ], dtype=np.float32)

        # Grab position: arm reaching to grab an object
        self.preset_positions['grab'] = np.array([
            -12.3,  # shoulder_pan
            60.9,   # shoulder_lift
            -38.3,  # elbow_flex
            73.0,   # wrist_flex
            -90.0,  # wrist_roll
            64.1    # gripper: open
        ], dtype=np.float32)

        # Drop position: arm extended to drop an object
        self.preset_positions['drop'] = np.array([
            -2.5,   # shoulder_pan
            -30.0,  # shoulder_lift
            -98.0,  # elbow_flex
            -107.6, # wrist_flex
            90.0,   # wrist_roll
            -3.3    # gripper: closed
        ], dtype=np.float32)

        print(f"  ✓ Initialized preset positions:")
        print(f"    A button → Home")
        print(f"    B button → Movement")
        print(f"    X button → Drop")
        print(f"    Y button → Grab")
        
    def apply_deadzone(self, value):
        """Apply dead zone to analog stick values"""
        if abs(value) < self.DEAD_ZONE:
            return 0.0
        # Re-scale to smooth transition
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEAD_ZONE) / (1.0 - self.DEAD_ZONE)
    
    def get_gamepad_input(self):
        """Read gamepad state and return action vector"""
        for event in pygame.event.get():
            pass

        rb = self.joystick.get_button(self.BTN_RB)
        lb = self.joystick.get_button(self.BTN_LB)
        a  = self.joystick.get_button(self.BTN_A)
        b  = self.joystick.get_button(self.BTN_B)
        x  = self.joystick.get_button(self.BTN_X)
        y  = self.joystick.get_button(self.BTN_Y)

        # Rising-edge detection (tap, not hold)
        rb_pressed = rb and not self._prev_rb
        lb_pressed = lb and not self._prev_lb
        a_pressed  = a  and not self._prev_a
        b_pressed  = b  and not self._prev_b
        x_pressed  = x  and not self._prev_x
        y_pressed  = y  and not self._prev_y

        self._prev_rb = rb
        self._prev_lb = lb
        self._prev_a  = a
        self._prev_b  = b
        self._prev_x  = x
        self._prev_y  = y

        # Check exit button
        if self.joystick.get_button(self.BTN_START):
            self.running = False
            return None

        # ── Mode switching ─────────────────────────────────────────
        if rb_pressed and lb_pressed:
            print("Mode: IDLE (both bumpers)")
            self.mode = None
        elif rb_pressed:
            if self.mode == "arm":
                print("Mode: IDLE")
                self.mode = None
            else:
                print("Mode: ARM")
                self.mode = "arm"
        elif lb_pressed:
            if self.mode == "motor":
                print("Mode: IDLE")
                self.mode = None
            else:
                print("Mode: MOTOR")
                self.mode = "motor"

        # ── Preset buttons (work in any mode) ─────────────────────
        if a_pressed:
            print("→ Moving to HOME position")
            return "preset_home"
        if b_pressed:
            print("→ Moving to MOVEMENT position")
            return "preset_movement"
        if x_pressed:
            print("→ Moving to DROP position")
            return "preset_drop"
        if y_pressed:
            print("→ Moving to GRAB position")
            return "preset_grab"

        # ── Motor mode ─────────────────────────────────────────────
        if self.mode == "motor":
            lt = (self.joystick.get_axis(self.AXIS_LT) + 1.0) / 2.0
            rt = (self.joystick.get_axis(self.AXIS_RT) + 1.0) / 2.0
            throttle = rt - lt  # positive = forward, negative = backward
            turn = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
            m1 = max(-1.0, min(1.0, throttle - turn))
            m2 = max(-1.0, min(1.0, throttle + turn))
            return {"motor": True, "m1": m1, "m2": m2}

        # ── Arm mode ───────────────────────────────────────────────
        if self.mode == "arm":
            left_x  = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
            left_y  = -self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_Y))
            right_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_Y))

            lt = (self.joystick.get_axis(self.AXIS_LT) + 1.0) / 2.0
            rt = (self.joystick.get_axis(self.AXIS_RT) + 1.0) / 2.0

            hat_x, hat_y = 0, 0
            if self.joystick.get_numhats() > 0:
                hat = self.joystick.get_hat(0)
                hat_x = -hat[0]
                hat_y =  hat[1]

            action = np.zeros(self.num_joints)
            if self.num_joints >= 6:
                action[0] = left_x  * self.max_speed   # shoulder_pan
                action[1] = left_y  * self.max_speed   # shoulder_lift
                action[2] = right_y * self.max_speed   # elbow_flex
                action[3] = -hat_y  * self.max_speed   # wrist_flex (D-pad up/down)
                action[4] = hat_x   * self.max_speed   # wrist_roll (D-pad left/right)
                if rt > 0.1:
                    action[5] = -rt * self.max_speed   # close gripper
                elif lt > 0.1:
                    action[5] =  lt * self.max_speed   # open gripper
            return action

        # ── Idle — nothing to do ───────────────────────────────────
        return np.zeros(self.num_joints)

    def _maybe_show_images(self) -> None:
        if not self._show_images or self._cam_buffer is None:
            return
        self._img_frame += 1
        if self._img_frame % self._img_interval != 0:
            return
        frames = self._cam_buffer.get_latest()
        for cam_name, img in frames.items():
            if img is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                # img is RGB from LeRobot — pygame wants RGB too, so no conversion needed
                surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                self._pygame_screen.blit(surface, (0, 0))
                pygame.display.flip()

    def run(self):
        """Main control loop"""
        print("\n" + "="*60)
        print("SO-101 GAMEPAD CONTROL")
        print("="*60)
        print("\nControls:")
        print("  ── MOTOR mode (tap LB to toggle) ─────────────────────")
        print("  RT:                  Drive forward")
        print("  LT:                  Drive backward")
        print("  Left Stick X:        Steer left / right")
        print("  ── ARM mode (tap RB to toggle) ───────────────────────")
        print("  Left Stick:          Shoulder pan & lift  (joints 0-1)")
        print("  Right Stick Y:       Elbow flex           (joint 2)")
        print("  D-Pad Up/Down:       Wrist flex           (joint 3)")
        print("  D-Pad Left/Right:    Wrist roll           (joint 4)")
        print("  LT:                  Open gripper")
        print("  RT:                  Close gripper")
        print("  ── General ───────────────────────────────────────────")
        print("  RB + LB:             Force idle (stop everything)")
        print("  A: HOME  B: MOVEMENT  X: DROP  Y: GRAB")
        print("  Start:               Exit")
        print("\nℹ️  Tap RB → ARM mode  |  Tap LB → MOTOR mode  |  Tap again → IDLE")
        print("Starting in 3 seconds...\n")
        time.sleep(3)
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get gamepad input
                action = self.get_gamepad_input()
                
                if action is None:
                    break

                # ── Motor command (local robot doesn't drive, just log) ──
                if isinstance(action, dict) and action.get("motor"):
                    # Motor mode not connected on local robot — log only
                    print(f"\r[MOTOR] m1={action['m1']:+.2f}  m2={action['m2']:+.2f}", end="", flush=True)

                # ── Preset handling ────────────────────────────────────
                elif isinstance(action, str):
                    preset_name = action.removeprefix("preset_")
                    target_position = self.preset_positions.get(preset_name, self.preset_positions['home']).copy()
                    target_position = np.clip(target_position, self.joint_limits_lower, self.joint_limits_upper)
                    action_dict = {key: float(target_position[i]) for i, key in enumerate(self.joint_keys)}
                    self.robot.send_action(action_dict)
                    self.current_position = target_position
                    time.sleep(0.5)  # Brief pause after preset movement

                # ── Arm continuous control ─────────────────────────────
                else:
                    # Calculate new target position (relative control)
                    target_position = self.current_position + action
                    target_position = np.clip(target_position, self.joint_limits_lower, self.joint_limits_upper)
                    action_dict = {key: float(target_position[i]) for i, key in enumerate(self.joint_keys)}
                    self.robot.send_action(action_dict)
                    self.current_position = target_position

                    # Display status (every ~1 second)
                    if int(time.time() * 30) % 30 == 0:
                        active = np.any(np.abs(action) > 0.001)
                        if active or self.mode == "arm":
                            status = "ACTIVE" if active else "READY"
                            print(f"[{status}] Position: {np.round(self.current_position, 2)}")
                
                # Maintain control frequency
                self._maybe_show_images()
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.control_dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        
        finally:
            print("\nShutting down...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self._cam_buffer is not None:
            self._cam_buffer.stop()
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        print("Disconnecting robot...")
        self.robot.disconnect()
        pygame.quit()
        print("✓ Cleanup complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Control SO-101 robot arm with Xbox controller"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO-101 (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="so101_follower",
        help="Robot ID (default: so101_follower)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=2.0,
        help="Maximum joint speed in degrees/step (default: 2.0)",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="Control frequency in Hz (default: 30)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="",
        help=(
            "Camera spec. Simple: 'front:/dev/video0,base:/dev/video2'  "
            "or lerobot-style: '{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30} }'"
        ),
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display camera frames in cv2 windows (requires opencv-python)",
    )
    parser.add_argument(
        "--image-hz",
        type=int,
        default=5,
        help="Camera display refresh rate in Hz (default: 5)",
    )

    args = parser.parse_args()
    
    calib_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / f"{args.robot_id}.json"
    
    if not calib_path.exists():
        print(f"\n⚠️  WARNING: Calibration file not found at {calib_path}")
        print("Please calibrate your robot first with:")
        print(f"  lerobot-calibrate --robot.type=so101_follower --robot.port={args.port} --robot.id={args.robot_id}")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Create controller and run
    controller = SO101GamepadController(
        robot_port=args.port,
        robot_id=args.robot_id,
        max_speed=args.max_speed,
        control_frequency=args.frequency,
        cameras=args.cameras,
        show_images=args.show_images,
        image_hz=args.image_hz,
    )
    
    controller.run()


if __name__ == "__main__":
    main()