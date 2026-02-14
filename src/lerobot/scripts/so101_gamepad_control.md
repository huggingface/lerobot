# SO-101 Xbox Gamepad Control - Complete Guide

Complete documentation for controlling your SO-101 robotic arm with an Xbox controller.

**Version:** 1.0  
**Author:** Custom LeRobot Extension  
**Date:** February 2026

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Controller Mapping](#controller-mapping)
4. [Features](#features)
5. [Understanding Calibration](#understanding-calibration)
6. [Customizing Preset Positions](#customizing-preset-positions)
7. [Custom Joint Limits](#custom-joint-limits)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)
10. [FAQ](#faq)

---

## Quick Start

### Prerequisites

- ✅ LeRobot installed with Feetech support
- ✅ SO-101 arm calibrated
- ✅ Xbox controller (or compatible gamepad)
- ✅ pygame library

### Install

```bash
pip install "lerobot[feetech]" pygame
```

### Run

```bash
python lerobot_gamepad_control.py --port COM6 --robot-id my_awesome_follower_arm
```

**Windows users:** Use `COM6` (or your port)  
**Linux users:** Use `/dev/ttyACM0` (or your port)

### Controls at a Glance

| Input | Function |
|-------|----------|
| Left Stick | Shoulder pan & lift |
| Right Stick | Elbow & wrist flex |
| D-Pad L/R | Wrist roll |
| LT / RT | Gripper close / open |
| **RB (HOLD)** | **Enable movement** |
| A / X / Y | Preset positions |
| Start | Exit |

⚠️ **IMPORTANT:** Hold RB (right bumper) to enable manual movement!

---

## Controller Mapping

### Detailed Control Table

| Control | Joint(s) | Function | Needs RB? |
|---------|----------|----------|-----------|
| **Left Stick X** | Shoulder Pan | Base rotation left/right | ✅ Yes |
| **Left Stick Y** | Shoulder Lift | Arm up/down | ✅ Yes |
| **Right Stick X** | Wrist Flex | Wrist pitch | ✅ Yes |
| **Right Stick Y** | Elbow Flex | Elbow bend | ✅ Yes |
| **D-Pad Left** | Wrist Roll | Rotate CCW | ✅ Yes |
| **D-Pad Right** | Wrist Roll | Rotate CW | ✅ Yes |
| **LT (Trigger)** | Gripper | Close | ✅ Yes |
| **RT (Trigger)** | Gripper | Open | ✅ Yes |
| **RB (Bumper)** | - | Enable (SAFETY) | - |
| **A Button** | All | HOME position (0°) | ❌ No |
| **X Button** | All | READY position | ❌ No |
| **Y Button** | All | VERTICAL position | ❌ No |
| **B Button** | All | Reset to home | ❌ No |
| **Start** | - | Exit program | ❌ No |

### Joint Index Reference

```
Joint 0: shoulder_pan   (Base rotation)
Joint 1: shoulder_lift  (Arm up/down)
Joint 2: elbow_flex     (Elbow bend)
Joint 3: wrist_flex     (Wrist pitch)
Joint 4: wrist_roll     (Wrist rotation)
Joint 5: gripper        (Open/close)
```

---

## Features

### 1. Safety Features

✅ **Dead-Man Switch (RB Button)**
- Must hold RB to enable manual movement
- Releasing RB stops all motion immediately
- Preset buttons (A/X/Y) work without RB

✅ **Position Limits**
- Automatically clips to calibrated safe ranges
- Prevents servo damage from over-extension
- Uses actual ranges from your calibration file

✅ **Speed Limiting**
- Max speed configurable (default 2°/step)
- Prevents sudden jerky movements
- Adjustable via `--max-speed` parameter

### 2. Preset Positions

Three instant-move positions accessible without holding RB:

**A Button - HOME**
**X Button - READY**
**Y Button - VERTICAL**

---

## Customizing Preset Positions

### Editing Presets

Open `lerobot_gamepad_control.py` and find `_initialize_presets()` method (around line 200):

```python
def _initialize_presets(self):
    """Initialize preset positions"""
    
    # HOME: All zeros
    self.preset_positions['home'] = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    
    # READY: Forward reach (X button)
    self.preset_positions['ready'] = np.array([
        0.0,    # shoulder_pan
        0.0,    # shoulder_lift
        -90.0,  # elbow_flex
        0.0,    # wrist_flex
        0.0,    # wrist_roll
        0.0     # gripper
    ], dtype=np.float32)
    
    # VERTICAL: Upward reach (Y button)
    self.preset_positions['vertical'] = np.array([
        0.0,    # shoulder_pan
        90.0,   # shoulder_lift
        0.0,    # elbow_flex
        0.0,    # wrist_flex
        0.0,    # wrist_roll
        0.0     # gripper
    ], dtype=np.float32)
```

### Example: Pick-and-Place Presets

```python
def _initialize_presets(self):
    # HOME: Safe start
    self.preset_positions['home'] = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    
    # READY (X): Above pick location
    self.preset_positions['ready'] = np.array([
        -25.0,  # Turn to pick area
        -20.0,  # Above object
        -80.0,  # Bent
        -30.0,  # Angled down
        0.0,    # Straight
        45.0    # Open gripper
    ], dtype=np.float32)
    
    # VERTICAL (Y): At pick location
    self.preset_positions['vertical'] = np.array([
        -25.0,  # Same pan
        -35.0,  # Lowered
        -80.0,  # Same
        -30.0,  # Same
        0.0,    # Same
        45.0    # Still open
    ], dtype=np.float32)
```

### Adding More Presets

You can add a 4th preset using the LB button:

**Step 1:** Add to `_initialize_presets()`
```python
self.preset_positions['storage'] = np.array([
    -90.0, -45.0, -120.0, -30.0, 0.0, 50.0
], dtype=np.float32)
```

**Step 2:** Add button handler in `get_gamepad_input()` (around line 250)
```python
# Check for LB button
if self.joystick.get_button(self.BTN_LB):
    print("→ Moving to STORAGE position")
    return "preset_storage"
```

**Step 3:** Update printed instructions in `run()` method
```python
print("  LB Button:          Move to STORAGE")
```

---

## Advanced Topics

### Command Line Options

```bash
lerobot_gamepad_control.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | string | `/dev/ttyACM0` | Serial port for SO-101 |
| `--robot-id` | string | `so101_follower` | Robot identifier (calibration name) |
| `--max-speed` | float | `2.0` | Max joint speed (degrees/step) |
| `--frequency` | int | `30` | Control loop frequency (Hz) |

**Examples:**

Slow, precise control:
```bash
lerobot_gamepad_control.py --port COM6 --max-speed 0.5 --frequency 60
```

Fast, responsive:
```bash
lerobot_gamepad_control.py --port COM6 --max-speed 3.5 --frequency 30
```

Different robot:
```bash
lerobot_gamepad_control.py --port COM7 --robot-id my_other_robot
```

### Speed Calculation

Max speed is in degrees per control step:
```
Actual speed (°/sec) = max_speed × control_frequency

Examples:
  max_speed=2.0, frequency=30Hz → 60°/second
  max_speed=1.0, frequency=60Hz → 60°/second (smoother)
  max_speed=3.0, frequency=30Hz → 90°/second (faster)
```

### Understanding the Control Loop

```
Every 1/30th second (at 30Hz):
  ├─ Read gamepad input
  ├─ Calculate action (max ±2.0° per joint)
  ├─ Add to current position
  ├─ Clip to safe limits
  └─ Send to robot

Result: Smooth incremental movement
```

### Custom Joint Mapping

Edit the script (around line 265) to change which stick controls which joint:

```python
# Default mapping
action[0] = left_x * self.max_speed      # Shoulder pan
action[1] = left_y * self.max_speed      # Shoulder lift
action[2] = right_y * self.max_speed     # Elbow flex
action[3] = right_x * self.max_speed     # Wrist flex
action[4] = hat_x * self.max_speed       # Wrist roll

# Example: Swap elbow and wrist
action[2] = right_x * self.max_speed     # Elbow flex
action[3] = right_y * self.max_speed     # Wrist flex
```

### Recording Demonstrations

Once comfortable with gamepad control, you can record datasets for training:

```python
# TODO: Add recording capability
# Save observations and actions to LeRobot dataset format
# See LeRobot documentation for dataset recording
```

(Feature not yet implemented in this script)

---

## FAQ

**Q: Do I need to hold RB all the time?**  
A: Only for manual control (sticks, triggers, D-pad). Preset buttons (A/X/Y) work without RB.

**Q: Why are my joint limits asymmetric?**  
A: This is normal! Joints have physical constraints. Example: elbow can flex -135° but only extend +15°.

**Q: Can I use a PlayStation controller?**  
A: Yes! Button indices might be different. Run the script and note which buttons trigger which actions, then edit the button constants.

**Q: What's the difference between 0° and servo center?**  
A: Your 0° is wherever you pointed the arm during calibration. Servo center (2048 units) is the servo's internal reference. The `homing_offset` connects them.

**Q: Can I change preset positions?**  
A: Yes! Edit `_initialize_presets()` in the script. Use manual control to find good positions, then copy the values.

**Q: How do I find my robot's port?**  
A: Run `lerobot-find-port` or manually check by unplugging/replugging and watching which port appears.

**Q: Why does the script use degrees instead of radians?**  
A: Degrees are more intuitive for humans. Internally, servos use units (0-4095), but LeRobot abstracts this to degrees.

**Q: Can I add more preset positions?**  
A: Yes! See "Customizing Preset Positions" section. You can add a 4th preset using LB button.

**Q: Is it safe to hold the stick in one direction?**  
A: Yes! Position limits prevent over-extension. The robot will stop at the calibrated safe limit.

**Q: What happens if I lose calibration?**  
A: Recalibrate with `lerobot-calibrate --robot.type=so101_follower --robot.port=COM6`. Always backup your calibration file!

**Q: Can I use this for data collection?**  
A: Currently, this script is for teleoperation only. Dataset recording functionality could be added.

**Q: Why 30Hz control frequency?**  
A: Good balance between smoothness and responsiveness. Higher (60Hz) is smoother but more CPU. Lower (15Hz) is jittery.

**Q: Do I need pygame for anything else?**  
A: No, pygame is only used for gamepad input. Rest of the code uses LeRobot.

---

## Support & Contributing

**Issues:** Report bugs or suggest features on GitHub  
**Documentation:** This guide covers all features  
**Source Code:** Available in LeRobot repository  

**Version History:**
- v1.0 (Feb 2026) - Initial release with full gamepad control

---


*End of Documentation*