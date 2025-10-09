# SO-101 MuJoCo Robot for LeRobot

LeRobot-compatible implementation of the SO-101 robot in MuJoCo simulation.

## Overview

This robot class enables behavior cloning data collection in simulation with the same control quality as the original teleop script.

## Key Features

- **Position-based action recording**: Records joint position targets (compatible with real robots)
- **High-frequency control**: 180 Hz control loop for smooth, stable motion
- **Jacobian-based XYZ control**: Cartesian space teleoperation via keyboard
- **Automatic vertical orientation**: Wrist maintains tool pointing downward
- **Gravity compensation**: Counters wrist droop during motion
- **Exact timing**: All frequencies are exact multiples (no accumulation errors)

## Architecture

```
Recording Loop (30 Hz)
  ↓
  ├─ get_observation() → {joint pos/vel, ee_pos, camera}
  ├─ keyboard input → velocity commands
  ├─ send_action() → runs 6 control iterations
  │    └─ Each control iteration (180 Hz):
  │         ├─ Compute Jacobian
  │         ├─ Solve for joint velocities (XYZ + tilt)
  │         ├─ Apply gravity compensation
  │         ├─ Rate limit + smooth
  │         ├─ Integrate to position targets
  │         └─ Step physics 2 times (360 Hz)
  └─ Return final position targets (for recording)
```

## Frequencies

| Rate | Purpose | Timestep |
|------|---------|----------|
| 360 Hz | Physics simulation | 2.78 ms |
| 180 Hz | Control loop | 5.56 ms |
| 30 Hz | Dataset recording | 33.33 ms |

All are exact multiples:
- 2 physics steps per control iteration
- 6 control iterations per recording frame

## Action Space

Joint position targets (radians):

```python
{
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float,
}
```

## Observation Space

```python
{
    # Joint positions (rad)
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float,

    # Joint velocities (rad/s)
    "shoulder_pan.vel": float,
    ... (all 6 joints)

    # End-effector position (m)
    "ee.pos_x": float,
    "ee.pos_y": float,
    "ee.pos_z": float,

    # Camera (RGB image)
    "camera_front": (128, 128, 3),  # uint8
}
```

## Files

- **`configuration_so101_mujoco.py`**: Configuration dataclass
- **`robot_so101_mujoco.py`**: Main Robot class
- **`test_so101_standalone.py`**: Interactive test with GLFW viewer
- **`test_control_validation.py`**: Automated validation tests
- **`README.md`**: This file

## Usage

### Standalone Testing

```bash
cd lerobot/src/lerobot/robots/so101_mujoco
python test_so101_standalone.py
```

**Controls:**
- W/S: Move +X/-X
- A/D: Move +Y/-Y
- E/Q: Move +Z/-Z
- [/]: Wrist roll left/right
- ,/.: Gripper close/open
- R: Reset to home
- ESC: Exit

### Validation Tests

```bash
python test_control_validation.py
```

Runs automated tests for:
- Initialization
- Observation structure
- XYZ motion correctness
- Vertical orientation maintenance
- Timing accuracy

### With lerobot_record (TODO)

```bash
lerobot-record \
    --robot.type=so101_mujoco \
    --robot.xml_path=gym-hil/gym_hil/assets/SO101/pick_scene.xml \
    --teleop.type=keyboard \
    --dataset.repo_id=yourname/so101_sim_dataset \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick and place cube"
```

## Control Logic Details

### XYZ Control (Primary Task)

Uses Jacobian pseudoinverse on first 3 DOF (pan, lift, elbow):

```python
J3 = Jp[:, [pan, lift, elbow]]  # 3×3 Jacobian
v_des = [vx, vy, vz]             # Desired velocity
dq = J3.T @ solve(J3@J3.T + λ²I, v_des)  # Damped least squares
```

### Wrist Tilt Correction (Secondary Task)

Maintains tool pointing downward using wrist_flex:

```python
tool_axis = R @ [0, -1, 0]  # Current tool direction
error = cross(tool_axis, [0, 0, -1])  # Error from vertical
w_xy = k_ori × error[:2]  # Correction angular velocity
dq_wf = Jr[:2, wf].T @ w_xy  # Map to wrist_flex
```

Automatically fades near:
- Singularities (low Jacobian singular value)
- Joint limits (directional check)

### Gravity Compensation

Adds feedforward term to wrist_flex:

```python
tau_gravity = data.qfrc_bias[wrist_flex]
dq_wf += k_gff × tau_gravity
```

### Rate Limiting & Smoothing

```python
dq = clip(dq, -v_max, +v_max)  # Hard velocity limits
dq_filt = (1-α) × dq_filt + α × dq  # Exponential filter
```

## Configuration Parameters

All control parameters match `orient_down.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lin_speed` | 0.04 m/s | XYZ velocity |
| `yaw_speed` | 1.20 rad/s | Wrist roll rate |
| `ori_gain` | 6.0 | Tilt correction strength |
| `lambda_pos` | 0.01 | XYZ solve damping |
| `vel_limit` | 0.5 rad/s | Arm joint limit |
| `vel_limit_wrist` | 8.0 rad/s | Wrist joint limit |
| `smooth_dq` | 0.30 | Arm smoothing |
| `smooth_dq_wrist` | 0.08 | Wrist smoothing |
| `wrist_gff_gain` | 0.5 | Gravity compensation |

## Differences from orient_down.py

1. **Chunked control**: Runs 6 control iterations per recording frame (orient_down runs continuously)
2. **Action output**: Returns joint position targets (orient_down just updates display)
3. **No viewer loop**: Uses external rendering (orient_down has GLFW loop)
4. **Frequencies**: 180 Hz control vs 200 Hz (for exact timing)

All **control logic is identical** - just packaged for LeRobot interface.

## Validation Test Results

All automated tests pass:

```bash
$ uv run python lerobot/src/lerobot/robots/so101_mujoco/test_control_validation.py
✓ Initialization test passed
✓ Observation structure test passed
✓ XYZ motion test passed
✓ Vertical orientation test passed
✓ Control stability test passed

Results: 5 passed, 0 failed
```

**Notes:**
- The SO-101 kinematics make perfect vertical tool orientation (alignment=1.0) impossible at some configurations
- Wrist tilt correction is active and maintains reasonable orientation during motion
- XYZ motion works correctly but wrist correction can interfere when tool is far from vertical
- All control frequencies are exact multiples (no timing drift)

## Next Steps

1. ✅ Implement robot class
2. ✅ Create standalone test
3. ✅ Write validation tests
4. ✅ Run validation tests (all passing)
5. ⬜ Integrate with `lerobot_record`
6. ⬜ Collect demonstration dataset
7. ⬜ Train policy
8. ⬜ Deploy and evaluate

## References

- Original control script: `gym-hil/gym_hil/so101/scripts/orient_down.py`
- MuJoCo model: `gym-hil/gym_hil/assets/SO101/pick_scene.xml`
- LeRobot docs: https://github.com/huggingface/lerobot
