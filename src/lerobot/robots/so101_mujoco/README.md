# SO-101 MuJoCo Robot for LeRobot

MuJoCo simulation of the SO-101 5-DOF robot arm with keyboard teleoperation for LeRobot dataset collection.

## Overview

This robot implementation enables intuitive keyboard teleoperation of the SO-101 arm in MuJoCo simulation, producing high-quality demonstration datasets compatible with LeRobot's behavior cloning pipeline. The system handles the unique challenges of a 5-DOF robot by making the end-effector always point downward and providing intuitive Cartesian control.

## Design Philosophy

### Why Point the Tool Down?

The SO-101 has only 5 DOF (pan, lift, elbow, wrist_flex, wrist_roll), which is insufficient for arbitrary 6-DOF end-effector control. Our solution:

1. **Constrain orientation**: Tool always points downward (toward -Z)
2. **Free up DOF**: This reduces the problem to 4 DOF (3 for XYZ position + 1 for wrist roll)
3. **Intuitive control**: Users control the wrist center point, not the gripper fingers
4. **Vertical maintenance**: Wrist_flex actively corrects tilt errors to keep tool vertical

This design makes teleoperation feel natural despite the kinematic limitations.

### End-Effector Placement

The end-effector is placed at the **wrist center** (`wrist_site`), not at the gripper fingers. This choice:

- Provides stable Jacobian conditioning (wrist has good manipulability)
- Keeps control point visible and predictable
- Avoids singularities near gripper closure
- Makes XYZ motion more intuitive (moving the "hand" rather than the "fingers")

## Keyboard Controls

The control scheme maps directly to Cartesian space:

| Key | Action | Frame |
|-----|--------|-------|
| **W/S** | Forward/Backward | +Y / -Y (world) |
| **A/D** | Left/Right | -X / +X (world) |
| **Q/E** | Up/Down | +Z / -Z (world) |
| **[/]** | Wrist roll | CCW / CW |
| **O/C** | Gripper | Open / Close |

**Design rationale**:
- WASD cluster for horizontal motion (familiar from gaming)
- QE for vertical (right hand can reach while left hand does WASD)
- Brackets for wrist roll (pinky reach)
- OC for gripper (mnemonic: Open/Close)

## LeRobot Integration

### Robot Interface

The `SO101MujocoRobot` class implements LeRobot's `Robot` base class:

```python
class SO101MujocoRobot(Robot):
    def connect(self) -> None:
        # Load MuJoCo model, setup rendering

    def get_observation(self) -> dict:
        # Return {joints, cameras, ee_pos}

    def send_action(self, action: dict) -> dict:
        # Execute action, return recorded action
```

### Teleoperation Flow

The recording loop (`lerobot_record.py`) works as follows:

1. **Keyboard input** → Captured by `LeRobotKeyboardTeleoperator`
2. **Convert to base action** → `_from_keyboard_to_base_action()` produces velocity dict
3. **Send action** → `send_action()` executes control and returns joint positions
4. **Record** → Joint positions saved to dataset

### Multi-Rate Control

LeRobot records at 30 Hz, but high-quality control needs higher frequency. Our solution:

**Recording cycle (30 Hz = 33.3 ms)**:
```
send_action(velocities) {
    for i in 0..5:  // 6 iterations = 180 Hz control
        _control_step(velocities) {
            - Compute Jacobian
            - Solve for joint velocities
            - Add wrist tilt correction
            - Apply gravity compensation
            - Rate limit and smooth
            - Step physics 2 times (360 Hz)
        }
    return final_joint_positions  // These get recorded
}
```

**Key insight**: We record the **target joint positions** at the end of each 30 Hz cycle, not the instantaneous state. These targets represent where we *commanded* the robot to be after executing the velocity for 33.3 ms.

### Action Recording Nuances

**During recording (teleop mode)**:
- Input: `{vx, vy, vz, yaw_rate, gripper_delta}` (velocities)
- Process: Run 6 control iterations @ 180 Hz
- Output: `{shoulder_pan.pos, ..., gripper.pos}` (joint positions)

The recorded actions are **desired joint positions** (`q_des`), computed via:
1. Jacobian maps velocities → joint velocities
2. Integration accumulates to position targets
3. Gripper uses rate control: `q_gripper += gripper_delta * dt`

**Important**: The gripper position must be synced from `data.ctrl` to `q_des` after rate control (in `robot_so101_mujoco.py`) to ensure it gets recorded correctly.

## Recording vs Replay

The system supports two distinct action modes, auto-detected based on dict keys:

### Recording Mode (Velocity → Position)

**Input**: `{vx, vy, vz, yaw_rate, gripper_delta}`

**Process**:
1. Run 6 control iterations @ 180 Hz
2. Each iteration: Jacobian → joint velocities → integrate → position targets
3. Physics runs at 360 Hz (2 steps per control iteration)

**Output**: `{shoulder_pan.pos, ..., gripper.pos}` → Saved to dataset

### Replay Mode (Position → Position)

**Input**: `{shoulder_pan.pos, ..., gripper.pos}` (from dataset)

**Process**:
1. Set `data.ctrl` directly to recorded positions
2. Step physics 12 times @ 360 Hz (one 30 Hz cycle)
3. No Jacobian computation, no IK

**Output**: Same positions (echoed back)

### Why Different APIs?

**Recording** needs high-frequency control because:
- Velocities must be converted to positions via numerical integration
- Orientation correction requires feedback (current orientation → correction torque)
- Gravity compensation adapts to configuration
- This is computationally intensive (Jacobian at 180 Hz)

**Replay** is simpler because:
- Dataset already contains desired positions
- Just track those positions with PD control (done by MuJoCo actuators)
- No IK needed
- Much faster (no 180 Hz loop)

The `send_action()` method auto-detects the mode by checking dict keys:
- Has velocity keys → Recording mode
- Has position keys → Replay mode

## Control Implementation

### Jacobian-Based XYZ Control

For position control of a 5-DOF arm, we use the first 3 joints (pan, lift, elbow) for XYZ:

```python
J3 = J_position[:, [shoulder_pan, shoulder_lift, elbow_flex]]  # 3×3
v_des = [vx, vy, vz]
dq = J3.T @ solve(J3@J3.T + λ²I, v_des)  # Damped least-squares
```

This is the **primary task**: achieve desired XYZ velocity.

### Wrist Tilt Correction (Secondary Task)

The wrist_flex joint actively maintains vertical orientation:

```python
tool_axis = R @ [0, -1, 0]  # Current tool direction (wrist frame)
error = cross(tool_axis, [0, 0, -1])  # Error from vertical
w_xy = k_ori * error[:2]  # Correction angular velocity (XY only)
dq_wrist_flex = J_rotation[:2, wrist_flex].T @ w_xy
```

This correction:
- Runs in the null-space of the primary task (doesn't interfere with XYZ)
- Fades near singularities (when Jacobian conditioning is poor)
- Fades near joint limits (directional - only when moving toward limit)
- Has a deadzone to avoid jitter

### Gravity Compensation

The wrist joint fights gravity. We add feedforward to reduce droop:

```python
tau_gravity = data.qfrc_bias[wrist_flex]  # Gravity torque
dq_wrist_flex += k_gff * tau_gravity
```

### Gripper Control

The gripper uses **rate control** (independent of arm):

```python
q_gripper_new = q_gripper_old + gripper_delta * dt
data.ctrl[gripper] = clip(q_gripper_new, [lo, hi])
q_des[gripper] = data.ctrl[gripper]  # Sync for recording!
```

**Critical**: Line 669 syncs `q_des` with `data.ctrl` so the actual gripper position gets recorded. Without this, the dataset would contain the stale home position (0.8).

### Timing and Physics

**Frequencies**:
- 360 Hz: Physics timestep (MuJoCo `dt = 1/360 ≈ 2.78 ms`)
- 180 Hz: Control loop (6× physics steps)
- 30 Hz: Recording (6× control iterations)

**Exact multiples** ensure no timing drift:
- `n_physics_per_control = 2` (exactly)
- `n_control_per_record = 6` (exactly)

During replay, we skip the 180 Hz loop and just run 12 physics steps at 360 Hz for one 30 Hz frame.

## Configuration Parameters

Key parameters in `SO101MujocoConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lin_speed` | 0.04 m/s | XYZ velocity per key press |
| `yaw_speed` | 1.20 rad/s | Wrist roll rate |
| `grip_speed` | 0.7 rad/s | Gripper rate |
| `ori_gain` | 6.0 | Tilt correction strength |
| `lambda_pos` | 0.01 | Jacobian damping for XYZ |
| `lambda_tilt` | 0.0001 | Jacobian damping for tilt |
| `vel_limit` | 0.5 rad/s | Joint velocity limit (arm) |
| `vel_limit_wrist` | 8.0 rad/s | Joint velocity limit (wrist) |
| `smooth_dq` | 0.30 | Velocity smoothing (arm) |
| `smooth_dq_wrist` | 0.08 | Velocity smoothing (wrist) |
| `wrist_gff_gain` | 0.5 | Gravity feedforward gain |

## Files

- `robot_so101_mujoco.py` - Main robot class (1000+ lines)
- `configuration_so101_mujoco.py` - Config dataclass
- `__init__.py` - Module exports

## Technical Notes

### Why 180 Hz Control?

- Must be a multiple of both 30 Hz (recording) and 360 Hz (physics)
- Higher than 30 Hz prevents jerky motion from velocity discretization
- 180 Hz = 6 control steps per frame = smooth integration
- Lower than original 200 Hz for exact integer ratios

### Why Record Joint Positions?

- Compatible with real robot interfaces (position control)
- Policies can learn joint-space trajectories
- Avoids IK during policy execution
- Simpler replay (just track positions)

### Observation Frequency

Observations are captured at 30 Hz (once per `get_observation()` call). The internal 180 Hz control doesn't generate observations - it only updates `q_des` which gets returned as the action.

## Differences from Standalone Teleop

The original `orient_down.py` script runs a continuous loop with GLFW rendering. This implementation adapts it for LeRobot:

**Similarities**:
- ✅ Identical Jacobian computation
- ✅ Same tilt correction logic
- ✅ Same gravity compensation
- ✅ Same rate limiting and smoothing

**Differences**:
- **Chunked execution**: Runs 6 control iterations per call (not continuous)
- **No viewer loop**: GLFW rendering happens separately
- **Returns actions**: Must return dict for recording
- **Dual mode**: Supports both velocity (record) and position (replay) actions

The core control quality is preserved - just the execution structure changed.

## Related Documentation

- **Main README**: `../../../../README.md` - Project overview and quick start
- **Collision Geometry**: `../../../../SO101/asset_processing.md` - Gripper collision fix
- **Config Example**: `../../../../configs/so101_mujoco_record.yaml` - Recording settings
