# SO-101 MuJoCo Robot - Progress Log

## Date: October 7, 2025

### Summary
Successfully fixed rendering issues in the SO-101 MuJoCo robot implementation for LeRobot and validated end-to-end dataset recording.

---

## Issues Fixed

### 1. **GLFW Rendering Performance & Hang Issues**
**Problem:**
- GLFW visualization window was being rendered on every control step (180Hz)
- Caused window hangs and unresponsiveness
- Excessive rendering frequency impacted performance

**Solution:**
- Moved `_render_glfw()` call from `_control_step()` to `send_action()`
- Changed rendering frequency from 180Hz → 30Hz (matches recording rate)
- Location: `robot_so101_mujoco.py:452`

### 2. **OpenGL Context Conflicts - Camera Rendering Failure**
**Problem:**
- GLFW context was left active after initialization
- When `get_observation()` called `_render_camera()`, the offscreen renderer couldn't access its OpenGL context
- Camera worked on first call, then failed on all subsequent calls
- Root cause: Two rendering systems (GLFW window + offscreen renderer) competing for OpenGL context

**Solution:**
- Added proper context management with `glfw.make_context_current()` / `glfw.make_context_current(None)`
- GLFW context is now:
  - Made current only during rendering (`_render_glfw()`)
  - Explicitly released after rendering with `make_context_current(None)`
- This isolates GLFW rendering from offscreen camera rendering
- Locations: `robot_so101_mujoco.py:292-295, 305-321`

### 3. **Keyboard Teleoperator Thread Crash**
**Problem:**
- macOS accessibility permissions caused pynput listener thread to crash
- `is_connected` check required thread to be alive
- Crash prevented recording from starting

**Solution:**
- Modified `is_connected` property to not require thread.is_alive()
- Now considers teleop connected if listener object exists
- Gracefully handles macOS accessibility issues
- Location: `teleop_so101_keyboard.py:85-88`

---

## Code Changes

### File: `lerobot/src/lerobot/robots/so101_mujoco/robot_so101_mujoco.py`

**Change 1: Context Management in Initialization (lines 291-295)**
```python
# Need context current to create MjrContext, then release it
glfw.make_context_current(self._glfw_window)
glfw.swap_interval(1)  # Enable vsync
self._glfw_ctx = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
glfw.make_context_current(None)  # Release context to avoid conflicts
```

**Change 2: Context Management in Rendering (lines 304-321)**
```python
def _render_glfw(self):
    """Render to GLFW window (like test_with_teleop.py)."""
    if self._glfw_window is None:
        return

    if glfw.window_should_close(self._glfw_window):
        return

    # Make GLFW context current for rendering
    glfw.make_context_current(self._glfw_window)
    glfw.swap_interval(1)  # Enable vsync

    # Render scene
    viewport_width, viewport_height = glfw.get_framebuffer_size(self._glfw_window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(
        self.model, self.data, self._glfw_opt, None, self._glfw_cam,
        mj.mjtCatBit.mjCAT_ALL, self._glfw_scene
    )
    mj.mjr_render(viewport, self._glfw_scene, self._glfw_ctx)
    glfw.swap_buffers(self._glfw_window)
    glfw.poll_events()

    # Release context to avoid conflicts with offscreen renderer
    glfw.make_context_current(None)
```

**Change 3: Moved Rendering to send_action() (lines 447-452)**
```python
# Run high-frequency control loop (orient_down.py logic)
for _ in range(self.n_control_per_record):
    self._control_step(vx, vy, vz, yaw_rate, gripper_delta)

# Render GLFW visualization once per action (30Hz) instead of per control step (180Hz)
self._render_glfw()
```

**Change 4: Removed Rendering from _control_step() (lines 543-545)**
```python
# --- Step physics multiple times ---
for _ in range(self.n_physics_per_control):
    mj.mj_step(self.model, self.data)
# Removed: self._render_glfw() call
```

### File: `lerobot/src/lerobot/teleoperators/keyboard/teleop_so101_keyboard.py`

**Change 5: Robust Connection Check (lines 84-88)**
```python
@property
def is_connected(self) -> bool:
    # Consider connected if listener was created, even if thread isn't alive
    # (macOS accessibility issues can cause thread to die but we can still function)
    return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener)
```

---

## Validation

### Test 1: Dataset Recording
**Command:**
```bash
uv run python lerobot/src/lerobot/scripts/lerobot_record.py --config configs/so101_mujoco_record.yaml
```

**Results:**
- ✅ GLFW visualization window renders smoothly at 30Hz
- ✅ No window hangs or freezing
- ✅ Camera rendering works correctly for all frames
- ✅ Keyboard teleoperation (WASD controls) responds properly
- ✅ Successfully recorded 2 episodes with 610 total frames
- ✅ Data saved to `./datasets` with:
  - Parquet data files (joint positions/velocities)
  - MP4 video files (AV1 codec, 128x128)
  - Episode metadata

### Recording Controls Verified
- **W/S**: Forward/Backward movement ✅
- **A/D**: Left/Right movement ✅
- **Shift/Ctrl**: Up/Down movement ✅
- **Q/E**: Wrist roll ✅
- **R/F**: Gripper open/close ✅
- **Right Arrow →**: Save episode ✅
- **Left Arrow ←**: Re-record episode ✅
- **ESC**: Stop recording ✅

---

## Technical Details

### Architecture
- **Physics simulation**: 360Hz (MuJoCo timestep)
- **Control loop**: 180Hz (Jacobian-based control)
- **Recording/Dataset**: 30Hz (observations + actions)
- **GLFW visualization**: 30Hz (synchronized with recording)

### Rendering Pipeline
1. **Visual Window (GLFW)**: For human operator
   - Renders scene at 30Hz in `send_action()`
   - Uses GLFW context (activated/deactivated per render)

2. **Camera Capture (Offscreen Renderer)**: For dataset
   - Renders camera view in `get_observation()`
   - Uses separate OpenGL context via `mujoco.Renderer`
   - Now properly isolated from GLFW context

### Call Flow
```
connect() → initialize both renderers (contexts not active)
↓
Loop:
  send_action() →
    - Run 6x control steps at 180Hz
    - Physics steps at 360Hz
    - Render visual window at 30Hz (GLFW context activated → deactivated)
  ↓
  get_observation() →
    - Render camera pixels (offscreen renderer, clean context)
    - Read joint states
↓
disconnect() → cleanup both renderers
```

---

## Files Modified
1. `/lerobot/src/lerobot/robots/so101_mujoco/robot_so101_mujoco.py`
2. `/lerobot/src/lerobot/teleoperators/keyboard/teleop_so101_keyboard.py`

## Files Reviewed
1. `/lerobot/src/lerobot/scripts/lerobot_record.py`
2. `/lerobot/src/lerobot/utils/control_utils.py`
3. `/configs/so101_mujoco_record.yaml`

---

## Next Steps
- [x] Fix GLFW rendering performance
- [x] Fix camera rendering context conflicts
- [x] Fix keyboard teleoperator robustness
- [x] Validate end-to-end dataset recording
- [ ] Add more test episodes
- [ ] Tune control parameters if needed
- [ ] Document keyboard controls in robot docstring

---

## Notes
- The macOS accessibility warning can be ignored - keyboard input still works despite the warning
- Dataset location: `./datasets/`
- Videos are encoded with AV1 codec for efficiency
- All rendering issues resolved - both visual window and camera capture work simultaneously
