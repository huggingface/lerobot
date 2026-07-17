# Unitree G1 — SONIC whole-body control

This package runs NVIDIA's **SONIC** whole-body controller (and the GR00T/Holosoma
locomotion controllers) on the Unitree G1, in MuJoCo simulation or on real hardware.
SONIC turns a high-level movement intent — or a streamed **SMPL** whole-body pose — into
50 Hz joint-position targets. It is a pure-Python/ONNX reimplementation of the SONIC
deploy stack (no `gear_sonic`/torch dependency).

## Controllers

Selected with `--robot.controller=<ClassName>`:

| Controller                     | Purpose                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------- |
| `SonicWholeBodyController`     | SONIC whole-body: locomotion (mode 0), 3-point VR teleop (mode 1), SMPL imitation (mode 2) |
| `GrootLocomotionController`    | GR00T locomotion policy                                                               |
| `HolosomaLocomotionController` | Holosoma locomotion policy                                                            |

On startup the controller **interpolates** from the robot's measured pose into the
policy's commanded target over ~3 s (no snap), and on disconnect (Ctrl-C) it performs a
**graceful damped settle** — holding pose while ramping stiffness to zero over
`--robot.graceful_stop_s` (default 1.5 s) instead of going instantly limp. Both apply in
every mode.

## Requirements

- `onnxruntime` (CPU) **or** `onnxruntime-gpu` (recommended — SONIC runs three ONNX
  sessions and is much smoother on GPU). Install the CUDA build that matches your
  driver (e.g. `onnxruntime-gpu==1.26.0` for a CUDA-12.x driver). Verify with:
  ```bash
  python -c "import onnxruntime as ort; print(ort.get_available_providers())"
  # expect CUDAExecutionProvider in the list for GPU
  ```
- `mujoco` for simulation (`is_simulation=True`).
- `pyzmq` only if you use the live SMPL stream (pico headset).
- The SONIC ONNX models are downloaded automatically from the `nvidia/GEAR-SONIC` Hub repo.

## Running

**Replay an SMPL dataset (motion imitation):**

```bash
lerobot-replay \
  --robot.type=unitree_g1 --robot.controller=SonicWholeBodyController \
  --dataset.repo_id=<user>/<smpl_dataset> --dataset.episode=0
```

**Keyboard teleop** (drives locomotion via the native keyboard teleoperator):

```bash
lerobot-teleoperate \
  --robot.type=unitree_g1 --robot.controller=SonicWholeBodyController \
  --teleop.type=keyboard
```

Controls: `WASD` move · `Q`/`E` turn · `1`–`8` mode · `9`/`0` speed · `-`/`=` height ·
`R` replan · `Space` emergency-stop.

**PICO headset teleop — SMPL whole-body** (mode 2, needs PICO Motion Trackers):

```bash
# 1) publisher (streams rt/smpl from full-body tracking)
python -m lerobot.teleoperators.pico_headset.pico_publisher --fps 50
# 2) controller
lerobot-teleoperate \
  --robot.type=unitree_g1 --robot.controller=SonicWholeBodyController \
  --teleop.type=pico_headset
```

**PICO headset teleop — 3-point VR** (mode 1, head + 2 controllers only, **no trackers**):

```bash
# 1) publisher (head + controllers -> 3-point targets + stick locomotion)
python -m lerobot.teleoperators.pico_headset.pico_publisher --fps 50 --headset-source devices
# 2) controller
lerobot-teleoperate \
  --robot.type=unitree_g1 --robot.controller=SonicWholeBodyController \
  --teleop.type=pico_headset --teleop.mode=vr3
```

3-point controls: left stick move · right stick X turn · right stick Y height ·
`A`+`B` / `X`+`Y` cycle locomotion mode (walk/run/squat/kneel/…) · hands+head track the
upper body. **Calibration**: stand in a neutral rest pose and press `A`+`B`+`X`+`Y` — the
publisher status line flips from `UNCALIBRATED` to `calibrated`. This maps your rest pose
onto the G1's neutral stance and is required before the hands track well; the SMPL
(mode 2) path is self-calibrating and needs no such step.

Both require the XRoboToolkit stack — see below.

## PICO headset / XRoboToolkit install

Live full-body teleop needs the **XRoboToolkit** system (a PC Service on your
workstation + a PICO app on the headset) and its Python binding, `xrobotoolkit_sdk`.
The full hardware + software walkthrough lives in the SONIC repo:
[`docs/source/getting_started/vr_teleop_setup.md`](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html).

Summary:

1. **PC Service** (workstation) — install and run it before connecting the headset.
   - Ubuntu 22.04 / 24.04 (x86_64): prebuilt `.deb` from the
     [XRoboToolkit-PC-Service releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases).
   - Jetson (aarch64): the arm64 `.deb`.
   - Windows (x64): the Windows PC Service build.
2. **PICO app** — install `XRoboToolkit-PICO-*.apk` on the headset (see the guide),
   enable Developer Mode. For **SMPL whole-body** (mode 2) you also need the PICO Motion
   Trackers paired/calibrated and "Full body" enabled; for **3-point** (mode 1,
   `--headset-source devices`) only Head + Controller + Send are required — no trackers.
3. **`xrobotoolkit_sdk`** — a pybind11/CMake build (not a pip package), from
   [`XRoboToolkit-PC-Service-Pybind`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind):
   - Linux x86_64: `pip install pybind11 cmake` then `bash setup_ubuntu.sh` (or the
     SONIC repo's `install_scripts/install_pico.sh`, which builds everything into a
     `.venv_teleop`).
   - Jetson aarch64: `bash setup_orin.sh` (builds `libPXREARobotSDK.so` from source).
   - Windows x64: `pip install pybind11` then `setup_windows.bat` (needs git + an
     MSVC/CMake toolchain; uses the prebuilt `PXREARobotSDK.dll`/`.lib`).
4. Connect PICO and workstation to the **same Wi-Fi**, open the XRoboToolkit app, enter
   the PC IP, and enable Head/Controller/Send (plus Full-body for SMPL mode 2).

### Platform support

| Platform                    | Live headset teleop | Notes                                       |
| --------------------------- | ------------------- | ------------------------------------------- |
| Linux x86_64                | ✅                  | Guided `install_pico.sh` (SONIC repo)       |
| Linux aarch64 (Jetson Orin) | ✅                  | `setup_orin.sh` builds the native lib       |
| Windows x64                 | ✅ (manual)         | `setup_windows.bat`; no one-shot env script |
| macOS                       | ❌                  | No PC Service / SDK build for Darwin        |

### No hardware required (any platform, incl. macOS/Windows)

The SMPL pipeline can be exercised without a headset or the SDK — the publisher emits
`rt/smpl` frames that the controller consumes exactly as it would from the headset:

```bash
# synthetic motion
python -m lerobot.teleoperators.pico_headset.pico_publisher --fake

# replay a canned SMPL clip
python -m lerobot.teleoperators.pico_headset.pico_publisher --motion-file <clip>.npz
```

## Notes

- SMPL **root motion** into the mode-2 anchor is opt-in (`SonicWholeBodyController(enable_smpl_root=True)`);
  it stays off by default (untested on hardware). When enabled, the per-frame root quat is
  spherically smoothed (`root_smoothing_alpha`, default 0.15) before it reaches the anchor,
  which removes the base-acceleration spikes the raw 30 Hz→50 Hz trajectory used to cause.
- Direct `rt/smpl` subscription without the pico teleoperator is available via
  `SonicWholeBodyController(enable_smpl_stream=True, smpl_host=..., smpl_port=...)`.
- 3-point (mode 1) uses the **headset-yaw frame** as its reference and the `A`+`B`+`X`+`Y`
  calibration to align to the G1 neutral stance. Calibration maps the operator's rest pose
  onto the G1's **standing** (`default_angles`) wrist/neck key-frame poses (position **and**
  orientation) computed by FK — the `default_angles` stand-in for gear_sonic's live
  measured-q recalibration, since the robot holds `default_angles` at calibration time.
  Re-aligning the arms only (preserving the neck level) is available via the calibrator's
  `recalibrate_wrists()`.
- 3-point **locomotion** from the PICO sticks follows gear_sonic's `PlannerLoop` exactly:
  a yaw accumulator on the right stick and **mode-dependent speed curves** on the left
  (slow `0.1+0.5·mag`, run `1.5+3·mag`, walk = planner default). Stick signs replicate
  gear_sonic's `get_controller_axes` usage (forward `+ly`, strafe `-lx`, turn `-rx`); since
  the publisher forwards the same raw SDK axes, this is the correct convention by construction.
- Startup interpolation and the graceful-stop settle are mode-agnostic; set
  `--robot.graceful_stop_s=0` to restore the old instant zero-torque on disconnect.
