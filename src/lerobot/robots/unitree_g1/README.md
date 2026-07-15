# Unitree G1 — SONIC whole-body control

This package runs NVIDIA's **SONIC** whole-body controller (and the GR00T/Holosoma
locomotion controllers) on the Unitree G1, in MuJoCo simulation or on real hardware.
SONIC turns a high-level movement intent — or a streamed **SMPL** whole-body pose — into
50 Hz joint-position targets. It is a pure-Python/ONNX reimplementation of the SONIC
deploy stack (no `gear_sonic`/torch dependency).

## Controllers

Selected with `--robot.controller=<ClassName>`:

| Controller | Purpose |
|---|---|
| `SonicWholeBodyController` | SONIC whole-body: locomotion, keyboard, and SMPL imitation (mode 2) |
| `GrootLocomotionController` | GR00T locomotion policy |
| `HolosomaLocomotionController` | Holosoma locomotion policy |

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

**PICO headset teleop** (live SMPL whole-body):
```bash
lerobot-teleoperate \
  --robot.type=unitree_g1 --robot.controller=SonicWholeBodyController \
  --teleop.type=pico_headset
```
This requires the XRoboToolkit stack — see below.

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
   enable Developer Mode, pair/calibrate the ankle motion trackers.
3. **`xrobotoolkit_sdk`** — a pybind11/CMake build (not a pip package), from
   [`XRoboToolkit-PC-Service-Pybind`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind):
   - Linux x86_64: `pip install pybind11 cmake` then `bash setup_ubuntu.sh` (or the
     SONIC repo's `install_scripts/install_pico.sh`, which builds everything into a
     `.venv_teleop`).
   - Jetson aarch64: `bash setup_orin.sh` (builds `libPXREARobotSDK.so` from source).
   - Windows x64: `pip install pybind11` then `setup_windows.bat` (needs git + an
     MSVC/CMake toolchain; uses the prebuilt `PXREARobotSDK.dll`/`.lib`).
4. Connect PICO and workstation to the **same Wi-Fi**, open the XRoboToolkit app, enter
   the PC IP, and enable Head/Controller/Full-body/Send.

### Platform support

| Platform | Live headset teleop | Notes |
|---|---|---|
| Linux x86_64 | ✅ | Guided `install_pico.sh` (SONIC repo) |
| Linux aarch64 (Jetson Orin) | ✅ | `setup_orin.sh` builds the native lib |
| Windows x64 | ✅ (manual) | `setup_windows.bat`; no one-shot env script |
| macOS | ❌ | No PC Service / SDK build for Darwin |

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
  it is off by default because an unsmoothed 30 Hz→50 Hz root trajectory can spike the
  base acceleration.
- Direct `rt/smpl` subscription without the pico teleoperator is available via
  `SonicWholeBodyController(enable_smpl_stream=True, smpl_host=..., smpl_port=...)`.
