# Unitree G1 — SONIC whole-body teleop (PICO headset)

Drive the G1 with the **SONIC** policy from live **full-body SMPL** streamed off a PICO
headset — running entirely through `lerobot-teleoperate` (no C++ deploy stack).

## Architecture

Two processes talk over one ZMQ channel (`rt/smpl`, TCP port `5560`):

```
PICO headset ──► XRoboToolkit PC Service ──► pico_manager_thread_server.py ──(rt/smpl)──► lerobot-teleoperate
   (gear_sonic env, publisher)                                                              (lerobot env, subscriber)
                                                                                            └─ SonicWholeBodyController (encode_mode=2)
```

- **Publisher** (`gear_sonic/scripts/pico_manager_thread_server.py`): reads PICO body
  tracking, converts to canonical SMPL joints, publishes each frame on `rt/smpl`.
  Lives in `gear_sonic` because it needs the XRoboToolkit SDK.
- **Subscriber** (this repo): the `pico_headset` teleoperator (or the
  `SONIC_SMPL_STREAM` fallback in `SonicWholeBodyController`) consumes `rt/smpl` and
  feeds the SONIC encoder's 10-frame (720-vec) whole-body reference window.

The subscriber never launches the publisher — you start it yourself. Until real
frames arrive, the robot stays in safe locomotion mode; it switches to whole-body
tracking automatically once frames flow.

## Install

**lerobot side (subscriber):**
```bash
# in your lerobot env
pip install -e ".[unitree_g1]"   # provides pyzmq, unitree_sdk2py, etc.
```

**publisher side (gear_sonic):** see the repo root `docs/TELEOP_QUICKSTART.md`
("One-time setup"). In short: install the gear_sonic teleop env, the
[XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases),
and the [PICO APK](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases).
On the PICO app set: PC Service IP = laptop Wi-Fi IP, Motion Tracker = **Full body**,
Data/Control = **Send**.

> The publisher and subscriber can run on the same laptop (use `127.0.0.1`) or on
> separate machines (point `--teleop.smpl_host` at the publisher's IP).

## Run

**1. Start the publisher** (gear_sonic env). Any `--manager` run publishes `rt/smpl`
on port 5560; the `--vis_*` flags only add the calibration/preview windows:
```bash
cd ~/Documents/sonic
python gear_sonic/scripts/pico_manager_thread_server.py --manager --vis_vr3pt --vis_smpl
# look for: [Manager] ZMQ 'rt/smpl' socket bound to port 5560
```

**2. Start teleoperate** (lerobot env).

Simulation:
```bash
lerobot-teleoperate \
  --robot.type=unitree_g1 --robot.is_simulation=true \
  --robot.controller=SonicWholeBodyController \
  --teleop.type=pico_headset --teleop.smpl_host=127.0.0.1 --teleop.smpl_port=5560 \
  --fps=50
```

Real robot:
```bash
lerobot-teleoperate \
  --robot.type=unitree_g1 --robot.is_simulation=false --robot.robot_ip=<ROBOT_IP> \
  --robot.controller=SonicWholeBodyController \
  --teleop.type=pico_headset --teleop.smpl_host=127.0.0.1 --teleop.smpl_port=5560 \
  --fps=50
```

> Skip `--display_data=true` with `pico_headset`: it would print all 720 `smpl.*`
> values every tick.

### Fallback: no teleoperator

To test the whole-body controller before wiring a teleop (e.g. to keep the
`unitree_g1` remote for e-stop), let the controller subscribe to `rt/smpl` directly:
```bash
SONIC_SMPL_STREAM=1 \
  lerobot-teleoperate \
    --robot.type=unitree_g1 --robot.is_simulation=false --robot.robot_ip=<ROBOT_IP> \
    --robot.controller=SonicWholeBodyController \
    --teleop.type=unitree_g1 --fps=50
# override endpoint with SONIC_SMPL_HOST / SONIC_SMPL_PORT
```

## Safety (real robot)

- The `pico_headset` teleop is **SMPL-only** — it does not pass a software e-stop.
  Keep the **hardware remote** in hand.
- Start in a neutral standing pose: the robot flips to whole-body tracking the
  instant real frames arrive.
- If frames drop, the controller **holds the last pose** (it does not snap to zero),
  but it won't auto-return to locomotion — exit via the hardware remote.
- Clear a ~3 m safety zone; move slowly at first.

## Standalone (no teleoperate)

`sonic.py` can consume the same stream directly for quick tests:
```bash
python examples/unitree_g1/sonic.py --smpl-stream --smpl-host 127.0.0.1 --smpl-port 5560
```

`smpl_stream.py` is a self-test that just prints window stats:
```bash
python examples/unitree_g1/smpl_stream.py --smpl-host 127.0.0.1 --smpl-port 5560
```
