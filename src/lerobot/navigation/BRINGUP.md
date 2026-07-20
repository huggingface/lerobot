# dog-nav on a real Unitree Go2 — bring-up guide

The synthetic scene (`--dry-run`) exists only to test the logic without a
robot. To run for real you need the dog, the GPU host, and the steps
below. Bring it up **in stages** — never start with autonomous motion on
untested hardware.

## 0. Prerequisites

- Unitree Go2 **EDU** (SDK access; the consumer Go2 can't be commanded).
- A GPU host (your 5090) on the **same network as the dog**. Over
  Ethernet the dog is on `192.168.123.x`; find your interface with
  `ip link` (e.g. `enp2s0`).
- A remote/controller in hand for a hardware e-stop at all times.

## 1. Get the branch onto the 5090

The branch `feat/unitree-go2` is local (not pushed to upstream). Either:

**Option A — your fork:**
```bash
# on the mac, one time:
git remote add fork git@github.com:<you>/lerobot.git
git push fork feat/unitree-go2
# on the 5090:
git clone git@github.com:<you>/lerobot.git && cd lerobot
git checkout feat/unitree-go2
```

**Option B — git bundle (no remote needed):**
```bash
# on the mac:
git bundle create go2-nav.bundle origin/main..feat/unitree-go2
# copy go2-nav.bundle to the 5090, then:
git clone https://github.com/huggingface/lerobot.git && cd lerobot
git fetch ../go2-nav.bundle feat/unitree-go2:feat/unitree-go2
git checkout feat/unitree-go2
```

## 2. Environment on the 5090

```bash
uv venv --python 3.12 .venv
uv pip install -e .                         # lerobot core (torch, etc.)
uv pip install transformers                 # SigLIP2
uv pip install unitree_sdk2py               # DDS to the dog (Linux only)
# LingBot-Map (geometry) — source install:
pip install -e 'git+https://github.com/robbyant/lingbot-map#egg=lingbot-map'
```

Smoke-test the code path with no dog:
```bash
.venv/bin/python -m lerobot.navigation.dog_cli --dry-run --command "go to the couch"
```

## 3. Stage 1 — verify DDS + sensors (NO motion)

Confirm the host talks to the dog and reads odometry + camera before
anything moves:
```python
from lerobot.robots.unitree_go2 import UnitreeGo2, UnitreeGo2Config
r = UnitreeGo2(UnitreeGo2Config(network_interface="enp2s0", stand_on_connect=False))
r.connect()
obs = r.get_observation()
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in obs.items()})
r.disconnect()
```
You want a real `front` image `(720, 1280, 3)` and non-garbage
`x.pos/y.pos/theta.pos`. If `theta.pos` doesn't change sign the way you
expect when you turn the dog by hand, tell me — the odometry sign
conventions may need a tweak for your firmware.

## 4. Stage 2 — teleop (low speed, hand on e-stop)

```bash
lerobot-teleoperate --robot.type=unitree_go2 \
    --robot.network_interface=enp2s0 --teleop.type=gamepad
```
Confirm forward/left/turn go the right way. This validates
`send_action`/`SportClient.Move` before the nav loop drives.

## 5. Stage 3 — MAP-ONLY (still no autonomous motion)

Build the map by teleoperating the dog around while the models run.
Query where things are; the dog never drives itself:
```bash
.venv/bin/python -m lerobot.navigation.dog_cli --map-only \
    --network-interface enp2s0 --device cuda --camera-hfov-deg 90
# teleop the dog around the room, then type object names:
#   couch      -> "couch is at (x, y, z) ..."  or  "not found yet"
```
Tune `--camera-hfov-deg` to your Go2 front camera so free-space carving
is correct (a wrong value only hurts dynamic removal, not the map).

## 6. Stage 4 — autonomous nav (open space, low speed, e-stop ready)

Only after 1–3 look right. Start in a clear area:
```bash
.venv/bin/python -m lerobot.navigation.dog_cli --live \
    --network-interface enp2s0 --device cuda \
    --max-lin-speed 0.3 --max-yaw-rate 0.6
# empty line -> one exploration step; type an object -> navigate to it.
```
`SafeBaseController` clamps speed, refuses moves into obstacle cells, and
latches an e-stop if keyframes go stale (>2 s). Ctrl-C stops the base.

## Known things to expect / tune on first hardware contact

- **Odometry sign conventions** (`position[0/1]`, `imu_state.rpy[2]`):
  verified in sim, not yet against live firmware — check in Stage 1.
- **Camera FOV / focal**: set `--camera-hfov-deg` from your camera.
- **Gait bob**: pose is planarized (yaw only); pitch/roll wobble is
  ignored for now. Fine at low speed; a full-SE(3) camera pose is the
  refinement if the map smears vertically.
- **Keyframe rate**: SAM2 isn't in this path; the per-tick cost is
  LingBot-Map + SigLIP2 on the 5090 (~tens of ms each). If ticks lag,
  drop camera resolution.
