# WebRTC remote control of an SO-100

Drive a real SO-100 that's plugged into a **robot**, from a **cloud** process, over
WebRTC. The cloud runs the control logic; the arm + camera stay on the user's
machine. The cloud `WebRTCProxyRobot` is a drop-in lerobot `Robot` — `get_observation()`
returns the remote arm's joints + camera and `send_action()` moves the remote motors,
so any standard `send_action`/`get_observation` control loop drives it. This example
keeps it self-contained: a tiny inline keyboard jog (no extra teleoperator dependency)
that proves the link end-to-end.

Implementation: [`src/lerobot/robots/webrtc_proxy`](../../src/lerobot/robots/webrtc_proxy/).

```
 robot (arm + camera)                    Cloud
 robot_daemon_so100.py                   signaling relay  +  cloud_teleop_so100.py
   SO100Follower ──┐                        │                 WebRTCProxyRobot
   (bus + camera)  ├── WebRTC ──────────────┴──────────────── + keyboard jog loop
   watchdog (P0) ──┘   media track (camera) + DataChannels (joints/action/control)
```

Needs the `webrtc` extra: `uv pip install --native-tls 'aiortc>=1.9.0,<2.0.0' 'aiohttp>=3.9.0,<4.0.0'`
(or `uv sync --extra webrtc`), plus `lerobot[hardware]` for the SO-100.

## 1. Onboarding (once, on the robot) — find the port + camera

```bash
uv run lerobot-find-port       # unplug/replug the bus -> /dev/tty.usbmodem...  -> PORT
uv run lerobot-find-cameras    # saves preview images   -> the camera index     -> CAMERA_INDEX
```
Put both into `robot_daemon_so100.py`. (These can also be driven from the cloud over the
control plane — see `python -m lerobot.robots.webrtc_proxy.sim_remote --rpc find_port --real-devices`.)

## 2. Start the three pieces

```bash
# (cloud) signaling relay — pairs the daemon with the controller
uv run python -m lerobot.robots.webrtc_proxy.signaling_server --port 8765

# (robot) serve the real arm; safes the arm if the cloud drops
uv run python examples/webrtc_remote_so100/robot_daemon_so100.py

# (cloud) drive it — web UI with a LIVE camera view (default)
uv run python examples/webrtc_remote_so100/cloud_teleop_so100.py
# -> open http://localhost:8088  (live remote camera + per-joint jog buttons)

# ...or a terminal keyboard jog instead:
uv run python examples/webrtc_remote_so100/cloud_teleop_so100.py --mode console
# -> type (then Enter):  1..6 select joint   +/- jog selected   q quit
```

The web panel (`panel.html`, served by the stdlib HTTP server) shows the remote
camera as a live MJPEG stream — pressing a jog button and watching the remote view
move is the end-to-end proof the WebRTC link works.

Same machine for a quick test (`ws://127.0.0.1:8765/ws`, `ICE_SERVERS=[]`). Real
cloud↔robot across the public internet: point `SIGNALING_URL` at the relay's public
address and add STUN/TURN urls to `ICE_SERVERS` (coturn) for NAT traversal.

## What this demonstrates

- **Drop-in Robot.** `cloud_teleop_so100.py` is a plain `send_action` /
  `get_observation` loop over `WebRTCProxyRobot` — no WebRTC-specific control code, and
  no special teleoperator. Swap the inline jog for `lerobot-record` to record a remote
  dataset, or a policy to run inference in the cloud.
- **IDs stay robot-local.** The cloud config declares only the logical schema (6 motors
  + camera `front` at a resolution); the serial port and camera index live on the robot.
- **P0 safety.** If the action stream stalls (network drop, cloud crash), the robot-side
  watchdog cuts motor torque so the arm goes limp instead of holding/straining.

## Not covered here (later milestones)

- Public-net NAT traversal (STUN/TURN/coturn) and K8s media deployment — M4.
- Multi-camera (one media track each) — currently one camera (`front`).
