# Unitree G1 — SONIC decoder whole-body control

This package runs NVIDIA's **SONIC** decoder on the Unitree G1, in MuJoCo simulation or
on real hardware, driven by a **64-D latent motion token**. It is a pure-Python/ONNX
reimplementation of the decode half of the SONIC deploy stack (no `gear_sonic`/torch
dependency): the decoder maps a 64-D latent token + proprioception history into 50 Hz
joint-position targets for the robot's PD controller. The encoder is bypassed — a policy
(e.g. `nepyope/sonic_walk`) emits the token directly.

## Controllers

Selected with `--robot.controller=<ClassName>`:

| Controller                     | Purpose                                             |
| ------------------------------ | --------------------------------------------------- |
| `SonicWholeBodyController`     | SONIC decoder driven by a 64-D latent motion token  |
| `GrootLocomotionController`    | GR00T locomotion policy                             |
| `HolosomaLocomotionController` | Holosoma locomotion policy                          |

The rest of this document covers the SONIC token path.

Each tick the `SonicWholeBodyController` takes a 64-D latent token
(`motion_token.0.pos … motion_token.63.pos`) and decodes it directly (encoder bypassed).
Before the first token arrives it holds a captured **neutral token** (a stable standing
pose), then holds the last token received between ticks (the ~30 Hz token stream vs. the
~50 Hz control loop). On startup the controller **interpolates** from the robot's measured
pose into the policy's commanded target over ~3 s (no snap).

## Requirements

- `onnxruntime` (CPU) **or** `onnxruntime-gpu` (recommended). Verify with:
  ```bash
  python -c "import onnxruntime as ort; print(ort.get_available_providers())"
  ```
- `mujoco` for simulation (`is_simulation=True`).
- The SONIC encoder/decoder ONNX models download automatically from the
  `nvidia/GEAR-SONIC` Hub repo.

## Architecture: controller always runs onboard

The controller runs **on the robot**, never on the laptop. The laptop is a thin client:
it negotiates the controller with `run_g1_server` (handshake), then PUSHes the 64-D token
and reads back the `observation.state` echo + camera frames over ZMQ.

## Running a rollout (real robot)

On the robot — host the SONIC decoder + camera onboard:

```bash
python -m lerobot.robots.unitree_g1.run_g1_server --handshake \
    --cameras "ego_view:/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.3-video-index0:640x480"
```

On the laptop — `lerobot-rollout` drives the thin client:

```bash
lerobot-rollout \
  --policy.path=nepyope/sonic_walk \
  --robot.type=unitree_g1 \
  --robot.is_simulation=false --robot.onboard=false \
  --robot.robot_ip=<ROBOT_IP> \
  --robot.controller=SonicWholeBodyController --robot.sonic_token_action=true \
  --robot.cameras='{ego_view: {type: zmq, server_address: <ROBOT_IP>, port: 5555, camera_name: ego_view, width: 640, height: 480, fps: 30}}' \
  --task="walk back and forth" --device=cuda
```

## Training a token policy (no pi05 code patch)

The SONIC token interface needs **no modeling changes** to pi05. A 64-D token action is
handled entirely by config: pi05 builds its action projections straight from config
(`action_in_proj = nn.Linear(max_action_dim, …)`, `action_out_proj = nn.Linear(…,
max_action_dim)`), pads the action to `max_action_dim`, then slices back to the dataset's
action dim. Set both dims to 64 and the pad/slice is a no-op, so the full 64-D token is
supervised.

Requirements:

1. The dataset carries a 64-D `action` and 64-D `observation.state` (the motion tokens).
2. Pass the dims to `lerobot-train`:

```bash
lerobot-train \
  --dataset.repo_id=nepyope/walk_back_and_forth \
  --policy.type=pi05 \
  --policy.max_action_dim=64 \
  --policy.max_state_dim=64 \
  --policy.chunk_size=50 --policy.n_action_steps=50
```

`nepyope/sonic_walk` was trained exactly this way (`config.json`: `max_action_dim=64`,
`max_state_dim=64`, `output_features.action.shape=[64]`). Same stock code path for train
and inference — the checkpoint's 64-wide `Linear`s load with unmodified pi05.

## Observation / action interface (token mode)

With `--robot.sonic_token_action=true` the robot advertises:

- action: 64-D `motion_token.{i}.pos` (the decoder consumes it directly),
- `observation.state`: 64-D `motion_token_state.{i}.pos` (the last commanded token,
  echoed so a token-output VLA closes the loop on its own previous token),

plus the ego camera image. The controller always runs onboard (or in sim); it is never
built on the laptop client.
