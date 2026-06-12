# Hans Robot S30 – LeRobot Integration

This directory contains example scripts for using the
[Hans Robot S30](https://www.hansrobot.com/) 6-DOF industrial arm with
🤗 LeRobot.

## Hardware overview

| Property | Value |
|---|---|
| DoF | 6 (J1 – J6) |
| Communication | TCP socket (port 10003) + XML-RPC (port 20000) |
| Protocol | Hans CPS (Controller Programming System) |
| Default controller IP | 192.168.115.11 |
| Joint units | Degrees |
| Teaching mode | Zero-force (gravity compensation) |

## Prerequisites

1. **Network connection** – connect your PC to the same subnet as the Hans
   controller (default: `192.168.115.x`).  Verify connectivity:
   ```bash
   ping 192.168.115.11
   ```

2. **LeRobot installation** – follow the
   [Installation Guide](https://huggingface.co/docs/lerobot/installation).

3. **Optional cameras** – any OpenCV-compatible USB camera(s).

## Robot adapter

The adapter lives in `src/lerobot/robots/hans_s30/` and exposes the standard
`Robot` interface:

```python
from lerobot.robots.hans_s30 import HansS30, HansS30RobotConfig

config = HansS30RobotConfig(ip="192.168.115.11", id="my_hans_s30")
robot = HansS30(config)
robot.connect()

obs = robot.get_observation()   # {"joint_1.pos": ..., ..., "joint_6.pos": ...}
robot.send_action(obs)          # echo current position as action
robot.disconnect()
```

### Configuration options

| Parameter | Default | Description |
|---|---|---|
| `ip` | `"192.168.115.11"` | Controller IP address |
| `port` | `10003` | CPS TCP port |
| `box_id` | `0` | Controller box ID |
| `robot_id` | `0` | Robot ID |
| `velocity` | `50.0` | Default joint velocity (deg/s) |
| `acc` | `50.0` | Default joint acceleration (deg/s²) |
| `speed_override` | `0.5` | Global speed ratio [0.01, 1.0] |
| `tcp_name` | `"TCP"` | Tool-frame name on controller |
| `ucs_name` | `"Base"` | User-frame name on controller |
| `electrify_wait_s` | `15.0` | Wait after power-on (s) |
| `controller_init_wait_s` | `20.0` | Wait after EtherCAT init (s) |
| `max_relative_target` | `None` | Per-joint displacement clamp (deg) |
| `cameras` | `{}` | Optional `{name: OpenCVCameraConfig}` dict |

## Scripts

### `teleoperate.py` – zero-force teaching

Switches the arm into gravity-compensation mode so the operator can manually
guide it while joint positions and camera frames are streamed to Rerun.

```bash
python examples/hans_s30/teleoperate.py
```

### `record.py` – collect a dataset

Records demonstrations via zero-force teaching and saves them as a
`LeRobotDataset`.  Edit the constants at the top of the file:

```python
ROBOT_IP        = "192.168.115.11"
HF_REPO_ID      = "<hf_username>/<dataset_repo_id>"
TASK_DESCRIPTION = "Pick up the red block and place it in the bin"
NUM_EPISODES     = 10
```

```bash
python examples/hans_s30/record.py
```

**Keyboard shortcuts during recording:**

| Key | Action |
|---|---|
| `→` / Space | Exit current episode early and save |
| `←` | Discard and re-record current episode |
| `Esc` | Stop recording |

### `replay.py` – replay a recorded episode

Plays back joint-position actions from a saved dataset.

```bash
python examples/hans_s30/replay.py
```

### `evaluate.py` – run a trained policy

Loads a policy from the Hugging Face Hub and runs it in closed loop.

```bash
python examples/hans_s30/evaluate.py
```

## Safety notes

- Set `speed_override` ≤ 0.3 and `max_relative_target` ≤ 10.0 (degrees)
  during initial testing.
- Ensure the work envelope is clear before enabling the robot.
- The controller's hardware emergency-stop button remains active at all times.
- `electrify_wait_s` and `controller_init_wait_s` must be long enough for your
  power supply; increase them if `GrpEnable` fails.

## Contributing

Please read the [LeRobot CONTRIBUTING guide](../../CONTRIBUTING.md) before
opening a pull request.  Run checks locally before submitting:

```bash
pre-commit run --all-files
pytest -sv tests/
```
