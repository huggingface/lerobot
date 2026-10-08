# Hardware configs for Carl's XLeRobot

Configs for the two SO-101 arms, their leaders, and the head RealSense.
Run them with stock lerobot CLIs:

```bash
lerobot-teleoperate --config_path=config/left-arm.yaml
lerobot-teleoperate --config_path=config/right-arm.yaml
lerobot-teleoperate --config_path=config/bi-arms.yaml   # both, one machine
```

`cart.yaml` uses `--robot.type=xlerobot`, which comes from the XLeRobot
plugin rather than from this repo — see below.

## Which machine runs what

```
         CART                                 OPERATOR STATION
  elroy — Jetson Orin Nano                 rosie — Raspberry Pi 5
  ───────────────────────               ─────────────────────
  follower arms, wrist cams,     wifi     leader arms, gamepad,
  head RealSense, base            <──>    the operator's screen
  cart-host.sh                            operator-teleop.sh
```

The GPU goes where the sensors and motors are, because that is where a
policy has to run: at full camera rate, with no network between seeing and
acting. Teleoperation is the only part that genuinely has two ends, so it is
the only part that crosses the wifi — and what crosses is small and
loss-tolerant (joint targets out, a downscaled view back).

Recording is done **on the cart** with the leaders temporarily plugged in
there (`cart-record.sh`), not across the link. Two reasons: the ZMQ host
conflates observations, so recording over it drops frames silently; and the
dataset should be captured through the same pipeline the policy will see at
inference, or you get a train/serve skew you cannot observe.

| file | runs on |
| --- | --- |
| `cart-host.sh` + `cart-host.yaml` | cart |
| `cart-record.sh` (uses `bi-arms.yaml`) | cart, leaders plugged in |
| `cart.yaml` | cart, everything local |
| `operator-teleop.sh` + `operator-client.yaml` | operator station |
| `left-arm.yaml`, `right-arm.yaml`, `bi-arms.yaml` | one machine, direct |

Files are named by role, not by hostname: the cart computer has already
swapped once.

### Prerequisites this arrangement adds

- **The Orin needs wifi on the cart.** Some Orin Nano dev kits ship without
  an M.2 wifi card. Check before mounting; a USB dongle also works.
- **The Pi needs a desktop session** to show the live view with rerun. Over
  SSH, `operator-teleop.sh --display=web` serves Foxglove instead.
- **The udev rules and calibration move to the Orin.** `/dev/cam_left` and
  `/dev/cam_right` come from `/etc/udev/rules.d/99-cameras.rules`, keyed on
  `ENV{ID_PATH}` — the port paths differ on the Orin, so regenerate them
  there. Copy
  `~/.cache/huggingface/lerobot/calibration/robots/` across as well.

## What this fork changes in lerobot

Deliberately almost nothing. Only `pyproject.toml`, so that merges from
upstream stay cheap:

- torch is routed to the CUDA wheel index only on `x86_64` linux. Upstream
  routes all linux there, and that index has no aarch64 wheels, so a Pi or
  Jetson cannot resolve torch at all.
- `pyrealsense2` may go to >=2.57.7 on ARM linux. Upstream pins <2.57.0, but
  the first aarch64 **cp312** wheels appear in 2.57.7, and lerobot requires
  Python >=3.12.
- `ipython` in dependencies, `[tool.uv] package = true` and
  `python-preference = "managed"`.

Expect `pyproject.toml` to conflict on upstream merges. Nothing else should.

## XLeRobot code is NOT in this repo

It used to be copied into `src/lerobot/robots/xlerobot/`,
`src/lerobot/model/SO101Robot.py` and `examples/`. That was removed — the
XLeRobot project now packages its robots as lerobot plugins, which is both
less work to maintain and free of merge conflicts.

lerobot auto-imports any installed package named `lerobot_robot_*`,
`lerobot_teleoperator_*`, etc. (`register_third_party_plugins()` in
`lerobot.utils.import_utils`), so an installed plugin makes its
`--robot.type=...` available on every standard CLI.

**Do not install anything from this repo directly.** The XLeRobot repo is a
uv workspace root that declares this fork plus its plugins, so one command
there builds the whole environment:

```bash
cd ../XLeRobot
uv sync                           # this fork (editable) + model + xlerobot + 2wheels
uv sync --extra mecanum --extra vr
uv sync --extra dataset           # adds torchcodec, needed to record
```

That assumes the two repos are **siblings**:

```
<somewhere>/
  lerobot/     <- this repo
  XLeRobot/    <- workspace root; its pyproject points at ../lerobot
```

Why that direction: the plugins all depend on `lerobot`, so declaring them
here would invert the dependency. Worse, this fork is itself named `lerobot`,
so uv would resolve the plugins' `lerobot` from PyPI and install upstream
lerobot beside the fork, both providing a `lerobot` module. The workspace
root maps `lerobot` to `../lerobot` for every member, which removes the
ambiguity by construction.

`uv sync` from *this* directory still works for lerobot alone, but it will
prune the plugins, and `--robot.type=xlerobot` then stops resolving. Sync
from XLeRobot.

Examples live in the XLeRobot repo under `software/examples/`.

## Python version

`pyrealsense2` publishes aarch64 wheels for cp39/cp310/cp312 only, and no
sdist. lerobot requires >=3.12. So on ARM, **Python 3.12 exactly**:

```bash
uv python pin 3.12
uv sync --extra feetech --extra intelrealsense --extra viz
```

Leave out `--extra dataset` until you need to record; it pulls `torchcodec`,
which on aarch64 needs >=0.11.0 and torch>=2.11.
