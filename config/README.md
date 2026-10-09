# Hardware configs for Carl's XLeRobot

| doc | when |
|---|---|
| **`RUNNING.md`** | **every session — what to start, on which machine, in what order** |
| `ORIN-SETUP.md` | one-time bring-up of a cart machine |
| this file | what the fork changes and why |

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
  ──────────────────────               ─────────────────────
  follower arms                           leader arms
  wrist cams + head RealSense             gamepad
  the base                                the rerun viewer
  the dataset                     actions
  the policy, later               <──────
                                  ──────>
                                   video
```

The GPU goes where the sensors and motors are, because that is where a
policy has to run: at full camera rate, with no network between seeing and
acting. So the Orin owns the robot, the cameras and the dataset, and the
only thing it does not have is the operator.

That makes the **leaders** the remote device, not the robot — the mirror
image of lekiwi. Two things cross the wifi and both tolerate loss: leader
actions one way, the operator's view the other. Camera frames go from USB
straight to disk without touching the network, so nothing can conflate or
drop them.

### The normal path

```
rosie   rerun --port 9876                        # viewer, first
rosie   ./config/operator-leader-host.sh         # leaders + gamepad
elroy   ./config/cart-teleop.sh --watch          # drive, record nothing
elroy   ./config/cart-record.sh --watch me/ds "task"
```

Run `cart-record.sh` from an SSH session opened *from the Pi*: lerobot's
episode keys (right arrow = end episode, left = re-record, escape = stop)
are read by the process on the cart, and it falls back to a terminal
listener when pynput cannot capture, so an SSH TTY puts them under the
operator's hands.

| file | runs on |
| --- | --- |
| `operator-leader-host.{sh,yaml}` | operator station |
| `cart-teleop.sh`, `cart-record.sh` | cart |
| `cart-remote.yaml` | cart — arms + head + base, once those motors exist |
| `_video.sh` | sourced by the cart scripts; `--watch` lives here |
| `left-arm.yaml`, `right-arm.yaml`, `bi-arms.yaml` | one machine, everything local |

### The two settings that fail silently

The teleoperator and the robot must agree on arm key naming and on which
joints exist. Nothing checks this at startup and a mismatch does not
raise — the link comes up, the loop runs at the right rate, and the arms do
not move.

`operator-leader-host.yaml` must have `remap_arm_prefix: true` (XLerobot
filters with `startswith("left_arm_")`) and `emit_head`/`emit_base` true.
`operator-leader-host.sh` echoes all four on startup for this reason, and
the leader host reports **17 action keys** when they are right — 12 arm,
2 head, 3 base.

### Prerequisites

- **udev rules and calibration live on the Orin now.** `/dev/cam_left` and
  `/dev/cam_right` are keyed on `ENV{ID_PATH}`, and the port paths differ on
  the Orin — read the real ones with
  `udevadm info -q property /dev/video0 | grep ID_PATH`. Copy
  `~/.cache/huggingface/lerobot/calibration/` across as well; the leaders'
  half has to be on the Pi and the followers' half on the Orin.
- **JetPack torch on the Orin.** The fork routes ARM to PyPI's CPU wheels,
  which is wrong for the machine that will run the policy.
- **The rerun viewer on the Pi** for `--watch`. Check `rerun --help` for
  your version's listen flag.

## What this fork changes in lerobot

Deliberately almost nothing. Only `pyproject.toml`, so that merges from
upstream stay cheap:

- torch is routed to the CUDA wheel index only on `x86_64` linux. Upstream
  routes all linux there, and `cu128` has no aarch64 wheels, so a Pi or
  Jetson cannot resolve torch at all.

  **This entry is live even when you sync from the XLeRobot workspace.**
  uv reads it as part of that resolution, so it is not dormant — adding a
  second index for `torch` without a disjoint marker makes uv refuse to
  resolve anything:

  ```
  Requirements contain conflicting indexes for package `torch` in split
  `... platform_machine == 'x86_64' and sys_platform == 'linux'`
  ```

  GPU torch for the Jetson is the `jetson` extra in the workspace
  (`uv sync --extra jetson`), scoped to `aarch64` for exactly that reason,
  routing to `cu132`. Note also that newer CUDA indexes **do** publish
  aarch64 wheels, so the premise above is specific to `cu128` rather than a
  permanent fact about ARM.
- `pyrealsense2` may go to >=2.57.7 on ARM linux. Upstream pins <2.57.0, but
  the first aarch64 **cp312** wheels appear in 2.57.7, and lerobot requires
  Python >=3.12.
- **ARM ceilings raised one minor version** for `torch` (<2.13),
  `torchvision` (<0.28) and `torchcodec` (<0.13). The cu132 index starts at
  torch 2.12.0 / torchvision 0.27.0 for aarch64 cp312 and publishes nothing
  older, so upstream's caps exclude every GPU build that exists for the
  Jetson — by exactly one notch each. torchcodec follows because upstream's
  own note says 0.12 needs torch 2.12, and `--extra dataset` cannot resolve
  on ARM otherwise.

  These **widen rather than shift**: 2.7 through 2.12 all stay legal on ARM,
  so rosie resolving from PyPI is unaffected. x86_64 keeps upstream's pins
  verbatim. The cost is that the Jetson now runs torch/torchcodec pairings
  lerobot has not tested — if dataset writing misbehaves, this is the first
  thing to suspect.
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

## Serial ports that exist but will not open

`lerobot-find-port` says the device is there; everything else says
"Could not connect on port ... Make sure you are using the correct port."
The port is almost always correct. Two things cause this on Ubuntu, and a
fresh machine usually has both:

**1. You are not in `dialout`.** `/dev/ttyACM*` is `root:dialout` mode 660.

```bash
id -nG | tr ' ' '\n' | grep -x dialout || sudo usermod -aG dialout $USER
```

Then **log out and back in** — group membership is established at login, so
a new shell in the same session still will not have it. `newgrp dialout`
works for one shell.

**2. ModemManager is probing the adapters.** These are CDC-ACM devices;
ModemManager opens every new one and talks AT commands at it for ~20
seconds. While it holds the port, your open fails in a way that is
indistinguishable from a permissions error — and because it lets go
eventually, a retry a minute later can succeed, which makes it look
intermittent and unexplainable.

```bash
sudo tee /etc/udev/rules.d/98-feetech-no-mm.rules >/dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ENV{ID_MM_DEVICE_IGNORE}="1"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=tty
```

`--action=add` matters; see `ORIN-SETUP.md`. If the machine has no cellular
modem at all, `sudo systemctl disable --now ModemManager` is simpler.

`config/scan-motors.sh` diagnoses both of these for you now.

## Python version

`pyrealsense2` publishes aarch64 wheels for cp39/cp310/cp312 only, and no
sdist. lerobot requires >=3.12. So on ARM, **Python 3.12 exactly**:

```bash
uv python pin 3.12
uv sync --extra feetech --extra intelrealsense --extra viz
```

Leave out `--extra dataset` until you need to record; it pulls `torchcodec`,
which on aarch64 needs >=0.11.0 and torch>=2.11.
