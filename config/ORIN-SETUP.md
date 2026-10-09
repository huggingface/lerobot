# Bringing up elroy (the Orin) as the cart

One step at a time, each with a check that has to pass before the next.
Everything here runs on the Orin unless a step says otherwise.

The aim of this document is a verified robot on one machine — arms, head,
base and three cameras — with no network, no leader arms and no dataset in
the picture. Those come after.

---

## 0. Find out what hardware you actually have

**Run this where the arms are plugged in right now** — on the Pi, before
moving anything. It is a hardware question, not a machine question, and
knowing the answer changes every later step.

```bash
./config/scan-motors.sh
```

It pings every id at every supported baud rate on both buses and maps what
answers onto the xlerobot layout:

```
bus1   1-6 left arm    7-8 head
bus2   1-6 right arm   7-9 base wheels
```

The three outcomes:

- **Arms, head and base all present** → the full `xlerobot` path works.
  Continue through this document as written.
- **Arms only** → `--robot.type=xlerobot` will fail on connect. It reads
  every motor unconditionally, base velocity included, so a missing head
  servo stops the arms too. Use `config/bi-arms.yaml` and
  `config/cart-remote-arms.yaml` until the rest is wired, and skip the
  calibration step below in favour of the per-arm one.
- **Something at the wrong baud rate** → `lerobot-setup-motors` for those
  servos. lerobot talks at 1000000 and will not find them otherwise.

A whole bus silent is usually the cable or the power. One id silent is
usually that servo or the daisy-chain link into it.

---

## 1. Python 3.12 and the workspace

```bash
git clone <your lerobot fork>  ~/GitHub/lerobot
git clone <your XLeRobot fork> ~/GitHub/XLeRobot     # must be siblings
cd ~/GitHub/lerobot && git checkout feature/xlerobot

cd ~/GitHub/XLeRobot
uv python pin 3.12
uv sync
```

**3.12 exactly.** `pyrealsense2` publishes aarch64 wheels for cp39, cp310
and cp312 only — no cp313, no cp314, and no sdist to build from. lerobot
needs ≥3.12, which leaves one version.

**Sync from XLeRobot, never from lerobot.** XLeRobot is the workspace root
and maps `lerobot` to `../lerobot`; syncing from the lerobot repo prunes
the plugins and `--robot.type=xlerobot` stops resolving.

Check:

```bash
uv run python -c "
from lerobot.robots import make_robot_from_config
from lerobot_robot_xlerobot import XLerobotConfig
print('xlerobot plugin OK')"
```

---

## 2. JetPack torch

`uv sync` gives you PyPI's aarch64 **CPU** torch. That is wrong for the
machine whose entire reason for being on the cart is the GPU — it will run
policies, slowly, on the CPU and nothing will tell you.

Install NVIDIA's JetPack wheels for your JetPack version into the same
environment, after `uv sync`, and confirm:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` here means inference will be useless later. Teleoperation and
recording do not care, so this can wait — but do not forget it.

---

## 3. Move the hardware

From the Pi to the Orin: both follower arm buses, both wrist cameras, the
head RealSense. **The leader arms stay on the Pi** — that is the whole
point of the split.

Give the RealSense a root port of its own. On the Pi, sharing a host
controller with a wrist camera killed that camera's frame reads the moment
the RealSense started streaming (`read failed (status=False)` →
`exceeded maximum consecutive read failures`). The Orin's controller map
differs, but the contention does not care:

```bash
lsusb -t          # 5000M = USB 3, 480M = USB 2
```

Check the serial buses came up:

```bash
ls -l /dev/serial/by-id/
```

Expect four entries while the leaders are still attached, two after they
move to the Pi:

| serial | role |
|---|---|
| `5A7A058116` | left follower → port1 |
| `5A68009991` | right follower → port2 |

---

## 4. Camera names

The configs use `/dev/cam_left` and `/dev/cam_right`, so the physical port
lives in one udev rule instead of three config files.

**Measured on elroy, 2026-10-09** — both wrist cameras on the external hub:

| | ID_PATH | node |
|---|---|---|
| left wrist | `platform-3610000.usb-usb-0:2.3.3:1.0` | video6 |
| right wrist | `platform-3610000.usb-usb-0:2.3.4:1.0` | video8 |
| head RealSense | `platform-3610000.usb-usb-0:1.1:1.0` and `:1.3` | video0–5 |

`config/99-cameras-elroy.rules` has those baked in:

```bash
sudo cp config/99-cameras-elroy.rules /etc/udev/rules.d/99-cameras.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/cam_*
```

You should see `cam_left -> ../video6` and `cam_right -> ../video8`.

If you ever need to redo this mapping — a new machine, a moved hub — the
reliable method is positional, because both Innomakers report serial
`SN0001` and nothing in their identity tells them apart. Unplug both, plug
in only the left one, and record which capture node appears:

```bash
for d in /dev/video*; do
  echo "$d  $(udevadm info -q property $d | grep -E '^(ID_PATH|ID_V4L_CAPABILITIES)=' | tr '\n' ' ')"
done
```

Two details that are easy to get wrong:

- **Each camera makes two nodes sharing one `ID_PATH`** — a capture node and
  a metadata node. Only the capture one has `:capture:` in
  `ID_V4L_CAPABILITIES`. Without that filter in the rule, the symlink lands
  on whichever node udev handles last, and you get a camera that opens fine
  and never delivers a frame.
- **Use `ENV{ID_PATH}`, not `KERNELS`.** On the Pi a `KERNELS` rule stopped
  working after a camera changed ports and would not come back even once the
  value was corrected — `udevadm test` showed the right value in the device
  chain and created no symlink. Never explained; `ID_PATH` has not done it.

The paths encode the *hub's* ports. Move a camera to a different hub port,
or the hub to a different Type-A port, and the rule silently stops matching.
Check `ls -l /dev/cam_*` after any replug.

### Formats

```bash
v4l2-ctl --device=/dev/cam_left --list-formats-ext
```

```
MJPG  1280x720  30.000 fps
YUYV  1280x720  10.000 fps    <- what OpenCV picks by default
```

Uncompressed 720p30 is ~55 MB/s, past USB 2.0. The configs set
`fourcc: MJPG` for this reason; without it you get
`failed to set fps=30 (actual_fps=10.0)`.

### RealSense permissions

The RealSense is addressed by serial number, so it needs no naming rule —
but libusb access may need librealsense's own permissions rules. Test first:

```bash
uv run --project ../XLeRobot lerobot-find-cameras realsense
```

If it reports no device while `lsusb` clearly shows it, that is a
permissions problem, not a wiring one. Install librealsense's
`99-realsense-libusb.rules` and make sure your user is in the `video` and
`plugdev` groups (`id -nG`), then re-plug the camera.

Headless check of all three at once:

```bash
uv run --project ../XLeRobot lerobot-find-cameras
```

---

## 5. Calibrate

Calibration lives on the machine the motors are attached to, so this is a
fresh calibration on the Orin even though the arms were calibrated on the
Pi. The per-arm files there are `so_follower/xlerobot_arms_{left,right}.json`
with motor names like `shoulder_pan`; `xlerobot` wants one file,
`robots/xlerobot/<id>.json`, with names like `left_arm_shoulder_pan`. Not
interchangeable — recalibrate rather than convert.

```bash
cd ~/GitHub/lerobot
uv run --project ../XLeRobot lerobot-calibrate \
  --robot.type=xlerobot \
  --robot.id=xlerobot \
  --robot.port1=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A058116-if00 \
  --robot.port2=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68009991-if00
```

It runs in two passes, and the prompts matter:

1. **bus1 — left arm and head.** Move them to the middle of their range,
   press ENTER, then move every joint through its full travel.
2. **bus2 — right arm.** Same. The three base wheels are handled
   automatically: homing offset 0 and a full-turn range, since a wheel has
   no end stops to find.

Homing offsets are per-unit physical facts. Swapping which arm sits behind
an id means recalibrating, not renaming.

Check:

```bash
C=$(uv run --project ../XLeRobot python -c \
  "from lerobot.utils.constants import HF_LEROBOT_CALIBRATION as C; print(C)")
find "$C" -name '*.json' | sed "s|$C/||" | sort
```

You want `robots/xlerobot/xlerobot.json`.

---

## 6. Self-test

The payoff. One machine, everything local.

```bash
./config/cart-selftest.sh
```

Read-only: connects both buses and all three cameras, reports the
observation rate, prints all 17 state values, and writes one frame per
camera to `./selftest-frames/`.

What to look for:

- **Observation rate ≥ 30 Hz.** Below that, recording cannot keep its
  contract. It is almost always a camera rather than the servos — confirm
  with `--no-cameras`, which should jump to several hundred Hz.
- **Joint values that vary between reads.** A dead servo often reads an
  exact constant while its neighbours jitter.
- **The PNGs.** Open them. A camera can be live, correctly sized and
  pointed at the wrong thing, and no automated check catches that.

Then the movement tests, each asking before it moves anything:

```bash
./config/cart-selftest.sh --head          # sweeps both head servos +/-12 deg
./config/cart-selftest.sh --arms          # nudges all 12 arm joints 5 deg
./config/cart-selftest.sh --base          # drives the wheels — ON BLOCKS
```

`--base` is the one that answers "can you see the base". It commands
forward, backward and both turns, then reads `x.vel` / `theta.vel` back out
of the observation. Those come from the wheels' `Present_Velocity` through
`_wheel_raw_to_body`, so a plausible number means the whole chain works:
command → kinematics → bus → servo → encoder → kinematics → observation.

Watch the wheels as well as the numbers. Body velocity is derived from all
three, so two correct wheels can mask a third spinning backwards.

---

## 7. Then, and only then

- `config/bi-arms.yaml` — arms and leaders on this one machine, to confirm
  teleoperation itself works before adding the network.
- `config/operator-leader-host.sh` on the Pi and `config/cart-teleop.sh`
  here — the split.
- `config/cart-record.sh` — recording.

Each of those adds exactly one new thing. When something breaks, it is the
thing you just added.
