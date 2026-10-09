# Running the robot

Day-to-day operation. `ORIN-SETUP.md` is the one-time bring-up; this is what
you do every session.

---

## At a glance

| where | what | why first |
|---|---|---|
| **Mac** | `rerun` viewer | the cart connects *out* to it; nothing listening means it back-pressures into the control loop |
| **rosie** | `./config/operator-leader-host.sh` | the cart waits 10 s for the first action and gives up |
| **elroy** | `./config/cart-teleop.sh` or `cart-record.sh` | needs both of the above |

All over SSH. Neither robot machine needs a screen — the Mac is the screen,
and the video originates on the cart, not the Pi.

---

## 1. Mac — the viewer (only if you want video)

```bash
uvx --from rerun-sdk==0.33.1 rerun
```

**No flags.** That one process is both the window and the listener — it
opens the viewer *and* serves gRPC on `0.0.0.0:9876`, which is what elroy
connects out to. There is no separate server to start. (`--port 9876` is
the default and changes nothing; the thing to avoid is `--serve-grpc`,
which is a server with *no* window and answers with "Connect by running
`rerun --connect ...`". You can make a working three-process setup out of
it, but there is no reason to.)

Pin the version. The SDK on elroy comes from `lerobot[viz]`; the viewer
here comes from PyPI's latest unless told otherwise, and the two halves of
one rerun release have to match. `cart-teleop.sh --watch` prints the exact
command with the right number in it.

**If the window opens but stays on the welcome screen**, the stream is not
arriving. Prove the link on its own, from elroy, with no robot in the way:

```bash
./config/video-test.sh          # moving test pattern, 3 fake cameras
```

Nothing appears? Then it is the link, in this order:

1. the viewer is running as plain `rerun`, not `--serve-grpc`
2. `nc -z -v 192.168.1.52 9876` from elroy answers
3. the Mac's firewall — System Settings → Network → Firewall. It blocks
   incoming connections to unknown binaries by default, and a
   `uvx`-launched rerun looks like a new binary whenever the cache moves.
   Either allow it when macOS asks, or turn the firewall off long enough
   to find out whether that is the answer.

Skip this whole step if you don't need to see what the robot sees.

---

## 2. rosie — the leaders

```bash
tmux new -s leaders                 # see below
cd ~/GitHub/lerobot
./config/operator-leader-host.sh
```

Expect:

```
serving: tcp://192.168.1.100:5557
         remap_arm_prefix: true
         emit_head: true
         emit_base: true
         ^ these must match the cart's config. A mismatch is silent.
Publishing on tcp://*:5557 at 60 Hz
17 action keys:
  ...
```

**17 keys** means arms, head and base. 12 means the head and base are
switched off and the gamepad will do nothing.

**Use tmux.** An SSH drop kills this process, and mid-recording that means
the cart's link stalls, the base zeroes and you lose the take. `Ctrl-B` then
`D` detaches; `tmux attach -t leaders` comes back.

---

## 3. elroy — the robot

```bash
cd ~/GitHub/lerobot
./config/cart-teleop.sh            # drive, record nothing
./config/cart-teleop.sh --watch    # + video to the Mac
```

Press ENTER at the calibration prompt to restore from file.

**Hold the leaders roughly where the followers are sitting before you
start.** The first action moves the followers to meet them.

**Cart on blocks** unless you mean it — the base is live.

### Recording

```bash
./config/cart-record.sh carlkesselman/<name> "<task description>" \
  --dataset.num_episodes=5 --dataset.episode_time_s=30
```

Episode keys are read by whichever process owns the TTY: right arrow ends an
episode, left re-records it, escape stops. Run it from an SSH session you
are attached to, inside tmux if you like — lerobot falls back to a terminal
listener when pynput cannot capture, so an SSH TTY works.

`push_to_hub` defaults to true. `hf auth login` once; `--dataset.push_to_hub=false`
to stay local.

---

## Controls

8BitDo SF30 Pro, wired, in its PlayStation-compatible mode.

| control | does |
|---|---|
| leader arms | the follower arms |
| D-pad | head pan / tilt |
| left stick | base forward/back, strafe |
| right stick X | base turn |
| R1 / L2 | base speed up / down |

If the pad power-cycles into a different input mode every one of those
becomes wrong at once, and the symptom is a cart that drives sideways rather
than any error. Re-probe:

```bash
uv run --project ../XLeRobot python -m \
  lerobot_teleoperator_xlerobot_leader_gamepad.probe_gamepad
```

---

## What good looks like

From the cadence summary printed on Ctrl-C:

```
effective cadence  29.5 Hz    (target 30 - the camera frame rate is the ceiling)
observe            ~28 ms     95% of all work; it is waiting for frames, not computing
teleop             0.05 ms    the whole network leader
send               ~1.5 ms
pacing headroom    4.3 ms     without video, 1.6 ms with
```

**Headroom is the number to watch.** Near zero means the loop is saturated
and has nothing left to absorb a hiccup. Video costs about 2.7 ms of it.

30 Hz is the ceiling, not a disappointment: `observe` blocks until the next
camera frame, and the cameras are 30 fps. A 4x reduction in pixels bought
only 13% off it, which is how we know it is latency-bound rather than
pixel-bound.

---

## When it goes wrong

| symptom | cause |
|---|---|
| **Arms do not move, loop runs at rate, no errors** | `remap_arm_prefix` disagrees across the link. `bi_so_follower` strips `left_`; `xlerobot` filters on `left_arm_`. |
| **Gamepad does nothing** | `emit_head`/`emit_base` false on rosie. Check the 17-key count. |
| `No action from the leader host ... within 10.0s` | leader host not running, or the wrong IP in `cart-remote.yaml` |
| `Leader link stalled` | wifi. Arms hold, base zeroes. Occasional is normal; constant is not. |
| `Failed to write 'Torque_Enable' on id_=N` | a dropped packet; `configure()` retries three times now. If it persists, run `./config/power-check.sh` before suspecting a servo. |
| `Failed to open OpenCVCamera(/dev/cam_left)` | usually a stale process: `pkill -f lerobot-teleoperate`. If the symlink is missing instead, `sudo udevadm trigger --action=add --subsystem-match=video4linux`. |
| **Script prints the header and exits silently** | `set -e` on a non-zero return. Shout about it; it is a bug. |
| **A fresh calibration starts** | you are on the arms-only config without one. It should refuse now — if it does not, Ctrl-C before the range-recording phase or you lose `robots/xlerobot/xlerobot.json`. |
| **CPU torch on elroy** | a bare `uv sync`. Use `./config/sync.sh`, which picks the right extras per machine. |

---

## Diagnostics

| | |
|---|---|
| `./config/cart-selftest.sh` | the cart alone: rate, 17 joint values, a frame per camera. `--head --arms --base` to move things. |
| `./config/leader-tap.sh` | the leader stream with no robot attached. Says which keys ever changed. |
| `./config/scan-motors.sh` | which motor ids answer, at which baud rate |
| `./config/power-check.sh` | voltage, temperature and load per servo, with torque applied one at a time |
| `./config/video-test.sh` | a synthetic camera feed to the viewer, no robot. Separates the video link from everything else. |
| `./config/sync.sh` | sync with the right extras for this machine |

---

## Order matters

Viewer, then leaders, then cart. Each waits on the one before:

- the cart gives up after 10 s with no leader host
- rerun's proxy may not replay what it buffered before a viewer attached
- starting the cart first just means doing it again
