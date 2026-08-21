# Async inference (and async teleop) for the Unitree G1

Status: partly implemented, and the topology landed differently than planned.

Async inference runs on hardware today via the **loopback-bridge** topology, not
the "onboard mode" of Step 2: `run_g1_server.py` keeps ownership of DDS on the
Jetson, and `robot_client` runs the robot class in a second process on the same
machine with `--robot.is_simulation=false --robot.robot_ip=127.0.0.1`. Everything
robot-side is loopback, so the controller thread and `rt/lowcmd` still stay
onboard — the goal of Step 2 — without splitting transport from simulation. See
`docs/source/unitree_g1.mdx` for the commands.

| Blocker                                       | State                                                                     |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| 1 — `robot_client` can't resolve `unitree_g1` | fixed (import added; ZMQ camera type registered too)                      |
| 2 — no onboard mode                           | sidestepped by the loopback bridge; still worth doing to drop the ZMQ hop |
| 3 — motion switcher                           | sidestepped: the bridge still releases it                                 |
| 4 — stale command                             | **still open**, and now the biggest one; see the note below               |
| 5 — `publish_lowcmd` CRC race                 | fixed (`_lowcmd_lock`)                                                    |

## The idea

Run the LeRobot async inference stack against the G1 so that a large policy
(GR00T, SmolVLA, pi0) can drive the arms and the locomotion velocity command
from a GPU machine, while the balance/WBC controller keeps running on the robot
at its own rate and stays responsible for not falling over.

Target topology:

```
   G1 onboard computer (Jetson)            laptop / GPU box
  ┌──────────────────────────────┐        ┌─────────────────────┐
  │ robot_client                 │        │ policy_server       │
  │  ├─ UnitreeG1 (Robot)        │ gRPC   │  ├─ checkpoint      │
  │  │   ├─ DDS -> rt/lowcmd     │◄──────►│  └─ predict_action_ │
  │  │   └─ controller thread    │  :8080 │      chunk()        │
  │  │       (50 Hz balance)     │        └─────────────────────┘
  │  └─ action queue             │
  └──────────────────────────────┘
```

The client owns the hardware and dials out to the server, so only the server
needs a reachable IP/port. On a tailnet that is trivial.

### Why the G1 is a good fit for this

`UnitreeG1` already runs the locomotion controller in its own thread at its own
`control_dt`, decoupled from `send_action` (`unitree_g1.py`, `_controller_loop`
and `send_action`). `send_action` only writes arm targets and pokes the joystick
command into the controller's input; the controller thread owns legs and waist.

That is exactly the layering async inference wants. The action queue drains at a
variable, jittery rate, and when it runs dry the client simply stops calling
`send_action` — the arms hold their last commanded position while the balance
loop keeps running locally. A network stall degrades manipulation, not balance.

**Except for one hole, described in "Blocker 4" below, which has to be fixed
before any of this is safe to run.**

## What exists today

| Piece                            | Where it runs                           | Transport                                            |
| -------------------------------- | --------------------------------------- | ---------------------------------------------------- |
| `run_g1_server.py`               | on the G1                               | DDS ↔ ZMQ bridge (ports 6000/6001, camera 5555)     |
| `UnitreeG1` robot class          | on the G1, second process               | ZMQ shims in `unitree_sdk2_socket.py`, over loopback |
| locomotion controller            | inside `UnitreeG1`, so on the G1        | —                                                    |
| `robot_client` / `policy_server` | client on the G1, server on the GPU box | gRPC                                                 |

(Originally the robot class ran on the laptop, so every lowcmd crossed the
network. Pointing it at `127.0.0.1` moved the whole robot-side process onto the
Jetson — the ZMQ hop is now loopback, and only token chunks cross the network.)

## Blockers

### Blocker 1 — `robot_client` cannot resolve `--robot.type=unitree_g1`

`src/lerobot/async_inference/robot_client.py` imports a handful of robots purely
for their draccus registration side effects:

```python
from lerobot.robots import (  # noqa: F401
    Robot, RobotConfig, bi_so_follower, koch_follower,
    make_robot_from_config, omx_follower, so_follower,
)
```

`lerobot/robots/__init__.py` only exports `Robot`, `RobotConfig` and
`make_robot_from_config`, so nothing pulls in `config_unitree_g1.py` and
`@RobotConfig.register_subclass("unitree_g1")` never fires. draccus then rejects
`--robot.type=unitree_g1` as an unknown choice.

`lerobot_record.py` already does this correctly (it imports
`unitree_g1 as unitree_g1_robot`).

Construction is _not_ a problem: `make_robot_from_config` has no G1 branch, but
the generic `make_device_from_device_class` fallback resolves `UnitreeG1Config`
→ `lerobot.robots.unitree_g1.UnitreeG1` correctly.

`SUPPORTED_ROBOTS` in `constants.py` also omits the G1, but the assert that used
it is commented out in the client, so it does not block anything.

**Fix:** add `unitree_g1` to the import list in `robot_client.py`. One line.

### Blocker 2 — there is no "run natively on the robot" mode

`is_simulation` is a single boolean controlling two orthogonal things — whether
MuJoCo is in the loop, and which transport talks to the motors:

| `is_simulation` | Channel classes      | `connect()` does                                                                                                                                 |
| --------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `True`          | real Unitree SDK DDS | `ChannelFactoryInitialize(0, "lo")`, spawns MuJoCo via `make_env("lerobot/unitree-g1-mujoco")`, and `_subscribe_lowstate` calls `sim_env.step()` |
| `False`         | ZMQ shims            | connects out to `config.robot_ip:6000/6001`                                                                                                      |

Neither is "onboard". With `True` you get real DDS but pinned to the loopback
interface plus a MuJoCo sim you do not want. With `False`, running on the Jetson
means DDS → JSON → ZMQ → JSON → DDS over loopback to reach a bus the process
could have joined directly.

**Fix:** split transport from simulation. See "Work plan" step 2.

### Blocker 3 — the motion switcher is only released by the bridge

`run_g1_server.py` releases Unitree's factory motion service before doing
anything else:

```python
msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()
status, result = msc.CheckMode()
while result is not None and "name" in result and result["name"]:
    msc.ReleaseMode()
    status, result = msc.CheckMode()
    time.sleep(1.0)
```

If `UnitreeG1` publishes `rt/lowcmd` directly without the bridge, nothing
releases that service, so it keeps ownership of the joints and the commands get
fought or ignored. Symptom: the controller loop logs a healthy 50 Hz while the
robot stands there doing nothing.

**Fix:** move this into `connect()` for the onboard path.

### Blocker 4 — stale locomotion command (SAFETY, fix this first)

`_update_controller_action` is the only writer of `controller_input`:

```python
def _update_controller_action(self, action: RobotAction) -> None:
    with self._controller_action_lock:
        for key in REMOTE_KEYS:
            if key in action:
                self.controller_input[key] = action[key]
```

Nothing ever decays or zeroes it, and the controller thread reads a snapshot
every `control_dt` forever. So when actions stop arriving — server stalls,
network drops, laptop sleeps, queue drains — `send_action` stops being called,
the arms hold position (fine), **but the last locomotion command persists
indefinitely and the G1 keeps walking**. The balance controller will happily
keep it upright the whole time it strolls into a wall.

This is benign in the current synchronous teleop loop because a dropped
connection kills the process and the controller thread with it. Put a network
and an action queue in between and it becomes a dead-man's-switch problem.

**This applies to policy inference too, not just teleop.**

**Fix:** a watchdog inside `_controller_loop` that zeroes the `remote.*` axes if
no action has been received for N ms. Put it in the controller loop, not the
client, so it holds regardless of which transport is driving.

**Update — this is worse under SONIC than the text above describes.** With
`SonicWholeBodyController` the stale thing is not a joystick axis, it is the
64-D token: `run_step` holds `self._last_token` and re-decodes it against fresh
proprioception every tick. So a dropped policy stream doesn't freeze the robot,
it keeps _executing the last intent_ — told to walk forward, it keeps walking
forward, balancing correctly the whole way into whatever is in front of it.
Nothing decays the token and nothing falls back to `neutral_token` (that is
seeded only on the first tick, when no token has arrived yet).

So the watchdog needs a controller-level hook, not just axis zeroing: on
timeout, either re-seed `neutral_token` (decodes to a stand) or command
`default_angles` directly. A timeout that zeroes `remote.*` does nothing for a
token-driven policy.

### Blocker 5 — `publish_lowcmd` CRC race

```python
self.msg.crc = self.crc.Crc(self.msg)
self.lowcmd_publisher.Write(self.msg)
```

Both the controller thread (joints 0–14) and `send_action` (arm joints 15–28)
call this with no lock. They write disjoint motor ranges so the targets do not
corrupt each other, but if one thread mutates `self.msg` between the other's CRC
computation and its `Write`, the published packet carries a stale CRC and the
robot drops it. Network latency currently hides this; running both threads
onboard tightens the timing and it shows up as jitter and dropped commands.

**Fix:** one lock around the mutate-CRC-write sequence.

## Observation / action plumbing to verify

Not blockers, but they fail _silently_, which is worse.

### The state vector is smaller than it looks

```python
@cached_property
def observation_features(self) -> dict[str, type | tuple]:
    return {**self._motors_ft, **self._cameras_ft}
```

`_motors_ft` is only the 29 `.q` values. But `get_observation()` also returns
`.dq`, `.tau`, IMU quaternion / gyro / accel / rpy, and `wireless_remote` bytes.

`map_robot_keys_to_lerobot_features` feeds `observation_features` into
`hw_to_dataset_features`, and `build_dataset_frame` iterates over the _declared
features_, not the raw observation. So all those extras are discarded.

Good news: the raw `wireless_remote` bytes will not crash the pickle/tensor
path. Bad news: if the checkpoint was trained on IMU or joint velocities, the
client hands the policy a 29-dim position-only state and you get bad behavior
with no error anywhere.

**Action:** confirm the checkpoint's expected `observation.state` dim before
running. If it needs more than 29, extend `observation_features` (and keep the
ordering identical to training).

### Action ordering is positional

```python
def _action_tensor_to_action_dict(self, action_tensor):
    return {key: action_tensor[i].item()
            for i, key in enumerate(self.robot.action_features)}
```

With a controller set, `action_features` is 14 arm joints + 4 remote axes = 18
dims, in dict insertion order. Must match training order exactly.

### Camera frames can be `None`

`cam.read_latest()` can return `None` before the first ZMQ frame lands, which
throws inside `resize_robot_observation_image` rather than skipping the frame.
Worth a guard.

## Work plan

Ordered so each step is independently testable.

**Step 0 — locomotion watchdog (Blocker 4).** Add a `last_action_time` updated
in `_update_controller_action`, and in `_controller_loop` zero the `remote.*`
axes when `time.monotonic() - last_action_time > timeout`. Make the timeout a
config field (suggest 200–500 ms). Test in sim.

**Step 1 — client import (Blocker 1).** Add `unitree_g1` to the `lerobot.robots`
import in `robot_client.py`. Validates end to end with the _existing_ topology:
`robot_client` on the laptop (ZMQ to the bridge), `policy_server` anywhere. This
proves the observation/action plumbing before touching transport.

**Step 2 — transport split (Blocker 2).** In `UnitreeG1Config`, replace the
`is_simulation` boolean with a mode and add a NIC field:

```python
mode: Literal["sim", "onboard", "remote"] = "sim"
network_interface: str | None = None   # e.g. "eth0", used by onboard
```

Keep `is_simulation` as a derived read-only property (`self.mode == "sim"`) so
existing sim and remote invocations keep working. In `__init__`, select real SDK
classes for both `sim` and `onboard`, ZMQ shims for `remote`. In `connect()`:

- `sim` — unchanged: `ChannelFactoryInitialize(0, "lo")` + `make_env(...)`
- `onboard` — `ChannelFactoryInitialize(0, network_interface)`, no MuJoCo
- `remote` — unchanged: `ChannelFactoryInitialize(0, config=self.config)`

Also guard the `sim_env.step()` call in `_subscribe_lowstate` on `mode == "sim"`.

**Step 3 — motion switcher (Blocker 3).** Call the release loop from `connect()`
when `mode == "onboard"`. Factor it out of `run_g1_server.py` into a shared
helper so there is one copy.

**Step 4 — lowcmd lock (Blocker 5).** Add `self._lowcmd_lock` and hold it across
the mutate/CRC/Write block in `publish_lowcmd`.

**Step 5 — run it onboard.** Install this fork on the Jetson with the async and
G1 extras (`uv sync --extra async --extra unitree_g1`); there are no console
scripts for either side, both are run as modules/scripts. Then:

```bash
# laptop / GPU box
uv run python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 --port=8080 --fps=30

# on the G1, under tmux so it survives SSH disconnect
uv run python src/lerobot/async_inference/robot_client.py \
    --robot.type=unitree_g1 \
    --robot.mode=onboard \
    --robot.network_interface=eth0 \
    --robot.controller=GrootLocomotionController \
    --server_address=<laptop-tailnet-ip>:8080 \
    --policy_type=<...> --pretrained_name_or_path=<...> \
    --actions_per_chunk=50 --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

Note `run_g1_server.py` is no longer needed for the inference path — the client
process _is_ the robot-side process. The bridge stays useful for the remote
mode and for anything that still runs the robot class off-board.

**Step 6 — tune.** `actions_per_chunk`, `chunk_size_threshold` and the
aggregation function are all tuned around SO-100-class arms at 30 fps. What
chunk-overlap blending does to a 29-DOF humanoid is genuinely uncharacterized.
Start with `latest_only` aggregation (no blending) to get a clean baseline
before enabling `weighted_average`.

## Extension: remote policy server (rented GPU / HF Jobs)

The laptop-on-the-LAN server works, but it means the GPU has to be in the room.
Question worth recording: can `policy_server` run on rented compute, e.g. HF
Jobs, with the robot dialing out to it?

**Not with the code as it stands.** LeRobot's Jobs integration is training-only
(`lerobot-train --job.target=<flavor>`); nothing in `src/lerobot/async_inference/`
knows about jobs. Jobs _can_ expose a port now (`hf jobs run --expose 8080`,
needs `huggingface_hub >= 1.19`, billed on top of the flavor), but the shape does
not match what the policy server speaks:

- **Auth/TLS.** The exposed port is reached through an HTTPS proxy gated on a
  Bearer HF token with read access to the job's namespace. `policy_server` calls
  `add_insecure_port` and `robot_client` opens a `grpc.insecure_channel` — plain
  h2c, no TLS, no call metadata. There is nowhere to put the token.
- **Streaming.** `SendObservations(stream Observation)` is a client-streaming
  RPC (`src/lerobot/transport/services.proto`). That needs end-to-end HTTP/2
  through the proxy, not the request/response gating it is built for.

What it would take, roughly in order of risk:

1. Verify the jobs proxy actually passes long-lived gRPC streams. If it only
   does unary HTTP, `SendObservations` has to be reshaped (unary per frame, or
   gRPC-web), which is a real protocol change, not a config flag.
2. TLS on the server (`add_secure_port` + creds) and `secure_channel` on the
   client.
3. Per-call auth metadata on the client (call credentials or an interceptor),
   plus config fields for the token and the `https://<job-id>--8080.hf.jobs`
   base URL.

**Cheaper answer that works today:** any GPU box reachable over plain TCP. Since
the robot dials out, only the server needs an address — so a tailnet-joined box
(we already had `100.77.244.54` in play) works with the existing insecure
channel and zero code changes. A Jobs container that joins the tailnet at
startup with an auth key would too, and skips the proxy entirely; that is
probably the shortest path to "GPU not in the room" if it ever matters.

**Prerequisite either way: Blocker 4.** Moving inference off the LAN widens the
stall window, and the client streams camera frames upstream at 30 fps. Without
the watchdog, a stall means the G1 keeps walking on its last token. Don't run
this over a WAN before the timeout exists.

## Extension: async teleop

Same transport, different producer. The `.proto` does not change.

**Server side is easy.** `policy_server.py` implements four gRPC methods —
`Ready`, `SendPolicyInstructions`, `SendObservations`, `GetActions`. A teleop
server implements the same four, backed by `Teleoperator.get_action()` instead
of `policy.predict_action_chunk()`. The observation stream flowing up becomes
the FPV/state feed for the operator instead of policy input.

**Client side is a fork, not a patch**, because the buffer policy inverts:

- `_aggregate_action_queues` exists to blend overlapping chunks; teleop wants a
  depth-1 overwrite buffer.
- `_ready_to_send_observation` gates on `qsize() / action_chunk_size`, which
  assumes you _want_ a backlog to ride out slow inference. A 50-action queue of
  the operator's past positions is 1.5 s of lag by design.
- `must_go` only exists to force an inference pass when the queue runs dry; an
  operator is always producing, so it is meaningless.
- Add a staleness deadline: if no command in N ms, hold pose and zero the
  locomotion axes (this is Blocker 4 again, and it matters more here).
- Optionally a 2–3 command smoothing buffer so a 330 Hz glove stream downsampled
  over a jittery network does not produce steppy motion. Smoothing depth traded
  against added latency.
- `RobotClientConfig.__post_init__` rejects empty `policy_type` and
  `pretrained_name_or_path`, so those need to become optional.

Estimate: ~200–300 lines across a new server file and a client fork.
`_action_tensor_to_action_dict` and the whole robot-facing end are reusable
verbatim, since teleop produces the same 18-dim arm-plus-remote vector.

**Transport caveat.** gRPC rides HTTP/2 over TCP, so a dropped packet
head-of-line-blocks everything behind it. Fine on LAN or a tailnet, painful over
the open internet — for teleop you would rather lose a command than delay all
subsequent ones. That is why WebRTC/SCTP with per-stream unreliable delivery is
the right answer for long-haul teleop. Build the gRPC version for the
same-network case; swap transport only if remote teleop becomes real.

## Open questions

- Which checkpoint, and what `observation.state` dim does it expect? Determines
  whether `observation_features` needs extending.
- What is the G1's internal NIC name for `ChannelFactoryInitialize`?
- Do the head cameras go through V4L on the Jetson, or the Android/PICO path? If
  the latter, the ZMQ camera server may still be needed onboard.
- Client-side device: does anything need `--client_device=cuda` on the Jetson,
  or is `cpu` fine given no on-device preprocessing?
