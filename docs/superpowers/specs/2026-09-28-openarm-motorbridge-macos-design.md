# Design: OpenArm on motorbridge (with macOS libusb PCAN backend)

## Problem

OpenArm's follower and teleoperator currently drive their Damiao CAN motors
through `DamiaoMotorsBus` (a `python-can`/SocketCAN backend), which is
Linux-only. We want OpenArm to use the `motorbridge` library instead (as the
reBot B601 already does), and to run on macOS. On macOS, the standard PCBUSB
(MacCAN) runtime only reaches the first channel of a multi-channel PEAK adapter,
which is insufficient for leader+follower or bimanual setups. The
`CarolinePascal/motorbridge` branch `feat/pcan-usb-fd-native` adds a libusb-based
PCAN backend that reaches both channels (classic CAN only).

## Current state (already scaffolded)

- A drop-in `MotorBridgeDamiaoBus` already exists in
  [`motorbridge_bus.py`](../../../src/lerobot/motors/damiao/motorbridge_bus.py):
  same degree-based MIT interface as `DamiaoMotorsBus`, backed by `motorbridge`.
  Its `connect()` uses `Controller.from_socketcanfd(port)` when `use_can_fd`,
  else `Controller(channel=port)`.
- It is **not** exported from
  [`motors/damiao/__init__.py`](../../../src/lerobot/motors/damiao/__init__.py)
  and **not** wired into the robot/teleop, which still build `DamiaoMotorsBus`.
- The follower/leader tests
  ([`test_openarm_follower.py`](../../../tests/robots/test_openarm_follower.py),
  [`test_openarm_leader.py`](../../../tests/teleoperators/test_openarm_leader.py))
  already expect `MotorBridgeDamiaoBus` and a `require_package` call in the
  robot/teleop modules, so they are currently red.
- `openarms = ["lerobot[damiao]"]` pulls only `python-can`, but
  `MotorBridgeDamiaoBus` calls `require_package("motorbridge", extra="openarms")`.
- Only the OpenArm follower and leader consume `DamiaoMotorsBus`.

## Chosen approach

- **Replace** the OpenArm backend with `MotorBridgeDamiaoBus` (motorbridge becomes
  the only OpenArm backend). Keep `DamiaoMotorsBus` (python-can) for its own tests
  / other potential users.
- **macOS auto-detection**: route through the branch's libusb PCAN backend and
  force classic CAN. Do **not** use the PCBUSB/MacCAN runtime.
- Scope: follower + leader, single + bimanual.

## Detailed design

### A. Wire OpenArm to `MotorBridgeDamiaoBus`

- Export `MotorBridgeDamiaoBus` from `motors/damiao/__init__.py` (keep
  `DamiaoMotorsBus`).
- `openarm_follower.py`: import `MotorBridgeDamiaoBus` and `require_package`; call
  `require_package("motorbridge", extra="openarms")` in `__init__`; construct
  `MotorBridgeDamiaoBus` with the same arguments. Remove the `DamiaoMotorsBus`
  import.
- `openarm_leader.py`: same switch; keep the `MotorState` import used only for
  type annotation in `get_action`.
- This turns the already-written follower/leader tests green.

### B. macOS -> libusb PCAN backend, classic CAN

- In `MotorBridgeDamiaoBus`, detect macOS via a mockable indirection over
  `platform.system() == "Darwin"`.
- On macOS:
  - Force classic CAN (ignore `use_can_fd`; the libusb PCAN backend is
    classic-CAN-only).
  - Select the libusb backend using the `pcanfd:` **channel prefix**:
    `Controller(channel="pcanfd:can0")` (assumption; can be switched to the
    `MOTORBRIDGE_PCAN_BACKEND=native` env var if the branch prefers that). A
    channel already prefixed with `pcanfd:` is left as-is.
  - This reaches both channels (`can0` -> BUS1, `can1` -> BUS2), enabling
    leader+follower and bimanual on one adapter.
  - Log a clear one-line message (classic CAN + libusb backend).
- On Linux: unchanged (`from_socketcanfd(port)` when FD, else
  `Controller(channel=port)`).

### C. Dependency

- Update the extra to install motorbridge:
  `openarms = ["lerobot[damiao]", "lerobot[motorbridge-dep]"]`.
- Do **not** pin the fork branch in `pyproject.toml`: the libusb `pcanfd:` backend
  is not in the PyPI `motorbridge>=0.5,<0.6` and must be built from source on
  macOS. The macOS build is documented instead (see D).

### D. Docs (`openarm.mdx`)

- Soften the "Linux Only" tip.
- Add a macOS section: build the fork branch with
  `cargo build --release -p motor_cli -p motor_abi --features motor_core/pcan-usb-fd`,
  then `pip install -e bindings/python`; note classic-CAN-only, both-channels, and
  the channel mapping (`can0` -> BUS1, `can1` -> BUS2).

### E. Tests

- Make the follower's `test_connect_selects_transport` platform-aware (patch the
  platform check to Linux) so FD/classic assertions hold regardless of host OS.
- Add a macOS test: platform = Darwin -> classic `Controller(channel="pcanfd:can0")`,
  `from_socketcanfd` not called, FD disabled. Mirror for the leader as needed.

## Non-goals (YAGNI)

- Not removing `DamiaoMotorsBus` (python-can); it stays and remains tested.
- No runtime backend-selection flag (we replace).
- No Windows support in this change.

## Open assumption to confirm

macOS selects the libusb backend via the `pcanfd:` channel prefix rather than the
`MOTORBRIDGE_PCAN_BACKEND=native` env var. Both select the same libusb backend per
the branch docs; the prefix avoids a global env var.
