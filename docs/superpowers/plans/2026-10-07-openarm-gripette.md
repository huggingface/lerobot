# OpenArm + Gripette gripper support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `openarm_gripette_follower` LeRobot robot that drives a 7-joint OpenArm (Damiao/CAN) arm together with Pollen's networked Gripette gripper (2 Feetech servos + camera) over gRPC.

**Architecture:** A new `OpenArmGripetteFollower(Robot)` composes an unmodified `OpenArmFollower` (configured with a gripper-less 7-joint `motor_config`) and a small `GripetteGripper` wrapper around Pollen's gRPC `GripperClient`. The gripper is exposed as a single `gripper.pos` (radians) plus a camera decoded from the gripper's 10 Hz `StreamState`.

**Tech Stack:** Python 3.12, draccus configs, `motorbridge` (arm CAN, via existing `MotorBridgeDamiaoBus`), `grpcio` + Pollen `gripette` package (gripper), `opencv-python-headless`/`numpy` (JPEG decode).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-07-openarm-gripette-design.md`.
- Single-arm only this iteration (no bimanual, no leader-side changes).
- Gripper exposed as a single `gripper.pos` in **Gripette-native radians**; no `gripper.load`, no 2-DOF.
- `gripper_motor_signs: tuple[float, float] = (1.0, -1.0)` default, tunable via config.
- Transport is gRPC via Pollen's `GripperClient`; the `gripette` package is an **external, guarded optional import** (`require_package("gripette", extra="gripette")`). It is not on PyPI — documented manual install; the `gripette` extra only pins the `grpcio` transport (reuse existing `grpcio-dep`).
- Dependency bounds policy: every new pyproject dependency has a lower **and** upper bound; reuse existing extras/groups, don't invent new ones unnecessarily.
- Imports: prefer top-level; guard optional deps with `TYPE_CHECKING or _foo_available` at module top + `require_package(...)` at use; reuse flags in `utils/import_utils.py` (don't call `is_package_available` ad-hoc).
- Keep changes minimal; do not modify `OpenArmFollower`.
- Env: run everything with the `lerobot-dev` conda env, forcing the workspace source on `PYTHONPATH` (the env's editable install points elsewhere):
  `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate lerobot-dev && export PYTHONPATH=/Users/caroline/Documents/lerobot-bis/src`
- Follower reference: `src/lerobot/robots/openarm_follower/` and tests `tests/robots/test_openarm_follower.py`.

---

### Task 1: Dependency plumbing (`gripette` extra + availability flag)

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`, around line 161/184)
- Modify: `src/lerobot/utils/import_utils.py:147` (hardware SDK flags block)

**Interfaces:**
- Produces: `lerobot.utils.import_utils._gripette_available: bool`; a `gripette` pyproject extra.

- [ ] **Step 1: Add the `gripette` extra**

In `pyproject.toml`, directly below the `openarms = [...]` line (≈184), add:

```toml
# Gripette gripper gRPC client. The Pollen `gripette` package is NOT on PyPI;
# install it manually (see docs/source/openarm.mdx). This extra only provides the
# gRPC transport.
gripette = ["lerobot[grpcio-dep]"]
```

- [ ] **Step 2: Add the availability flag**

In `src/lerobot/utils/import_utils.py`, in the "Hardware SDKs" block just after the `_motorbridge_available` lines (≈148), add:

```python
_gripette_available = is_package_available("gripette")
```

- [ ] **Step 3: Verify imports resolve**

Run: `python -c "from lerobot.utils.import_utils import _gripette_available; print(_gripette_available)"`
Expected: prints `False` (package not installed) with no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/lerobot/utils/import_utils.py
git commit -m "feat(openarm-gripette): add gripette optional extra + availability flag"
```

---

### Task 2: Config dataclass

**Files:**
- Create: `src/lerobot/robots/openarm_gripette_follower/__init__.py`
- Create: `src/lerobot/robots/openarm_gripette_follower/config_openarm_gripette_follower.py`
- Test: `tests/robots/test_openarm_gripette_follower.py`

**Interfaces:**
- Consumes: `lerobot.robots.openarm_follower.OpenArmFollowerConfigBase`, `lerobot.robots.config.RobotConfig`.
- Produces:
  - `OpenArmGripetteFollowerConfig` registered as `"openarm_gripette_follower"`, fields:
    - `arm_config: OpenArmFollowerConfigBase` (nested; its `gripper` motor is stripped by the robot)
    - `gripette_addr: str` (default `"localhost:50051"`)
    - `gripper_motor_signs: tuple[float, float] = (1.0, -1.0)`
    - `use_gripper_camera: bool = True`
    - `gripper_camera_name: str = "gripette"`

- [ ] **Step 1: Write the failing test**

Create `tests/robots/test_openarm_gripette_follower.py`:

```python
#!/usr/bin/env python
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.robots.openarm_gripette_follower import OpenArmGripetteFollowerConfig


def test_config_defaults_and_registration():
    cfg = OpenArmGripetteFollowerConfig(
        arm_config=OpenArmFollowerConfigBase(port="can0", side="right"),
        gripette_addr="10.0.0.5:50051",
    )
    assert cfg.type == "openarm_gripette_follower"
    assert cfg.gripper_motor_signs == (1.0, -1.0)
    assert cfg.use_gripper_camera is True
    assert cfg.gripper_camera_name == "gripette"
    assert cfg.arm_config.port == "can0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py::test_config_defaults_and_registration -v`
Expected: FAIL with `ModuleNotFoundError: ...openarm_gripette_follower`.

- [ ] **Step 3: Write the config**

Create `src/lerobot/robots/openarm_gripette_follower/config_openarm_gripette_follower.py`:

```python
#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from ..config import RobotConfig
from ..openarm_follower import OpenArmFollowerConfigBase


@RobotConfig.register_subclass("openarm_gripette_follower")
@dataclass
class OpenArmGripetteFollowerConfig(RobotConfig):
    """OpenArm 7-joint follower with a networked Gripette gripper (gRPC)."""

    arm_config: OpenArmFollowerConfigBase = field(
        default_factory=lambda: OpenArmFollowerConfigBase(port="can0")
    )
    # gRPC target of the Gripette server, e.g. "192.168.1.x:50051".
    gripette_addr: str = "localhost:50051"
    # Per-motor sign applied to the single gripper.pos command (mirrored mounting).
    gripper_motor_signs: tuple[float, float] = (1.0, -1.0)
    # Expose the Gripette's 10 Hz JPEG stream as a camera observation.
    use_gripper_camera: bool = True
    gripper_camera_name: str = "gripette"
```

Create `src/lerobot/robots/openarm_gripette_follower/__init__.py`:

```python
#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .config_openarm_gripette_follower import OpenArmGripetteFollowerConfig
from .openarm_gripette_follower import OpenArmGripetteFollower

__all__ = ["OpenArmGripetteFollower", "OpenArmGripetteFollowerConfig"]
```

> Note: the `__init__` imports `OpenArmGripetteFollower` (created in Task 4). Until Task 4 lands, run the Task 2 test by importing the config module directly if needed: `from lerobot.robots.openarm_gripette_follower.config_openarm_gripette_follower import OpenArmGripetteFollowerConfig`. The final test file imports from the package.

- [ ] **Step 4: Temporarily point the test at the config module, run, then revert after Task 4**

For now, change the test import to the direct module path to validate the config in isolation:

```python
from lerobot.robots.openarm_gripette_follower.config_openarm_gripette_follower import (
    OpenArmGripetteFollowerConfig,
)
```

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py::test_config_defaults_and_registration -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/robots/openarm_gripette_follower/config_openarm_gripette_follower.py \
        tests/robots/test_openarm_gripette_follower.py
git commit -m "feat(openarm-gripette): add OpenArmGripetteFollowerConfig"
```

(The package `__init__.py` is committed in Task 4 once the robot class exists.)

---

### Task 3: Gripette gRPC wrapper (`GripetteGripper`)

**Files:**
- Create: `src/lerobot/robots/openarm_gripette_follower/gripette_gripper.py`
- Test: `tests/robots/test_openarm_gripette_follower.py` (add cases)

**Interfaces:**
- Consumes: Pollen `gripette.client.GripperClient` (guarded import), `cv2`, `numpy`.
- Produces class `GripetteGripper`:
  - `__init__(self, addr: str, motor_signs: tuple[float, float], use_camera: bool, camera_name: str)`
  - `connect(self) -> None` — open client, `torque_on()`, start daemon stream thread
  - `is_connected: bool`
  - `send_goal(self, pos: float) -> float` — `client.move(sign1*pos, sign2*pos)`; returns `pos`
  - `read_observation(self) -> dict[str, Any]` — `{"gripper.pos": float}` plus `{camera_name: np.ndarray}` when `use_camera`
  - `camera_feature(self) -> tuple[int, int, int] | None` — `(H, W, 3)` or `None`
  - `disconnect(self, disable_torque: bool) -> None`
  - `_store(self, frame) -> None` — decode JPEG + cache latest pos/image (unit-tested directly)

- [ ] **Step 1: Write the failing tests**

Append to `tests/robots/test_openarm_gripette_follower.py`:

```python
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

_GRIP = "lerobot.robots.openarm_gripette_follower.gripette_gripper"


def _jpeg(h=8, w=6):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # strong blue channel in BGR
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes(), (h, w)


def test_store_decodes_jpeg_and_sign_corrects_position():
    from lerobot.robots.openarm_gripette_follower.gripette_gripper import GripetteGripper

    g = GripetteGripper(addr="x:1", motor_signs=(1.0, -1.0), use_camera=True, camera_name="gripette")
    jpeg, (h, w) = _jpeg()
    g._store(SimpleNamespace(jpeg_data=jpeg, motor1=0.42, motor2=-0.42))

    obs = g.read_observation()
    assert obs["gripper.pos"] == pytest.approx(0.42)       # sign1 * motor1
    assert obs["gripette"].shape == (h, w, 3)
    # decoded back to RGB: red channel (index 0) should carry the blue we encoded
    assert obs["gripette"][0, 0, 2] > 100


def test_send_goal_applies_signs():
    from lerobot.robots.openarm_gripette_follower.gripette_gripper import GripetteGripper

    g = GripetteGripper(addr="x:1", motor_signs=(1.0, -1.0), use_camera=False, camera_name="gripette")
    g._client = MagicMock()
    sent = g.send_goal(0.3)
    g._client.move.assert_called_once_with(0.3, -0.3)
    assert sent == pytest.approx(0.3)


def test_connect_disconnect_torque_and_thread():
    from lerobot.robots.openarm_gripette_follower.gripette_gripper import GripetteGripper

    jpeg, _ = _jpeg()
    frame = SimpleNamespace(jpeg_data=jpeg, motor1=0.1, motor2=-0.1)
    started = threading.Event()

    def fake_stream():
        started.set()
        yield frame
        while True:  # keep the stream "open" like the real server
            import time as _t

            _t.sleep(0.01)

    client = MagicMock()
    client.stream.side_effect = fake_stream

    with patch(f"{_GRIP}.GripperClient", return_value=client):
        g = GripetteGripper(addr="x:1", motor_signs=(1.0, -1.0), use_camera=True, camera_name="gripette")
        g.connect()
        assert started.wait(timeout=2.0)
        client.torque_on.assert_called_once()
        assert g.is_connected
        g.disconnect(disable_torque=True)
        client.torque_off.assert_called_once()
        client.close.assert_called_once()
        assert not g.is_connected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py -k "store or send_goal or connect_disconnect" -v`
Expected: FAIL with `ModuleNotFoundError: ...gripette_gripper`.

- [ ] **Step 3: Write the wrapper**

Create `src/lerobot/robots/openarm_gripette_follower/gripette_gripper.py`:

```python
#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper around Pollen's Gripette gRPC client for use inside LeRobot.

Exposes the 2-motor gripper as a single ``gripper.pos`` (radians) and decodes the
server's 10 Hz JPEG ``StreamState`` into an RGB camera frame. A background daemon
thread consumes the stream and caches the latest position + image.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.utils.import_utils import _gripette_available, require_package

if TYPE_CHECKING or _gripette_available:
    from gripette.client import GripperClient
else:
    GripperClient = None

logger = logging.getLogger(__name__)


class GripetteGripper:
    def __init__(
        self,
        addr: str,
        motor_signs: tuple[float, float],
        use_camera: bool,
        camera_name: str,
    ):
        require_package("gripette", extra="gripette")
        self.addr = addr
        self.motor_signs = motor_signs
        self.use_camera = use_camera
        self.camera_name = camera_name

        self._client: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_pos: float = 0.0
        self._latest_image: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def connect(self) -> None:
        self._client = GripperClient(self.addr)
        self._client.connect()
        self._client.torque_on()
        self._stop.clear()
        self._thread = threading.Thread(target=self._consume_stream, daemon=True)
        self._thread.start()

    def _consume_stream(self) -> None:
        assert self._client is not None
        try:
            for frame in self._client.stream():
                if self._stop.is_set():
                    break
                self._store(frame)
        except Exception:
            if not self._stop.is_set():
                logger.exception("Gripette stream closed unexpectedly.")

    def _store(self, frame) -> None:
        image = None
        if self.use_camera and frame.jpeg_data:
            bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest_pos = self.motor_signs[0] * frame.motor1
            if image is not None:
                self._latest_image = image

    def send_goal(self, pos: float) -> float:
        assert self._client is not None
        self._client.move(self.motor_signs[0] * pos, self.motor_signs[1] * pos)
        return pos

    def read_observation(self) -> dict[str, Any]:
        with self._lock:
            obs: dict[str, Any] = {"gripper.pos": self._latest_pos}
            if self.use_camera and self._latest_image is not None:
                obs[self.camera_name] = self._latest_image
        return obs

    def camera_feature(self) -> tuple[int, int, int] | None:
        with self._lock:
            if self.use_camera and self._latest_image is not None:
                h, w, _ = self._latest_image.shape
                return (h, w, 3)
        return None

    def disconnect(self, disable_torque: bool) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                if disable_torque:
                    self._client.torque_off()
            except Exception:
                logger.exception("Failed to disable Gripette torque on disconnect.")
            self._client.close()
        self._client = None
        self._thread = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py -k "store or send_goal or connect_disconnect" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/robots/openarm_gripette_follower/gripette_gripper.py \
        tests/robots/test_openarm_gripette_follower.py
git commit -m "feat(openarm-gripette): add GripetteGripper gRPC wrapper"
```

---

### Task 4: `OpenArmGripetteFollower` robot (composition)

**Files:**
- Create: `src/lerobot/robots/openarm_gripette_follower/openarm_gripette_follower.py`
- Modify: `src/lerobot/robots/openarm_gripette_follower/__init__.py` (already written in Task 2; commit now)
- Test: `tests/robots/test_openarm_gripette_follower.py` (add cases)

**Interfaces:**
- Consumes: `OpenArmGripetteFollowerConfig`, `OpenArmFollower`, `OpenArmFollowerConfig`, `GripetteGripper`.
- Produces class `OpenArmGripetteFollower(Robot)` (`name = "openarm_gripette_follower"`) with the standard `Robot` API, where:
  - `action_features` = arm joints 1–7 `*.pos` + `gripper.pos`
  - `observation_features` = `arm.observation_features` + `gripper.pos` + (optional) `{gripper_camera_name: (H,W,3)}`
  - `send_action` routes `joint_*` to the arm and `gripper.pos` to the gripper
  - `get_observation` merges arm obs + gripper obs
  - `connect/disconnect/is_connected/is_calibrated/calibrate/configure` delegate to the arm; gripper connects/disconnects alongside

- [ ] **Step 1: Write the failing tests**

Append to `tests/robots/test_openarm_gripette_follower.py`. This reuses the openarm follower mocking pattern (patch `MotorBridgeController`) and patches `GripetteGripper`:

```python
import math

_ADAPTER = "lerobot.motors.damiao.motorbridge_bus"
_PKG = "lerobot.robots.openarm_gripette_follower.openarm_gripette_follower"
_ARM = "lerobot.robots.openarm_follower.openarm_follower"

_ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]


def _make_robot(use_camera=True):
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
    from lerobot.robots.openarm_gripette_follower import (
        OpenArmGripetteFollower,
        OpenArmGripetteFollowerConfig,
    )

    cfg = OpenArmGripetteFollowerConfig(
        arm_config=OpenArmFollowerConfigBase(
            port="can0",
            side="right",
            joint_limits=dict.fromkeys(_ARM_JOINTS + ["gripper"], (-180.0, 180.0)),
        ),
        gripette_addr="x:1",
        use_gripper_camera=use_camera,
    )
    with (
        patch(f"{_ARM}.require_package", lambda *a, **k: None),
        patch(f"{_PKG}.GripetteGripper") as gripper_cls,
    ):
        robot = OpenArmGripetteFollower(cfg)
    return robot, gripper_cls


def test_arm_bus_excludes_gripper_motor():
    robot, _ = _make_robot()
    assert set(robot.arm.bus.motors) == set(_ARM_JOINTS)  # no "gripper" on CAN


def test_feature_maps():
    robot, gripper_cls = _make_robot(use_camera=True)
    robot._gripper.camera_feature.return_value = (8, 6, 3)
    expected_motors = {f"{j}.pos" for j in _ARM_JOINTS} | {"gripper.pos"}
    assert set(robot.action_features) == expected_motors
    obs_keys = set(robot.observation_features)
    assert expected_motors <= obs_keys
    assert "gripette" in obs_keys


def test_send_action_routes_arm_and_gripper():
    robot, _ = _make_robot()
    robot.arm = MagicMock()
    robot.arm.send_action.return_value = {"joint_1.pos": 10.0}
    robot._gripper.send_goal.return_value = 0.3
    out = robot.send_action({"joint_1.pos": 10.0, "gripper.pos": 0.3})
    robot.arm.send_action.assert_called_once_with({"joint_1.pos": 10.0})
    robot._gripper.send_goal.assert_called_once_with(0.3)
    assert out["joint_1.pos"] == 10.0
    assert out["gripper.pos"] == 0.3


def test_get_observation_merges():
    robot, _ = _make_robot()
    robot.arm = MagicMock()
    robot.arm.get_observation.return_value = {"joint_1.pos": 5.0}
    robot._gripper.read_observation.return_value = {"gripper.pos": 0.2, "gripette": "IMG"}
    obs = robot.get_observation()
    assert obs["joint_1.pos"] == 5.0
    assert obs["gripper.pos"] == 0.2
    assert obs["gripette"] == "IMG"


def test_connect_disconnect_delegate():
    robot, _ = _make_robot()
    robot.arm = MagicMock()
    robot.arm.is_connected = True
    robot._gripper.is_connected = True
    robot.connect(calibrate=False)
    robot.arm.connect.assert_called_once_with(calibrate=False)
    robot._gripper.connect.assert_called_once()
    assert robot.is_connected
    robot.disconnect()
    robot._gripper.disconnect.assert_called_once()
    robot.arm.disconnect.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py -k "arm_bus or feature_maps or routes or merges or delegate" -v`
Expected: FAIL (ImportError for `OpenArmGripetteFollower`).

- [ ] **Step 3: Write the robot**

Create `src/lerobot/robots/openarm_gripette_follower/openarm_gripette_follower.py`:

```python
#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import fields
from functools import cached_property

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected

from ..openarm_follower import OpenArmFollower, OpenArmFollowerConfig, OpenArmFollowerConfigBase
from ..robot import Robot
from .config_openarm_gripette_follower import OpenArmGripetteFollowerConfig
from .gripette_gripper import GripetteGripper

logger = logging.getLogger(__name__)


class OpenArmGripetteFollower(Robot):
    """7-joint OpenArm (Damiao/CAN) arm + networked Gripette gripper (gRPC)."""

    config_class = OpenArmGripetteFollowerConfig
    name = "openarm_gripette_follower"

    def __init__(self, config: OpenArmGripetteFollowerConfig):
        super().__init__(config)
        self.config = config

        # Build an OpenArmFollower with the gripper removed from the CAN bus.
        base = config.arm_config
        arm_motor_config = {k: v for k, v in base.motor_config.items() if k != "gripper"}
        arm_cfg = OpenArmFollowerConfig(
            id=f"{config.id}_arm" if config.id else None,
            calibration_dir=config.calibration_dir,
            **{f.name: getattr(base, f.name) for f in fields(OpenArmFollowerConfigBase)},
        )
        arm_cfg.motor_config = arm_motor_config
        self.arm = OpenArmFollower(arm_cfg)

        self._gripper = GripetteGripper(
            addr=config.gripette_addr,
            motor_signs=config.gripper_motor_signs,
            use_camera=config.use_gripper_camera,
            camera_name=config.gripper_camera_name,
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self.arm.action_features, "gripper.pos": float}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {**self.arm.observation_features, "gripper.pos": float}
        cam = self._gripper.camera_feature()
        if cam is not None:
            features[self.config.gripper_camera_name] = cam
        return features

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and self._gripper.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.arm.connect(calibrate=calibrate)
        self._gripper.connect()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated

    def calibrate(self) -> None:
        self.arm.calibrate()

    def configure(self) -> None:
        self.arm.configure()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: RobotObservation = dict(self.arm.get_observation())
        obs.update(self._gripper.read_observation())
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        arm_action = {k: v for k, v in action.items() if k != "gripper.pos"}
        sent: RobotAction = dict(self.arm.send_action(arm_action))
        if "gripper.pos" in action:
            sent["gripper.pos"] = self._gripper.send_goal(action["gripper.pos"])
        return sent

    def disconnect(self) -> None:
        self._gripper.disconnect(disable_torque=self.config.arm_config.disable_torque_on_disconnect)
        self.arm.disconnect()
        logger.info(f"{self} disconnected.")
```

Commit the package `__init__.py` from Task 2 now (it imports this class).

- [ ] **Step 4: Revert the Task-2 test import to the package path**

In `tests/robots/test_openarm_gripette_follower.py`, the first test should import from the package (now that `__init__` resolves):

```python
from lerobot.robots.openarm_gripette_follower import OpenArmGripetteFollowerConfig
```

- [ ] **Step 5: Run the full test file**

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py -v`
Expected: PASS (all tests: config, wrapper x3, robot x5).

- [ ] **Step 6: Lint + mypy the new package**

Run: `ruff check src/lerobot/robots/openarm_gripette_follower/`
Run: `mypy src/lerobot/robots/openarm_gripette_follower/` (expect: Success; fix any `assert ... is not None` / typing issues inline like the existing motorbridge bus does)

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/robots/openarm_gripette_follower/openarm_gripette_follower.py \
        src/lerobot/robots/openarm_gripette_follower/__init__.py \
        tests/robots/test_openarm_gripette_follower.py
git commit -m "feat(openarm-gripette): add OpenArmGripetteFollower composite robot"
```

---

### Task 5: CLI registration wiring

**Files:**
- Modify: `src/lerobot/robots/utils.py:64-72` (add dispatch branch)
- Modify (add `openarm_gripette_follower,` to each `from lerobot.robots import (...)` block): `src/lerobot/scripts/lerobot_record.py`, `lerobot_teleoperate.py`, `lerobot_calibrate.py`, `lerobot_replay.py`, `lerobot_rollout.py`, `lerobot_find_joint_limits.py`
- Test: `tests/robots/test_openarm_gripette_follower.py` (add dispatch case)

**Interfaces:**
- Consumes: `make_robot_from_config`, `OpenArmGripetteFollowerConfig`.
- Produces: `make_robot_from_config(OpenArmGripetteFollowerConfig(...)) -> OpenArmGripetteFollower`.

- [ ] **Step 1: Write the failing test**

Append to `tests/robots/test_openarm_gripette_follower.py`:

```python
def test_make_robot_from_config_dispatches():
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
    from lerobot.robots.openarm_gripette_follower import (
        OpenArmGripetteFollower,
        OpenArmGripetteFollowerConfig,
    )

    cfg = OpenArmGripetteFollowerConfig(arm_config=OpenArmFollowerConfigBase(port="can0"))
    with (
        patch(f"{_ARM}.require_package", lambda *a, **k: None),
        patch(f"{_PKG}.GripetteGripper"),
    ):
        robot = make_robot_from_config(cfg)
    assert isinstance(robot, OpenArmGripetteFollower)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py::test_make_robot_from_config_dispatches -v`
Expected: FAIL (falls through to `make_device_from_device_class` → ValueError).

- [ ] **Step 3: Add the dispatch branch**

In `src/lerobot/robots/utils.py`, after the `bi_openarm_follower` branch (≈72), add:

```python
    elif config.type == "openarm_gripette_follower":
        from .openarm_gripette_follower import OpenArmGripetteFollower

        return OpenArmGripetteFollower(config)
```

- [ ] **Step 4: Register for the CLI parsers**

In each of the six scripts listed above, add `openarm_gripette_follower,` to the `from lerobot.robots import (` block (alphabetically next to `openarm_follower`). Example for `lerobot_record.py`:

```python
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    openarm_gripette_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
```

- [ ] **Step 5: Run the dispatch test + full file**

Run: `python -m pytest tests/robots/test_openarm_gripette_follower.py -v`
Expected: PASS (all).

- [ ] **Step 6: Smoke-test CLI registration**

Run: `python -c "from lerobot.robots import openarm_gripette_follower; from lerobot.robots.config import RobotConfig; assert 'openarm_gripette_follower' in RobotConfig.get_known_choices(); print('registered')"`
Expected: prints `registered`. (If `get_known_choices` is unavailable in this draccus version, instead assert `OpenArmGripetteFollowerConfig(...).type == "openarm_gripette_follower"`.)

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/robots/utils.py src/lerobot/scripts/*.py tests/robots/test_openarm_gripette_follower.py
git commit -m "feat(openarm-gripette): register openarm_gripette_follower for the CLI"
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/source/openarm.mdx` (add a section after "Bimanual Teleoperation" / before "Recording Data")

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the Gripette section**

Insert into `docs/source/openarm.mdx`:

```markdown
### OpenArm + Gripette gripper

`openarm_gripette_follower` drives a 7-joint OpenArm arm (Damiao/CAN) together with
Pollen's [Gripette](https://github.com/pollen-robotics/grabette/tree/develop/packages/gripette)
gripper, which runs its own gRPC server (on a Raspberry Pi) and streams a camera.

The gripper is **not** on the CAN bus: install the Gripette client and point LeRobot
at the running gripper server.

```bash
# Gripette client (not on PyPI):
pip install "gripette @ git+https://github.com/pollen-robotics/grabette.git#subdirectory=packages/gripette"
# gRPC transport:
pip install -e ".[gripette]"
```

```bash
lerobot-teleoperate \
    --robot.type=openarm_gripette_follower \
    --robot.arm_config.port=can0 \
    --robot.arm_config.side=right \
    --robot.gripette_addr=192.168.1.42:50051 \
    --robot.id=my_openarm_gripette \
    --teleop.type=openarm_mini \
    --teleop.port=/dev/tty.usbmodemXXXX \
    --teleop.side=right \
    --teleop.id=my_mini_right
```

The gripper is exposed as a single `gripper.pos` in radians; the Gripette's 10 Hz
camera appears as the `gripette` observation (toggle with `--robot.use_gripper_camera`).
Flip `--robot.gripper_motor_signs "(1.0, -1.0)"` if the two servos move the wrong way.
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/openarm.mdx
git commit -m "docs(openarm): document openarm_gripette_follower"
```

---

## Self-Review

**Spec coverage:**
- gRPC transport via `GripperClient` → Task 3. ✓
- Dedicated single-arm follower (compose, stock `OpenArmFollower` untouched) → Task 4. ✓
- Single `gripper.pos` in radians → Tasks 3–4. ✓
- Camera from `StreamState` (background thread, JPEG decode) → Task 3. ✓
- External guarded `gripette` dep + PyPI caveat (manual install, grpcio extra) → Tasks 1, 6. ✓
- No `gripper.load`, no 2-DOF, no bimanual/leader → honored (out of scope). ✓
- `gripper_motor_signs` default `(1.0,-1.0)`, tunable → Task 2, surfaced in Task 6. ✓
- CLI registration → Task 5. ✓
- Hardware-free mocked tests → Tasks 2–5. ✓
- Docs → Task 6. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands include expected output.

**Type consistency:** `GripetteGripper` method names (`connect`, `is_connected`, `send_goal`, `read_observation`, `camera_feature`, `disconnect`, `_store`) are used identically in Tasks 3 and 4. `OpenArmGripetteFollowerConfig` field names (`arm_config`, `gripette_addr`, `gripper_motor_signs`, `use_gripper_camera`, `gripper_camera_name`) are consistent across Tasks 2, 4, 5, 6. The robot attribute names `self.arm` and `self._gripper` match the tests in Tasks 4–5.
```
