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

"""Control-plane (device discovery) tests."""

import pytest

from lerobot.robots.webrtc_proxy.control import ControlServer, FindPortError, SyntheticInventory
from lerobot.robots.webrtc_proxy.protocol import RpcRequest

aiortc = pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")


# ---- pure server logic (no transport) -------------------------------------
def test_find_port_diff_logic():
    inv = SyntheticInventory(ports=["/dev/a", "/dev/b", "/dev/c"])
    srv = ControlServer(inv)
    srv._dispatch(RpcRequest(1, "find_port_begin", {}))
    inv.simulate_unplug("/dev/b")
    assert srv._dispatch(RpcRequest(2, "find_port_result", {})) == {"port": "/dev/b"}


def test_find_port_ambiguous_raises():
    srv = ControlServer(SyntheticInventory(ports=["/dev/a", "/dev/b"]))
    srv._dispatch(RpcRequest(1, "find_port_begin", {}))
    with pytest.raises(FindPortError):
        srv._dispatch(RpcRequest(2, "find_port_result", {}))


def test_set_camera_plan_invokes_callback():
    seen = {}
    srv = ControlServer(SyntheticInventory(), on_camera_plan=seen.update)
    srv._dispatch(RpcRequest(1, "set_camera_plan", {"width": 320, "height": 240, "fps": 30}))
    assert seen == {"width": 320, "height": 240, "fps": 30}


def test_local_inventory_lists_real_ports():
    pytest.importorskip("serial", reason="find_available_ports needs pyserial (lerobot[hardware])")
    from lerobot.robots.webrtc_proxy.control import LocalDeviceInventory

    ports = LocalDeviceInventory().list_ports()
    assert isinstance(ports, list)
    assert all(isinstance(p, str) for p in ports)


# ---- end-to-end over the real control channel -----------------------------
def test_discovery_rpc_over_link(webrtc_link):
    inv = SyntheticInventory(
        ports=["/dev/tty.usbmodem-A", "/dev/tty.usbmodem-B"],
        cameras=[{"type": "opencv", "id": 0, "name": "front cam"}],
    )
    with webrtc_link(inventory=inv) as link:
        robot = link.robot
        assert set(robot.list_ports()) == {"/dev/tty.usbmodem-A", "/dev/tty.usbmodem-B"}
        assert robot.list_cameras() == [{"type": "opencv", "id": 0, "name": "front cam"}]

        before = robot.find_port_begin()
        assert "/dev/tty.usbmodem-B" in before
        inv.simulate_unplug("/dev/tty.usbmodem-B")  # the test holds the daemon's inventory
        assert robot.find_port_result() == "/dev/tty.usbmodem-B"


def test_real_inventory_over_link(webrtc_link):
    pytest.importorskip("serial", reason="find_available_ports needs pyserial (lerobot[hardware])")
    from lerobot.robots.webrtc_proxy.control import LocalDeviceInventory

    with webrtc_link(inventory=LocalDeviceInventory()) as link:
        ports = link.robot.list_ports()
        assert isinstance(ports, list) and all(isinstance(p, str) for p in ports)


def test_control_rpc_error_propagates(webrtc_link):
    with webrtc_link(inventory=SyntheticInventory()) as link, pytest.raises(RuntimeError):
        link.robot.find_port_result()  # before begin -> server raises -> RuntimeError on cloud
