# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Helper to find the USB port associated with your MotorsBus.

Example:

```shell
lerobot-find-port
```

On Windows the bus shows up as a `COM<n>` port, and the machine usually exposes other
serial ports too (Bluetooth links, motherboard/virtual COM ports). Those are neither
stable nor related to the arm, and they flap in and out of existence, so only USB
serial adapters are considered when diffing the port list. Unplugging the adapter also
takes a moment to be reflected by the driver, hence the polling below.
"""

import platform
import time
from pathlib import Path

# How long to wait for the driver to publish/retire a port before giving up. Windows
# enumerates a USB serial adapter's COM port asynchronously, so a single check right
# after the cable moves often still reports the old list.
_WINDOWS_PORT_CHANGE_TIMEOUT_S = 5.0
_POSIX_PORT_CHANGE_TIMEOUT_S = 0.5
_POLL_INTERVAL_S = 0.2

# hwid markers of ports that are never a motor bus.
_NON_USB_HWID_MARKERS = ("BTHENUM", "BLUETOOTH", "RFCOMM")


def _list_port_infos():
    """Return `pyserial`'s ListPortInfo objects for every serial port on this machine."""
    from lerobot.utils.import_utils import require_package

    require_package("pyserial", extra="hardware", import_name="serial")
    from serial.tools import list_ports

    return list(list_ports.comports())


def is_usb_serial_port(port_info) -> bool:
    """Whether a `pyserial` port is a USB serial adapter (as opposed to Bluetooth/virtual)."""
    hwid = (getattr(port_info, "hwid", "") or "").upper()
    if any(marker in hwid for marker in _NON_USB_HWID_MARKERS):
        return False
    return "USB" in hwid


def describe_port(port_info) -> str:
    """One human-readable line for a port, e.g. `COM3 — USB-SERIAL CH340 (USB)`."""
    kind = "USB" if is_usb_serial_port(port_info) else "non-USB"
    description = (getattr(port_info, "description", "") or "").strip()
    return f"{port_info.device} — {description or 'unknown device'} ({kind})"


def find_available_ports():
    from lerobot.utils.import_utils import require_package

    require_package("pyserial", extra="hardware", import_name="serial")

    if platform.system() == "Windows":
        # List COM ports using pyserial
        return [port.device for port in _list_port_infos()]
    # List /dev/tty* ports for Unix-based systems
    return [str(path) for path in Path("/dev").glob("tty*")]


def _candidate_devices(port_infos, all_devices: list[str]) -> set[str]:
    """The device names to diff between "before" and "after".

    Windows exposes Bluetooth and virtual ports next to the adapter; diffing the whole
    list turns a single unplug into several changes. Prefer USB serial adapters when at
    least one is plugged in, and fall back to every port otherwise (e.g. a board behind
    an unusual virtual-COM driver).
    """
    if platform.system() != "Windows":
        return set(all_devices)
    usb_devices = {info.device for info in port_infos if is_usb_serial_port(info)}
    return usb_devices or set(all_devices)


def _print_windows_port_notes(port_infos) -> None:
    """Spell out what was found on Windows, which is where users get confused."""
    for info in port_infos:
        print(f"  - {describe_port(info)}")
    non_usb = [info for info in port_infos if not is_usb_serial_port(info)]
    if non_usb and len(non_usb) == len(port_infos):
        print(
            "  ! No USB serial adapter was found — the ports above are Bluetooth or virtual COM ports. "
            "Plug the arm's USB cable in (and install the CH340/CP210x driver if the port never appears)."
        )


def _snapshot() -> tuple[list, list[str]]:
    """Return (port infos, device names), using pyserial on Windows and `/dev` elsewhere."""
    if platform.system() == "Windows":
        port_infos = _list_port_infos()
        return port_infos, [info.device for info in port_infos]
    return [], find_available_ports()


def _wait_for_devices_to_change(before: list[str], timeout_s: float) -> set[str]:
    """Poll the port list until it differs from `before` or `timeout_s` elapses."""
    deadline = time.monotonic() + timeout_s
    before_set = set(before)
    while True:
        port_infos, devices = _snapshot()
        current = _candidate_devices(port_infos, devices)
        if current != before_set:
            return current
        if time.monotonic() >= deadline:
            return current
        time.sleep(_POLL_INTERVAL_S)


def find_port():
    is_windows = platform.system() == "Windows"
    print("Finding all available ports for the MotorsBus.")
    port_infos_before, ports_before = _snapshot()
    print("Ports before disconnecting:", ports_before)
    if is_windows:
        _print_windows_port_notes(port_infos_before)

    print("Remove the USB cable from your MotorsBus and press Enter when done.")
    input()  # Wait for user to disconnect the device

    timeout_s = _WINDOWS_PORT_CHANGE_TIMEOUT_S if is_windows else _POSIX_PORT_CHANGE_TIMEOUT_S
    ports_after = _wait_for_devices_to_change(ports_before, timeout_s)
    ports_diff = sorted(_candidate_devices(port_infos_before, ports_before) - ports_after)

    if len(ports_diff) == 1:
        port = ports_diff[0]
        description = {info.device: describe_port(info) for info in port_infos_before}.get(port, port)
        print(f"The port of this MotorsBus is '{port}'")
        if is_windows:
            print(f"  ({description})")
        print("Reconnect the USB cable.")
    elif len(ports_diff) == 0:
        message = f"Could not detect the port. No difference was found ({ports_diff})."
        if is_windows:
            message += (
                " On Windows this usually means the adapter's COM port did not change: make sure the arm "
                "is powered, that no other program holds the port (Arduino IDE, a motor debug tool, an "
                "earlier lerobot run), and check the port disappears from Device Manager when unplugged."
            )
        raise OSError(message)
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


def main():
    find_port()


if __name__ == "__main__":
    main()
