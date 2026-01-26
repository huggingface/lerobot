# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Setup and debug CAN interfaces for Damiao motors (e.g., OpenArms).

Examples:

Setup CAN interfaces with CAN FD:
```shell
lerobot-setup-can --mode=setup --interfaces=can0,can1,can2,can3
```

Test motors on a single interface:
```shell
lerobot-setup-can --mode=test --interfaces=can0
```

Test motors on all interfaces:
```shell
lerobot-setup-can --mode=test --interfaces=can0,can1,can2,can3
```

Speed test:
```shell
lerobot-setup-can --mode=speed --interfaces=can0
```
"""

import subprocess
import sys
import time
from dataclasses import dataclass, field

import draccus

from lerobot.utils.import_utils import is_package_available

MOTOR_NAMES = {
    0x01: "joint_1",
    0x02: "joint_2",
    0x03: "joint_3",
    0x04: "joint_4",
    0x05: "joint_5",
    0x06: "joint_6",
    0x07: "joint_7",
    0x08: "gripper",
}


@dataclass
class CANSetupConfig:
    mode: str = "test"
    interfaces: str = "can0"  # Comma-separated, e.g. "can0,can1,can2,can3"
    bitrate: int = 1000000
    data_bitrate: int = 5000000
    use_fd: bool = True
    motor_ids: list[int] = field(default_factory=lambda: list(range(0x01, 0x09)))
    timeout: float = 1.0
    speed_iterations: int = 100

    def get_interfaces(self) -> list[str]:
        return [i.strip() for i in self.interfaces.split(",") if i.strip()]


def check_interface_status(interface: str) -> tuple[bool, str, bool]:
    """Check if CAN interface is UP and configured."""
    try:
        result = subprocess.run(["ip", "link", "show", interface], capture_output=True, text=True)  # nosec B607
        if result.returncode != 0:
            return False, "Interface not found", False

        output = result.stdout
        is_up = "UP" in output
        is_fd = "fd on" in output.lower() or "canfd" in output.lower()
        status = "UP" if is_up else "DOWN"
        if is_fd:
            status += " (CAN FD)"

        return is_up, status, is_fd
    except FileNotFoundError:
        return False, "ip command not found", False


def setup_interface(interface: str, bitrate: int, data_bitrate: int, use_fd: bool) -> bool:
    """Configure a CAN interface."""
    try:
        subprocess.run(["sudo", "ip", "link", "set", interface, "down"], check=False, capture_output=True)  # nosec B607

        cmd = ["sudo", "ip", "link", "set", interface, "type", "can", "bitrate", str(bitrate)]
        if use_fd:
            cmd.extend(["dbitrate", str(data_bitrate), "fd", "on"])

        result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B607
        if result.returncode != 0:
            print(f"  ✗ Failed to configure: {result.stderr}")
            return False

        result = subprocess.run(  # nosec B607
            ["sudo", "ip", "link", "set", interface, "up"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ✗ Failed to bring up: {result.stderr}")
            return False

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_motor(bus, motor_id: int, timeout: float, use_fd: bool):
    """Test a single motor and return responses."""
    import can

    enable_msg = can.Message(
        arbitration_id=motor_id,
        data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC],
        is_extended_id=False,
        is_fd=use_fd,
    )

    try:
        bus.send(enable_msg)
    except Exception as e:
        return None, f"Send error: {e}"

    responses = []
    start_time = time.time()

    while time.time() - start_time < timeout:
        msg = bus.recv(timeout=0.1)
        if msg:
            responses.append((msg.arbitration_id, msg.data.hex(), getattr(msg, "is_fd", False)))

    disable_msg = can.Message(
        arbitration_id=motor_id,
        data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD],
        is_extended_id=False,
        is_fd=use_fd,
    )
    try:
        bus.send(disable_msg)
    except Exception:
        print(f"Error sending message to motor 0x{motor_id:02X}")

    return responses, None


def test_interface(cfg: CANSetupConfig, interface: str):
    """Test all motors on a CAN interface."""
    import can

    is_up, status, _ = check_interface_status(interface)
    print(f"\n{interface}: {status}")

    if not is_up:
        print(f"  ⚠ Interface is not UP. Run: lerobot-setup-can --mode=setup --interfaces {interface}")
        return {}

    try:
        kwargs = {"channel": interface, "interface": "socketcan", "bitrate": cfg.bitrate}
        if cfg.use_fd:
            kwargs.update({"data_bitrate": cfg.data_bitrate, "fd": True})
        bus = can.interface.Bus(**kwargs)
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return {}

    results = {}
    try:
        while bus.recv(timeout=0.01):
            pass

        for motor_id in cfg.motor_ids:
            motor_name = MOTOR_NAMES.get(motor_id, f"motor_0x{motor_id:02X}")
            responses, error = test_motor(bus, motor_id, cfg.timeout, cfg.use_fd)

            if error:
                print(f"  Motor 0x{motor_id:02X} ({motor_name}): ✗ {error}")
                results[motor_id] = {"found": False, "error": error}
            elif responses:
                print(f"  Motor 0x{motor_id:02X} ({motor_name}): ✓ FOUND")
                for resp_id, data, is_fd in responses:
                    fd_flag = " [FD]" if is_fd else ""
                    print(f"    → Response 0x{resp_id:02X}{fd_flag}: {data}")
                results[motor_id] = {"found": True, "responses": responses}
            else:
                print(f"  Motor 0x{motor_id:02X} ({motor_name}): ✗ No response")
                results[motor_id] = {"found": False}

            time.sleep(0.05)
    finally:
        bus.shutdown()

    found = sum(1 for r in results.values() if r.get("found"))
    print(f"\n  Summary: {found}/{len(cfg.motor_ids)} motors found")
    return results


def speed_test(cfg: CANSetupConfig, interface: str):
    """Test communication speed with motors."""
    import can

    is_up, status, _ = check_interface_status(interface)
    if not is_up:
        print(f"{interface}: {status} - skipping")
        return

    print(f"\n{interface}: Running speed test ({cfg.speed_iterations} iterations)...")

    try:
        kwargs = {"channel": interface, "interface": "socketcan", "bitrate": cfg.bitrate}
        if cfg.use_fd:
            kwargs.update({"data_bitrate": cfg.data_bitrate, "fd": True})
        bus = can.interface.Bus(**kwargs)
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return

    responding_motor = None
    for motor_id in cfg.motor_ids:
        responses, _ = test_motor(bus, motor_id, 0.5, cfg.use_fd)
        if responses:
            responding_motor = motor_id
            break

    if not responding_motor:
        print("  ✗ No responding motors found")
        bus.shutdown()
        return

    print(f"  Testing with motor 0x{responding_motor:02X}...")
    latencies = []

    for _ in range(cfg.speed_iterations):
        start = time.perf_counter()
        msg = can.Message(
            arbitration_id=responding_motor,
            data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC],
            is_extended_id=False,
            is_fd=cfg.use_fd,
        )
        bus.send(msg)
        resp = bus.recv(timeout=0.1)
        if resp:
            latencies.append((time.perf_counter() - start) * 1000)

    bus.shutdown()

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        hz = 1000.0 / avg_latency if avg_latency > 0 else 0
        print(f"  ✓ Success rate: {len(latencies)}/{cfg.speed_iterations}")
        print(f"  ✓ Avg latency: {avg_latency:.2f} ms")
        print(f"  ✓ Max frequency: {hz:.1f} Hz")
    else:
        print("  ✗ No successful responses")


def run_setup(cfg: CANSetupConfig):
    """Setup CAN interfaces."""
    print("=" * 50)
    print("CAN Interface Setup")
    print("=" * 50)
    print(f"Mode: {'CAN FD' if cfg.use_fd else 'CAN 2.0'}")
    print(f"Bitrate: {cfg.bitrate / 1_000_000:.1f} Mbps")
    if cfg.use_fd:
        print(f"Data bitrate: {cfg.data_bitrate / 1_000_000:.1f} Mbps")
    print()

    interfaces = cfg.get_interfaces()
    for interface in interfaces:
        print(f"Configuring {interface}...")
        if setup_interface(interface, cfg.bitrate, cfg.data_bitrate, cfg.use_fd):
            is_up, status, _ = check_interface_status(interface)
            print(f"  ✓ {interface}: {status}")
        else:
            print(f"  ✗ {interface}: Failed")

    print("\nSetup complete!")
    print("\nNext: Test motors with:")
    print(f"  lerobot-setup-can --mode=test --interfaces {','.join(interfaces)}")


def run_test(cfg: CANSetupConfig):
    """Test motors on CAN interfaces."""
    print("=" * 50)
    print("CAN Motor Test")
    print("=" * 50)
    print(f"Testing motors 0x{min(cfg.motor_ids):02X}-0x{max(cfg.motor_ids):02X}")
    print(f"Mode: {'CAN FD' if cfg.use_fd else 'CAN 2.0'}")
    print()

    interfaces = cfg.get_interfaces()
    all_results = {}
    for interface in interfaces:
        all_results[interface] = test_interface(cfg, interface)

    total_found = sum(sum(1 for r in res.values() if r.get("found")) for res in all_results.values())

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total motors found: {total_found}")

    if total_found == 0:
        print("\n⚠ No motors found! Check:")
        print("  1. Motors are powered (24V)")
        print("  2. CAN wiring (CANH, CANL, GND)")
        print("  3. Motor timeout parameter > 0 (use Damiao tools)")
        print("  4. 120Ω termination at both cable ends")
        print(f"  5. Interface configured: lerobot-setup-can --mode=setup --interfaces {interfaces[0]}")


def run_speed(cfg: CANSetupConfig):
    """Run speed tests on CAN interfaces."""
    print("=" * 50)
    print("CAN Speed Test")
    print("=" * 50)

    for interface in cfg.get_interfaces():
        speed_test(cfg, interface)


@draccus.wrap()
def setup_can(cfg: CANSetupConfig):
    if not is_package_available("can"):
        print("Error: python-can not installed. Install with: pip install python-can")
        sys.exit(1)

    if cfg.mode == "setup":
        run_setup(cfg)
    elif cfg.mode == "test":
        run_test(cfg)
    elif cfg.mode == "speed":
        run_speed(cfg)
    else:
        print(f"Unknown mode: {cfg.mode}")
        print("Available modes: setup, test, speed")
        sys.exit(1)


def main():
    setup_can()


if __name__ == "__main__":
    main()
