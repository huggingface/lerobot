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

"""Prepare two identified YAM SocketCAN adapters before starting a controller.

Only down interfaces are configured/enabled. No motor packets are sent. Run
with all robot controllers stopped; this is startup setup, not fault recovery.
"""

import argparse
import json
import os
import shutil
import subprocess
from typing import Any

from lerobot.robots.bi_yam_follower.config_bi_yam_follower import YamArmConfig
from lerobot.robots.bi_yam_follower.yam_arm import verify_adapter


def _link(ip: str, port: str) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output([ip, "-json", "-details", "link", "show", "dev", port], text=True)
    )[0]


def _validate(info: dict[str, Any], port: str) -> None:
    link = info.get("linkinfo", {})
    if link.get("info_kind") != "can":
        raise ValueError(f"{port} is not a CAN interface")
    data = link.get("info_data", {})
    if "UP" in info["flags"] and (
        data.get("bittiming", {}).get("bitrate") != 1_000_000
        or data.get("state") != "ERROR-ACTIVE"
        or info.get("mtu") != 16
    ):
        raise ValueError(
            f"{port} is already up with an unexpected bitrate, mode, or CAN error state; not resetting it"
        )


def prepare_can(left: YamArmConfig, right: YamArmConfig) -> None:
    if left.port == right.port:
        raise ValueError("YAM requires two distinct CAN interfaces")
    ip = shutil.which("ip")
    if ip is None:
        raise RuntimeError("This helper requires Linux SocketCAN and iproute2 (ip)")
    arms = (left, right)
    # Validate both identities and existing configurations before changing either.
    for arm in arms:
        if not arm.expected_adapter_serial:
            raise ValueError("Provide both adapter serials to verify left/right assignment")
        verify_adapter(arm)
        _validate(_link(ip, arm.port), arm.port)
    for arm in arms:
        info = _link(ip, arm.port)
        _validate(info, arm.port)
        if "UP" not in info["flags"]:
            command = [ip] if os.geteuid() == 0 else ["sudo", "-n", ip]
            subprocess.run(
                [*command, "link", "set", arm.port, "type", "can", "bitrate", "1000000", "fd", "off"],
                check=True,
            )
            subprocess.run([*command, "link", "set", arm.port, "up"], check=True)
            info = _link(ip, arm.port)
            _validate(info, arm.port)
            if "UP" not in info["flags"]:
                raise RuntimeError(f"{arm.port} did not come up")
        print(f"{arm.port}: UP, classic CAN at 1 Mbit/s, adapter serial verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", default="can0")
    parser.add_argument("--right-port", default="can1")
    parser.add_argument("--left-serial", required=True)
    parser.add_argument("--right-serial", required=True)
    args = parser.parse_args()
    prepare_can(
        YamArmConfig(port=args.left_port, expected_adapter_serial=args.left_serial),
        YamArmConfig(port=args.right_port, expected_adapter_serial=args.right_serial),
    )


if __name__ == "__main__":
    main()
