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
Safe Linux V4L2 device discovery for OpenCV cameras.

This module intentionally avoids opening /dev/video* devices. Some OpenCV/V4L2
backend combinations can change the active camera format when VideoCapture opens
a device, so default discovery only enumerates device identifiers.
"""

from pathlib import Path
from typing import Any


def _unknown_stream_profile() -> dict[str, None]:
    return {
        "format": None,
        "fourcc": None,
        "width": None,
        "height": None,
        "fps": None,
    }


def _read_device_name(sysfs_video_device: Path) -> str | None:
    name_path = sysfs_video_device / "name"
    try:
        name = name_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name or None


def _camera_info(dev_path: Path, name: str | None) -> dict[str, Any]:
    display_name = f"{name} @ {dev_path}" if name else f"OpenCV Camera @ {dev_path}"
    return {
        "name": display_name,
        "type": "OpenCV",
        "id": str(dev_path),
        "backend_api": "V4L2",
        "default_stream_profile": _unknown_stream_profile(),
    }


def find_linux_video_devices(
    sysfs_video4linux_path: Path = Path("/sys/class/video4linux"),
    dev_path: Path = Path("/dev"),
) -> list[dict[str, Any]]:
    """
    Enumerate Linux video devices without opening camera device files.

    Stream profile values are intentionally unknown because querying them through
    OpenCV opens the device and can mutate active settings on some backends.
    """
    sysfs_devices = sorted(sysfs_video4linux_path.glob("video*"), key=lambda path: path.name)
    if sysfs_devices:
        return [
            _camera_info(dev_path / sysfs_device.name, _read_device_name(sysfs_device))
            for sysfs_device in sysfs_devices
        ]

    return [_camera_info(path, None) for path in sorted(dev_path.glob("video*"), key=lambda path: path.name)]
