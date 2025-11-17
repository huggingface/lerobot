"""
Helper to list all available USB ports and cameras in a sorted, numbered format.
Returns data in a format suitable for Rust consumption.

Example:

```shell
lerobot-auto-config
```
"""

import platform
import json
import cv2
from pathlib import Path


def find_available_ports():
    from serial.tools import list_ports  # Part of pyserial library

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def find_available_cameras():
    """Find all available camera indices."""
    available_cameras = []
    
    # Test a wider range of camera indices
    # Most systems have 0-3, but some might have more
    for i in range(12):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except Exception:
            # Skip any problematic camera indices
            continue
    
    return available_cameras


def get_config_data():
    """Return all available ports and cameras as a sorted list."""
    ports = sorted(find_available_ports())
    cameras = sorted(find_available_cameras())
    
    return {
        "ports": ports,
        "cameras": cameras
    }


def main():
    # For Rust consumption, output as JSON
    config_data = get_config_data()
    print(json.dumps(config_data))


if __name__ == "__main__":
    main()
