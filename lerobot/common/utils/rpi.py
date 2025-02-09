import platform
import subprocess
import traceback
from functools import cache
from typing import Tuple
from warnings import warn


@cache
def is_raspberry_pi() -> bool:
    """
    Check that the system is running on a raspberry pi and is running Bookworm or later
    """
    is_rpi_os = True
    os_name, os_version, os_machine = get_system_info()
    if os_name == "Linux" and os_machine in ["armv6l", "armv7l", "aarch64"]:
        if os_version != "Unknown":
            if os_version.isdecimal() and int(os_version) < 12:
                warn("This code has not been tested RaspberryPi OS older than Bookworm", stacklevel=1)
        else:
            is_rpi_os = False
    else:
        is_rpi_os = False
    return is_rpi_os


@cache
def get_system_info() -> Tuple[str, str, str]:
    os_name = platform.system()
    os_version = get_os_version()
    os_machine = platform.machine()
    return os_name, os_version, os_machine


@cache
def get_os_version() -> str:
    try:
        # This command works on systems where `cat` and `/etc/os-release` are available
        output = subprocess.check_output(["cat", "/etc/os-release"]).decode("utf-8")
        for line in output.split("\n"):
            if line.startswith("VERSION_ID="):
                version = line.split("=")[1].strip('"')
                return version
    except Exception as e:
        print(f"Failed to retrieve OS version: {e}")
        return "Unknown"


@cache
def check_csi_cam():
    csi_cam_available = True
    try:
        from picamera2 import Picamera2
    except Exception:
        print(
            "Error trying to import picamera2. Either the module has not been "
            "installed or external dependencies have not been installed properly. "
            "Check RPI_SETUP.md for the full installation procedure."
        )
        traceback.print_exc()
        print()
        csi_cam_available = False
    tmp_pi_cam = Picamera2()
    if tmp_pi_cam.is_open:
        tmp_pi_cam.close()
    else:
        csi_cam_available = False
    del tmp_pi_cam
    return csi_cam_available
