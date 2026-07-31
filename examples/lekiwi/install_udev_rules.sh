#!/usr/bin/env bash
#
# Install stable /dev symlinks for the LeKiwi hardware via udev.
#
# Why this exists:
#   - The Orbbec Gemini 335L exposes two UVC interfaces whose color and depth nodes both report
#     V4L index 0, so /dev/v4l/by-id collides and /dev/videoN shuffles on replug. We pin the RGB
#     color node by matching its USB interface number (04) + capture node (index 0).
#   - The Sonix wrist cam has a single interface, so vendor+product + index 0 is unique.
#   - The Feetech motor bus (WCH USB-serial adapter) enumerates as /dev/ttyACM0, but that number
#     is not guaranteed; we pin it to /dev/lekiwi_bus.
#
# Result (stable names, survive replug/reboot/port changes):
#   /dev/lekiwi_orbbec  -> Orbbec Gemini 335L RGB color stream
#   /dev/lekiwi_wrist   -> Sonix USB2.0_CAM1 wrist camera
#   /dev/lekiwi_bus     -> Feetech motor bus serial port
#
# Usage:
#   bash examples/lekiwi/install_udev_rules.sh          # uses sudo internally
#   sudo bash examples/lekiwi/install_udev_rules.sh     # if already root
#
# NOTE ON REPRODUCIBILITY / OTHER MACHINES:
#   The camera rules match by USB vendor/product/interface, which are model-specific (they work on
#   any identical camera). The motor-bus rule matches only the WCH vendor:product (1a86:55d3); this
#   is unique on a LeKiwi host where the follower bus is the only WCH adapter. If you have MORE than
#   one WCH adapter on the same machine (e.g. a leader arm plugged into the same host), pin it by its
#   serial instead: find it with
#       udevadm info -q property -n /dev/ttyACM0 | grep ID_SERIAL_SHORT
#   then add  ATTRS{serial}=="<SERIAL>"  to the motor-bus rule below.

set -euo pipefail

RULES_PATH="/etc/udev/rules.d/99-lekiwi-cameras.rules"

SUDO=""
if [[ $EUID -ne 0 ]]; then
    SUDO="sudo"
fi

echo "Writing udev rules to ${RULES_PATH} ..."
${SUDO} tee "${RULES_PATH}" >/dev/null <<'EOF'
# LeKiwi stable device symlinks. Managed by examples/lekiwi/install_udev_rules.sh.
#
# udev requires all ATTRS{} in one rule to match the SAME parent device, so we cannot mix the
# interface attr (bInterfaceNumber) with usb-device attrs (idVendor/serial).

# Orbbec Gemini 335L RGB color: interface 04 + primary capture node (index 0) uniquely selects it.
SUBSYSTEM=="video4linux", ATTR{index}=="0", ATTRS{bInterfaceNumber}=="04", SYMLINK+="lekiwi_orbbec"

# Sonix USB2.0_CAM1 wrist: single interface, so vendor+product + index 0 is unique.
SUBSYSTEM=="video4linux", ATTR{index}=="0", ATTRS{idVendor}=="05a3", ATTRS{idProduct}=="9230", SYMLINK+="lekiwi_wrist"

# Feetech motor bus (WCH USB-serial adapter). Pin the serial port to a stable name.
# If multiple WCH adapters share this host, also match ATTRS{serial}=="<your-serial>".
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", SYMLINK+="lekiwi_bus"
EOF

echo "Reloading udev rules and re-triggering ..."
${SUDO} udevadm control --reload-rules
${SUDO} udevadm trigger --subsystem-match=video4linux --subsystem-match=tty
${SUDO} udevadm settle

echo
echo "Done. Current LeKiwi symlinks (missing ones just mean that device isn't plugged in):"
ls -l /dev/lekiwi_orbbec /dev/lekiwi_wrist /dev/lekiwi_bus 2>&1 || true
