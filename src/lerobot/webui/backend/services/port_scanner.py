"""Port scanning service wrapping lerobot_find_port logic."""

from pathlib import Path
from typing import List

from lerobot.webui.backend.models.setup import PortInfo

# Feetech motor controllers on macOS show up as /dev/cu.usbmodem*
FEETECH_PORT_PREFIX = "/dev/cu.usbmodem"


class PortScannerService:
    """Service for scanning and detecting serial ports."""

    def list_ports(self) -> List[PortInfo]:
        """List available Feetech motor controller ports.

        Only returns ports matching /dev/cu.usbmodem* which are Feetech motor controllers.

        Returns:
            List of PortInfo objects.
        """
        ports = [str(p) for p in Path("/dev").glob("cu.usbmodem*")]

        return [
            PortInfo(
                port=port,
                description="Feetech Motor Controller",
                hwid=None,
            )
            for port in sorted(ports)
        ]

    def detect_port_change(self, ports_before: List[str], ports_after: List[str]) -> tuple[List[str], List[str]]:
        """Detect which ports were added or removed.

        Args:
            ports_before: List of ports before change.
            ports_after: List of ports after change.

        Returns:
            Tuple of (removed_ports, added_ports).
        """
        removed = [p for p in ports_before if p not in ports_after]
        added = [p for p in ports_after if p not in ports_before]

        return removed, added
