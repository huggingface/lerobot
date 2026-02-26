"""Calibration service for checking and managing device calibrations."""

from pathlib import Path
from typing import List, Optional

from lerobot.webui.backend.models.config import Config
from lerobot.webui.backend.models.system import CalibrationStatus


class CalibrationService:
    """Service for managing device calibrations."""

    def __init__(self):
        """Initialize CalibrationService."""
        # Calibration files are stored in ~/.cache/huggingface/lerobot/calibration/
        self.calibration_base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"

    def check_calibration(
        self, device_type: str, device_id: str, robot_type: str, port: Optional[str] = None
    ) -> CalibrationStatus:
        """Check if calibration file exists for a device.

        Args:
            device_type: "robot" or "teleoperator".
            device_id: Device identifier (e.g., "left_follower", "bimanual_leader").
            robot_type: Robot/teleoperator type (e.g., "so101_follower", "bi_so101_leader").
            port: Optional device port.

        Returns:
            CalibrationStatus object.
        """
        # Path format: ~/.cache/huggingface/lerobot/calibration/{robots|teleoperators}/{type}/{id}.json
        category = "robots" if device_type == "robot" else "teleoperators"
        calibration_path = self.calibration_base_path / category / robot_type / f"{device_id}.json"

        is_calibrated = calibration_path.exists()

        return CalibrationStatus(
            device_type=device_type,
            device_id=device_id,
            robot_type=robot_type,
            port=port,
            is_calibrated=is_calibrated,
            calibration_path=str(calibration_path) if is_calibrated else None,
        )

    def list_missing_calibrations(self, config: Config) -> List[CalibrationStatus]:
        """List all devices that need calibration based on config.

        Args:
            config: Configuration object.

        Returns:
            List of CalibrationStatus for devices missing calibration.
        """
        missing = []

        if config.mode == "bimanual":
            # Check bimanual devices
            devices = [
                ("robot", "bimanual_follower", "bi_so101_follower", None),  # Bimanual follower doesn't need port
                ("teleoperator", "bimanual_leader", "bi_so101_leader", None),
            ]

            for device_type, device_id, robot_type, port in devices:
                status = self.check_calibration(device_type, device_id, robot_type, port)
                if not status.is_calibrated:
                    missing.append(status)

        else:
            # Check single arm devices
            devices = [
                ("robot", "single_follower", "so101_follower", config.single_arm.follower_port),
                ("teleoperator", "single_leader", "so101_leader", config.single_arm.leader_port),
            ]

            for device_type, device_id, robot_type, port in devices:
                if port:  # Only check if port is configured
                    status = self.check_calibration(device_type, device_id, robot_type, port)
                    if not status.is_calibrated:
                        missing.append(status)

        return missing

    def list_calibration_files(self, category: str, robot_type: str) -> List[str]:
        """List available calibration files for a given category and robot type.

        Args:
            category: "robots" or "teleoperators".
            robot_type: Robot type (e.g., "so101_follower", "bi_so101_follower").

        Returns:
            List of calibration filenames (e.g., ["left_follower.json", "right_follower.json"]).
        """
        cal_dir = self.calibration_base_path / category / robot_type
        if not cal_dir.exists():
            return []
        return sorted([f.name for f in cal_dir.glob("*.json")])

    def build_calibration_command(
        self, device_type: str, device_id: str, robot_type: str, port: str
    ) -> List[str]:
        """Build calibration command for a device.

        Args:
            device_type: "robot" or "teleoperator".
            device_id: Device identifier.
            robot_type: Robot/teleoperator type.
            port: Device port.

        Returns:
            Command as list of strings.
        """
        # lerobot-calibrate uses --robot.* or --teleop.* flags
        prefix = "robot" if device_type == "robot" else "teleop"

        return [
            "lerobot-calibrate",
            f"--{prefix}.type={robot_type}",
            f"--{prefix}.port={port}",
            f"--{prefix}.id={device_id}",
        ]
