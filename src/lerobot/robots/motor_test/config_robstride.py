#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Dict, Optional

from lerobot.cameras import CameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("robstride_test")
@dataclass
class RobstrideTestConfig(RobotConfig):
    """Configuration pour un robot de test avec moteurs Robstride sur bus CAN."""

    # Interface CAN (Linux: "can0", "can1", etc. / slcan si USB-sérial)
    port: str = "can0"
    can_interface: str = "socketcan"

    # CAN classique pour commencer (pas FD)
    use_can_fd: bool = False
    can_bitrate: int = 1_000_000

    # Sécurité / confort
    disable_torque_on_disconnect: bool = True

    # Limite de déplacement relatif optionnelle
    max_relative_target: Optional[float | Dict[str, float]] = None

    # Caméras (laisse vide pour le test moteur)
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)

    # Config moteur : {nom_joint: (send_can_id, recv_can_id, motor_type_str)}
    motor_config: Dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x01, "ELO5"),  # à adapter au vrai type si besoin
        }
    )

    # Gains MIT (si ton RobstrideBus en fait quelque chose)
    position_kp: list[float] = field(default_factory=lambda: [5.0])
    position_kd: list[float] = field(default_factory=lambda: [1.0])

    # Calibration
    calibration_mode: str = "manual"
    zero_position_on_connect: bool = False
