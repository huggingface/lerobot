# src/lerobot/robots/bi_piper_follower/bi_piper_follower.py
#!/usr/bin/env python

import logging
from contextlib import suppress
from functools import cached_property
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.robots.bi_piper_follower.config_bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.piper.piper_sdk_interface import PiperSDKInterface
from lerobot.cameras.utils import make_cameras_from_configs

logger = logging.getLogger(__name__)


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",  # la nueva junta (equivale a la que antes era 0)
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _leader_order_to_positions(action: dict[str, float], prefix: str) -> list[float]:
    # Devuelve la lista de 7 valores en el orden de joints de Piper (0..5 + gripper)
    vals = []
    for name in JOINT_NAMES:
        key = f"{prefix}{name}.pos"
        vals.append(action.get(key, 0.0))
    return vals


def _sdk_status_to_leader_keys(status: dict[str, Any], prefix: str) -> dict[str, float]:
    # status: {'joint_0.pos', ..., 'joint_6.pos'} → mapea a leader keys
    # Mapeo 1:1 de índice Piper → nombre semantic leader
    out = {}
    for i, name in enumerate(JOINT_NAMES):
        joint_key = f"joint_{i}.pos"
        out[f"{prefix}{name}.pos"] = float(status.get(joint_key, 0.0))
    return out


class BiPiperFollower(Robot):
    """
    Bimanual Piper follower: 2 brazos Piper (izquierdo y derecho).
    Consume acciones 6DOF + gripper por brazo con nombres 'left_*' y 'right_*'
    en el mismo orden semántico que el leader (SO100PiperLeader).
    """

    config_class = BiPiperFollowerConfig
    name = "bi_piper_follower"

    def __init__(self, config: BiPiperFollowerConfig):
        super().__init__(config)
        self.config = config

        # Inicializa dos SDK (puertos CAN distintos)
        self.left_sdk = PiperSDKInterface(port=config.left_port)
        self.right_sdk = PiperSDKInterface(port=config.right_port)

        self.cameras = make_cameras_from_configs(config.cameras or {})

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Llaves que espera este follower, compatibles con el teleoperador bimanual
        left = {f"left_{name}.pos": float for name in JOINT_NAMES}
        right = {f"right_{name}.pos": float for name in JOINT_NAMES}
        return left | right

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Reportamos observaciones motor por motor con las mismas llaves que usamos en action_features
        # (más las cámaras si existieran)
        motors = {k: float for k in self.action_features}
        cams = {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}
        return motors | cams

    @property
    def is_connected(self) -> bool:
        # El SDK de Piper se considera conectado tras inicializar; además, revisa cámaras si hay
        return True and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:  # calibrate no aplica aquí; se mantiene firma
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        # Lee estado de ambos SDK y mapea a llaves 'left_*' / 'right_*'
        obs = {}
        left_status = self.left_sdk.get_status()
        right_status = self.right_sdk.get_status()
        obs.update(_sdk_status_to_leader_keys(left_status, "left_"))
        obs.update(_sdk_status_to_leader_keys(right_status, "right_"))

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Construye vectores en orden Piper y los envía a cada brazo
        left_positions = _leader_order_to_positions(action, "left_")
        right_positions = _leader_order_to_positions(action, "right_")

        # Nota: PiperSDKInterface ya maneja inversiones/escala por dentro
        self.left_sdk.set_joint_positions(left_positions)
        self.right_sdk.set_joint_positions(right_positions)

        return action

    def disconnect(self) -> None:
        # Manda a detener y cierra SDK y cámaras
        with suppress(Exception):
            self.left_sdk.disconnect()
        with suppress(Exception):
            self.right_sdk.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)