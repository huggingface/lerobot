# src/lerobot/teleoperators/bi_so100_piper_leader/bi_so100_piper_leader.py
#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.motors.feetech.feetech import OperatingMode
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_bi_so100_piper_leader import BiSO100PiperLeaderConfig

logger = logging.getLogger(__name__)


class SO100PiperLeader(Teleoperator):
    """
    Leader de un solo brazo con 6DOF + gripper para Piper.
    Equivale a SO-100 pero añade una junta intermedia (p.ej. 'forearm_roll') que
    antes se rellenaba con 0 cuando se controlaba Piper.
    """

    # Puedes crear también un config propio si más adelante deseas parametrizar IDs.
    # Por ahora seguimos el patrón de SO100Leader con IDs fijos 1..7.
    config_class = BiSO100PiperLeaderConfig  # se usa solo para tipado
    name = "so100_piper_leader"

    def __init__(self, port: str, *, leader_id: str | None = None, calibration: dict[str, MotorCalibration] | None = None):
        # Teleoperator core init (id/calibración):
        class _TmpCfg:
            def __init__(self, _id, _cal_dir=None):
                self.id = _id
                self.calibration_dir = _cal_dir

        super().__init__(_TmpCfg(leader_id))
        if calibration:
            self.calibration = calibration

        # 6DOF + gripper: añadimos 'forearm_roll' como 4ª junta (índice 3)
        self.bus = FeetechMotorsBus(
            port=port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "forearm_roll": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),  # NUEVA JUNTA
                "wrist_flex": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.bus.motors}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch entre calibración en motor y archivo, o no existe archivo de calibración. Iniciando calibración."
            )
            self.calibrate()

        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        # Mismo flujo que SO100Leader: offsets + rango + escritura archivo
        if self.calibration:
            user_input = input(
                f"ENTER = usar archivo de calibración de {self.id}; o escribe 'c' para recalibrar: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Escribiendo calibración del id %s a los motores", self.id)
                self.bus.write_calibration(self.calibration)
                return

        logger.info("Calibrando %s", self)
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Mueve el brazo a la mitad de su rango y presiona ENTER...")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [m for m in self.bus.motors if m != full_turn_motor]
        print("Mueve todas las juntas (menos 'wrist_roll') a través de su rango completo. ENTER para parar...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibración guardada en {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Conecta SOLO el motor '{motor}' a la controladora y presiona ENTER...")
            self.bus.setup_motor(motor)
            print(f"Motor '{motor}' configurado con id {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s read action: %.1fms", self, dt_ms)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # (pendiente si se quisiera force feedback)
        return

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.bus.disconnect()
        logger.info("%s disconnected.", self)


class BiSO100PiperLeader(Teleoperator):
    """
    Bimanual: compone 2 SO100PiperLeader (izquierdo y derecho) y expone acciones con prefijos 'left_' / 'right_'.
    """

    config_class = BiSO100PiperLeaderConfig
    name = "bi_so100_piper_leader"

    def __init__(self, config: BiSO100PiperLeaderConfig):
        super().__init__(config)
        self.config = config

        # Instanciamos dos brazos con 6DOF + gripper
        self.left_arm = SO100PiperLeader(port=config.left_arm_port, leader_id=f"{config.id}_left" if config.id else None)
        self.right_arm = SO100PiperLeader(port=config.right_arm_port, leader_id=f"{config.id}_right" if config.id else None)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{k}": float for k in self.left_arm.action_features} | {
            f"right_{k}": float for k in self.right_arm.action_features
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, Any]:
        act = {}
        left = self.left_arm.get_action()
        right = self.right_arm.get_action()
        act.update({f"left_{k}": v for k, v in left.items()})
        act.update({f"right_{k}": v for k, v in right.items()})
        return act

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Si en el futuro hay feedback, separar por prefijo y reenviar
        return

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()