"""B601-DM gravity feedforward for MIT position control."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)
_URDF_RELATIVE_PATH = Path("urdf/DM/urdf/ReBot_Arm_DM.urdf")


def _candidate_urdf_paths() -> list[Path]:
    candidates: list[Path] = []
    configured_urdf = os.environ.get("REBOT_GRAVITY_URDF")
    if configured_urdf:
        candidates.append(Path(configured_urdf).expanduser())

    roots: list[Path] = []
    configured_root = os.environ.get("REBOT_GRAVITY_SDK_ROOT")
    if configured_root:
        roots.append(Path(configured_root).expanduser())
    try:
        import reBotArm_control_py

        roots.append(Path(reBotArm_control_py.__file__).resolve().parent.parent)
    except (ImportError, AttributeError, OSError):
        pass

    for root in roots:
        candidates.append(root / _URDF_RELATIVE_PATH)
    return candidates


def resolve_b601_dm_urdf() -> Path:
    for candidate in _candidate_urdf_paths():
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _candidate_urdf_paths())
    raise FileNotFoundError(
        "B601-DM gravity model URDF was not found. Set REBOT_GRAVITY_URDF or "
        f"REBOT_GRAVITY_SDK_ROOT; searched: {searched}"
    )


class B601GravityFeedforward:
    """Evaluate the SDK-compatible generalized gravity vector ``g(q)``."""

    def __init__(self) -> None:
        try:
            import pinocchio as pin
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "B601-DM MIT gravity feedforward requires Pinocchio. Install the reBot extra "
                "(`pip install 'lerobot[rebot]'`)."
            ) from error

        self._pin = pin
        self.urdf_path = resolve_b601_dm_urdf()
        self._model = pin.buildModelFromUrdf(str(self.urdf_path))
        self._data = self._model.createData()
        if self._model.nq < len(ARM_JOINTS) or self._model.nv < len(ARM_JOINTS):
            raise ValueError(
                "B601-DM gravity model must expose at least six joints; "
                f"got nq={self._model.nq}, nv={self._model.nv}"
            )

    def torque(self, present_positions_deg: Mapping[str, float]) -> dict[str, float]:
        q_deg = np.asarray([present_positions_deg[name] for name in ARM_JOINTS], dtype=np.float64)
        if not np.all(np.isfinite(q_deg)):
            raise RuntimeError(f"Non-finite B601 feedback; refusing gravity command: {q_deg.tolist()}")

        q_model = np.zeros(self._model.nq, dtype=np.float64)
        q_model[: len(ARM_JOINTS)] = np.deg2rad(q_deg)
        self._pin.computeGeneralizedGravity(self._model, self._data, q_model)
        tau = np.asarray(self._data.g[: len(ARM_JOINTS)], dtype=np.float64)
        return {name: float(value) for name, value in zip(ARM_JOINTS, tau, strict=True)}
