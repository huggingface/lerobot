"""Regression tests for ``lerobot.model.kinematics.RobotKinematics``.

Uses the exo_left URDF from the ``lerobot/unitree-g1-mujoco`` HF repo
(same pattern as ``src/lerobot/robots/unitree_g1/g1_kinematics.py``).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Skip the whole module if optional deps are missing.
pytest.importorskip("placo")
huggingface_hub = pytest.importorskip("huggingface_hub")

import lerobot.processor  # noqa: F401  break circular import
from lerobot.model.kinematics import RobotKinematics

URDF_REPO = "lerobot/unitree-g1-mujoco"
URDF_REL = "assets/exo_left.urdf"
TARGET_FRAME = "ee"
# Joint order: base -> tip
JOINT_NAMES = [
    "shoulder_pitch", "shoulder_yaw", "shoulder_roll",
    "elbow_flex", "wrist_roll",
]
# A mid-workspace seed where the IK convergence bug manifests clearly.
SEED_POSE_DEG = np.array([20.0, 10.0, 0.0, 45.0, 0.0])


@pytest.fixture(scope="module")
def kin() -> RobotKinematics:
    repo_path = huggingface_hub.snapshot_download(URDF_REPO)
    return RobotKinematics(
        urdf_path=os.path.join(repo_path, URDF_REL),
        target_frame_name=TARGET_FRAME,
        joint_names=JOINT_NAMES,
    )


def test_inverse_kinematics_converges_in_one_call(kin: RobotKinematics):
    """One call to ``inverse_kinematics`` should drive the EE within 1 mm
    of the requested pose.

    placo's ``solver.solve(True)`` performs a single Newton step. From a
    non-trivial seed that one step leaves significant residual position
    error -- the teleop loop pattern (one IK call per frame with the
    motor's lagging position as seed via
    ``initial_guess_current_joints=True``) never converges as a result.
    """
    q0 = SEED_POSE_DEG.copy()
    T0 = kin.forward_kinematics(q0)

    # 100 mm +Y translation -- well within reach from the seed.
    T1 = T0.copy()
    T1[1, 3] += 0.10

    q1 = kin.inverse_kinematics(q0, T1)
    achieved = kin.forward_kinematics(q1)

    pos_err_mm = float(np.linalg.norm(achieved[:3, 3] - T1[:3, 3]) * 1000.0)
    assert pos_err_mm < 1.0, (
        f"After one inverse_kinematics() call, position error is "
        f"{pos_err_mm:.2f} mm (>1 mm). The solver is not converging in "
        f"one call -- it needs to iterate."
    )
