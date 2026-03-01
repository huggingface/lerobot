"""Tests for manipulability metrics (near-singularity detection).

Validation ladder tests (M0–M5) plus unit tests for the core module.

Two test tiers:
  1. Pure unit tests (no placo) — always run.
  2. Integration tests (placo + minimal URDF) — skipped if placo is absent.
  3. SO-101 integration tests — skipped if placo or SO-101 URDF is absent.

To run:
    pytest tests/test_manipulability.py -v

With SO-101 URDF (set env var):
    SO101_URDF_PATH=./SO101/so101_new_calib.urdf pytest tests/test_manipulability.py -v
"""

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# M0: Environment setup — verify imports
# ---------------------------------------------------------------------------

try:
    import placo  # noqa: F401

    HAS_PLACO = True
except ImportError:
    HAS_PLACO = False

from lerobot.model.manipulability import (
    ManipulabilityResult,
    compute_manipulability,
    extract_translational_jacobian,
)

needs_placo = pytest.mark.skipif(not HAS_PLACO, reason="placo not installed")

URDF_PATH = os.environ.get(
    "SO101_URDF_PATH",
    os.path.join(os.path.dirname(__file__), "..", "SO101", "so101_new_calib.urdf"),
)
HAS_URDF = os.path.isfile(URDF_PATH)
needs_urdf = pytest.mark.skipif(not HAS_URDF, reason=f"URDF not found at {URDF_PATH}")

SO101_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
SO101_ARM_NAMES = SO101_MOTOR_NAMES[:5]

# Minimal 2-DOF planar arm URDF for integration tests (no mesh assets needed)
PLANAR_URDF = """<?xml version="1.0"?>
<robot name="planar_arm">
  <link name="base_link"><inertial><origin xyz="0 0 0"/><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="link1"><inertial><origin xyz="0.5 0 0"/><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="link2"><inertial><origin xyz="0.5 0 0"/><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="ee_frame"><inertial><origin xyz="0 0 0"/><mass value="0.001"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
  <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="10"/></joint>
  <joint name="joint2" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="1.0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="10"/></joint>
  <joint name="ee_joint" type="fixed"><parent link="link2"/><child link="ee_frame"/><origin xyz="1.0 0 0" rpy="0 0 0"/></joint>
</robot>"""


# ---------------------------------------------------------------------------
# Pure unit tests for compute_manipulability (no placo needed)
# ---------------------------------------------------------------------------


class TestComputeManipulability:
    """Unit tests for the core manipulability functions."""

    def test_identity_jacobian(self):
        """A 3×3 identity Jv should give σ_min=1, cond=1."""
        jv = np.eye(3)
        result = compute_manipulability(jv)
        assert isinstance(result, ManipulabilityResult)
        assert abs(result.sigma_min - 1.0) < 1e-10
        assert abs(result.sigma_max - 1.0) < 1e-10
        assert abs(result.condition_number - 1.0) < 1e-10

    def test_singular_jacobian(self):
        """A rank-deficient Jv should give σ_min≈0, cond=∞."""
        jv = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float)
        result = compute_manipulability(jv)
        assert result.sigma_min < 1e-14
        assert result.condition_number == float("inf")

    def test_scaled_jacobian(self):
        """Scaling a column should change σ but not rank."""
        jv = np.array([[2, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, 0.5, 0, 0]], dtype=float)
        result = compute_manipulability(jv)
        assert abs(result.sigma_min - 0.5) < 1e-10
        assert abs(result.sigma_max - 3.0) < 1e-10
        assert abs(result.condition_number - 6.0) < 1e-10

    def test_3x5_shape(self):
        """Typical SO-101 Jv shape: 3×5."""
        rng = np.random.default_rng(42)
        jv = rng.standard_normal((3, 5))
        result = compute_manipulability(jv)
        assert result.singular_values.shape == (3,)
        assert result.sigma_min > 0
        assert result.condition_number >= 1.0

    def test_no_nan_random(self):
        """No NaN/Inf in ManipulabilityResult for 100 random Jv matrices."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            jv = rng.standard_normal((3, 5))
            result = compute_manipulability(jv)
            assert np.isfinite(result.sigma_min)
            assert np.isfinite(result.sigma_max)
            assert result.condition_number >= 1.0


class TestExtractTranslationalJacobian:
    """Tests for extracting Jv from the 6×n_joints Jacobian."""

    def test_full_extraction(self):
        """Extract top 3 rows from a 6×5 matrix."""
        j = np.arange(30, dtype=float).reshape(6, 5)
        jv = extract_translational_jacobian(j)
        assert jv.shape == (3, 5)
        np.testing.assert_array_equal(jv, j[:3, :])

    def test_arm_indices_extraction(self):
        """Extract top 3 rows, specific columns only."""
        j = np.arange(36, dtype=float).reshape(6, 6)
        jv = extract_translational_jacobian(j, arm_joint_indices=[0, 1, 2, 3, 4])
        assert jv.shape == (3, 5)
        np.testing.assert_array_equal(jv, j[:3, :5])


# ---------------------------------------------------------------------------
# Integration tests with minimal URDF (placo required, no mesh assets)
# ---------------------------------------------------------------------------


@needs_placo
class TestWithPlanarArm:
    """Integration tests using a minimal 2-DOF planar arm URDF.

    Validates the full pipeline: RobotKinematics -> compute_frame_jacobian ->
    extract_translational_jacobian -> compute_manipulability, including the
    critical finite-difference Jacobian gate (M4).
    """

    @pytest.fixture(autouse=True)
    def setup_kinematics(self, tmp_path):
        from lerobot.model.kinematics import RobotKinematics

        urdf_file = tmp_path / "planar_arm.urdf"
        urdf_file.write_text(PLANAR_URDF)
        self.kin = RobotKinematics(
            urdf_path=str(urdf_file),
            target_frame_name="ee_frame",
            joint_names=["joint1", "joint2"],
        )
        self.motor_names = ["joint1", "joint2"]

    # M1: URDF loads, dimensions correct
    def test_urdf_loads(self):
        """M1: URDF loads and joint names match."""
        urdf_names = list(self.kin.robot.joint_names())
        for name in self.motor_names:
            assert name in urdf_names

    def test_frame_exists(self):
        """M1: ee_frame exists in the model."""
        assert "ee_frame" in list(self.kin.robot.frame_names())

    def test_v_offsets(self):
        """M1: get_joint_v_offsets returns valid indices."""
        offsets = self.kin.get_joint_v_offsets()
        assert len(offsets) == 2
        assert all(isinstance(o, int) and o >= 0 for o in offsets)

    # M2: FK sanity
    def test_fk_sanity(self):
        """M2: FK produces expected positions for known configurations."""
        pos_home = self.kin.forward_kinematics(np.array([0.0, 0.0]))[:3, 3]
        np.testing.assert_allclose(pos_home, [2.0, 0.0, 0.0], atol=1e-6)

        pos_90 = self.kin.forward_kinematics(np.array([90.0, 0.0]))[:3, 3]
        np.testing.assert_allclose(pos_90, [0.0, 2.0, 0.0], atol=1e-6)

    # M3: Jacobian shape
    def test_jacobian_shape(self):
        """M3: compute_frame_jacobian returns (6, n_joints)."""
        j = self.kin.compute_frame_jacobian(np.array([45.0, 30.0]))
        assert j.shape == (6, 2), f"Expected (6,2), got {j.shape}"
        jv = extract_translational_jacobian(j)
        assert jv.shape == (3, 2)

    def test_jacobian_subset(self):
        """M3: Can request a subset of joints."""
        j1 = self.kin.compute_frame_jacobian(np.array([45.0, 30.0]), joint_names=["joint1"])
        assert j1.shape == (6, 1)

    # M4: Finite-difference Jacobian validation (CRITICAL GATE)
    def test_jacobian_finite_diff(self):
        """M4: Analytical Jv matches finite-difference Jv. CRITICAL GATE."""
        eps = 1e-6
        rng = np.random.default_rng(42)
        max_error = 0

        for _ in range(20):
            q = rng.uniform(-90, 90, size=2)
            j_arm = self.kin.compute_frame_jacobian(q)
            jv_analytical = extract_translational_jacobian(j_arm)

            pos_0 = self.kin.forward_kinematics(q)[:3, 3].copy()
            jv_fd = np.zeros((3, 2))
            for j in range(2):
                q_plus = q.copy()
                q_plus[j] += eps
                pos_plus = self.kin.forward_kinematics(q_plus)[:3, 3].copy()
                jv_fd[:, j] = (pos_plus - pos_0) / np.deg2rad(eps)

            error = np.linalg.norm(jv_analytical - jv_fd)
            max_error = max(max_error, error)

        assert max_error < 1e-3, f"FD error {max_error:.6f} exceeds threshold"

    # M5: sigma_min / cond behavior
    def test_sigma_min_at_singularity(self):
        """M5: sigma_min approaches 0 at q2=180 (fully folded back)."""
        q_sing = np.array([0.0, 180.0])
        j = self.kin.compute_frame_jacobian(q_sing)
        jv = extract_translational_jacobian(j)
        result = compute_manipulability(jv)
        assert result.sigma_min < 1e-6
        assert result.condition_number > 1e6

    def test_sigma_min_positive_away_from_singularity(self):
        """M5: sigma_min > 0 at q2=90 (well-conditioned)."""
        q_fold = np.array([0.0, 90.0])
        j = self.kin.compute_frame_jacobian(q_fold)
        jv = extract_translational_jacobian(j)
        result = compute_manipulability(jv)
        assert result.sigma_min > 0.1
        assert result.condition_number < 10

    def test_sigma_min_decreases_toward_singularity(self):
        """M5: sigma_min at q2=170 < sigma_min at q2=90."""
        r_good = compute_manipulability(
            extract_translational_jacobian(self.kin.compute_frame_jacobian(np.array([0.0, 90.0])))
        )
        r_bad = compute_manipulability(
            extract_translational_jacobian(self.kin.compute_frame_jacobian(np.array([0.0, 170.0])))
        )
        assert r_bad.sigma_min < r_good.sigma_min
        assert r_bad.condition_number > r_good.condition_number

    # FK consistency
    def test_fk_consistency(self):
        """FK from compute_frame_jacobian path matches forward_kinematics."""
        rng = np.random.default_rng(99)
        for _ in range(10):
            q = rng.uniform(-90, 90, size=2)
            t_fk = self.kin.forward_kinematics(q)
            self.kin.compute_frame_jacobian(q)
            t_jac = self.kin.robot.get_T_world_frame("ee_frame")
            np.testing.assert_allclose(t_fk[:3, 3], t_jac[:3, 3], atol=1e-6)

    # Performance
    def test_performance_budget(self):
        """p99 latency < 500us for Jacobian + SVD."""
        import time

        rng = np.random.default_rng(42)
        times = []
        for _ in range(1000):
            q = rng.uniform(-90, 90, size=2)
            t0 = time.perf_counter()
            j = self.kin.compute_frame_jacobian(q)
            jv = extract_translational_jacobian(j)
            _ = compute_manipulability(jv)
            times.append(time.perf_counter() - t0)

        p99 = np.percentile(np.array(times) * 1e6, 99)
        assert p99 < 500, f"p99 latency {p99:.1f}us exceeds 500us budget"


# ---------------------------------------------------------------------------
# SO-101 integration tests (require placo + SO-101 URDF with mesh assets)
# ---------------------------------------------------------------------------


@needs_placo
@needs_urdf
class TestWithSO101:
    """Integration tests using the actual SO-101 URDF."""

    @pytest.fixture(autouse=True)
    def setup_kinematics(self):
        from lerobot.model.kinematics import RobotKinematics

        self.kin = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="gripper_frame_link",
            joint_names=SO101_MOTOR_NAMES,
        )

    def test_urdf_loads(self):
        urdf_names = list(self.kin.robot.joint_names())
        for name in SO101_MOTOR_NAMES:
            assert name in urdf_names

    def test_jacobian_shape(self):
        q = np.zeros(len(SO101_MOTOR_NAMES), dtype=float)
        j_arm = self.kin.compute_frame_jacobian(q, joint_names=SO101_ARM_NAMES)
        assert j_arm.shape == (6, 5)
        jv = extract_translational_jacobian(j_arm)
        assert jv.shape == (3, 5)

    def test_jacobian_finite_diff(self):
        """M4 on real SO-101 URDF."""
        eps = 1e-6
        rng = np.random.default_rng(42)
        max_error = 0

        for _ in range(10):
            q = rng.uniform(-30, 30, size=len(SO101_MOTOR_NAMES))
            q[-1] = 0.0
            j_arm = self.kin.compute_frame_jacobian(q, joint_names=SO101_ARM_NAMES)
            jv_a = extract_translational_jacobian(j_arm)

            pos_0 = self.kin.forward_kinematics(q)[:3, 3].copy()
            jv_fd = np.zeros((3, 5))
            for j in range(5):
                q_p = q.copy()
                q_p[j] += eps
                pos_p = self.kin.forward_kinematics(q_p)[:3, 3].copy()
                jv_fd[:, j] = (pos_p - pos_0) / np.deg2rad(eps)

            error = np.linalg.norm(jv_a - jv_fd)
            max_error = max(max_error, error)

        assert max_error < 1e-3, f"FD error {max_error:.6f}"

    def test_gripper_excluded(self):
        """Varying gripper does not change arm Jv."""
        q1 = np.zeros(len(SO101_MOTOR_NAMES), dtype=float)
        q2 = q1.copy()
        q2[-1] = 50.0
        jv1 = extract_translational_jacobian(
            self.kin.compute_frame_jacobian(q1, joint_names=SO101_ARM_NAMES)
        )
        jv2 = extract_translational_jacobian(
            self.kin.compute_frame_jacobian(q2, joint_names=SO101_ARM_NAMES)
        )
        np.testing.assert_allclose(jv1, jv2, atol=1e-8)

    def test_performance_budget(self):
        import time

        rng = np.random.default_rng(42)
        times = []
        for _ in range(1000):
            q = rng.uniform(-45, 45, size=len(SO101_MOTOR_NAMES))
            t0 = time.perf_counter()
            j = self.kin.compute_frame_jacobian(q, joint_names=SO101_ARM_NAMES)
            jv = extract_translational_jacobian(j)
            _ = compute_manipulability(jv)
            times.append(time.perf_counter() - t0)

        p99 = np.percentile(np.array(times) * 1e6, 99)
        assert p99 < 500, f"p99 latency {p99:.1f}us exceeds 500us budget"
