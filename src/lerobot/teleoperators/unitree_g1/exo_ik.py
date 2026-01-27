"""Exoskeleton IK helper: FK → IK pipeline and Meshcat visualization."""

import logging
import os

import numpy as np

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex

from .exo_calib import JOINTS

logger = logging.getLogger(__name__)


def _get_frame_id(model, name: str) -> int | None:
    """Get frame ID if it exists, else None."""
    try:
        fid = model.getFrameId(name)
        if 0 <= fid < model.nframes:
            return fid
    except Exception:
        pass
    return None


class MeshcatMarkers:
    """Helper for managing Meshcat markers."""

    def __init__(self, viewer):
        self.viewer = viewer

    def add_sphere(self, path: str, radius: float, rgba: tuple):
        """Add a colored sphere marker."""
        import meshcat.geometry as mg

        color_int = int(rgba[0] * 255) << 16 | int(rgba[1] * 255) << 8 | int(rgba[2] * 255)
        self.viewer[path].set_object(
            mg.Sphere(radius),
            mg.MeshPhongMaterial(color=color_int, opacity=rgba[3], transparent=rgba[3] < 1.0),
        )

    def add_axes(self, path: str, axis_len: float = 0.1, axis_w: int = 6):
        """Add XYZ axes visualization."""
        import meshcat.geometry as mg

        pts = np.array([
            [0, 0, 0], [axis_len, 0, 0],
            [0, 0, 0], [0, axis_len, 0],
            [0, 0, 0], [0, 0, axis_len],
        ], dtype=np.float32).T
        cols = np.array([
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ], dtype=np.float32).T
        self.viewer[path].set_object(
            mg.LineSegments(
                mg.PointsGeometry(position=pts, color=cols),
                mg.LineBasicMaterial(linewidth=axis_w, vertexColors=True),
            )
        )

    def set_transform(self, path: str, T: np.ndarray):
        """Set 4x4 transform for a marker."""
        self.viewer[path].set_transform(T)


class ExoskeletonIKHelper:
    """
    Helper class for IK-based teleoperation.

    Loads exoskeleton URDFs and computes FK to get end-effector poses,
    then uses G1 IK to solve for joint angles.
    """

    def __init__(self, frozen_joints: list[str] | None = None):
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError("IK mode requires pinocchio. Install with: pip install pin") from e

        self.pin = pin
        self.ee_frame = "ee"
        self.frozen_joints = frozen_joints or []

        # Load G1 IK solver (downloads lerobot/unitree-g1-mujoco repo)
        from lerobot.robots.unitree_g1.robot_kinematic_processor import G1_29_ArmIK

        self.g1_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        self.robot_g1 = self.g1_ik.reduced_robot
        self.robot_g1.data = self.robot_g1.model.createData()
        self.q_g1 = pin.neutral(self.robot_g1.model)

        # Reuse repo_path from G1 IK solver
        assets_dir = os.path.join(self.g1_ik.repo_path, "assets")

        # Create symlinks for URDF package:// paths
        for pkg_name, mesh_folder in [("assets_left", "meshes_exo_left"), ("assets_right", "meshes_exo_right")]:
            symlink_path = os.path.join(assets_dir, pkg_name)
            target_path = os.path.join(assets_dir, mesh_folder)
            if not os.path.exists(symlink_path) and os.path.exists(target_path):
                try:
                    os.symlink(target_path, symlink_path)
                    logger.info(f"Created symlink: {symlink_path} -> {target_path}")
                except OSError as e:
                    logger.warning(f"Could not create symlink {symlink_path}: {e}")

        # Build frozen joint index map
        self.frozen_joint_indices = {}
        for jname in self.frozen_joints:
            if jname in self.robot_g1.model.names:
                jid = self.robot_g1.model.getJointId(jname)
                idx_q = self.robot_g1.model.idx_qs[jid]
                self.frozen_joint_indices[jname] = idx_q
                logger.info(f"Freezing joint: {jname} (q_idx={idx_q})")

        # G1 end-effector frame IDs
        self.left_ee_id = _get_frame_id(self.robot_g1.model, "L_ee")
        self.right_ee_id = _get_frame_id(self.robot_g1.model, "R_ee")

        # Load exoskeleton models
        self._load_exo_models(assets_dir)

        # Visualization (initialized lazily)
        self.viewer = None
        self.markers: MeshcatMarkers | None = None
        self.viz_g1 = None
        self.viz_exo_left = None
        self.viz_exo_right = None
        self.left_offset = np.array([0.6, 0.3, 0.0])
        self.right_offset = np.array([0.6, -0.3, 0.0])

    def _load_exo_models(self, assets_dir: str):
        """Load exoskeleton URDF models."""
        pin = self.pin
        self.exo_left = None
        self.exo_right = None
        self.exo_left_ee_id = None
        self.exo_right_ee_id = None
        self.q_exo_left = None
        self.q_exo_right = None
        self.exo_left_joint_map = {}
        self.exo_right_joint_map = {}

        left_urdf = os.path.join(assets_dir, "exo_left.urdf")
        right_urdf = os.path.join(assets_dir, "exo_right.urdf")

        if os.path.exists(left_urdf):
            self.exo_left = pin.RobotWrapper.BuildFromURDF(left_urdf, assets_dir)
            self.q_exo_left = pin.neutral(self.exo_left.model)
            self.exo_left_ee_id = self._find_ee_frame(self.exo_left.model)
            self.exo_left_joint_map = self._build_joint_map(self.exo_left)
            logger.info(f"Loaded left exo URDF: {left_urdf}")
        else:
            logger.warning(f"Left exo URDF not found: {left_urdf}")

        if os.path.exists(right_urdf):
            self.exo_right = pin.RobotWrapper.BuildFromURDF(right_urdf, assets_dir)
            self.q_exo_right = pin.neutral(self.exo_right.model)
            self.exo_right_ee_id = self._find_ee_frame(self.exo_right.model)
            self.exo_right_joint_map = self._build_joint_map(self.exo_right)
            logger.info(f"Loaded right exo URDF: {right_urdf}")
        else:
            logger.warning(f"Right exo URDF not found: {right_urdf}")

    def _find_ee_frame(self, model) -> int:
        """Find end-effector frame in model."""
        ee_id = _get_frame_id(model, self.ee_frame)
        if ee_id is not None:
            return ee_id
        for fid in reversed(range(model.nframes)):
            if model.frames[fid].type == self.pin.FrameType.BODY:
                return fid
        return 0

    def _build_joint_map(self, robot) -> dict[str, int]:
        """Build mapping from joint names to q indices."""
        joint_map = {}
        for jname in [j[0] for j in JOINTS]:
            if jname in robot.model.names:
                jid = robot.model.getJointId(jname)
                joint_map[jname] = robot.model.idx_qs[jid]
        return joint_map

    def init_visualization(self, show_axes: bool = True):
        """Initialize Meshcat visualization for G1 and exoskeletons."""
        try:
            from pinocchio.visualize import MeshcatVisualizer
        except ImportError as e:
            logger.warning(f"Meshcat visualization not available: {e}")
            return

        pin = self.pin

        # G1 visualization
        self.viz_g1 = MeshcatVisualizer(
            self.robot_g1.model, self.robot_g1.collision_model, self.robot_g1.visual_model
        )
        self.viz_g1.initViewer(open=True)
        self.viz_g1.loadViewerModel("g1")
        self.viz_g1.display(self.q_g1)
        self.viewer = self.viz_g1.viewer
        self.markers = MeshcatMarkers(self.viewer)

        # Left exo visualization
        if self.exo_left is not None:
            self.viz_exo_left = MeshcatVisualizer(
                self.exo_left.model, self.exo_left.collision_model, self.exo_left.visual_model
            )
            self.viz_exo_left.initViewer(open=False)
            self.viz_exo_left.viewer = self.viewer
            self.viz_exo_left.loadViewerModel("exo_left")
            T = np.eye(4)
            T[:3, 3] = self.left_offset
            self.viewer["exo_left"].set_transform(T)
            self.viz_exo_left.display(self.q_exo_left)

        # Right exo visualization
        if self.exo_right is not None:
            self.viz_exo_right = MeshcatVisualizer(
                self.exo_right.model, self.exo_right.collision_model, self.exo_right.visual_model
            )
            self.viz_exo_right.initViewer(open=False)
            self.viz_exo_right.viewer = self.viewer
            self.viz_exo_right.loadViewerModel("exo_right")
            T = np.eye(4)
            T[:3, 3] = self.right_offset
            self.viewer["exo_right"].set_transform(T)
            self.viz_exo_right.display(self.q_exo_right)

        # Add markers
        self.markers.add_sphere("markers/left_exo_ee", 0.012, (0.2, 1.0, 0.2, 0.9))
        self.markers.add_sphere("markers/left_g1_ee", 0.015, (1.0, 0.2, 0.2, 0.9))
        self.markers.add_sphere("markers/left_ik_target", 0.015, (0.1, 0.3, 1.0, 0.9))
        self.markers.add_sphere("markers/right_exo_ee", 0.012, (0.2, 1.0, 0.2, 0.9))
        self.markers.add_sphere("markers/right_g1_ee", 0.015, (1.0, 0.2, 0.2, 0.9))
        self.markers.add_sphere("markers/right_ik_target", 0.015, (0.1, 0.3, 1.0, 0.9))

        if show_axes:
            self.markers.add_axes("markers/left_exo_axes", 0.06)
            self.markers.add_axes("markers/left_g1_axes", 0.08)
            self.markers.add_axes("markers/right_exo_axes", 0.06)
            self.markers.add_axes("markers/right_g1_axes", 0.08)

        logger.info(f"Meshcat visualization initialized: {self.viewer.url()}")
        print(f"\n🌐 Meshcat URL: {self.viewer.url()}\n")

    def update_visualization(self):
        """Update Meshcat display with current joint states."""
        if self.viewer is None:
            return

        pin = self.pin

        # Update G1
        if self.viz_g1 is not None:
            self.viz_g1.display(self.q_g1)
            pin.forwardKinematics(self.robot_g1.model, self.robot_g1.data, self.q_g1)
            pin.updateFramePlacements(self.robot_g1.model, self.robot_g1.data)

            if self.left_ee_id is not None:
                T = self.robot_g1.data.oMf[self.left_ee_id].homogeneous
                self.markers.set_transform("markers/left_g1_ee", T)
                self.markers.set_transform("markers/left_g1_axes", T)

            if self.right_ee_id is not None:
                T = self.robot_g1.data.oMf[self.right_ee_id].homogeneous
                self.markers.set_transform("markers/right_g1_ee", T)
                self.markers.set_transform("markers/right_g1_axes", T)

        # Update left exo
        if self.viz_exo_left is not None and self.exo_left is not None:
            self.viz_exo_left.display(self.q_exo_left)
            pin.forwardKinematics(self.exo_left.model, self.exo_left.data, self.q_exo_left)
            pin.updateFramePlacements(self.exo_left.model, self.exo_left.data)
            T_exo_ee = self.exo_left.data.oMf[self.exo_left_ee_id]
            T_world = pin.SE3(np.eye(3), self.left_offset) * T_exo_ee
            self.markers.set_transform("markers/left_exo_ee", T_world.homogeneous)
            self.markers.set_transform("markers/left_exo_axes", T_world.homogeneous)

            T_target = np.eye(4)
            T_target[:3, :3] = T_exo_ee.rotation
            T_target[:3, 3] = self.left_offset + T_exo_ee.translation
            self.markers.set_transform("markers/left_ik_target", T_target)

        # Update right exo
        if self.viz_exo_right is not None and self.exo_right is not None:
            self.viz_exo_right.display(self.q_exo_right)
            pin.forwardKinematics(self.exo_right.model, self.exo_right.data, self.q_exo_right)
            pin.updateFramePlacements(self.exo_right.model, self.exo_right.data)
            T_exo_ee = self.exo_right.data.oMf[self.exo_right_ee_id]
            T_world = pin.SE3(np.eye(3), self.right_offset) * T_exo_ee
            self.markers.set_transform("markers/right_exo_ee", T_world.homogeneous)
            self.markers.set_transform("markers/right_exo_axes", T_world.homogeneous)

            T_target = np.eye(4)
            T_target[:3, :3] = T_exo_ee.rotation
            T_target[:3, 3] = self.right_offset + T_exo_ee.translation
            self.markers.set_transform("markers/right_ik_target", T_target)

    def compute_g1_joints_from_exo(
        self,
        left_angles: dict[str, float],
        right_angles: dict[str, float],
    ) -> dict[str, float]:
        """Compute G1 joint angles from exoskeleton joint angles using IK."""
        pin = self.pin

        # Update left exo FK
        left_target = None
        if self.exo_left is not None and left_angles:
            for name, angle in left_angles.items():
                if name in self.exo_left_joint_map:
                    self.q_exo_left[self.exo_left_joint_map[name]] = float(angle)

            pin.forwardKinematics(self.exo_left.model, self.exo_left.data, self.q_exo_left)
            pin.updateFramePlacements(self.exo_left.model, self.exo_left.data)
            T_exo_ee = self.exo_left.data.oMf[self.exo_left_ee_id]
            left_target = np.eye(4)
            left_target[:3, :3] = T_exo_ee.rotation
            left_target[:3, 3] = self.left_offset + T_exo_ee.translation

        # Update right exo FK
        right_target = None
        if self.exo_right is not None and right_angles:
            for name, angle in right_angles.items():
                if name in self.exo_right_joint_map:
                    self.q_exo_right[self.exo_right_joint_map[name]] = float(angle)

            pin.forwardKinematics(self.exo_right.model, self.exo_right.data, self.q_exo_right)
            pin.updateFramePlacements(self.exo_right.model, self.exo_right.data)
            T_exo_ee = self.exo_right.data.oMf[self.exo_right_ee_id]
            right_target = np.eye(4)
            right_target[:3, :3] = T_exo_ee.rotation
            right_target[:3, 3] = self.right_offset + T_exo_ee.translation

        # Fallback to current G1 poses
        pin.forwardKinematics(self.robot_g1.model, self.robot_g1.data, self.q_g1)
        pin.updateFramePlacements(self.robot_g1.model, self.robot_g1.data)

        if left_target is None and self.left_ee_id is not None:
            left_target = self.robot_g1.data.oMf[self.left_ee_id].homogeneous
        if right_target is None and self.right_ee_id is not None:
            right_target = self.robot_g1.data.oMf[self.right_ee_id].homogeneous

        if left_target is None or right_target is None:
            logger.warning("Missing IK targets, returning current pose")
            return {}

        # Save frozen joint values
        frozen_values = {name: self.q_g1[idx] for name, idx in self.frozen_joint_indices.items()}

        # Solve IK
        self.q_g1, _ = self.g1_ik.solve_ik(left_target, right_target, current_lr_arm_motor_q=self.q_g1)

        # Restore frozen values
        for name, idx in self.frozen_joint_indices.items():
            self.q_g1[idx] = frozen_values[name]

        # Convert to action dict
        return {f"{joint.name}.q": float(self.q_g1[i]) for i, joint in enumerate(G1_29_JointArmIndex) if i < len(self.q_g1)}

