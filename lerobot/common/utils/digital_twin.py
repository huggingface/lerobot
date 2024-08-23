# ruff: noqa: N803, N806
import logging

import numpy as np
import open3d as o3d

GREEN = np.array([0.5, 1.0, 0.5])
RED = np.array([1.0, 0.1, 0.1])
PURPLE = np.array([0.8, 0, 0.8])
LIGHT_GRAY = np.array([0.8, 0.8, 0.8])


def skew_symmetric(w):
    """Creates the skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues_rotation(w, theta):
    """Computes the rotation matrix using Rodrigues' formula."""
    w_hat = skew_symmetric(w)
    return np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat


def screw_axis_to_transform(S, theta):
    """Converts a screw axis to a 4x4 transformation matrix."""
    S_w = S[:3]
    S_v = S[3:]
    if np.allclose(S_w, 0) and np.linalg.norm(S_v) == 1:  # Pure translation
        T = np.eye(4)
        T[:3, 3] = S_v * theta
    elif np.linalg.norm(S_w) == 1:  # Rotation and translation
        R = rodrigues_rotation(S_w, theta)
        t = (
            np.eye(3) * theta
            + (1 - np.cos(theta)) * skew_symmetric(S_w)
            + (theta - np.sin(theta)) * skew_symmetric(S_w) @ skew_symmetric(S_w)
        ) @ S_v
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    return T


class DigitalTwin:
    def __init__(self):
        # Frames. All origins of robot link frames are at the center of the respective motor axis.
        # (W)orld
        # (B)ase # Bo is at the center of the motor axis of the shoulder pan motor.
        # (S)houlder  # So is at the point where the shoulder pan motor connects to the shoulder lift motor
        # (H)umerus  # Ho is on the center of the shoulder lift motor axis
        # (F)orearm  # Fo is on the center of the elbow lift motor axis
        #  w(R)ist  # Wo is on the center of the wrist lift motor axis
        # (G)ripper  # Go is at the point where the wrist twist motor connects to the gripper

        # Gripper
        self.gripper_X0 = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.gripper = o3d.geometry.OrientedBoundingBox(
            center=[0, 0, 0],
            R=np.eye(3),
            extent=[0.07, 0.036, 0.035],  # (0.07, 0.035, 0.036)
        )
        # self.gripper.compute_vertex_normals()
        # self.gripper.paint_uniform_color(GREEN)
        self.gripper.color = GREEN
        # Screw axis of gripper frame wrt base frame.
        self.S_BG = np.array([1, 0, 0, 0, 0.018, 0])
        # Gripper origin to centroid transform.
        self.X_GoGc = np.array(
            [
                [1, 0, 0, 0.035],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # 0-position humerus frame pose wrt base.
        self.X_BoGo = np.array(
            [
                [1, 0, 0, 0.253],
                [0, 1, 0, 0],
                [0, 0, 1, 0.018],
                [0, 0, 0, 1],
            ]
        )

        # Wrist
        self.wrist_X0 = np.array(
            [
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.wrist = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.042, 0.027, 0.02],  # (0.025, 0.042, 0.02)
            )
        )
        self.wrist.compute_vertex_normals()
        self.wrist.paint_uniform_color(GREEN)
        # Screw axis of wrist frame wrt base frame.
        self.S_BR = np.array([0, 1, 0, -0.018, 0, +0.21])
        # 0-position origin to centroid transform.
        self.X_RoRc = np.array(
            [
                [1, 0, 0, 0.0035],
                [0, 1, 0, -0.002],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # 0-position wrist frame pose wrt base.
        self.X_BR = np.array(
            [
                [1, 0, 0, 0.210],
                [0, 1, 0, 0],
                [0, 0, 1, 0.018],
                [0, 0, 0, 1],
            ]
        )

        # Forearm
        self.forearm = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.09, 0.035, 0.024],
            )
        )
        self.forearm.compute_vertex_normals()
        self.forearm.paint_uniform_color(GREEN)
        # Screw axis of forearm frame wrt base frame.
        self.S_BF = np.array([0, 1, 0, -0.020, 0, +0.109])
        # Forearm origin + centroid transform.
        self.X_FoFc = np.array(
            [
                [1, 0, 0, 0.036],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # 0-position forearm frame pose wrt base.
        self.X_BF = np.array(
            [
                [1, 0, 0, 0.109],
                [0, 1, 0, 0],
                [0, 0, 1, 0.020],
                [0, 0, 0, 1],
            ]
        )

        # Humerus
        self.humerus = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.125, 0.045, 0.025],
            )
        )
        self.humerus.compute_vertex_normals()
        self.humerus.paint_uniform_color(GREEN)
        # Screw axis of humerus frame wrt base frame.
        self.S_BH = np.array([0, -1, 0, 0.036, 0, 0])
        # Humerus origin to centroid transform.
        self.X_HoHc = np.array(
            [
                [1, 0, 0, 0.0475],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # 0-position humerus frame pose wrt base.
        self.X_BH = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.036],
                [0, 0, 0, 1],
            ]
        )

        # Screw axis of shoulder frame wrt Base frame.
        self.shoulder = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.05, 0.04, 0.007],  # (0.035, 0.05, 0.04)
            )
        )
        self.shoulder.compute_vertex_normals()
        self.shoulder.paint_uniform_color(GREEN)
        self.S_BS = np.array([0, 0, -1, 0, 0, 0])
        self.X_SoSc = np.array(
            [
                [1, 0, 0, -0.017],
                [0, 1, 0, 0],
                [0, 0, 1, 0.0035],
                [0, 0, 0, 1],
            ]
        )
        self.X_BS = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.02],
                [0, 0, 0, 1],
            ]
        )

        # Base to world transform.
        # o3d seems to be aligning the box frame so that it is longest to longest to shortest on xyz.
        self.base_X0 = np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.base = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(
                center=[0, 0, 0],
                R=np.eye(3),
                extent=[0.05, 0.04, 0.035],  # (0.035, 0.05, 0.04)
            )
        )
        self.base.compute_vertex_normals()
        self.base.paint_uniform_color(GREEN)
        self.X_BoBc = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0.015],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.X_WoBo = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.02],
                [0, 0, 0, 1],
            ]
        )

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Create a floor
        self.floor = o3d.geometry.TriangleMesh.create_box(width=10.0, height=10.0, depth=0.001)  # x, y, z
        self.floor.translate([-5.0, -5.0, -0.001])
        self.floor.compute_vertex_normals()
        self.floor.paint_uniform_color(LIGHT_GRAY)

        # Grid parameters
        grid_size = 10.0  # Size of the grid in each dimension
        grid_spacing = 0.1  # 10 cm spacing

        # Create grid lines
        lines = []
        colors = [[0, 0, 0] for _ in range(int(2 * grid_size / grid_spacing))]  # Black color for all lines

        # Lines along X-axis
        for i in range(int(grid_size / grid_spacing) + 1):
            x = -grid_size / 2 + i * grid_spacing
            lines.append([[-grid_size / 2, x, 0], [grid_size / 2, x, 0]])

        # Lines along Y-axis
        for i in range(int(grid_size / grid_spacing) + 1):
            y = -grid_size / 2 + i * grid_spacing
            lines.append([[y, -grid_size / 2, 0], [y, grid_size / 2, 0]])

        # Create LineSet geometry
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3)),
            lines=o3d.utility.Vector2iVector(np.arange(len(lines) * 2).reshape(-1, 2)),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Create a visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Digital Twin", width=768, height=512)
        self.vis.add_geometry(self.floor)
        self.vis.add_geometry(self.base)
        self.vis.add_geometry(self.shoulder)
        self.vis.add_geometry(self.humerus)
        self.vis.add_geometry(self.forearm)
        self.vis.add_geometry(self.wrist)
        self.vis.add_geometry(self.gripper)
        self.vis.add_geometry(coordinate_frame)
        self.vis.add_geometry(line_set)

        self._do_quit = False
        self.vis.register_key_callback(ord("Q"), self.set_quit)

        # Hide a bunch of waypoints under the plane.
        self.waypoints = [o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.005) for i in range(1000)]
        for waypoint in self.waypoints:
            waypoint.translate([0, 0, -0.1])
            waypoint.compute_vertex_normals()
            waypoint.paint_uniform_color(PURPLE)
            self.vis.add_geometry(waypoint)

        # Set initial view control
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.05)
        view_control.set_up([0, 0, 1])
        view_control.set_front([-1, -0.3, 0.8])  # Set camera looking towards the cube
        view_control.set_lookat([0, 0, 0])  # Set camera focus point

    def set_quit(self, *_):
        self._do_quit = True

    def quit_signal_is_set(self):
        return self._do_quit

    def fk_base(self):
        return self.X_WoBo @ self.X_BoBc @ self.base_X0

    def fk_shoulder(self, robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return self.X_WoBo @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0]) @ self.X_SoSc @ self.X_BS

    def fk_humerus(self, robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ self.X_HoHc
            @ self.X_BH
        )

    def fk_forearm(self, robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])
            @ self.X_FoFc
            @ self.X_BF
        )

    def fk_wrist(self, robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(self.S_BR, robot_pos_rad[3])
            @ self.X_RoRc
            @ self.X_BR
            @ self.wrist_X0
        )

    def fk_gripper(self, robot_pos_deg):
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(self.S_BR, robot_pos_rad[3])
            @ screw_axis_to_transform(self.S_BG, robot_pos_rad[4])
            @ self.X_GoGc
            @ self.X_BoGo
            @ self.gripper_X0
        )

    def set_object_pose(self, obj, X_WO):
        if obj is self.gripper:
            self.gripper.center = X_WO[:3, 3]
            self.gripper.R = X_WO[:3, :3]
            return
        box = obj.get_oriented_bounding_box()
        obj.translate(-box.center)
        obj.rotate(np.linalg.inv(box.R))
        obj.transform(X_WO)

    def set_twin_pose(self, follower_pos, follower_pos_trajectory=None):
        # follower_pos *= 0
        self.set_object_pose(self.base, self.fk_base())
        self.vis.update_geometry(self.base)
        self.set_object_pose(self.shoulder, self.fk_shoulder(follower_pos))
        self.vis.update_geometry(self.shoulder)
        self.set_object_pose(self.humerus, self.fk_humerus(follower_pos))
        self.vis.update_geometry(self.humerus)
        self.set_object_pose(self.forearm, self.fk_forearm(follower_pos))
        self.vis.update_geometry(self.forearm)
        self.set_object_pose(self.wrist, self.fk_wrist(follower_pos))
        self.vis.update_geometry(self.wrist)
        self.set_object_pose(self.gripper, self.fk_gripper(follower_pos))
        # self.gripper.paint_uniform_color(
        #     np.clip((1 - follower_pos[-1] / 50) * RED + (follower_pos[-1] / 50) * GREEN, 0, 1)
        # )
        self.gripper.color = np.clip(
            (1 - follower_pos[-1] / 50) * RED + (follower_pos[-1] / 50) * GREEN, 0, 1
        )
        self.vis.update_geometry(self.gripper)

        if follower_pos_trajectory is not None:
            self._set_gripper_waypoints(follower_pos_trajectory)

        # Update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()

    def _set_gripper_waypoints(self, follower_pos_trajectory):
        if follower_pos_trajectory.shape[0] > len(self.waypoints):
            logging.warning("Not enough waypoint objects loaded into the scene to show the full trajectory.")
        # Rest all waypoints to below the floor.
        for waypoint in self.waypoints:
            waypoint.translate([0, 0, -0.1], relative=False)
        # Set the necessary number of waypoints.
        for waypoint, follower_pos in zip(self.waypoints, follower_pos_trajectory, strict=False):
            pos = self.fk_gripper(follower_pos)[:3, 3]
            waypoint.translate(pos, relative=False)
            self.vis.update_geometry(waypoint)

    def __del__(self):
        self.vis.destroy_window()
