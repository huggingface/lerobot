#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SO-101 MuJoCo Robot for LeRobot.

This implements the SO-101 robot in MuJoCo simulation with control logic
from orient_down.py, adapted for LeRobot's recording interface.

Key features:
- Keyboard teleoperation → joint position targets
- High-frequency control (180 Hz) with position recording (30 Hz)
- Jacobian-based XYZ control with wrist vertical orientation
- Gravity compensation
"""

import logging
from functools import cached_property
from pathlib import Path
from typing import Any

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from .configuration_so101_mujoco import SO101MujocoConfig

logger = logging.getLogger(__name__)


class SO101MujocoRobot(Robot):
    """SO-101 robot in MuJoCo simulation."""

    config_class = SO101MujocoConfig
    name = "so101_mujoco"

    # Joint names (order matters)
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def __init__(self, config: SO101MujocoConfig):
        super().__init__(config)
        self.config = config

        # MuJoCo model and data (initially None until connect())
        self.model: mj.MjModel | None = None
        self.data: mj.MjData | None = None
        self._renderers: dict[str, mj.Renderer] = {}  # One renderer per camera
        self._viewer = None  # GLFW viewer window

        # GLFW rendering (like test_with_teleop.py)
        self._glfw_window = None
        self._glfw_cam = None
        self._glfw_opt = None
        self._glfw_scene = None
        self._glfw_ctx = None

        # Joint/actuator/site IDs (set in connect())
        self.dof_ids: dict[str, int] = {}
        self.act_ids: dict[str, int] = {}
        self.ee_site_id: int = -1

        # Control state
        self.q_des: np.ndarray | None = None  # Desired joint positions
        self.dq_filt: np.ndarray | None = None  # Filtered joint velocities
        self.j_lo: np.ndarray | None = None  # Joint lower limits
        self.j_hi: np.ndarray | None = None  # Joint upper limits

        # Keyboard velocity commands (set by _from_keyboard_to_base_action)
        self._keyboard_velocities: dict[str, float] = {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "yaw_rate": 0.0,
            "gripper_delta": 0.0,
        }

        # Timing
        self.control_dt = 1.0 / config.control_fps
        self.physics_dt = 1.0 / config.physics_fps
        self.n_physics_per_control = int(self.control_dt / self.physics_dt)
        self.n_control_per_record = int((1.0 / config.record_fps) / self.control_dt)

        # Camera tracking (we render MuJoCo cameras directly, not using LeRobot camera abstraction)
        # This dict just tracks cameras for lerobot_record
        self.cameras = {f"camera_{name}": None for name in config.camera_names}

        logger.info(
            f"SO101MujocoRobot initialized: "
            f"record={config.record_fps}Hz, control={config.control_fps}Hz, "
            f"physics={config.physics_fps}Hz"
        )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation structure for dataset creation."""
        features = {
            # Joint positions
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
            # Joint velocities
            "shoulder_pan.vel": float,
            "shoulder_lift.vel": float,
            "elbow_flex.vel": float,
            "wrist_flex.vel": float,
            "wrist_roll.vel": float,
            "gripper.vel": float,
            # End-effector position
            "ee.pos_x": float,
            "ee.pos_y": float,
            "ee.pos_z": float,
        }
        # Add cameras
        for cam_name in self.config.camera_names:
            features[f"camera_{cam_name}"] = (self.config.camera_height, self.config.camera_width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action structure (joint position targets)."""
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected (model loaded)."""
        return self.model is not None and self.data is not None

    def connect(self, calibrate: bool = True) -> None:
        """Load MuJoCo model and initialize control state."""
        if self.is_connected:
            logger.warning(f"{self} already connected")
            return

        # Load model
        xml_path = Path(self.config.xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)

        # Override physics timestep
        self.model.opt.timestep = self.physics_dt

        # Setup renderers for each camera
        for cam_name in self.config.camera_names:
            self._renderers[cam_name] = mj.Renderer(
                self.model,
                height=self.config.camera_height,
                width=self.config.camera_width
            )

        # Map joint/actuator/site IDs
        self._setup_ids()

        # Get joint limits
        self.j_lo = self.model.jnt_range[:, 0].copy()
        self.j_hi = self.model.jnt_range[:, 1].copy()

        # Initialize control state
        self.dq_filt = np.zeros(self.model.nv)

        # Find and set home position
        self._initialize_home_position()

        # Initialize GLFW rendering (like test_with_teleop.py)
        self._init_glfw_rendering()

        logger.info(f"{self} connected successfully")

    def launch_viewer(self) -> None:
        """Launch the MuJoCo GLFW viewer window."""
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot must be connected before launching viewer")

        if self._viewer is not None:
            logger.warning("Viewer already launched")
            return

        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Initial sync to make viewer responsive
        self._viewer.sync()
        logger.info("MuJoCo viewer window opened")

    def _setup_ids(self):
        """Map joint, actuator, and site names to MuJoCo IDs."""
        for joint_name in self.JOINT_NAMES:
            self.dof_ids[joint_name] = self.model.jnt_dofadr[
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            ]
            self.act_ids[joint_name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, joint_name)

        # Store robot joint indices for clipping (needed because model has extra DOFs for block)
        self.robot_qpos_indices = np.array([self.dof_ids[name] for name in self.JOINT_NAMES])

        self.ee_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.config.ee_site_name)

        if self.ee_site_id < 0:
            raise RuntimeError(f"Site '{self.config.ee_site_name}' not found in model")

    def _initialize_home_position(self):
        """
        Set a good home position with tool pointing down.

        With new robot orientation (base at X=0.2, pointing in +Y direction):
        - EE positioned in forward workspace (+Y direction)
        - Tool pointing downward with good alignment
        - Good Jacobian conditioning for manipulation
        """
        # Set joint angles to home configuration
        q_home = self.data.qpos.copy()
        q_home[self.dof_ids["shoulder_pan"]] = 0.0
        q_home[self.dof_ids["shoulder_lift"]] = -0.3
        q_home[self.dof_ids["elbow_flex"]] = 0.6
        q_home[self.dof_ids["wrist_flex"]] = 1.2
        q_home[self.dof_ids["wrist_roll"]] = 0.0
        q_home[self.dof_ids["gripper"]] = 0.8

        # Set initial state
        self.data.qpos[:] = q_home
        self.data.qvel[:] = 0.0
        self.q_des = q_home.copy()

        # Set actuator targets
        for joint_name in self.JOINT_NAMES:
            self.data.ctrl[self.act_ids[joint_name]] = self.q_des[self.dof_ids[joint_name]]

        # Forward kinematics
        mj.mj_forward(self.model, self.data)

        robot_q = [q_home[self.dof_ids[name]] for name in self.JOINT_NAMES]
        ee_pos = self.data.site_xpos[self.ee_site_id]
        logger.info(f"Home position initialized: {robot_q}")
        logger.info(f"Home EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

    def _init_glfw_rendering(self):
        """Initialize GLFW window and rendering context (like test_with_teleop.py)."""
        # Keep offscreen renderer for camera images - it has a separate OpenGL context

        if not glfw.init():
            logger.warning("GLFW init failed - running without visualization")
            return

        # Create window - don't steal keyboard focus
        window_width, window_height = 1280, 720
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)  # Don't steal focus on creation
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)  # Don't steal focus when shown
        self._glfw_window = glfw.create_window(
            window_width, window_height,
            "SO-101 MuJoCo Recording",
            None, None
        )
        # Reset hints to default
        glfw.default_window_hints()
        if not self._glfw_window:
            glfw.terminate()
            logger.warning("Failed to create GLFW window - running without visualization")
            return

        # Set up MuJoCo rendering structures
        self._glfw_cam = mj.MjvCamera()
        self._glfw_opt = mj.MjvOption()
        mj.mjv_defaultCamera(self._glfw_cam)
        self._glfw_cam.distance = 1.3
        self._glfw_cam.azimuth = 140
        self._glfw_cam.elevation = -20

        self._glfw_scene = mj.MjvScene(self.model, maxgeom=10000)

        # Need context current to create MjrContext, then release it
        glfw.make_context_current(self._glfw_window)
        glfw.swap_interval(1)  # Enable vsync
        self._glfw_ctx = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
        glfw.make_context_current(None)  # Release context to avoid conflicts

        logger.info("GLFW visualization window created")

    def _render_glfw(self):
        """Render camera views to GLFW window in a grid layout."""
        if self._glfw_window is None:
            return

        # Check if window should close
        if glfw.window_should_close(self._glfw_window):
            return

        # Make GLFW context current for rendering
        glfw.make_context_current(self._glfw_window)
        glfw.swap_interval(1)  # Enable vsync

        # Get window dimensions
        viewport_width, viewport_height = glfw.get_framebuffer_size(self._glfw_window)

        # Layout: 2x2 grid for 3 cameras (top-left: top, top-right: front, bottom-left: wrist)
        cam_width = viewport_width // 2
        cam_height = viewport_height // 2

        # Render each camera to its grid position
        camera_positions = {
            "top": (0, cam_height),           # Top-left
            "front": (cam_width, cam_height), # Top-right
            "wrist": (0, 0)                   # Bottom-left
        }

        for cam_name in self.config.camera_names:
            if cam_name not in camera_positions:
                continue

            x, y = camera_positions[cam_name]
            viewport = mj.MjrRect(x, y, cam_width, cam_height)

            # Update scene with this camera
            mj.mjv_updateScene(
                self.model, self.data, self._glfw_opt, None, self._glfw_cam,
                mj.mjtCatBit.mjCAT_ALL, self._glfw_scene
            )

            # Override camera to use the named camera
            cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                self._glfw_cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self._glfw_cam.fixedcamid = cam_id
                mj.mjv_updateScene(
                    self.model, self.data, self._glfw_opt, None, self._glfw_cam,
                    mj.mjtCatBit.mjCAT_ALL, self._glfw_scene
                )

            # Render this camera view
            mj.mjr_render(viewport, self._glfw_scene, self._glfw_ctx)

        glfw.swap_buffers(self._glfw_window)
        glfw.poll_events()

        # Release context to avoid conflicts with offscreen renderer
        glfw.make_context_current(None)

    @property
    def is_calibrated(self) -> bool:
        """Simulation doesn't need calibration."""
        return True

    def calibrate(self) -> None:
        """No-op for simulation."""
        pass

    def configure(self) -> None:
        """No-op for simulation (configuration done at load time)."""
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get current robot state."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Update forward kinematics
        mj.mj_forward(self.model, self.data)

        # Get joint positions and velocities
        obs = {}
        for joint_name in self.JOINT_NAMES:
            dof_id = self.dof_ids[joint_name]
            obs[f"{joint_name}.pos"] = float(self.data.qpos[dof_id])
            obs[f"{joint_name}.vel"] = float(self.data.qvel[dof_id])

        # Get end-effector position
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obs["ee.pos_x"] = float(ee_pos[0])
        obs["ee.pos_y"] = float(ee_pos[1])
        obs["ee.pos_z"] = float(ee_pos[2])

        # Render all cameras
        for cam_name in self.config.camera_names:
            obs[f"camera_{cam_name}"] = self._render_camera(cam_name)

        return obs

    def _render_camera(self, camera_name: str) -> np.ndarray:
        """Render camera view."""
        # If renderer doesn't exist, return dummy image
        if camera_name not in self._renderers:
            return np.zeros(
                (self.config.camera_height, self.config.camera_width, 3),
                dtype=np.uint8
            )

        renderer = self._renderers[camera_name]

        # Update scene
        renderer.update_scene(self.data, camera=camera_name)
        pixels = renderer.render()

        # Convert to uint8 RGB
        return pixels.astype(np.uint8)

    def _from_keyboard_to_base_action(self, keyboard_action: dict) -> dict:
        """Convert keyboard input to velocity commands (stored for send_action).

        Controls:
        - W/S: +Y / -Y (world frame, forward/backward)
        - A/D: -X / +X (world frame, left/right)
        - Q/E: +Z / -Z (up/down)
        - [ / ]: Wrist roll left/right
        - O / C: Gripper open/close
        """
        # W/S for Y direction (forward/backward)
        vy = self.config.lin_speed if keyboard_action.get("w") else 0.0
        vy -= self.config.lin_speed if keyboard_action.get("s") else 0.0

        # A/D for X direction (left/right)
        vx = -self.config.lin_speed if keyboard_action.get("a") else 0.0
        vx += self.config.lin_speed if keyboard_action.get("d") else 0.0

        # Q/E for Z direction (up/down)
        vz = self.config.lin_speed if keyboard_action.get("q") else 0.0
        vz -= self.config.lin_speed if keyboard_action.get("e") else 0.0

        # [ / ] for wrist roll
        yaw_rate = -self.config.yaw_speed if keyboard_action.get("[") else 0.0
        yaw_rate += self.config.yaw_speed if keyboard_action.get("]") else 0.0

        # O / C for gripper
        gripper_delta = self.config.grip_speed if keyboard_action.get("o") else 0.0
        gripper_delta -= self.config.grip_speed if keyboard_action.get("c") else 0.0

        # Store for backward compatibility with test scripts
        self._keyboard_velocities = {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "yaw_rate": yaw_rate,
            "gripper_delta": gripper_delta,
        }

        # Return velocity dict for lerobot_record
        return self._keyboard_velocities.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Execute velocity commands for 1/30s using high-frequency control.

        Action dict should contain velocity keys: 'vx', 'vy', 'vz', 'yaw_rate', 'gripper_delta'
        (These are set by _from_keyboard_to_base_action or can be provided directly)

        Returns the final joint position targets (for recording).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Get velocity commands from action dict (set by _from_keyboard_to_base_action)
        # If action is empty, use stored keyboard velocities (backward compatibility with test scripts)
        if action:
            vx = action.get("vx", 0.0)
            vy = action.get("vy", 0.0)
            vz = action.get("vz", 0.0)
            yaw_rate = action.get("yaw_rate", 0.0)
            gripper_delta = action.get("gripper_delta", 0.0)
        else:
            # Fallback to stored velocities for test scripts
            vel = self._keyboard_velocities
            vx, vy, vz = vel["vx"], vel["vy"], vel["vz"]
            yaw_rate = vel["yaw_rate"]
            gripper_delta = vel["gripper_delta"]

        # Run high-frequency control loop (orient_down.py logic)
        for _ in range(self.n_control_per_record):
            self._control_step(vx, vy, vz, yaw_rate, gripper_delta)

        # Render GLFW visualization once per action (30Hz) instead of per control step (180Hz)
        self._render_glfw()

        # Return final target positions (what we commanded)
        action_to_record = {}
        for joint_name in self.JOINT_NAMES:
            action_to_record[f"{joint_name}.pos"] = float(
                self.q_des[self.dof_ids[joint_name]]
            )

        return action_to_record

    def _control_step(self, vx: float, vy: float, vz: float, yaw_rate: float, gripper_delta: float):
        """
        Single control iteration (orient_down.py logic).

        This runs at control_fps (180 Hz) and:
        1. Computes Jacobian at current configuration
        2. Solves for joint velocities to achieve XYZ velocity
        3. Adds wrist tilt correction for vertical orientation
        4. Applies gravity compensation
        5. Rate limits and smooths
        6. Integrates to get position targets
        7. Steps physics n_physics_per_control times
        """
        # Forward kinematics
        mj.mj_forward(self.model, self.data)

        # Get Jacobians at end-effector
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, Jp, Jr, self.ee_site_id)

        # --- PRIMARY: XYZ control via (pan, lift, elbow) ---
        arm_cols = [
            self.dof_ids["shoulder_pan"],
            self.dof_ids["shoulder_lift"],
            self.dof_ids["elbow_flex"]
        ]
        J3 = Jp[:, arm_cols]
        v_des = np.array([vx, vy, vz])
        A = J3 @ J3.T + (self.config.lambda_pos ** 2) * np.eye(3)
        dq3 = J3.T @ np.linalg.solve(A, v_des)
        dq = np.zeros(self.model.nv)
        dq[arm_cols] = dq3

        # --- SECONDARY: Wrist tilt correction for vertical orientation ---
        dq = self._add_wrist_tilt_correction(dq, Jr, J3, vx, vy, vz)

        # --- Independent wrist roll ---
        dq[self.dof_ids["wrist_roll"]] += yaw_rate

        # --- Gravity compensation for wrist flex ---
        tau_g = self.data.qfrc_bias[self.dof_ids["wrist_flex"]]
        dq[self.dof_ids["wrist_flex"]] += self.config.wrist_gff_gain * tau_g

        # --- Rate limiting ---
        dq_lim = self.config.vel_limit * np.ones(self.model.nv)
        dq_lim[self.dof_ids["wrist_flex"]] = self.config.vel_limit_wrist
        dq_lim[self.dof_ids["wrist_roll"]] = self.config.vel_limit_wrist
        dq = np.clip(dq, -dq_lim, dq_lim)

        # --- Smoothing ---
        alpha = self.config.smooth_dq * np.ones(self.model.nv)
        alpha[self.dof_ids["wrist_flex"]] = self.config.smooth_dq_wrist
        alpha[self.dof_ids["wrist_roll"]] = self.config.smooth_dq_wrist
        self.dq_filt = (1.0 - alpha) * self.dq_filt + alpha * dq

        # --- Integrate to get position targets ---
        # Use mj_integratePos to handle quaternions correctly for free joints
        mj.mj_integratePos(self.model, self.q_des, self.dq_filt, self.control_dt)
        # Only clip robot joints (not block freejoint)
        self.q_des[self.robot_qpos_indices] = np.clip(
            self.q_des[self.robot_qpos_indices],
            self.j_lo[self.robot_qpos_indices],
            self.j_hi[self.robot_qpos_indices]
        )

        # --- Send to actuators (arm joints only, gripper handled separately) ---
        arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        for joint_name in arm_joint_names:
            self.data.ctrl[self.act_ids[joint_name]] = self.q_des[self.dof_ids[joint_name]]

        # --- Gripper rate control (separate from arm) ---
        gidx = self.act_ids["gripper"]
        gdof = self.dof_ids["gripper"]
        self.data.ctrl[gidx] = np.clip(
            self.data.ctrl[gidx] + gripper_delta * self.control_dt,
            self.j_lo[gdof],
            self.j_hi[gdof]
        )

        # --- Step physics multiple times ---
        for _ in range(self.n_physics_per_control):
            mj.mj_step(self.model, self.data)

    def _add_wrist_tilt_correction(
        self, dq: np.ndarray, Jr: np.ndarray, J3: np.ndarray, vx: float, vy: float, vz: float
    ) -> np.ndarray:
        """
        Add wrist-flex correction to maintain vertical tool orientation.

        From orient_down.py lines 200-246.
        """
        # Get current orientation
        R = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        tool_axis = np.array(self.config.tool_axis_site)
        a_tool = R @ tool_axis

        # Error vector (cross product with desired -Z direction)
        e = np.cross(a_tool, np.array([0, 0, -1.0]))
        err_xy = e[:2]
        err_mag = float(np.linalg.norm(err_xy))

        # Jacobian column for wrist_flex
        wf_dof = self.dof_ids["wrist_flex"]
        jcol = Jr[:2, wf_dof]
        jnorm2 = float(jcol.T @ jcol)

        # Fade tilt control near singularities
        s = np.linalg.svd(J3, compute_uv=False)
        smin = float(np.min(s)) if s.size else 0.0
        sing_scale = np.clip(smin / 0.10, 0.0, 1.0)

        # Fade near joint limits (directional)
        q = self.data.qpos[wf_dof]
        lo = self.j_lo[wf_dof]
        hi = self.j_hi[wf_dof]
        limit_threshold = 0.1  # radians

        # Compute desired correction
        moving = (vx != 0.0) or (vy != 0.0) or (vz != 0.0)

        if (moving or err_mag > self.config.tilt_deadzone) and sing_scale > 1e-3 and jnorm2 > 1e-8:
            w_xy = self.config.ori_gain * sing_scale * err_xy
            nrm = np.linalg.norm(w_xy)
            if nrm > self.config.tilt_wmax:
                w_xy *= self.config.tilt_wmax / (nrm + 1e-9)

            dq_wf = float(jcol.T @ w_xy) / (jnorm2 + self.config.lambda_tilt ** 2)

            # Check if moving toward limit
            dist_lo = abs(q - lo)
            dist_hi = abs(hi - q)

            if dist_lo < limit_threshold and dq_wf < 0:
                lim_scale = dist_lo / limit_threshold
            elif dist_hi < limit_threshold and dq_wf > 0:
                lim_scale = dist_hi / limit_threshold
            else:
                lim_scale = 1.0

            dq[wf_dof] += lim_scale * dq_wf

        return dq

    def reset_to_home_position(self) -> None:
        """
        Reset robot arm to home position at the start of each episode.

        With new robot orientation (base at X=0.2, pointing in +Y direction):
        - Home position points end-effector forward (+Y) and slightly up
        - Gripper open and ready for manipulation
        - Good reach into workspace ahead

        This ensures consistent starting conditions for each recorded episode.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Set joint positions to home configuration
        # Note: These values are relative to the robot's LOCAL frame, which is now
        # rotated 90° CCW in the world frame
        self.data.qpos[self.dof_ids["shoulder_pan"]] = 0.0    # Centered
        self.data.qpos[self.dof_ids["shoulder_lift"]] = -0.3  # Arm slightly forward
        self.data.qpos[self.dof_ids["elbow_flex"]] = 0.6      # Elbow bent
        self.data.qpos[self.dof_ids["wrist_flex"]] = 1.2      # Wrist down
        self.data.qpos[self.dof_ids["wrist_roll"]] = 0.0      # Aligned
        self.data.qpos[self.dof_ids["gripper"]] = 0.8         # Open

        # Reset velocities to zero
        self.data.qvel[self.dof_ids["shoulder_pan"]] = 0.0
        self.data.qvel[self.dof_ids["shoulder_lift"]] = 0.0
        self.data.qvel[self.dof_ids["elbow_flex"]] = 0.0
        self.data.qvel[self.dof_ids["wrist_flex"]] = 0.0
        self.data.qvel[self.dof_ids["wrist_roll"]] = 0.0
        self.data.qvel[self.dof_ids["gripper"]] = 0.0

        # Update q_des to match
        self.q_des = self.data.qpos.copy()

        # Set actuator targets
        for joint_name in self.JOINT_NAMES:
            self.data.ctrl[self.act_ids[joint_name]] = self.q_des[self.dof_ids[joint_name]]

        # Reset filtered velocity to zero
        self.dq_filt = np.zeros(self.model.nv)

        # Forward kinematics to update derived quantities
        mj.mj_forward(self.model, self.data)

        ee_pos = self.data.site_xpos[self.ee_site_id]
        logger.info(f"Robot reset to home position - EE at: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

    def reset_block_position(self, x_range: tuple[float, float] = (0.0, 0.4),
                            y_range: tuple[float, float] = (0.0, 0.3)) -> None:
        """
        Randomize block position within specified ranges.

        With new robot orientation (robot base at X=0.2, pointing in +Y direction):
        - X range: ±0.2m from robot centerline (robot is at X=0.2)
        - Y range: forward workspace from base edge to 30cm forward

        Args:
            x_range: (min_x, max_x) in meters, default (0.0, 0.4) - ±0.2m left/right of robot
            y_range: (min_y, max_y) in meters, default (0.0, 0.3) - forward workspace
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Find block body ID
        block_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "block")
        if block_body_id < 0:
            logger.warning("Block body not found in model - skipping reset")
            return

        # Get qpos address for block's freejoint (7 DOF: 3 pos + 4 quat)
        block_jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "block")
        block_qpos_adr = self.model.jnt_qposadr[block_jnt_id]

        # Randomize position
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = 0.012  # Cube half-size to sit on floor

        # Set position
        self.data.qpos[block_qpos_adr:block_qpos_adr + 3] = [x, y, z]

        # Reset orientation to upright (identity quaternion: w=1, x=0, y=0, z=0)
        self.data.qpos[block_qpos_adr + 3:block_qpos_adr + 7] = [1, 0, 0, 0]

        # Get qvel address and reset velocities
        block_qvel_adr = self.model.jnt_dofadr[block_jnt_id]
        self.data.qvel[block_qvel_adr:block_qvel_adr + 6] = 0.0

        # Forward to update derived quantities
        mj.mj_forward(self.model, self.data)

        logger.info(f"Block reset to position: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def get_block_position(self) -> tuple[float, float, float] | None:
        """Get current block position for episode metadata."""
        if not self.is_connected:
            return None

        block_jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "block")
        if block_jnt_id < 0:
            return None

        block_qpos_adr = self.model.jnt_qposadr[block_jnt_id]
        pos = self.data.qpos[block_qpos_adr:block_qpos_adr + 3]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def disconnect(self) -> None:
        """Close MuJoCo model and renderer."""
        if not self.is_connected:
            return

        # Close GLFW window
        if self._glfw_window is not None:
            glfw.terminate()
            self._glfw_window = None
            self._glfw_cam = None
            self._glfw_opt = None
            self._glfw_scene = None
            self._glfw_ctx = None

        # Close viewer
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception as e:
                logger.warning(f"Error closing viewer: {e}")
            finally:
                self._viewer = None

        # Close all renderers
        for cam_name, renderer in self._renderers.items():
            try:
                renderer.close()
            except Exception as e:
                logger.warning(f"Error closing renderer for {cam_name}: {e}")
        self._renderers.clear()

        self.model = None
        self.data = None
        self.q_des = None
        self.dq_filt = None

        logger.info(f"{self} disconnected")
