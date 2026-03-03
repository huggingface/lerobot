#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
import sys
import threading
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ....robot import Robot
from .config import PantheraArmConfig

logger = logging.getLogger(__name__)

try:
    import pinocchio as pin
except Exception:
    pin = None


class PantheraArm(Robot):
    """Panthera arm wrapper with polar/cartesian end-effector action interface. https://github.com/HighTorque-Robotics"""

    config_class = PantheraArmConfig
    name = "panthera_arm"
    supports_shared_bus = False

    def __init__(self, config: PantheraArmConfig):
        super().__init__(config)
        self.config = config
        self.camera_configs = dict(config.cameras)
        self.cameras = make_cameras_from_configs(self.camera_configs)
        self._last_camera_obs: dict[str, np.ndarray] = {
            name: self._make_blank_camera_obs(name) for name in self.cameras
        }
        self._robot = None
        self._target_pos: np.ndarray | None = None
        self._target_rot: np.ndarray | None = None
        self._gripper_target: float = 0.0
        self._target_lock = threading.Lock()
        self._impedance_thread: threading.Thread | None = None
        self._impedance_stop_event = threading.Event()
        self._impedance_enabled_event = threading.Event()
        self._pin_data = None
        self._last_impedance_error_log_s: float = 0.0
        self._last_limit_warn_log_s: float = 0.0

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {
            "joint1.pos": float,
            "joint2.pos": float,
            "joint3.pos": float,
            "joint4.pos": float,
            "joint5.pos": float,
            "joint6.pos": float,
            "joint1.vel": float,
            "joint2.vel": float,
            "joint3.vel": float,
            "joint4.vel": float,
            "joint5.vel": float,
            "joint6.vel": float,
            "joint1.torque": float,
            "joint2.torque": float,
            "joint3.torque": float,
            "joint4.torque": float,
            "joint5.torque": float,
            "joint6.torque": float,
            "gripper.pos": float,
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
        }
        features.update(self._cameras_ft)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "radial": float,
            "orbit": float,
            "delta_z": float,
            "delta_roll": float,
            "delta_pitch": float,
            "delta_yaw": float,
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and all(cam.is_connected for cam in self.cameras.values())

    def _make_blank_camera_obs(self, cam_key: str) -> np.ndarray:
        cam_config = self.camera_configs.get(cam_key)
        height = getattr(cam_config, "height", None) or 1
        width = getattr(cam_config, "width", None) or 1
        return np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        camera_features: dict[str, tuple[int, int, int]] = {}
        for cam_key, cam_config in self.camera_configs.items():
            height = getattr(cam_config, "height", None) or 1
            width = getattr(cam_config, "width", None) or 1
            camera_features[cam_key] = (height, width, 3)
        return camera_features

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        sdk_python_dir = self._resolve_sdk_python_dir()
        panthera_cls = self._import_panthera_class(sdk_python_dir)
        config_path = self._resolve_config_path(sdk_python_dir)
        self._robot = panthera_cls(config_path) if config_path else panthera_cls()

        if self.config.use_cartesian_impedance and self.config.run_startup_sequence:
            self._run_impedance_startup_sequence()

        fk = self._robot.forward_kinematics()
        if fk is None:
            raise RuntimeError("Panthera forward_kinematics() returned None during connect.")

        with self._target_lock:
            self._target_pos = np.array(fk["position"], dtype=float)
            self._target_rot = np.array(fk["rotation"], dtype=float)
            self._gripper_target = float(self._robot.get_current_pos_gripper())

        for cam in self.cameras.values():
            cam.connect()

        if self.config.use_cartesian_impedance:
            self._start_impedance_loop()

        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        assert self._robot is not None
        q = np.asarray(self._robot.get_current_pos(), dtype=float)
        dq = np.asarray(self._robot.get_current_vel(), dtype=float)
        tau = np.asarray(self._robot.get_current_torque(), dtype=float)
        fk = self._robot.forward_kinematics(q)
        pos = np.asarray(fk["position"], dtype=float) if fk is not None else np.zeros(3, dtype=float)

        obs = {
            "joint1.pos": float(q[0]),
            "joint2.pos": float(q[1]),
            "joint3.pos": float(q[2]),
            "joint4.pos": float(q[3]),
            "joint5.pos": float(q[4]),
            "joint6.pos": float(q[5]),
            "joint1.vel": float(dq[0]),
            "joint2.vel": float(dq[1]),
            "joint3.vel": float(dq[2]),
            "joint4.vel": float(dq[3]),
            "joint5.vel": float(dq[4]),
            "joint6.vel": float(dq[5]),
            "joint1.torque": float(tau[0]),
            "joint2.torque": float(tau[1]),
            "joint3.torque": float(tau[2]),
            "joint4.torque": float(tau[3]),
            "joint5.torque": float(tau[4]),
            "joint6.torque": float(tau[5]),
            "gripper.pos": float(self._robot.get_current_pos_gripper()),
            "ee.x": float(pos[0]),
            "ee.y": float(pos[1]),
            "ee.z": float(pos[2]),
        }
        for name, cam in self.cameras.items():
            try:
                frame = cam.async_read()
            except Exception as exc:
                logger.warning("Failed to read Panthera camera %s (%s); using cached frame", name, exc)
                frame = self._last_camera_obs.get(name)
                if frame is None:
                    frame = self._make_blank_camera_obs(name)
                    self._last_camera_obs[name] = frame
            else:
                self._last_camera_obs[name] = frame
            obs[name] = frame
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self._robot is not None
        assert self._target_pos is not None and self._target_rot is not None

        radial = float(action.get("radial", 0.0))
        orbit = float(action.get("orbit", 0.0))
        delta_z = float(action.get("delta_z", 0.0))
        delta_roll = float(action.get("delta_roll", 0.0))
        delta_pitch = float(action.get("delta_pitch", 0.0))
        delta_yaw = float(action.get("delta_yaw", 0.0))
        gripper = float(action.get("gripper", 1.0))

        with self._target_lock:
            self._apply_polar_delta(radial=radial, orbit=orbit, delta_z=delta_z)
            self._apply_rotation_delta(
                delta_roll=delta_roll,
                delta_pitch=delta_pitch,
                delta_yaw=delta_yaw,
            )
            self._apply_gripper_delta(gripper, command_immediately=not self.config.use_cartesian_impedance)
            target_pos = np.array(self._target_pos, dtype=float)
            target_rot = np.array(self._target_rot, dtype=float)

        if self.config.use_cartesian_impedance:
            return {
                "radial": radial,
                "orbit": orbit,
                "delta_z": delta_z,
                "delta_roll": delta_roll,
                "delta_pitch": delta_pitch,
                "delta_yaw": delta_yaw,
                "gripper": gripper,
            }

        q_cur = np.asarray(self._robot.get_current_pos(), dtype=float)
        q_goal = self._robot.inverse_kinematics(
            target_position=target_pos,
            target_rotation=target_rot,
            init_q=q_cur,
            max_iter=self.config.ik_max_iter,
            eps=self.config.ik_eps,
            damping=self.config.ik_damping,
            adaptive_damping=self.config.ik_adaptive_damping,
            multi_init=self.config.ik_multi_init,
        )

        if q_goal is None:
            logger.debug("Panthera IK failed for target %s", target_pos.tolist())
            return {
                "radial": radial,
                "orbit": orbit,
                "delta_z": delta_z,
                "delta_roll": delta_roll,
                "delta_pitch": delta_pitch,
                "delta_yaw": delta_yaw,
                "gripper": gripper,
            }

        vel = self.config.joint_velocity
        if len(vel) != 6:
            raise ValueError("panthera_arm.joint_velocity must contain exactly 6 elements.")

        self._robot.Joint_Pos_Vel(
            pos=np.asarray(q_goal, dtype=float),
            vel=np.asarray(vel, dtype=float),
            max_tqu=np.asarray(self.config.max_torque, dtype=float) if self.config.max_torque else None,
            iswait=False,
        )

        return {
            "radial": radial,
            "orbit": orbit,
            "delta_z": delta_z,
            "delta_roll": delta_roll,
            "delta_pitch": delta_pitch,
            "delta_yaw": delta_yaw,
            "gripper": gripper,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        assert self._robot is not None

        if self._impedance_thread is not None:
            self._impedance_enabled_event.clear()
            self._impedance_stop_event.set()
            self._impedance_thread.join(timeout=1.5)
            self._impedance_thread = None
            self._impedance_stop_event.clear()

        if self.config.stop_on_disconnect and hasattr(self._robot, "set_stop"):
            try:
                self._robot.set_stop()
            except Exception:
                logger.debug("Panthera set_stop failed during disconnect", exc_info=True)
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                logger.warning("Failed to disconnect Panthera camera", exc_info=True)
        self._robot = None
        with self._target_lock:
            self._target_pos = None
            self._target_rot = None
        self._pin_data = None
        logger.info("%s disconnected.", self)

    def _apply_polar_delta(self, radial: float, orbit: float, delta_z: float) -> None:
        assert self._target_pos is not None

        x, y, z = self._target_pos
        radius = float(np.hypot(x, y))
        angle = float(np.arctan2(y, x))

        radius += radial * self.config.radial_step_m
        radius = float(np.clip(radius, self.config.min_radius_m, self.config.max_radius_m))
        angle += orbit * self.config.polar_angle_step_rad
        z += delta_z * self.config.vertical_step_m
        z = float(np.clip(z, self.config.min_z_m, self.config.max_z_m))

        self._target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), z], dtype=float)

    def _apply_rotation_delta(self, delta_roll: float, delta_pitch: float, delta_yaw: float) -> None:
        assert self._target_rot is not None
        if delta_roll == 0.0 and delta_pitch == 0.0 and delta_yaw == 0.0:
            return
        d_rot = Rot.from_euler(
            "xyz",
            [
                delta_roll * self.config.rotation_step_deg,
                delta_pitch * self.config.rotation_step_deg,
                delta_yaw * self.config.rotation_step_deg,
            ],
            degrees=True,
        ).as_matrix()
        self._target_rot = self._target_rot @ d_rot

    def _apply_gripper_delta(self, gripper_action: float, command_immediately: bool = True) -> None:
        assert self._robot is not None
        if gripper_action > 1.5:
            self._gripper_target += self.config.gripper_step
        elif gripper_action < 0.5:
            self._gripper_target -= self.config.gripper_step
        else:
            return

        lower = -0.1
        upper = 2.0
        if hasattr(self._robot, "gripper_limits") and self._robot.gripper_limits:
            lower = float(self._robot.gripper_limits.get("lower", lower))
            upper = float(self._robot.gripper_limits.get("upper", upper))
        self._gripper_target = float(np.clip(self._gripper_target, lower, upper))
        if not command_immediately:
            return
        self._robot.gripper_control(
            pos=self._gripper_target,
            vel=self.config.gripper_velocity,
            max_tqu=self.config.gripper_max_torque,
        )

    def _start_impedance_loop(self) -> None:
        assert self._robot is not None
        if pin is None:
            raise ImportError(
                "pinocchio is required for panthera_arm cartesian impedance mode. "
                "Install pinocchio or disable `use_cartesian_impedance`."
            )
        required_attrs = ("model", "joint_names")
        missing_attrs = [name for name in required_attrs if not hasattr(self._robot, name)]
        if missing_attrs:
            raise RuntimeError(
                "Panthera SDK instance is missing required attributes for impedance mode: "
                f"{missing_attrs}"
            )
        required_methods = ("get_Gravity", "get_friction_compensation", "pos_vel_tqe_kp_kd")
        missing_methods = [name for name in required_methods if not hasattr(self._robot, name)]
        if missing_methods:
            raise RuntimeError(
                "Panthera SDK instance is missing required methods for impedance mode: "
                f"{missing_methods}"
            )
        self._validate_impedance_config()
        self._pin_data = self._robot.model.createData()
        self._impedance_enabled_event.set()
        self._impedance_stop_event.clear()
        self._impedance_thread = threading.Thread(
            target=self._impedance_loop,
            name="panthera_impedance_loop",
            daemon=True,
        )
        self._impedance_thread.start()
        logger.info(
            "Panthera cartesian impedance loop started at %.1f Hz.",
            self.config.impedance_control_hz,
        )

    def _run_impedance_startup_sequence(self) -> None:
        assert self._robot is not None
        robot = self._robot

        required_methods = ("Joint_Pos_Vel", "inverse_kinematics", "moveJ", "moveL", "rotation_matrix_from_euler")
        missing = [name for name in required_methods if not hasattr(robot, name)]
        if missing:
            logger.warning(
                "Skipping Panthera startup sequence because SDK methods are missing: %s",
                missing,
            )
            return

        try:
            motor_count = int(getattr(robot, "motor_count", 6))
            if motor_count <= 0:
                return

            zero_pos = [0.0] * motor_count
            zero_vel = list(self.config.joint_velocity[:motor_count])
            if len(zero_vel) < motor_count:
                zero_vel.extend([0.5] * (motor_count - len(zero_vel)))

            max_torque = None
            if self.config.max_torque:
                max_torque = list(self.config.max_torque[:motor_count])
                if len(max_torque) < motor_count:
                    max_torque.extend([10.0] * (motor_count - len(max_torque)))

            robot.Joint_Pos_Vel(zero_pos, zero_vel, max_torque, iswait=True)

            if len(self.config.startup_home_pos_m) != 3 or len(self.config.startup_home_euler_rad) != 3:
                raise ValueError(
                    "panthera_arm.startup_home_pos_m and startup_home_euler_rad must contain exactly 3 elements."
                )

            home_rot = robot.rotation_matrix_from_euler(*self.config.startup_home_euler_rad)
            q_init = robot.inverse_kinematics(
                target_position=self.config.startup_home_pos_m,
                target_rotation=home_rot,
                init_q=np.asarray(robot.get_current_pos(), dtype=float),
            )
            if q_init is not None:
                robot.moveJ(
                    q_init,
                    duration=self.config.startup_movej_duration_s,
                    max_tqu=max_torque,
                    iswait=True,
                )

            fk = robot.forward_kinematics()
            if fk is not None:
                p_lift = np.asarray(fk["position"], dtype=float)
                p_lift[2] += float(self.config.startup_lift_m)
                robot.moveL(
                    target_position=p_lift,
                    target_rotation=np.asarray(fk["rotation"], dtype=float),
                    duration=self.config.startup_lift_duration_s,
                    use_spline=True,
                )
        except Exception:
            logger.warning("Panthera startup sequence failed; continuing without startup pose sequence.", exc_info=True)

    def _validate_impedance_config(self) -> None:
        if self.config.impedance_control_hz <= 0:
            raise ValueError("panthera_arm.impedance_control_hz must be > 0.")
        if self.config.joint_limit_margin_rad < 0:
            raise ValueError("panthera_arm.joint_limit_margin_rad must be >= 0.")
        if self.config.impedance_max_consecutive_errors <= 0:
            raise ValueError("panthera_arm.impedance_max_consecutive_errors must be > 0.")
        if self.config.impedance_error_log_interval_s <= 0:
            raise ValueError("panthera_arm.impedance_error_log_interval_s must be > 0.")

        cartesian_vectors = (
            ("impedance_k_pos", self.config.impedance_k_pos),
            ("impedance_k_rot", self.config.impedance_k_rot),
            ("impedance_b_pos", self.config.impedance_b_pos),
            ("impedance_b_rot", self.config.impedance_b_rot),
        )
        for name, values in cartesian_vectors:
            if len(values) != 3:
                raise ValueError(f"panthera_arm.{name} must contain exactly 3 elements.")

        joint_vectors = (
            ("joint_damping", self.config.joint_damping),
            ("tau_limit", self.config.tau_limit),
            ("friction_fc", self.config.friction_fc),
            ("friction_fv", self.config.friction_fv),
        )
        for name, values in joint_vectors:
            if len(values) != 6:
                raise ValueError(f"panthera_arm.{name} must contain exactly 6 elements.")
        if len(self.config.tool_offset_m) != 3:
            raise ValueError("panthera_arm.tool_offset_m must contain exactly 3 elements.")
        if len(self.config.startup_home_pos_m) != 3:
            raise ValueError("panthera_arm.startup_home_pos_m must contain exactly 3 elements.")
        if len(self.config.startup_home_euler_rad) != 3:
            raise ValueError("panthera_arm.startup_home_euler_rad must contain exactly 3 elements.")

    def _impedance_loop(self) -> None:
        assert self._robot is not None
        robot = self._robot
        dt = 1.0 / self.config.impedance_control_hz
        consecutive_errors = 0
        alpha_dq = 1.0
        if self.config.dq_lpf_cutoff_hz > 0:
            alpha_dq = (2.0 * np.pi * self.config.dq_lpf_cutoff_hz * dt) / (
                1.0 + 2.0 * np.pi * self.config.dq_lpf_cutoff_hz * dt
            )

        k_cart = np.asarray(self.config.impedance_k_pos + self.config.impedance_k_rot, dtype=float)
        b_cart = np.asarray(self.config.impedance_b_pos + self.config.impedance_b_rot, dtype=float)
        tau_limit = np.asarray(self.config.tau_limit, dtype=float)
        joint_damping = np.asarray(self.config.joint_damping, dtype=float)
        friction_fc = np.asarray(self.config.friction_fc, dtype=float)
        friction_fv = np.asarray(self.config.friction_fv, dtype=float)
        damping_lambda_sq = float(self.config.impedance_lambda_damping) ** 2

        dq_filtered = np.zeros(6, dtype=float)
        dq_initialized = False
        with self._target_lock:
            if self._target_pos is None or self._target_rot is None:
                return
            last_feasible_pos = np.array(self._target_pos, dtype=float)
            last_feasible_rot = np.array(self._target_rot, dtype=float)

        joint_lower: np.ndarray | None = None
        joint_upper: np.ndarray | None = None
        if (
            self.config.enforce_joint_limit_margin
            and hasattr(robot, "joint_limits")
            and getattr(robot, "joint_limits", None)
        ):
            lower = np.asarray(robot.joint_limits.get("lower"), dtype=float)
            upper = np.asarray(robot.joint_limits.get("upper"), dtype=float)
            if lower.shape[0] >= 6 and upper.shape[0] >= 6:
                margin = float(self.config.joint_limit_margin_rad)
                joint_lower = lower[:6] + margin
                joint_upper = upper[:6] - margin
            else:
                logger.warning("Panthera joint_limits has unexpected shape; skipping joint limit margin checks.")

        while not self._impedance_stop_event.is_set():
            t0 = time.perf_counter()
            if not self._impedance_enabled_event.is_set():
                time.sleep(min(dt, 0.01))
                continue
            try:
                q = np.asarray(robot.get_current_pos(), dtype=float)
                dq_raw = np.asarray(robot.get_current_vel(), dtype=float)
                if q.shape[0] != 6 or dq_raw.shape[0] != 6:
                    raise RuntimeError(
                        "Panthera impedance mode expects 6 arm joints. "
                        f"Got q={q.shape}, dq={dq_raw.shape}"
                    )

                if not dq_initialized:
                    dq_filtered[:] = dq_raw
                    dq_initialized = True
                else:
                    dq_filtered[:] = alpha_dq * dq_raw + (1.0 - alpha_dq) * dq_filtered
                dq = dq_filtered

                if joint_lower is not None and joint_upper is not None:
                    violated = (q < joint_lower) | (q > joint_upper)
                    if np.any(violated):
                        with self._target_lock:
                            self._target_pos = np.array(last_feasible_pos, dtype=float)
                            self._target_rot = np.array(last_feasible_rot, dtype=float)

                        violated_parts = []
                        for i in range(6):
                            if violated[i]:
                                side = "lower" if q[i] < joint_lower[i] else "upper"
                                violated_parts.append(f"j{i+1}:{side}:{q[i]:+.3f}")

                        logger.error(
                            "Panthera joint limit margin violation detected (%s). Triggering immediate hardware stop.",
                            ", ".join(violated_parts),
                        )
                        try:
                            if hasattr(robot, "set_stop"):
                                robot.set_stop()
                            else:
                                logger.error(
                                    "Panthera SDK does not expose set_stop(); cannot issue immediate hardware stop."
                                )
                        except Exception:
                            logger.warning("Panthera set_stop failed during joint-limit emergency stop.", exc_info=True)
                        self._impedance_enabled_event.clear()
                        self._impedance_stop_event.set()
                        break

                p_cur, r_cur, jacobian = self._compute_fk_and_jacobian(q)
                with self._target_lock:
                    if self._target_pos is None or self._target_rot is None:
                        continue
                    target_pos = np.array(self._target_pos, dtype=float)
                    target_rot = np.array(self._target_rot, dtype=float)
                    gripper_target = float(self._gripper_target)
                last_feasible_pos = target_pos.copy()
                last_feasible_rot = target_rot.copy()

                err_pos = target_pos - p_cur
                err_rot = self._orientation_error_axis_angle(target_rot, r_cur)
                err = np.concatenate([err_pos, err_rot])

                cartesian_vel = jacobian @ dq
                cartesian_force = k_cart * err - b_cart * cartesian_vel
                jjt = jacobian @ jacobian.T
                alpha = np.linalg.solve(jjt + damping_lambda_sq * np.eye(6), cartesian_force)
                tau_cart = jacobian.T @ alpha

                tau_joint_damp = -joint_damping * dq
                tau_gravity = np.asarray(robot.get_Gravity(q), dtype=float)
                tau_coriolis = np.zeros(6, dtype=float)
                if self.config.enable_coriolis_comp and hasattr(robot, "get_Coriolis_vector"):
                    tau_coriolis = np.asarray(robot.get_Coriolis_vector(q, dq), dtype=float)
                tau_friction = np.asarray(
                    robot.get_friction_compensation(
                        dq,
                        friction_fc,
                        friction_fv,
                        self.config.friction_vel_threshold,
                    ),
                    dtype=float,
                )

                tau_cmd = np.clip(tau_cart + tau_joint_damp + tau_gravity + tau_coriolis + tau_friction, -tau_limit, tau_limit)
                zero = np.zeros(6, dtype=float)

                # Match the manufacturer script: set gripper motor target before arm torque command.
                if hasattr(robot, "Motors") and hasattr(robot, "gripper_id"):
                    robot.Motors[robot.gripper_id - 1].pos_vel_tqe_kp_kd(
                        gripper_target,
                        0.0,
                        0.0,
                        self.config.gripper_kp,
                        self.config.gripper_kd,
                    )
                else:
                    robot.gripper_control(
                        pos=gripper_target,
                        vel=self.config.gripper_velocity,
                        max_tqu=self.config.gripper_max_torque,
                    )

                robot.pos_vel_tqe_kp_kd(
                    zero.tolist(),
                    zero.tolist(),
                    tau_cmd.tolist(),
                    zero.tolist(),
                    zero.tolist(),
                )
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                now_s = time.perf_counter()
                if now_s - self._last_impedance_error_log_s >= self.config.impedance_error_log_interval_s:
                    logger.warning("Panthera impedance loop iteration failed: %s", exc, exc_info=True)
                    self._last_impedance_error_log_s = now_s
                if self.config.impedance_fail_safe_stop and consecutive_errors >= self.config.impedance_max_consecutive_errors:
                    logger.error(
                        "Panthera impedance loop entering fail-safe stop after %d consecutive errors.",
                        consecutive_errors,
                    )
                    try:
                        if hasattr(robot, "set_stop"):
                            robot.set_stop()
                    except Exception:
                        logger.warning("Panthera fail-safe stop command failed.", exc_info=True)
                    self._impedance_enabled_event.clear()
                    self._impedance_stop_event.set()
                    break

            sleep_s = dt - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _compute_fk_and_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._robot is not None
        assert self._pin_data is not None
        if pin is None:
            raise RuntimeError("pinocchio is required to compute Jacobian for impedance mode.")

        q_pin = np.zeros(self._robot.model.nq, dtype=float)
        for i, name in enumerate(self._robot.joint_names):
            joint_id = self._robot.model.getJointId(name)
            q_pin[self._robot.model.joints[joint_id].idx_q] = float(q[i])

        pin.computeJointJacobians(self._robot.model, self._pin_data, q_pin)
        last_joint_id = self._robot.model.getJointId(self._robot.joint_names[-1])
        last_tf = self._pin_data.oMi[last_joint_id]
        rot = np.asarray(last_tf.rotation, dtype=float)
        pos = np.asarray(last_tf.translation, dtype=float)

        tool_offset = np.asarray(self.config.tool_offset_m, dtype=float)
        tcp_offset_world = rot @ tool_offset
        tcp_pos = pos + tcp_offset_world

        jac_full = pin.getJointJacobian(
            self._robot.model,
            self._pin_data,
            last_joint_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        jac_tcp = np.asarray(jac_full, dtype=float).copy()
        jac_tcp[:3, :] -= self._skew(tcp_offset_world) @ jac_full[3:, :]

        cols = [
            self._robot.model.joints[self._robot.model.getJointId(name)].idx_v for name in self._robot.joint_names
        ]
        jacobian = jac_tcp[:, cols]

        return tcp_pos, rot, jacobian

    @staticmethod
    def _orientation_error_axis_angle(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
        rot_err = current_rot.T @ target_rot
        rotvec = Rot.from_matrix(rot_err).as_rotvec()
        return current_rot @ rotvec

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=float,
        )

    def _resolve_sdk_python_dir(self) -> Path:
        sdk_python_dir = Path(self.config.sdk_python_dir).expanduser()
        if not sdk_python_dir.is_absolute():
            sdk_python_dir = (Path.cwd() / sdk_python_dir).resolve()
        if not sdk_python_dir.exists():
            raise FileNotFoundError(
                "Panthera SDK python directory not found: "
                f"{sdk_python_dir}. Set `sdk_python_dir` in panthera_arm config."
            )
        return sdk_python_dir

    def _resolve_config_path(self, sdk_python_dir: Path) -> str | None:
        if not self.config.config_path:
            return None

        path_in_cfg = Path(self.config.config_path).expanduser()
        candidates: list[Path] = []
        if path_in_cfg.is_absolute():
            candidates.append(path_in_cfg.resolve())
        else:
            candidates.append((Path.cwd() / path_in_cfg).resolve())
            # Manufacturer reference keeps config files under panthera_python/robot_param.
            candidates.append((sdk_python_dir.parent / path_in_cfg).resolve())

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            "Panthera config file not found. Checked: "
            + ", ".join(str(path) for path in candidates)
            + ". Update `config_path` in panthera_arm config."
        )

    def _import_panthera_class(self, sdk_python_dir: Path):
        sdk_path = str(sdk_python_dir)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)

        try:
            from Panthera_lib import Panthera
        except Exception as exc:
            raise ImportError(
                "Failed to import Panthera SDK (`from Panthera_lib import Panthera`). "
                f"Verify sdk_python_dir and dependencies. sdk_python_dir={sdk_python_dir}"
            ) from exc
        return Panthera
