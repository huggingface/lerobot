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

"""MuJoCo-backed SO-101 stand-in.

Implements the ``Robot`` contract against a MuJoCo simulation so the async RTC
rollout pipeline can run without physical hardware. Observation/action keys are
kept identical to ``so101_follower`` so the same policy and processors apply.
"""

import logging
from functools import cached_property

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_sim_so101 import SimSO101Config

logger = logging.getLogger(__name__)

# Motor order is the source of truth shared with so_follower.py.
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY_MOTORS = MOTORS[:-1]
GRIPPER = "gripper"


class SimSO101(Robot):
    config_class = SimSO101Config
    name = "sim_so101"

    def __init__(self, config: SimSO101Config):
        super().__init__(config)
        self.config = config
        self._model = None
        self._data = None
        self._renderers: dict[str, object] = {}
        self._actuator_ids: dict[str, int] = {}
        self._qpos_addr: dict[str, int] = {}
        self._gripper_range: tuple[float, float] | None = None
        self._n_substeps = 1
        self._success_qpos_addr: int | None = None
        self._success_dofadr: int | None = None
        self._success_rest_z: float | None = None
        self._box_bounds: tuple[float, float, float, float, float] | None = None
        self._belt_qpos_addr: int | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTORS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {key: (cam.height, cam.width, 3) for key, cam in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def cameras(self) -> dict[str, object]:
        return self.config.cameras

    @property
    def is_connected(self) -> bool:
        return self._data is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import mujoco

        self._model = mujoco.MjModel.from_xml_path(self.config.mjcf_path)
        self._data = mujoco.MjData(self._model)

        # Slide the whole pick-and-place layout to honor belt_distance (gap from the
        # origin to the belt's near edge). Shifting the conveyor frame, the moving belt
        # surface and the box by the same delta preserves their relative spacing (the
        # box stays just beyond the belt's far edge); the cube's start x is matched to
        # the belt centre after the keyframe reset below. No-op without a "belt" body.
        belt_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "belt")
        belt_center_x: float | None = None
        if belt_body_id >= 0:
            belt_top_geom = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "belt_top")
            belt_half_x = float(self._model.geom_size[belt_top_geom][0])
            belt_center_x = self.config.belt_distance + belt_half_x
            dx = belt_center_x - float(self._model.body_pos[belt_body_id][0])
            for body_name in ("conveyor_frame", "belt", "box"):
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    self._model.body_pos[bid][0] += dx

        # If the MJCF defines a "home" keyframe, start there instead of the MJCF's
        # qpos=0 default. Joint `ref` attributes don't help here: they only recalibrate
        # what a given qpos *number* means, not the actual rest configuration (`ref`
        # leaves the compiled qpos=ref pose geometrically identical to qpos=0 without it),
        # so a keyframe is the only way to give the rollout a deliberate starting pose
        # (e.g. one where an eye-in-hand camera actually faces the work area instead of
        # wherever zero-angle-on-every-joint happens to point it).
        home_key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, home_key_id)

        # Place the cube relative to the (possibly shifted) belt: always on the belt
        # centre line in x, and in y either parked directly in front of the robot
        # (y=0, graspable for static eval) when the belt is stopped, or fed from the
        # -y end so it travels through the reachable region when the belt is running.
        if belt_center_x is not None:
            cube_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
            if cube_joint_id >= 0:
                cube_qadr = int(self._model.jnt_qposadr[cube_joint_id])
                self._data.qpos[cube_qadr] = belt_center_x
                self._data.qpos[cube_qadr + 1] = -0.20 if self.config.belt_speed != 0 else 0.0

        # Conveyor belt (e.g. scene_cube.xml's "belt"): a velocity actuator's ctrl is a
        # standing command, not a per-step one, so set it once here rather than in
        # send_action(). No-op on scenes without a "belt_motor" actuator.
        belt_act_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "belt_motor")
        if belt_act_id >= 0:
            self._data.ctrl[belt_act_id] = self.config.belt_speed
            # The slide joint would physically translate the belt body (and its rendered
            # geom) through world space, so the green surface itself drifts off-screen
            # within seconds. Instead we run it as a treadmill: send_action() snaps the
            # slide position back to 0 every control step while leaving its velocity
            # alone, so the contact still sees a moving surface (friction drags resting
            # objects at belt speed) but the surface never visibly moves — the separate
            # static frame geoms are what make it read as a conveyor.
            belt_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "belt_slide")
            self._belt_qpos_addr = int(self._model.jnt_qposadr[belt_joint_id])

        # Resolve so101 motor -> MuJoCo actuator id + joint qpos address. Fail fast
        # if the scene's naming doesn't match joint_map.
        for motor, sim_name in self.config.joint_map.items():
            act_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, sim_name)
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            if act_id < 0 or joint_id < 0:
                raise ValueError(
                    f"MJCF '{self.config.mjcf_path}' has no actuator/joint named '{sim_name}' "
                    f"(mapped from motor '{motor}'). Check SimSO101Config.joint_map."
                )
            self._actuator_ids[motor] = act_id
            self._qpos_addr[motor] = int(self._model.jnt_qposadr[joint_id])

        gripper_joint = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, self.config.joint_map[GRIPPER]
        )
        self._gripper_range = tuple(float(x) for x in self._model.jnt_range[gripper_joint])

        self._n_substeps = max(1, round((1.0 / self.config.control_fps) / self._model.opt.timestep))

        for key, cam in self.config.cameras.items():
            self._renderers[key] = mujoco.Renderer(self._model, height=cam.height, width=cam.width)

        mujoco.mj_forward(self._model, self._data)

        if self.config.success is not None:
            body_name = self.config.success.body_name
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(
                    f"MJCF '{self.config.mjcf_path}' has no body named '{body_name}' "
                    f"(SimSO101Config.success.body_name)."
                )
            joint_id = self._model.body_jntadr[body_id]
            if joint_id < 0 or self._model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(
                    f"Body '{body_name}' has no freejoint — success tracking needs a freejoint "
                    f"body (e.g. the 'cube' body in scene_cube.xml) to read its z position from."
                )
            self._success_qpos_addr = int(self._model.jnt_qposadr[joint_id])
            self._success_dofadr = int(self._model.jnt_dofadr[joint_id])
            self._success_rest_z = float(self._data.qpos[self._success_qpos_addr + 2])

            criterion = self.config.success.criterion
            if criterion == "place_in_box":
                # Precompute the box's world-frame AABB from its (axis-aligned) geoms:
                # horizontal footprint + top rim. check_success() then counts the body
                # as "placed" when it's settled within that footprint and below the rim
                # — the solid walls mean anything inside the footprint and below the rim
                # is physically in the cavity, not beside/on the box.
                box_name = self.config.success.box_body_name
                box_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, box_name)
                if box_id < 0:
                    raise ValueError(
                        f"MJCF '{self.config.mjcf_path}' has no body named '{box_name}' "
                        f"(SimSO101Config.success.box_body_name), required for criterion "
                        f"'place_in_box'."
                    )
                geom_ids = [g for g in range(self._model.ngeom) if self._model.geom_bodyid[g] == box_id]
                if not geom_ids:
                    raise ValueError(f"Box body '{box_name}' has no geoms to bound its cavity.")
                xmin = ymin = float("inf")
                xmax = ymax = rim = float("-inf")
                for g in geom_ids:
                    gx, gy, gz = (float(v) for v in self._data.geom_xpos[g])
                    hx, hy, hz = (float(v) for v in self._model.geom_size[g])
                    xmin, xmax = min(xmin, gx - hx), max(xmax, gx + hx)
                    ymin, ymax = min(ymin, gy - hy), max(ymax, gy + hy)
                    rim = max(rim, gz + hz)
                self._box_bounds = (xmin, xmax, ymin, ymax, rim)
            elif criterion != "lift":
                raise ValueError(
                    f"Unknown success criterion '{criterion}' "
                    f"(SimSO101Config.success.criterion); expected 'lift' or 'place_in_box'."
                )

        logger.info(f"{self} connected (sim, {self._n_substeps} substeps per control step).")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}
        for motor in BODY_MOTORS:
            rad = float(self._data.qpos[self._qpos_addr[motor]])
            obs[f"{motor}.pos"] = float(np.rad2deg(rad)) if self.config.use_degrees else rad
        obs[f"{GRIPPER}.pos"] = self._gripper_sim_to_robot(float(self._data.qpos[self._qpos_addr[GRIPPER]]))

        for key, renderer in self._renderers.items():
            renderer.update_scene(self._data, camera=self.config.cameras[key].mujoco_name)
            obs[key] = renderer.render()  # (H, W, 3) uint8

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        import mujoco

        for motor in BODY_MOTORS:
            val = action[f"{motor}.pos"]
            self._data.ctrl[self._actuator_ids[motor]] = (
                float(np.deg2rad(val)) if self.config.use_degrees else val
            )
        self._data.ctrl[self._actuator_ids[GRIPPER]] = self._gripper_robot_to_sim(action[f"{GRIPPER}.pos"])

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        if self._belt_qpos_addr is not None:
            # Treadmill: zero the belt's position (not its velocity) so the surface
            # drags objects without visibly translating. mj_forward re-derives
            # xpos/cam_xpos now, since get_observation() renders before the next step.
            self._data.qpos[self._belt_qpos_addr] = 0.0
            mujoco.mj_forward(self._model, self._data)

        return {key: val for key, val in action.items() if key.endswith(".pos")}

    @check_if_not_connected
    def check_success(self) -> bool:
        """Whether the configured success criterion is currently met.

        Privileged sim-state read for evaluation only — never passed through
        `get_observation`, so it has no effect on the policy. Returns False
        when `SimSO101Config.success` is unset (no tracked body).
        """
        if self._success_qpos_addr is None:
            return False
        addr = self._success_qpos_addr
        if self.config.success.criterion == "place_in_box":
            cx, cy, cz = (float(self._data.qpos[addr + i]) for i in range(3))
            xmin, xmax, ymin, ymax, rim = self._box_bounds
            if not (xmin < cx < xmax and ymin < cy < ymax and cz < rim):
                return False
            vel = self._data.qvel[self._success_dofadr : self._success_dofadr + 3]
            return bool(np.linalg.norm(vel) < self.config.success.settle_speed_mps)
        # default: "lift"
        z = float(self._data.qpos[addr + 2])
        return (z - self._success_rest_z) >= self.config.success.height_m

    def _gripper_sim_to_robot(self, rad: float) -> float:
        lo, hi = self._gripper_range
        return 100.0 * (rad - lo) / (hi - lo)

    def _gripper_robot_to_sim(self, val: float) -> float:
        lo, hi = self._gripper_range
        return lo + (val / 100.0) * (hi - lo)

    @check_if_not_connected
    def disconnect(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}
        self._model = None
        self._data = None
        self._success_qpos_addr = None
        self._success_dofadr = None
        self._success_rest_z = None
        self._box_bounds = None
        logger.info(f"{self} disconnected.")
