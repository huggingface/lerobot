from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class Task2Sim:
    """Minimal MuJoCo wrapper used by the local bimanual simulation bridge."""

    def __init__(
        self,
        xml_path: str | Path,
        robot_dofs: int = 6,
        render_size: tuple[int, int] | None = (480, 640),
        cube_raise_z: float = 0.05,
        substeps: int = 1,
        launch_viewer: bool = True,
        show_sites: bool = True,
        use_home_pose: bool = False,
        home_qpos: np.ndarray | None = None,
        home_ctrl: np.ndarray | None = None,
    ):
        self.xml_path = Path(xml_path).resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")

        self.robot_dofs = int(robot_dofs)
        self.substeps = int(substeps)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.ctrl_range = np.asarray(self.model.actuator_ctrlrange, dtype=float).copy()

        mujoco.mj_resetData(self.model, self.data)

        self.num_arms = max(1, int(self.model.nu) // self.robot_dofs)
        self.active_arm = 0
        self.viewer = None
        self.images = {}

        if render_size is None:
            self._renderer = None
        else:
            height, width = render_size

        # if use_home_pose and home_qpos is not None:
        #     self.apply_home_pose(home_qpos, home_ctrl)

        self._raise_cube_if_possible(float(cube_raise_z))
        # mujoco.mj_forward(self.model, self.data)

        if launch_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self._key_callback,
            )
            if show_sites:
                try:
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SITE] = 1
                except Exception:
                    pass

    def _key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except Exception:
            return

        if key == "t" and self.num_arms > 0:
            self.active_arm = (self.active_arm + 1) % self.num_arms
            print(f"[keyboard] active_arm={self.active_arm + 1}/{self.num_arms}")

    def _raise_cube_if_possible(self, z: float) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if body_id < 0:
            return

        joint_count = int(self.model.body_jntnum[body_id])
        if joint_count <= 0:
            return

        joint_id = int(self.model.body_jntadr[body_id])
        if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            return

        qadr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qadr : qadr + 3] = np.array([0.2, 0.2, z], dtype=float)
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def apply_home_pose(self, home_qpos: np.ndarray, home_ctrl: np.ndarray | None = None) -> None:
        qpos = np.asarray(home_qpos, dtype=float).reshape(-1)
        ctrl = qpos if home_ctrl is None else np.asarray(home_ctrl, dtype=float).reshape(-1)

        joint_count = min(self.robot_dofs, qpos.size)
        for arm_index in range(self.num_arms):
            for joint_offset in range(joint_count):
                actuator_index = arm_index * self.robot_dofs + joint_offset
                if actuator_index >= int(self.model.nu):
                    break

                joint_id = int(self.model.actuator_trnid[actuator_index, 0])
                qadr = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[qadr] = qpos[joint_offset]

                dofadr = int(self.model.jnt_dofadr[joint_id])
                if 0 <= dofadr < int(self.model.nv):
                    self.data.qvel[dofadr] = 0.0

        if int(self.model.nu) > 0:
            ctrl_target = self.data.ctrl.copy()
            for arm_index in range(self.num_arms):
                start = arm_index * self.robot_dofs
                end = min(start + min(self.robot_dofs, ctrl.size), int(self.model.nu))
                if start >= end:
                    continue
                ctrl_target[start:end] = np.clip(
                    ctrl[: end - start],
                    self.ctrl_range[start:end, 0],
                    self.ctrl_range[start:end, 1],
                )
            self.data.ctrl[:] = ctrl_target

        mujoco.mj_forward(self.model, self.data)

    def _render_images(self):
        if self._renderer is None:
            return {}

        for camera_name in ("camera_front", "camera_top", "front", "top", "camera_vizu", "vizu"):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                print(camera_name)
                continue
            self._renderer.update_scene(self.data, camera=camera_name)
            renderer_var = self._renderer.render().copy()
            self.images[camera_name] = renderer_var
        
    def get_images(self):
        return self.images

    def step(self) -> None:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            self._render_images()
        if self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
