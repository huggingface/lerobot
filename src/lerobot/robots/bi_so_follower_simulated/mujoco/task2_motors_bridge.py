from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mujoco_task2 import Task2Sim


@dataclass(frozen=True)
class _HomePose:
    qpos: np.ndarray
    ctrl: np.ndarray


@dataclass
class _SharedState:
    qpos_deg: np.ndarray
    images: dict[str, np.ndarray]


class Task2SharedBackend:
    """Shared MuJoCo backend for one or more per-arm bus views."""

    def __init__(
        self,
        xml_path: str,
        robot_dofs: int = 6,
        render_size: tuple[int, int] | None = (480, 640),
        realtime: bool = True,
        slowmo: float = 1.0,
        launch_viewer: bool = False,
    ):
        self.xml_path = Path(xml_path).resolve()
        self.sim = Task2Sim(
            xml_path=self.xml_path,
            robot_dofs=robot_dofs,
            render_size=render_size,
            launch_viewer=launch_viewer,
            show_sites=True,
        )
        self.model = self.sim.model
        self.data = self.sim.data
        self.render_size = render_size

        self.robot_dofs = int(robot_dofs)
        self.nu = int(self.model.nu)
        self.num_arms = max(1, self.nu // self.robot_dofs)

        self.realtime = bool(realtime)
        self.slowmo = float(slowmo)

        self._ctrl_target = np.zeros(self.nu, dtype=float)
        # self._apply_startup_pose()

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._refcount = 0
        self._state = _SharedState(
            qpos_deg=np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32),
            images={},
        )

    def _home_key_id(self) -> int:
        if int(self.model.nkey) <= 0:
            return -1
        try:
            return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        except Exception:
            return -1

    def _read_actuated_joint_qpos_rad(self) -> np.ndarray:
        qpos_rad = np.zeros(self.nu, dtype=float)
        for actuator_index in range(self.nu):
            joint_id = int(self.model.actuator_trnid[actuator_index, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            qpos_rad[actuator_index] = float(self.data.qpos[qadr])
        return qpos_rad

    def _extract_first_arm_home_pose(self) -> _HomePose | None:
        key_id = self._home_key_id()
        if key_id < 0:
            return None

        model_key_qpos = np.asarray(self.model.key_qpos[key_id], dtype=float)
        model_key_ctrl = np.asarray(self.model.key_ctrl[key_id], dtype=float)

        qpos = np.zeros(self.robot_dofs, dtype=float)
        ctrl = np.zeros(self.robot_dofs, dtype=float)

        for joint_offset in range(self.robot_dofs):
            actuator_index = joint_offset
            if actuator_index >= self.nu:
                break

            joint_id = int(self.model.actuator_trnid[actuator_index, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            if qadr >= model_key_qpos.size:
                return None

            qpos[joint_offset] = float(model_key_qpos[qadr])
            if actuator_index < model_key_ctrl.size:
                ctrl[joint_offset] = float(model_key_ctrl[actuator_index])
            else:
                ctrl[joint_offset] = qpos[joint_offset]

        return _HomePose(qpos=qpos, ctrl=ctrl)

    def _apply_startup_pose(self) -> None:
        home_pose = self._extract_first_arm_home_pose()
        if home_pose is not None:
            self.sim.apply_home_pose(home_pose.qpos, home_pose.ctrl)

        self._ctrl_target[:] = np.asarray(self.data.ctrl, dtype=float)
        if self.nu > 0:
            lo = self.model.actuator_ctrlrange[:, 0]
            hi = self.model.actuator_ctrlrange[:, 1]
            self._ctrl_target[:] = np.clip(self._ctrl_target, lo, hi)
            self.data.ctrl[:] = self._ctrl_target

        mujoco.mj_forward(self.model, self.data)

    def start(self) -> None:
        with self._lock:
            self._refcount += 1
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._refcount = max(0, self._refcount - 1)
            if self._refcount > 0:
                return
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        self.sim.close()

    def set_arm_target_deg(self, arm_index: int, q_deg: np.ndarray) -> None:
        q_deg = np.asarray(q_deg, dtype=float).reshape(-1)
        if q_deg.size != self.robot_dofs:
            raise ValueError(f"Expected {self.robot_dofs} values, got {q_deg.size}")

        start = arm_index * self.robot_dofs
        end = start + self.robot_dofs
        if end > self.nu:
            raise ValueError(f"arm_index {arm_index} out of range for nu={self.nu}")

        q_rad = np.deg2rad(q_deg)
        with self._lock:
            lo = self.model.actuator_ctrlrange[start:end, 0]
            hi = self.model.actuator_ctrlrange[start:end, 1]
            self._ctrl_target[start:end] = np.clip(q_rad, lo, hi)

    def get_state(self) -> _SharedState:
        with self._lock:
            return _SharedState(
                qpos_deg=self._state.qpos_deg.copy(),
                images={name: image.copy() for name, image in self._state.images.items()},
            )

    def _loop(self) -> None:
        '''
            the renderer has to be initialised here otherwise it will give a Context error
            initialisation and rendering should be done in the same thread
        '''
        dt = float(self.model.opt.timestep) * int(self.sim.substeps)
        self.sim._renderer = mujoco.Renderer(self.model, height = self.render_size[1], width = self.render_size[0])

        while True:
            with self._lock:
                if not self._running:
                    break
                if self.nu > 0:
                    self.data.ctrl[:] = self._ctrl_target

            tick_start = time.time()
            self.sim.step()

            with self._lock:
                self._state.qpos_deg = np.rad2deg(self._read_actuated_joint_qpos_rad()).astype(np.float32)
                self._state.images = self.sim.get_images()

            if self.realtime:
                elapsed = time.time() - tick_start
                time.sleep(max(0.0, dt * self.slowmo - elapsed))


class Task2ArmBus:
    """Per-arm bus wrapper matching the expected read/write interface."""

    def __init__(self, backend: Task2SharedBackend, arm_index: int):
        self.backend = backend
        self.arm_index = int(arm_index)
        self.robot_dofs = backend.robot_dofs

        base = "joint_"
        suffix = "" if arm_index == 0 else "_r"
        self._motor_names = [f"{base}{i}{suffix}" for i in range(1, self.robot_dofs + 1)]

    @property
    def motor_names(self) -> list[str]:
        return self._motor_names

    def connect(self) -> None:
        self.backend.start()

    def disconnect(self) -> None:
        self.backend.stop()

    def read(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        state = self.backend.get_state()
        start = self.arm_index * self.robot_dofs
        end = start + self.robot_dofs
        return state.qpos_deg[start:end], state.images

    def write(self, *args, **kwargs) -> None:
        del kwargs
        if len(args) == 1:
            values = args[0]
        elif len(args) == 2:
            values = args[1]
        else:
            raise TypeError("write expects write(values) or write(group_name, values)")

        self.backend.set_arm_target_deg(self.arm_index, np.asarray(values, dtype=float))


def make_task2_bimanual_buses(
    xml_path: str,
    robot_dofs: int = 6,
    render_size: tuple[int, int] | None = (480, 640),
    realtime: bool = True,
    slowmo: float = 1.0,
    launch_viewer: bool = False,
) -> tuple[Task2SharedBackend, dict[str, Task2ArmBus]]:
    backend = Task2SharedBackend(
        xml_path=xml_path,
        robot_dofs=robot_dofs,
        render_size=render_size,
        realtime=realtime,
        slowmo=slowmo,
        launch_viewer=launch_viewer,
    )
    buses = {f"arm{i}": Task2ArmBus(backend, i) for i in range(backend.num_arms)}
    return backend, buses
