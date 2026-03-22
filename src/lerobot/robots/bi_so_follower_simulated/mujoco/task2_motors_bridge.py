# task2_motors_bridge.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

# IMPORTANT: adjust this import if your Task2Sim file has a different name
from mujoco_task2 import Task2Sim


@dataclass
class _SharedState:
    qpos_deg: np.ndarray          # shape (nu,) in degrees (actuator order)
    images: dict[str, np.ndarray] # {"front": HxWx3, "top": HxWx3}


class Task2SharedBackend:
    """
    One MuJoCo sim instance (with 1 or 2 arms).
    Multiple Task2ArmBus objects share this backend.
    """

    def __init__(
        self,
        xml_path: str,
        robot_dofs: int = 6,
        render_size: tuple[int, int] | None = (480, 640),
        realtime: bool = True,
        slowmo: float = 1.0,
        launch_viewer: bool = False,
    ):
        xml_path_resolved = str(Path(xml_path).resolve())
        self.sim = Task2Sim(
            xml_path=xml_path_resolved,
            robot_dofs=robot_dofs,
            launch_viewer=launch_viewer,
            show_sites=True,
            use_home_pose=False,
        )
        self.model = self.sim.model
        self.data = self.sim.data

        self.robot_dofs = robot_dofs
        self.nu = int(self.model.nu)
        self.num_arms = max(1, self.nu // robot_dofs)

        # Shared target for ALL actuators (radians). Initialize to "home" so the model holds
        # the XML-defined pose (and, if present, replicate the `home` keyframe pose from `so_arm100.xml`).
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self._ctrl_target = np.zeros(self.nu, dtype=float)

        def _find_home_joint_from_xml(path: str, dofs: int) -> np.ndarray | None:
            visited: set[str] = set()

            def parse_file(p: Path) -> np.ndarray | None:
                p = p.resolve()
                p_str = str(p)
                if p_str in visited or not p.exists():
                    return None
                visited.add(p_str)

                try:
                    root = ET.parse(p_str).getroot()
                except Exception:
                    return None

                best = None
                best_score = -1
                for key in root.findall(".//key"):
                    if key.get("name") != "home":
                        continue
                    qpos = key.get("qpos")
                    if not qpos:
                        continue
                    try:
                        vals = [float(x) for x in qpos.split()]
                    except Exception:
                        continue

                    if len(vals) < dofs:
                        continue

                    # Prefer keys that look like a single-arm pose (either fixed-base or free-base).
                    score = 0
                    if len(vals) == dofs:
                        score = 3
                    elif len(vals) == dofs + 7:
                        score = 3
                    elif len(vals) >= dofs:
                        score = 1

                    if score > best_score:
                        best = np.asarray(vals[-dofs:], dtype=float)
                        best_score = score
                        if best_score >= 3:
                            break

                if best is not None:
                    return best

                for inc in root.findall(".//include"):
                    inc_file = inc.get("file")
                    if not inc_file:
                        continue
                    found = parse_file((p.parent / inc_file).resolve())
                    if found is not None:
                        return found

                return None

            return parse_file(Path(path))

        home_joint = _find_home_joint_from_xml(xml_path_resolved, self.robot_dofs)

        # Set ctrl target to current joint angles (XML qpos0), and optionally overwrite to `home`.
        qpos_rad = np.zeros(self.nu, dtype=float)
        for a in range(self.nu):
            j_id = int(self.model.actuator_trnid[a, 0])
            qadr = int(self.model.jnt_qposadr[j_id])
            qpos_rad[a] = float(self.data.qpos[qadr])

        if home_joint is not None and self.num_arms > 0:
            # Apply home joint angles to ALL arms (including right arm which may not be in the keyframe).
            for arm in range(self.num_arms):
                for j in range(self.robot_dofs):
                    a = arm * self.robot_dofs + j
                    if a >= self.nu:
                        break
                    j_id = int(self.model.actuator_trnid[a, 0])
                    qadr = int(self.model.jnt_qposadr[j_id])
                    self.data.qpos[qadr] = home_joint[j]
                    dofadr = int(self.model.jnt_dofadr[j_id])
                    if dofadr >= 0 and dofadr < self.model.nv:
                        self.data.qvel[dofadr] = 0.0
                    qpos_rad[a] = home_joint[j]
            mujoco.mj_forward(self.model, self.data)

        self._ctrl_target[:] = np.clip(qpos_rad, lo, hi)
        if self.nu > 0:
            self.data.ctrl[:] = self._ctrl_target[:]

        # Renderer (optional but useful for LeRobot cameras)
        if render_size is None:
            self._renderer = None
        else:
            h, w = render_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._refcount = 0

        self.realtime = realtime
        self.slowmo = float(slowmo)

        # Cache of last state (degrees + images)
        self._state = _SharedState(
            qpos_deg=np.rad2deg(qpos_rad).astype(np.float32),
            images={},
        )

    def start(self):
        with self._lock:
            self._refcount += 1
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            self._refcount = max(0, self._refcount - 1)
            if self._refcount > 0:
                return
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        self.sim.close()

    def set_arm_target_deg(self, arm_index: int, q_deg: np.ndarray):
        """Set goal for one arm (degrees)."""
        q_deg = np.asarray(q_deg, dtype=float).reshape(-1)
        if q_deg.size != self.robot_dofs:
            raise ValueError(f"Expected {self.robot_dofs} values, got {q_deg.size}")

        s = arm_index * self.robot_dofs
        e = s + self.robot_dofs
        if e > self.nu:
            raise ValueError(f"arm_index {arm_index} out of range for nu={self.nu}")

        q_rad = np.deg2rad(q_deg)

        with self._lock:
            lo = self.model.actuator_ctrlrange[s:e, 0]
            hi = self.model.actuator_ctrlrange[s:e, 1]
            self._ctrl_target[s:e] = np.clip(q_rad, lo, hi)

    def get_state(self) -> _SharedState:
        with self._lock:
            # return copies so caller can’t race the sim thread
            return _SharedState(
                qpos_deg=self._state.qpos_deg.copy(),
                images={k: v.copy() for k, v in self._state.images.items()},
            )

    def _loop(self):
        dt = float(self.model.opt.timestep) * int(self.sim.substeps)

        while True:
            with self._lock:
                if not self._running:
                    break
                # apply ctrl targets
                self.data.ctrl[:] = self._ctrl_target[:]

            t0 = time.time()
            self.sim.step()

            # --- Update cached state (read each actuator's joint angle robustly) ---
            with self._lock:
                qpos_rad = np.zeros(self.nu, dtype=float)

                for a in range(self.nu):
                    # actuator_trnid[a, 0] is the JOINT id that this actuator drives
                    j_id = int(self.model.actuator_trnid[a, 0])
                    qadr = int(self.model.jnt_qposadr[j_id])
                    qpos_rad[a] = float(self.data.qpos[qadr])

                self._state.qpos_deg = np.rad2deg(qpos_rad).astype(np.float32)

                # Render cameras if present (try common names)
                imgs = {}
                if self._renderer is not None:
                    for cam in ("camera_front", "camera_top", "front", "top"):
                        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                        if cam_id >= 0:
                            self._renderer.update_scene(self.data, camera=cam)
                            imgs[cam] = self._renderer.render().copy()

                self._state.images = imgs
            # -------------------------------------------------------------

            if self.realtime:
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt * self.slowmo - elapsed))


class Task2ArmBus:
    """
    A per-arm view of the shared backend.
    Implements connect/disconnect/read/write like a motors bus.
    """

    def __init__(self, backend: Task2SharedBackend, arm_index: int):
        self.backend = backend
        self.arm_index = int(arm_index)
        self.robot_dofs = backend.robot_dofs

        # Optional: expose motor_names like other buses do
        base = "joint_"
        suffix = "" if arm_index == 0 else "_r"
        self._motor_names = [f"{base}{i}{suffix}" for i in range(1, self.robot_dofs + 1)]

    @property
    def motor_names(self) -> list[str]:
        return self._motor_names

    def connect(self):
        self.backend.start()

    def disconnect(self):
        self.backend.stop()

    def read(self):
        """
        Return (qpos_deg_for_this_arm, images_dict).
        Matches what SimulatedRobot expects: q_pos, rendered_images = sim.read()
        """
        st = self.backend.get_state()
        s = self.arm_index * self.robot_dofs
        e = s + self.robot_dofs
        return st.qpos_deg[s:e], st.images

    def write(self, *args, **kwargs):
        """
        Accept BOTH calling styles:
          - write(values)
          - write("Goal_Position", values)
        """
        if len(args) == 1:
            values = args[0]
        elif len(args) == 2:
            # group_name = args[0]  # ignored
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
):
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
