from __future__ import annotations

import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer

@dataclass
class Observation:
    time: float
    joint_qpos: np.ndarray
    joint_qvel: np.ndarray
    cube_pos: np.ndarray
    ee_pos: np.ndarray | None


class Task2Sim:
    """
    Supervisor Task 2: single file with subtasks a/b/c/d.

    a) init + open viewer only (frozen, no stepping)
    b) add simulation loop (cube falls)
    c) add get_observation() + print joint state and cube position
    d) "do something": IK control from script (move end-effector to cube then to target)
    """

    def __init__(
        self,
        xml_path: str | Path,
        robot_dofs: int = 6,
        cube_raise_z: float = 0.05,
        substeps: int = 1,
        launch_viewer: bool = True,
        show_sites: bool = True,
        use_home_pose: bool = False,
        home_qpos: np.ndarray | None = None,
    ):
        self.xml_path = Path(xml_path).resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")

        self.robot_dofs = int(robot_dofs)
        self.substeps = int(substeps)

        # Load MuJoCo model + data
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Actuator ranges for safety clipping
        self.ctrl_range = self.model.actuator_ctrlrange.copy()  # (nu,2)

        mujoco.mj_resetData(self.model, self.data)

        # Number of arms in model (assumes nu is multiple of robot_dofs)
        self.num_arms = max(1, self.model.nu // self.robot_dofs)
        self.active_arm = 0  # 0 = arm1, 1 = arm2


        # Optional: start from a nicer pose
        if use_home_pose:
            if home_qpos is None:
                # Reasonable default that fits your typical ranges
                home_qpos = np.array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0], dtype=float)
            self.home_qpos = home_qpos.copy()
            self._apply_home_pose(home_qpos)

        # Optional: raise cube so it visibly falls in task b/c
        self._raise_cube_if_possible(z=float(cube_raise_z))
        mujoco.mj_forward(self.model, self.data)

        # --- Keyboard control state ---
        self.paused = False
        self.selected_joint = 0          # 0..5
        self.key_step = 0.05             # radians per keypress (tune this)

        lo = self.ctrl_range[:, 0]
        hi = self.ctrl_range[:, 1]
        # Initialize ctrl_target from current actuator controls (more reliable than qpos indexing)
        self.ctrl_target = 0.5 * (lo + hi)  # default mid-range
        if self.model.nu > 0:
            self.ctrl_target[:] = np.clip(self.data.ctrl.copy(), lo, hi)
        # ----------------------------------------

        self.viewer = None
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

    # ------------------ helpers ------------------
    def _apply_home_pose(self, home_qpos: np.ndarray):
        home_qpos = np.asarray(home_qpos, dtype=float)
        n = min(self.robot_dofs, home_qpos.shape[0])

        # Apply same home pose to ALL arms (assumes arm joints are packed first in qpos/ctrl)
        for arm in range(self.num_arms):
            qs = arm * self.robot_dofs
            qe = qs + n
            if qe <= self.model.nq:
                self.data.qpos[qs:qe] = home_qpos[:n]

        # Hold via actuators
        nu = self.model.nu
        if nu > 0:
            ctrl = self.data.ctrl.copy()
            for arm in range(self.num_arms):
                cs = arm * self.robot_dofs
                ce = min(cs + n, nu)
                if cs < nu:
                    ctrl[cs:ce] = np.clip(
                        home_qpos[: (ce - cs)],
                        self.ctrl_range[cs:ce, 0],
                        self.ctrl_range[cs:ce, 1],
                    )
            self.data.ctrl[:] = ctrl

        mujoco.mj_forward(self.model, self.data)


    def _raise_cube_if_possible(self, z: float) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if body_id < 0:
            return

        jnt_num = int(self.model.body_jntnum[body_id])
        if jnt_num <= 0:
            return

        jnt_id = int(self.model.body_jntadr[body_id])  # first joint of cube body
        if int(self.model.jnt_type[jnt_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            return

        qadr = int(self.model.jnt_qposadr[jnt_id])
        self.data.qpos[qadr : qadr + 3] = np.array([0.2, 0.2, z], dtype=float)
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        lo = self.ctrl_range[:, 0]
        hi = self.ctrl_range[:, 1]
        return np.clip(ctrl, lo, hi)

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    def _key_callback(self, keycode: int):
        try:
            k = chr(keycode)
        except Exception:
            return

        # SPACE toggles pause
        if k == " ":
            self.paused = not self.paused
            print(f"[keyboard] paused={self.paused}")
            return

        k = k.lower()

        if k == "t":
            self.active_arm = (self.active_arm + 1) % self.num_arms
            print(f"[keyboard] active_arm={self.active_arm+1}/{self.num_arms}")
            return

        # Select joint 1..6
        if k in ("1", "2", "3", "4", "5", "6"):
            self.selected_joint = int(k) - 1
            print(f"[keyboard] selected_joint=joint_{self.selected_joint + 1}")
            return

        idx = self.active_arm * self.robot_dofs + self.selected_joint

        # Jog selected joint
        if k == "j":      # decrease
            self.ctrl_target[idx] -= self.key_step
        elif k == "l":    # increase
            self.ctrl_target[idx] += self.key_step
        elif k == "0":    # reset home
            if hasattr(self, "home_qpos") and self.home_qpos is not None:
                self._apply_home_pose(self.home_qpos)
                lo = self.ctrl_range[:, 0]
                hi = self.ctrl_range[:, 1]
                m = min(self.model.nu, self.robot_dofs * self.num_arms)
                self.ctrl_target[:m] = np.clip(self.data.qpos[:m], lo[:m], hi[:m])
                print("[keyboard] reset to home")
            else:
                print("[keyboard] no home pose set (run with --use-home-pose)")
            return
        else:
            return

        # Clip and keep
        self.ctrl_target[:] = self._clip_ctrl(self.ctrl_target)

    def run_keyboard_control(self, realtime: bool = True, slowmo: float = 1.0):
        print("\nKeyboard controls:")
        print("  1..6 : select joint")
        print("  J/L  : -/+ selected joint")
        print("  SPACE: pause/unpause")
        print("  0    : reset to home pose\n")
        print("  T    : toggle active arm (arm1/arm2)")
        try:
            while self.viewer is None or self.viewer.is_running():
                if not self.paused:
                    self.data.ctrl[:] = self._clip_ctrl(self.ctrl_target)
                    self.run_one_control_tick(realtime=realtime, slowmo=slowmo)
                else:
                    if self.viewer is not None:
                        self.viewer.sync()
                    time.sleep(0.01)
        finally:
            self.close()


    # ------------------ Task b: step loop ------------------
    def step(self):
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def run_sim(self, realtime: bool = True, slowmo: float = 1.0):
        dt = float(self.model.opt.timestep) * self.substeps
        try:
            while self.viewer is None or self.viewer.is_running():
                t0 = time.time()
                self.step()
                if realtime:
                    elapsed = time.time() - t0
                    time.sleep(max(0.0, dt * float(slowmo) - elapsed))
        finally:
            self.close()

    # ------------------ Task a: frozen viewer ------------------
    def run_frozen(self):
        if self.viewer is None:
            raise RuntimeError("Viewer not launched")
        try:
            while self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.01)
        finally:
            self.close()

    # ------------------ Task c: observations ------------------
    def get_observation(self, ee_site: str | None = None) -> Observation:
        d = self.robot_dofs * self.num_arms
        joint_qpos = self.data.qpos[:d].copy()
        joint_qvel = self.data.qvel[:d].copy()

        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_pos = self.data.body(cube_id).xpos.copy() if cube_id >= 0 else np.array([np.nan, np.nan, np.nan])

        ee_pos = None
        if ee_site:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
            if sid >= 0:
                ee_pos = self.data.site(sid).xpos.copy()

        return Observation(
            time=float(self.data.time),
            joint_qpos=joint_qpos,
            joint_qvel=joint_qvel,
            cube_pos=cube_pos,
            ee_pos=ee_pos,
        )

    def run_sim_with_print(self, print_every_sec: float = 0.5, realtime: bool = True, slowmo: float = 1.0, ee_site: str | None = None):
        dt = float(self.model.opt.timestep) * self.substeps
        next_print = 0.0
        try:
            while self.viewer is None or self.viewer.is_running():
                t0 = time.time()
                self.step()

                obs = self.get_observation(ee_site=ee_site)
                if obs.time >= next_print:
                    q = np.array2string(obs.joint_qpos, precision=3, floatmode="fixed")
                    c = np.array2string(obs.cube_pos, precision=3, floatmode="fixed")
                    if obs.ee_pos is not None:
                        ee = np.array2string(obs.ee_pos, precision=3, floatmode="fixed")
                        print(f"t={obs.time:7.3f}  q={q}  cube={c}  ee={ee}")
                    else:
                        print(f"t={obs.time:7.3f}  q={q}  cube={c}")
                    next_print = obs.time + float(print_every_sec)

                if realtime:
                    elapsed = time.time() - t0
                    time.sleep(max(0.0, dt * float(slowmo) - elapsed))
        finally:
            self.close()

    # ------------------ Task d: IK "do something" ------------------
    def _ik_step_site_position_only(
        self,
        site_id: int,
        x_target: np.ndarray,
        damping: float,
        step_size: float,
    ) -> np.ndarray:
        """
        One Damped-Least-Squares IK step for POSITION ONLY.
        Returns a joint target vector of length robot_dofs.
        """
        # Current EE position
        x_cur = self.data.site(site_id).xpos.copy()
        e = (x_target - x_cur).astype(float)

        # Jacobian wrt qvel (nv)
        Jp = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, Jp, None, site_id)

        if self.model.nv < self.robot_dofs:
            raise RuntimeError(f"nv={self.model.nv} < robot_dofs={self.robot_dofs}")

        # assume robot joint velocities are first robot_dofs entries
        J = Jp[:, : self.robot_dofs]  # (3, dofs)

        # DLS: dq = J^T (J J^T + λ^2 I)^-1 e
        A = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, e)

        q = self.data.qpos[: self.robot_dofs].copy()
        q_target = q + step_size * dq

        # clip to actuator ranges for safety (assumes actuator i corresponds to joint i)
        m = min(self.model.nu, self.robot_dofs)
        for i in range(m):
            lo, hi = self.ctrl_range[i]
            q_target[i] = float(np.clip(q_target[i], lo, hi))

        return q_target

    def run_task_d_ik(
        self,
        ee_site: str = "ee_site",
        realtime: bool = True,
        slowmo: float = 1.0,
        damping: float = 0.3,
        step_size: float = 0.2,
        tol: float = 0.01,
        max_ticks_per_waypoint: int = 400,
        above_z: float = 0.12,
        down_z: float = 0.03,
        dwell_ticks: int = 30,
    ):
        """
        Task d: script controls arm to do something meaningful:
        Move EE above cube -> down -> up -> above target -> down -> up.
        """
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if sid < 0:
            raise ValueError(f"Site '{ee_site}' not found. Add it in XML or pass correct name.")

        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_id < 0:
            raise ValueError("Body 'cube' not found.")

        # target position (prefer a 'target_region' geom if present)
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_region")
        target_pos = self.data.geom_xpos[gid].copy() if gid >= 0 else np.array([-0.2, 0.2, 0.0], dtype=float)

        # ensure ctrl has correct length
        if self.model.nu == 0:
            raise RuntimeError("Model has nu=0 actuators; cannot control via data.ctrl")

        # init ctrl to mid-range (stable)
        lo = self.ctrl_range[:, 0]
        hi = self.ctrl_range[:, 1]
        ctrl = 0.5 * (lo + hi)
        self.data.ctrl[:] = self._clip_ctrl(ctrl)

        def drive_to(x_target: np.ndarray):
            nonlocal ctrl
            for _ in range(max_ticks_per_waypoint):
                x_cur = self.data.site(sid).xpos.copy()
                if np.linalg.norm(x_target - x_cur) < tol:
                    break

                q_target = self._ik_step_site_position_only(
                    site_id=sid,
                    x_target=x_target,
                    damping=damping,
                    step_size=step_size,
                )

                m = min(self.model.nu, self.robot_dofs)
                ctrl[:m] = q_target[:m]
                self.data.ctrl[:] = self._clip_ctrl(ctrl)

                # step physics
                self.run_one_control_tick(realtime=realtime, slowmo=slowmo)

                if self.viewer is not None and not self.viewer.is_running():
                    return

            # dwell a bit so motion is easier to see
            for _ in range(dwell_ticks):
                self.run_one_control_tick(realtime=realtime, slowmo=slowmo)
                if self.viewer is not None and not self.viewer.is_running():
                    return

        # build waypoints
        cube_pos = self.data.body(cube_id).xpos.copy()
        above = np.array([0.0, 0.0, above_z], dtype=float)
        down = np.array([0.0, 0.0, down_z], dtype=float)

        waypoints = [
            cube_pos + above,
            cube_pos + down,
            cube_pos + above,
            target_pos + above,
            target_pos + down,
            target_pos + above,
        ]

        try:
            for wp in waypoints:
                drive_to(wp)

            # keep running so you can inspect final pose
            while self.viewer is None or self.viewer.is_running():
                self.run_one_control_tick(realtime=realtime, slowmo=slowmo)
        finally:
            self.close()

    def run_one_control_tick(self, realtime: bool, slowmo: float):
        dt = float(self.model.opt.timestep) * self.substeps
        t0 = time.time()
        self.step()
        if realtime:
            elapsed = time.time() - t0
            time.sleep(max(0.0, dt * float(slowmo) - elapsed))


def main():
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(here / "lerobot_pick_place_cube.xml"))

    ap.add_argument("--task", choices=["a", "b", "c", "d", "k"], default="b")
    ap.add_argument("--robot-dofs", type=int, default=6)
    ap.add_argument("--cube-z", type=float, default=0.05)
    ap.add_argument("--substeps", type=int, default=1)

    ap.add_argument("--key-step", type=float, default=0.06)

    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--slowmo", type=float, default=1.0, help=">1 slower visuals, <1 faster")

    # Task c
    ap.add_argument("--print-every", type=float, default=0.5)
    ap.add_argument("--ee-site", default="ee_site")

    # Task d (IK)
    ap.add_argument("--damping", type=float, default=0.2)
    ap.add_argument("--step-size", type=float, default=0.4)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--max-ticks", type=int, default=400)
    ap.add_argument("--above-z", type=float, default=0.12)
    ap.add_argument("--down-z", type=float, default=0.03)
    ap.add_argument("--dwell", type=int, default=30)

    # Optional nicer start
    ap.add_argument("--use-home-pose", action="store_true")

    args = ap.parse_args()

    sim = Task2Sim(
        xml_path=args.xml,
        robot_dofs=args.robot_dofs,
        cube_raise_z=args.cube_z,
        substeps=args.substeps,
        launch_viewer=True,
        use_home_pose=args.use_home_pose,
    )

    sim.key_step = args.key_step

    realtime = not args.no_realtime

    if args.task == "a":
        sim.run_frozen()
    elif args.task == "b":
        sim.run_sim(realtime=realtime, slowmo=args.slowmo)
    elif args.task == "c":
        sim.run_sim_with_print(
            print_every_sec=args.print_every,
            realtime=realtime,
            slowmo=args.slowmo,
            ee_site=args.ee_site,
        )
    elif args.task == "k":
        sim.run_keyboard_control(realtime=realtime, slowmo=args.slowmo)
    else:
        sim.run_task_d_ik(
            ee_site=args.ee_site,
            realtime=realtime,
            slowmo=args.slowmo,
            damping=args.damping,
            step_size=args.step_size,
            tol=args.tol,
            max_ticks_per_waypoint=args.max_ticks,
            above_z=args.above_z,
            down_z=args.down_z,
            dwell_ticks=args.dwell,
        )


if __name__ == "__main__":
    main()
