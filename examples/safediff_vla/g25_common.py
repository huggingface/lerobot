"""G25 shared physics snapshot (policy rollouts and VLABench expert runs): per step EE pose/velocity, object pose/velocity, finger width, gripper-object contact count, is_grasped, object-in-EE-frame transform T_EE_object. Positions in the WORLD frame (T_EE_object is frame-invariant); quaternions wxyz."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R

_CACHE: dict = {}


def _ids(e):
    ph = e.physics; key = (id(ph), e.task.target_entity)
    if key not in _CACHE:
        ent = e.task.entities[e.task.target_entity]; og = {int(ph.bind(g).element_id) for g in ent.geoms}; rg = {int(ph.bind(g).element_id) for g in e.task.robot.gripper_geoms if getattr(g, "name", None) is not None or True}
        _CACHE.clear(); _CACHE[key] = (og, rg)
    return _CACHE[key]


def snapshot(e, container: str | None = None) -> dict:
    ph = e.physics; task = e.task; ent = task.entities[task.target_entity]; ee_p = np.array(task.robot.get_end_effector_pos(ph), dtype=float, copy=True); ee_q = np.array(task.robot.get_end_effector_quat(ph), dtype=float, copy=True)
    op, oq = ent.get_pose(ph); op = np.array(op, dtype=float, copy=True); oq = np.array(oq, dtype=float, copy=True); vl, va = ent.get_velocity(ph); vl = np.array(vl, dtype=float, copy=True); va = np.array(va, dtype=float, copy=True)
    nm = ph.named.data.xpos; width = float(np.linalg.norm(np.asarray(nm["franka/left_finger"]) - np.asarray(nm["franka/right_finger"]))); og, rg = _ids(e); c = ph.data.contact[:ph.data.ncon]; g1, g2 = np.asarray(c.geom1), np.asarray(c.geom2)
    ncon = int(sum(1 for a, b in zip(g1, g2) if (int(a) in og and int(b) in rg) or (int(b) in og and int(a) in rg)))
    Re = R.from_quat([ee_q[1], ee_q[2], ee_q[3], ee_q[0]]); Ro = R.from_quat([oq[1], oq[2], oq[3], oq[0]]); t_rel = Re.inv().apply(op - ee_p); r_rel = (Re.inv() * Ro).as_rotvec()
    d = {"ee_p": ee_p, "ee_q": ee_q, "obj_p": op, "obj_q": oq, "obj_v": vl, "obj_w": va, "width": width, "ncon": ncon, "grasped": bool(ent.is_grasped(ph, task.robot)), "t_rel": t_rel, "r_rel": r_rel}
    if container is not None and container in task.entities: d["cont_p"] = np.array(task.entities[container].get_xpos(ph), dtype=float, copy=True)
    return d
