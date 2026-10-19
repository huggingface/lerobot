"""G28 helpers: pinhole projection of a world point into a (moving) MuJoCo camera from physics state; target segmentation pixel count; frame rendering. Camera ids (policy order): front=2 ('forward'), right=0, wrist=3 ('franka/Franka_wrist_cam')."""
from __future__ import annotations
import numpy as np

FRONT, RIGHT, WRIST = 2, 0, 3


def cam_proj(ph, cam_id: int, xyz, res: int = 480):
    R = np.array(ph.data.cam_xmat[cam_id]).reshape(3, 3); o = np.array(ph.data.cam_xpos[cam_id]); fovy = float(ph.model.cam_fovy[cam_id]); p = R.T @ (np.asarray(xyz, float) - o); f = (res / 2) / np.tan(np.deg2rad(fovy) / 2); d = -p[2]
    if d <= 1e-6: return None
    return float(f * p[0] / d + res / 2), float(res / 2 - f * p[1] / d), float(d)


def target_pixels(e, cam_id: int, res: int = 240):
    ph = e.physics; ent = e.task.entities[e.task.target_entity]; gid = {int(ph.bind(g).element_id) for g in ent.geoms}; seg = ph.render(height=res, width=res, camera_id=cam_id, segmentation=True); ids, typ = seg[..., 0], seg[..., 1]
    m = np.isin(ids, list(gid)) & (typ == 5)
    return int(m.sum()), m


def frame(e, cam_id: int, res: int = 240):
    return e.physics.render(height=res, width=res, camera_id=cam_id)
