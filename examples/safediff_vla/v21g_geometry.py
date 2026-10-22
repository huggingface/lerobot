"""[v2.1 anchor phase] Shared geometry for the observable 3D reference: calibrated two-view triangulation (front + right FIXED cameras).
Frames: pixel (u,v) in the 480x480 original render -> camera ray (MuJoCo cam: x right, y up, looks along -z; pinhole, fovy 45 deg, square image) -> world -> robot-base frame (world - base, base=(0,-0.4,0.78), the fixed robot mount) in metres.
Calibration source: outputs/data/safediff_vla_g1_counterfactual/camera_calibration.json (physics cam_xpos/cam_xmat/cam_fovy; identical for every scene, verified poker==fruit)."""
from __future__ import annotations
import numpy as np
from v21_referent_table import cam_params, project  # noqa: F401

CAMS = cam_params("select_poker")  # [front, right, wrist]


def ray(cam_i, u, v, size=480):
    c, R, fovy = CAMS[cam_i]; f = (size / 2) / np.tan(np.deg2rad(fovy) / 2); d = R @ np.array([(u - size / 2) / f, -(v - size / 2) / f, -1.0]); return c, d / np.linalg.norm(d)


def triangulate(uv_front, uv_right):
    """closest-approach midpoint of the two rays (world frame, m) + ray gap (m)."""
    c1, d1 = ray(0, *uv_front); c2, d2 = ray(1, *uv_right); A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]]); b = np.array([(c2 - c1) @ d1, (c2 - c1) @ d2])
    try: s, t = np.linalg.solve(A, b)
    except np.linalg.LinAlgError: return None, 1e9
    p1, p2 = c1 + s * d1, c2 + t * d2
    if s < 0 or t < 0: return None, 1e9
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))
