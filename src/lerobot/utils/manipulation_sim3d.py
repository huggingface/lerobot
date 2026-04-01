# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Rerun 3D preview: URDF STL meshes when available, solid object cubes, plan path."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lerobot.model.kinematics import RobotKinematics

logger = logging.getLogger(__name__)

_ARM_MESH_PALETTE = np.array(
    [
        [88, 118, 198],
        [98, 128, 208],
        [108, 138, 218],
        [118, 148, 228],
        [100, 125, 190],
        [130, 155, 210],
        [160, 175, 205],
        [190, 195, 210],
    ],
    dtype=np.uint8,
)


def _safe_entity_name(link_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", link_name)


def _log_transform(path: str, T: np.ndarray) -> None:
    import rerun as rr

    T = np.asarray(T, dtype=np.float64)
    t = T[:3, 3]
    Rm = T[:3, :3]
    try:
        from scipy.spatial.transform import Rotation as R_scipy

        q = R_scipy.from_matrix(Rm).as_quat()
        quat = rr.Quaternion(xyzw=q)
        rr.log(path, rr.Transform3D(translation=t, quaternion=quat))
    except Exception:
        try:
            rr.log(path, rr.Transform3D(translation=t, mat3x3=Rm))
        except Exception as e:
            logger.debug("Rerun Transform3D log failed for %s: %s", path, e)


def _link_vertex_colors(n_verts: int, link_name: str) -> np.ndarray:
    h = hashlib.md5(link_name.encode()).hexdigest()
    idx = int(h, 16) % len(_ARM_MESH_PALETTE)
    c = _ARM_MESH_PALETTE[idx]
    return np.tile(c.reshape(1, 3), (n_verts, 1))


def _log_mesh3d(rr_mod, path: str, positions: np.ndarray, triangles: np.ndarray, colors: np.ndarray) -> None:
    pos = np.asarray(positions, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int32)
    if pos.size == 0 or tri.size == 0:
        return
    idx = tri.reshape(-1) if tri.ndim == 2 else tri
    cols = np.asarray(colors, dtype=np.uint8)
    attempts: list[dict] = [
        {"vertex_positions": pos, "triangle_indices": idx, "vertex_colors": cols},
        {"vertex_positions": pos, "indices": idx, "vertex_colors": cols},
        {"vertex_positions": pos, "triangle_indices": idx},
        {"vertex_positions": pos, "indices": idx},
    ]
    last_err: Exception | None = None
    for kw in attempts:
        try:
            rr_mod.log(path, rr_mod.Mesh3D(**kw))
            return
        except Exception as e:
            last_err = e
            continue
    logger.debug("Mesh3D log failed for %s: %s", path, last_err)


def log_manipulation_sim3d(
    *,
    frame_sequence: int,
    kinematics: RobotKinematics,
    joint_deg: np.ndarray,
    object_centers_base: np.ndarray | None = None,
    object_half_sizes_base: np.ndarray | None = None,
    object_labels: list[str] | None = None,
    focus_object_index: int | None = None,
    planned_ee_positions_base: np.ndarray | None = None,
    plan_summary: str | None = None,
) -> None:
    """Log a 3D scene slice on the same ``frame`` timeline as camera streams.

    Args:
        frame_sequence: Rerun ``frame`` sequence index.
        kinematics: Placo wrapper (must match live URDF).
        joint_deg: Current arm joint angles in degrees (5-DOF SO arm).
        object_centers_base: (N, 3) object centers in robot base frame (m).
        object_half_sizes_base: (N, 3) half-extents for axis-aligned boxes (m).
        object_labels: Optional labels per object.
        focus_object_index: Highlight this row in ``object_centers_base`` (pick target).
        planned_ee_positions_base: (M, 3) planned EE path (e.g. waypoint origins).
        plan_summary: Short text (e.g. pick label + action).
    """
    import rerun as rr

    rr.set_time("frame", sequence=int(frame_sequence))

    # Ensure the 3D scene has at least one stable primitive every frame so the user can
    # add a 3D view and immediately see something under `sim3d/*` even if other logs fail.
    try:
        rr.log(
            "sim3d/_origin",
            rr.Points3D(positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64), radii=0.01),
        )
    except Exception:
        pass

    try:
        from lerobot.utils.urdf_visual_meshes import (
            load_link_visual_meshes_cached,
            transform_mesh_to_world,
        )

        chain = kinematics.get_link_transforms_chain(joint_deg)
        link_meshes = load_link_visual_meshes_cached(kinematics.urdf_dir)
        meshed_any = False
        if chain and link_meshes:
            for name, T in chain:
                packed = link_meshes.get(name)
                if packed is None:
                    continue
                V_link, F = packed
                if V_link.size == 0:
                    continue
                V_w = transform_mesh_to_world(V_link, T)
                cols = _link_vertex_colors(V_w.shape[0], name)
                _log_mesh3d(
                    rr,
                    f"sim3d/robot/mesh/{_safe_entity_name(name)}",
                    V_w,
                    F,
                    cols,
                )
                meshed_any = True
        if chain and not meshed_any:
            pts = np.stack([T[:3, 3] for _, T in chain], axis=0)
            if pts.shape[0] >= 2:
                rr.log(
                    "sim3d/robot/arm_chain",
                    rr.LineStrips3D(
                        strips=[pts],
                        radii=0.004,
                        colors=np.array([[100, 140, 255, 255]], dtype=np.uint8),
                    ),
                )
            for name, T in chain:
                _log_transform(f"sim3d/robot/links/{_safe_entity_name(name)}", T)
    except Exception as e:
        logger.warning("display_sim3d: failed to log robot chain: %s", e)

    # Ground grid (robot base / URDF world XY, Z=0) so scale is obvious in the 3D view.
    try:
        half_w = 0.4
        n = 10
        grid_strips: list[np.ndarray] = []
        for i in range(n + 1):
            t = -half_w + (2.0 * half_w) * i / n
            grid_strips.append(np.array([[t, -half_w, 0.0], [t, half_w, 0.0]], dtype=np.float64))
        for i in range(n + 1):
            t = -half_w + (2.0 * half_w) * i / n
            grid_strips.append(np.array([[-half_w, t, 0.0], [half_w, t, 0.0]], dtype=np.float64))
        rr.log(
            "sim3d/workspace/ground_grid",
            rr.LineStrips3D(
                strips=grid_strips,
                radii=0.001,
                colors=np.tile(np.array([[160, 160, 180, 140]], dtype=np.uint8), (len(grid_strips), 1)),
            ),
        )
    except Exception as e:
        logger.debug("display_sim3d: ground grid: %s", e)

    # End-effector frame + 5 cm RGB axes (X red, Y green, Z blue).
    try:
        T_ee = np.asarray(kinematics.forward_kinematics(joint_deg), dtype=np.float64)
        _log_transform("sim3d/robot/ee_frame", T_ee)
        t = T_ee[:3, 3]
        Rm = T_ee[:3, :3]
        L = 0.05
        strips_axes = [
            np.stack([t, t + Rm[:, 0] * L], axis=0),
            np.stack([t, t + Rm[:, 1] * L], axis=0),
            np.stack([t, t + Rm[:, 2] * L], axis=0),
        ]
        rr.log(
            "sim3d/robot/ee_axes",
            rr.LineStrips3D(
                strips=strips_axes,
                radii=0.004,
                colors=np.array([[255, 80, 80, 255], [80, 255, 80, 255], [100, 150, 255, 255]], dtype=np.uint8),
            ),
        )
    except Exception as e:
        logger.debug("display_sim3d: ee marker: %s", e)

    if object_centers_base is not None and object_centers_base.size > 0:
        from lerobot.utils.urdf_visual_meshes import axis_aligned_cube_mesh

        C = np.atleast_2d(np.asarray(object_centers_base, dtype=np.float64))
        n = C.shape[0]
        H = None
        if object_half_sizes_base is not None:
            H = np.atleast_2d(np.asarray(object_half_sizes_base, dtype=np.float64))
            if H.shape[0] == 1 and n > 1:
                H = np.repeat(H, n, axis=0)
        try:
            fi = int(focus_object_index) if focus_object_index is not None else None
            if H is not None and H.shape[0] == n and H.shape[1] == 3:
                for i in range(n):
                    Vc, Fc = axis_aligned_cube_mesh(C[i], H[i])
                    is_focus = fi is not None and i == fi
                    base_rgb = np.array([[255, 120, 70]], dtype=np.uint8)
                    if is_focus:
                        base_rgb = np.array([[255, 45, 220]], dtype=np.uint8)
                    cols = np.tile(base_rgb, (Vc.shape[0], 1))
                    if object_labels and i < len(object_labels):
                        tag = _safe_entity_name(str(object_labels[i]))
                    else:
                        tag = f"obj{i}"
                    _log_mesh3d(
                        rr,
                        f"sim3d/objects/cube_{i}_{tag}",
                        Vc,
                        Fc,
                        cols,
                    )
            else:
                rr.log(
                    "sim3d/objects/centers",
                    rr.Points3D(
                        positions=C,
                        radii=0.012,
                        colors=np.tile(np.array([[220, 110, 75]], dtype=np.uint8), (n, 1)),
                    ),
                )
                if fi is not None and 0 <= fi < n:
                    rr.log(
                        "sim3d/objects/focus_target",
                        rr.Points3D(
                            positions=C[fi : fi + 1],
                            radii=0.028,
                            colors=np.array([[255, 40, 255]], dtype=np.uint8),
                        ),
                    )
        except Exception as e:
            logger.warning("display_sim3d: failed to log objects: %s", e)
        # Avoid relying on text archetypes (version-dependent); keep labels in plan_summary upstream.

    if planned_ee_positions_base is not None:
        P = np.asarray(planned_ee_positions_base, dtype=np.float64)
        if P.ndim == 2 and P.shape[0] >= 2 and P.shape[1] == 3:
            try:
                rr.log(
                    "sim3d/plan/ee_path",
                    rr.LineStrips3D(
                        strips=[P],
                        radii=0.003,
                        colors=np.array([[80, 255, 120, 255]], dtype=np.uint8),
                    ),
                )
            except Exception as e:
                logger.warning("display_sim3d: failed to log ee path: %s", e)
        if P.ndim == 2 and P.shape[0] >= 1 and P.shape[1] == 3:
            try:
                rr.log(
                    "sim3d/plan/ee_targets",
                    rr.Points3D(
                        positions=P,
                        radii=0.018,
                        colors=np.tile(np.array([[50, 220, 100]], dtype=np.uint8), (P.shape[0], 1)),
                    ),
                )
            except Exception as e:
                logger.warning("display_sim3d: failed to log ee targets: %s", e)

    # Do not log text here (archetype varies across rerun-sdk versions).
