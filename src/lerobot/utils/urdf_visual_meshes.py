# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Load triangle meshes from URDF ``<visual><mesh>`` (STL) for Rerun / debug viz."""

from __future__ import annotations

import logging
import os
import re
import struct
import xml.etree.ElementTree as ET
from typing import Any


import numpy as np

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}


def _children_tag(parent: Any, short: str) -> list[Any]:
    return [c for c in list(parent) if c.tag == short or c.tag.endswith("}" + short)]


def _parse_floats(s: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not s or not str(s).strip():
        return default
    parts = re.split(r"[\s,]+", str(s).strip())
    out: list[float] = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    if not out:
        return default
    return tuple(out)


def _urdf_visual_origin_to_T(origin_el: Any | None) -> np.ndarray:
    """4x4 transform from link frame to visual (mesh) frame."""
    if origin_el is None:
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
    else:
        xyz = _parse_floats(origin_el.get("xyz"), (0.0, 0.0, 0.0))
        rpy = _parse_floats(origin_el.get("rpy"), (0.0, 0.0, 0.0))
        if len(xyz) < 3:
            xyz = tuple(list(xyz) + [0.0] * (3 - len(xyz)))
        if len(rpy) < 3:
            rpy = tuple(list(rpy) + [0.0] * (3 - len(rpy)))
    try:
        from scipy.spatial.transform import Rotation as R_scipy

        Rm = R_scipy.from_euler("xyz", [rpy[0], rpy[1], rpy[2]], degrees=False).as_matrix()
    except Exception:
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
        cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
        cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        Rm = Rz @ Ry @ Rx
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.array(xyz[:3], dtype=np.float64)
    return T


def _resolve_mesh_path(urdf_dir: str, filename: str) -> str | None:
    fn = (filename or "").strip()
    if not fn:
        return None
    urdf_dir = os.path.abspath(urdf_dir)
    candidates: list[str] = []
    if fn.startswith("package://"):
        rest = fn.split("://", 1)[1]
        slash = rest.find("/")
        rel = rest[slash + 1 :] if slash >= 0 else rest
        candidates.extend(
            [
                os.path.join(urdf_dir, rel),
                os.path.join(urdf_dir, os.path.basename(rel)),
                os.path.join(urdf_dir, "meshes", os.path.basename(rel)),
            ]
        )
    else:
        candidates.extend(
            [
                os.path.join(urdf_dir, fn),
                os.path.join(urdf_dir, os.path.basename(fn)),
                os.path.join(urdf_dir, "meshes", os.path.basename(fn)),
            ]
        )
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_stl_triangles(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL file as (V, F) with unique vertices per triangle (STL style)."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) >= 5 and raw[:5].lower().startswith(b"solid"):
        try:
            return _load_stl_ascii(raw.decode("utf-8", errors="ignore"))
        except Exception:
            pass
    return _load_stl_binary(raw)


def _load_stl_binary(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(data) < 84:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    n_decl = struct.unpack_from("<I", data, 80)[0]
    tri_bytes = max(0, len(data) - 84)
    n_from_size = tri_bytes // 50
    if n_from_size <= 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    # Some exporters write a wrong triangle count; trust file size when inconsistent.
    if n_decl == 0 or n_decl > n_from_size or n_decl < n_from_size // 2:
        n_tri = n_from_size
    else:
        n_tri = n_decl
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    o = 84
    vi = 0
    for _ in range(n_tri):
        if o + 50 > len(data):
            break
        # skip normal (3xf32)
        o += 12
        tri = []
        for _k in range(3):
            x, y, z = struct.unpack_from("<fff", data, o)
            o += 12
            verts.append([x, y, z])
            tri.append(vi)
            vi += 1
        faces.append(tri)
        o += 2  # attribute
    if not verts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    return V, F


def _load_stl_ascii(text: str) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    vi = 0
    for m in re.finditer(
        r"facet\s+normal\s+[^\n]+\s+outer\s+loop\s+(.*?)\s+endloop",
        text,
        re.DOTALL | re.IGNORECASE,
    ):
        block = m.group(1)
        tri: list[int] = []
        for vm in re.finditer(r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", block):
            verts.append([float(vm.group(1)), float(vm.group(2)), float(vm.group(3))])
            tri.append(vi)
            vi += 1
        if len(tri) == 3:
            faces.append(tri)
    if not verts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def parse_link_visual_meshes(urdf_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``link_name -> (vertices in link frame Nx3, triangle_indices Mx3)``."""
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    # Match ``link`` / ``{ns}link`` (some URDFs declare xmlns).
    links = list(root.findall("link"))
    if not links:
        links = [el for el in root.iter() if el.tag.endswith("}link") or el.tag == "link"]

    for link in links:
        lname = link.get("name")
        if not lname:
            continue
        for vis in _children_tag(link, "visual"):
            geoms = _children_tag(vis, "geometry")
            geom = geoms[0] if geoms else None
            if geom is None:
                continue
            meshes = _children_tag(geom, "mesh")
            mesh_el = meshes[0] if meshes else None
            if mesh_el is None:
                continue
            fn = mesh_el.get("filename")
            if not fn:
                continue
            scale_t = _parse_floats(mesh_el.get("scale"), (1.0, 1.0, 1.0))
            sx = float(scale_t[0]) if len(scale_t) > 0 else 1.0
            sy = float(scale_t[1]) if len(scale_t) > 1 else sx
            sz = float(scale_t[2]) if len(scale_t) > 2 else sx

            mpath = _resolve_mesh_path(urdf_dir, fn)
            if mpath is None:
                logger.debug("URDF mesh not found for link %r: %s", lname, fn)
                continue
            try:
                V, F = load_stl_triangles(mpath)
            except Exception as e:
                logger.debug("Failed to load STL %s: %s", mpath, e)
                continue
            if V.size == 0:
                continue
            V = V * np.array([sx, sy, sz], dtype=np.float64)
            origins = _children_tag(vis, "origin")
            origin_el = origins[0] if origins else None
            T_lv = _urdf_visual_origin_to_T(origin_el)
            R = T_lv[:3, :3]
            t = T_lv[:3, 3]
            V = (V @ R.T) + t

            out.setdefault(lname, []).append((V.astype(np.float64), F.astype(np.int32)))

    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lname, parts in out.items():
        if not parts:
            continue
        v_off = 0
        all_v: list[np.ndarray] = []
        all_f: list[np.ndarray] = []
        for V, F in parts:
            all_v.append(V)
            all_f.append(F + v_off)
            v_off += V.shape[0]
        merged[lname] = (np.vstack(all_v), np.vstack(all_f))
    return merged


def load_link_visual_meshes_cached(urdf_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cached meshes for the ``robot.urdf`` inside ``urdf_dir`` (placo layout)."""
    from lerobot.utils.urdf_chain import robot_urdf_file_in_dir

    urdf_path = os.path.abspath(robot_urdf_file_in_dir(urdf_dir))
    try:
        mtime = os.path.getmtime(urdf_path)
    except OSError:
        return {}
    key = urdf_path
    hit = _cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        meshes = parse_link_visual_meshes(urdf_path)
    except Exception as e:
        logger.warning("Could not parse URDF visuals from %s: %s", urdf_path, e)
        meshes = {}
    _cache[key] = (mtime, meshes)
    if meshes:
        logger.info("Loaded STL visuals for %d URDF link(s) from %s.", len(meshes), urdf_path)
    else:
        logger.info("No STL link visuals found in %s (stick-figure fallback in sim3d).", urdf_path)
    return meshes


def transform_mesh_to_world(V: np.ndarray, T_world_link: np.ndarray) -> np.ndarray:
    """Apply 4x4 world-from-link transform to Nx3 vertices (row vectors)."""
    if V.size == 0:
        return V
    T = np.asarray(T_world_link, dtype=np.float64)
    n = V.shape[0]
    Ph = np.ones((4, n), dtype=np.float64)
    Ph[:3, :] = V.T
    W = (T @ Ph)[:3, :].T
    return W


def axis_aligned_cube_mesh(center: np.ndarray, half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed box mesh: center and half-extents in meters (axis-aligned, world frame)."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = np.maximum(np.asarray(half_extents, dtype=np.float64).reshape(3), 1e-4)
    # Unit cube corners ±0.5 → scale by 2h, translate to center.
    corners = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    V = corners * (2.0 * h) + c
    # 12 triangles (CCW outward)
    F = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return V, F
