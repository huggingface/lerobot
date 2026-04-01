#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import io
import struct
import tempfile

import numpy as np

from lerobot.utils.urdf_visual_meshes import (
    axis_aligned_cube_mesh,
    load_stl_triangles,
    parse_link_visual_meshes,
)


def _minimal_binary_stl() -> bytes:
    header = b" " * 80
    n = 1
    buf = io.BytesIO()
    buf.write(header)
    buf.write(struct.pack("<I", n))
    normal = struct.pack("<fff", 0.0, 0.0, 1.0)
    v0 = struct.pack("<fff", 0.0, 0.0, 0.0)
    v1 = struct.pack("<fff", 1.0, 0.0, 0.0)
    v2 = struct.pack("<fff", 0.0, 1.0, 0.0)
    attr = struct.pack("<H", 0)
    buf.write(normal + v0 + v1 + v2 + attr)
    return buf.getvalue()


def test_load_stl_binary_triangle():
    raw = _minimal_binary_stl()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(raw)
        path = f.name
    V, F = load_stl_triangles(path)
    assert V.shape == (3, 3)
    assert F.shape == (1, 3)
    assert F.max() == 2


def test_axis_aligned_cube_mesh():
    c = np.array([0.1, -0.2, 0.3])
    h = np.array([0.02, 0.03, 0.04])
    V, F = axis_aligned_cube_mesh(c, h)
    assert V.shape == (8, 3)
    assert F.shape == (12, 3)
    assert np.allclose(V.mean(axis=0), c, atol=1e-6)


def test_parse_minimal_urdf_with_mesh(tmp_path):
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    stl_path = mesh_dir / "link1.stl"
    stl_path.write_bytes(_minimal_binary_stl())
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="r">
  <link name="link1">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/link1.stl" scale="1 1 1"/>
      </geometry>
    </visual>
  </link>
</robot>
"""
    )
    meshes = parse_link_visual_meshes(str(urdf))
    assert "link1" in meshes
    V, F = meshes["link1"]
    assert V.shape[0] >= 3
    assert F.size >= 9
