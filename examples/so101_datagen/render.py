#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replays recorded states for the cameras: color frames, and segmentation where boxes are needed.

Every episode compiles a new model, so the renderer is rebuilt per episode. The OpenGL context is
not: one per process is created once and reused, since creating and freeing contexts per episode
leaks memory and costs tens of milliseconds.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import SceneInfo

# Dataset camera name -> camera in the model.
CAMERAS = {"front": "front", "wrist": "wrist_cam"}
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

_GL_CONTEXT = None


def _gl_context(width: int, height: int):
    global _GL_CONTEXT
    if _GL_CONTEXT is None:
        from mujoco.rendering.classic import gl_context

        _GL_CONTEXT = gl_context.GLContext(width, height)
    _GL_CONTEXT.make_current()
    return _GL_CONTEXT


@dataclass
class Visible:
    """An object or container as one camera sees it, in [0, 1] image coordinates."""

    name: str
    label: str
    pixels: int
    bbox: list[float]
    center: list[float]


class StateRenderer:
    """Renders a compiled scene at given ``qpos``, without stepping physics."""

    def __init__(self, model: mujoco.MjModel, height: int, width: int):
        self.model = model
        self.data = mujoco.MjData(model)
        self.height, self.width = height, width
        _gl_context(width, height)
        self._rect = mujoco.MjrRect(0, 0, width, height)
        self._option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera_ids = {name: model.camera(cam).id for name, cam in CAMERAS.items()}
        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        # Segmentation gets a context without multisampling: at object edges, multisampling blends
        # the id-encoding colors into ids of other geoms.
        samples = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0
        self._seg_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        model.vis.quality.offsamples = samples
        self._seg_scene = mujoco.MjvScene(model, maxgeom=10000)
        self._seg_scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self._seg_scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        self._rgb = np.empty((height, width, 3), dtype=np.uint8)
        # Geom id -> index of the object or container it belongs to, -1 for anything else.
        self._owner = np.full(model.ngeom + 1, -1, dtype=np.int64)

    def set_state(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        # Positions of bodies, cameras and lights are all rendering needs.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _render(self, camera: str, scene: mujoco.MjvScene, context: mujoco.MjrContext) -> np.ndarray:
        self._camera.fixedcamid = self._camera_ids[camera]
        mujoco.mjv_updateScene(
            self.model, self.data, self._option, None, self._camera, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        mujoco.mjr_render(self._rect, scene, context)
        mujoco.mjr_readPixels(self._rgb, None, self._rect, context)
        # OpenGL rows start at the bottom.
        return np.flipud(self._rgb)

    def rgb(self, camera: str) -> np.ndarray:
        return self._render(camera, self._scene, self._context).copy()

    def visible(self, camera: str, info: SceneInfo, min_pixels: int) -> dict[str, Visible]:
        """Objects and containers seen by ``camera``, from a segmentation render: occlusion included."""
        image = self._render(camera, self._seg_scene, self._seg_context).astype(np.int64)
        segid = (image[..., 0] + image[..., 1] * 256 + image[..., 2] * 65536).ravel()
        # Segmentation ids are scene geom indices plus one (zero is the background): map them to model geoms.
        scene = self._seg_scene
        to_geom = np.full(scene.ngeom + 2, -1, dtype=np.int64)
        for geom in scene.geoms[: scene.ngeom]:
            if geom.segid >= 0 and geom.objtype == GEOM:
                to_geom[geom.segid + 1] = geom.objid
        geoms = to_geom[np.minimum(segid, scene.ngeom + 1)]

        names = list(info.geoms)
        owner = self._owner
        owner[:] = -1
        for k, name in enumerate(names):
            owner[info.geoms[name]] = k
        flat = np.flatnonzero(owner[geoms] >= 0)
        items = owner[geoms[flat]]
        rows, cols = np.divmod(flat, self.width)
        h, w = self.height, self.width
        out = {}
        for k in np.unique(items):
            mine = items == k
            pixels = int(mine.sum())
            if pixels < min_pixels:
                continue
            r, c = rows[mine], cols[mine]
            name = names[k]
            bbox = [c.min() / w, r.min() / h, (c.max() + 1) / w, (r.max() + 1) / h]
            out[name] = Visible(
                name=name,
                label=info.labels[name],
                pixels=pixels,
                bbox=[round(float(v), 4) for v in bbox],
                center=[round(float(c.mean() + 0.5) / w, 4), round(float(r.mean() + 0.5) / h, 4)],
            )
        return out

    def close(self) -> None:
        self._context.free()
        self._seg_context.free()
