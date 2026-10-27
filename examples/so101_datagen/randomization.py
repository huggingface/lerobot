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

"""Seeded samplers for every scene parameter. They only return parameters; ``scene.py`` applies them."""

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from .assets import MeshAsset
from .config import GenerationConfig, Range
from .scene import (
    BackgroundParams,
    CameraParams,
    ContainerParams,
    DynamicsParams,
    LightParams,
    Look,
    ObjectParams,
    SceneParams,
)

# Named colors, so labels such as "red cube" stay true under jitter.
PALETTE = {
    "red": (0.85, 0.12, 0.10),
    "green": (0.15, 0.65, 0.20),
    "blue": (0.12, 0.30, 0.85),
    "yellow": (0.95, 0.82, 0.10),
    "orange": (0.95, 0.45, 0.08),
    "purple": (0.50, 0.20, 0.70),
    "pink": (0.95, 0.45, 0.65),
    "white": (0.92, 0.92, 0.90),
    "black": (0.08, 0.08, 0.08),
    "brown": (0.45, 0.28, 0.12),
    "gray": (0.50, 0.50, 0.50),
    "cyan": (0.10, 0.70, 0.80),
}
# Colors of printed SO-101 arms.
ARM_COLORS = [
    (1.0, 0.82, 0.12),
    (0.30, 0.62, 0.68),
    (0.92, 0.92, 0.90),
    (0.12, 0.12, 0.12),
    (0.95, 0.45, 0.08),
]
TEXTURE_KINDS = ("checker", "stripes", "grain", "noise", "dots")
CONTAINER_NOUNS = ("box", "bin", "container")
TRAY_NOUNS = ("tray", "box")


@dataclass
class Footprint:
    """A placed item on the table, for overlap checks."""

    x: float
    y: float
    radius: float


def uniform(rng: np.random.Generator, r: Range) -> float:
    return float(rng.uniform(r[0], r[1]))


def sample_scene(
    rng: np.random.Generator,
    cfg: GenerationConfig,
    meshes: dict[str, MeshAsset],
    containers: int,
    min_objects: int,
) -> SceneParams | None:
    """Sample one scene with ``containers`` containers and at least ``min_objects`` graspable objects.

    Returns None when the layout can't be placed, so the caller draws another seed.
    """
    placed: list[Footprint] = []
    used_labels: set[str] = set()
    container_params = []
    for i in range(containers):
        container = _sample_container(rng, cfg, f"container_{i}", used_labels)
        spot = _place(rng, cfg, placed, container.inner_half + container.thickness)
        if spot is None:
            return None
        container.pos = spot
        container_params.append(container)

    count = max(min_objects, int(rng.integers(cfg.objects.count[0], cfg.objects.count[1] + 1)))
    objects = []
    for i in range(count):
        obj = _sample_object(rng, cfg, meshes, f"object_{i}", used_labels)
        footprint = float(
            np.hypot(obj.half_size[0], obj.half_size[1] if obj.shape != "cylinder" else obj.half_size[0])
        )
        spot = _place(rng, cfg, placed, footprint)
        if spot is None:
            if i < min_objects:
                return None
            break
        obj.pos = spot
        objects.append(obj)

    return SceneParams(
        objects=objects,
        containers=container_params,
        lights=_sample_lights(rng, cfg),
        background=_sample_background(rng, cfg),
        front_camera=_sample_front_camera(rng, cfg),
        wrist_offset=(rng.uniform(-1, 1, 3) * cfg.wrist_camera.pos_jitter).tolist(),
        wrist_rotvec=np.deg2rad(rng.uniform(-1, 1, 3) * cfg.wrist_camera.rot_jitter).tolist(),
        wrist_fovy=uniform(rng, cfg.wrist_camera.fovy),
        dynamics=DynamicsParams(
            kp_scale=rng.uniform(*cfg.dynamics.kp_scale, 6).tolist(),
            damping_scale=rng.uniform(*cfg.dynamics.damping_scale, 6).tolist(),
            frictionloss_scale=rng.uniform(*cfg.dynamics.frictionloss_scale, 6).tolist(),
            arm_rgba=[*ARM_COLORS[rng.integers(len(ARM_COLORS))], 1.0],
        ),
    )


def _place(rng: np.random.Generator, cfg: GenerationConfig, placed: list[Footprint], radius: float):
    """Rejection-sample a free spot in the reachable annulus."""
    for _ in range(100):
        r = uniform(rng, cfg.layout.radius)
        angle = np.deg2rad(uniform(rng, cfg.layout.angle))
        x, y = r * np.cos(angle), r * np.sin(angle)
        if all(np.hypot(x - p.x, y - p.y) > radius + p.radius + cfg.layout.min_gap for p in placed):
            placed.append(Footprint(x, y, radius))
            return [float(x), float(y)]
    return None


def _color(
    rng: np.random.Generator, cfg: GenerationConfig, exclude: Sequence[str] = ()
) -> tuple[str, list[float]]:
    names = [name for name in PALETTE if name not in exclude]
    name = names[rng.integers(len(names))]
    rgb = np.clip(np.asarray(PALETTE[name]) + rng.uniform(-1, 1, 3) * cfg.objects.color_jitter, 0, 1)
    return name, [*rgb.tolist(), 1.0]


def _texture(rng: np.random.Generator, light: float, repeat: Range = (1.0, 1.0)) -> dict:
    """A pattern in shades of white, tinted by the material color, so the label's color stays true."""
    low = rng.uniform(light, 1.0)
    return {
        "kind": TEXTURE_KINDS[rng.integers(len(TEXTURE_KINDS))],
        "seed": int(rng.integers(2**31)),
        "rgb1": [1.0, 1.0, 1.0],
        "rgb2": [low, low, low],
        "repeat": uniform(rng, repeat),
    }


def _look(
    rng: np.random.Generator, rgba: list[float], texture_prob: float, repeat: Range = (1.0, 1.0)
) -> Look:
    return Look(
        rgba=rgba,
        texture=_texture(rng, 0.55, repeat) if rng.random() < texture_prob else None,
        specular=float(rng.uniform(0.0, 0.6)),
        shininess=float(rng.uniform(0.0, 0.8)),
    )


def _sample_object(
    rng: np.random.Generator,
    cfg: GenerationConfig,
    meshes: dict[str, MeshAsset],
    name: str,
    used_labels: set[str],
) -> ObjectParams:
    oc = cfg.objects
    mass = uniform(rng, oc.mass)
    friction = uniform(rng, oc.friction)
    if meshes and rng.random() < oc.mesh_prob:
        candidates = [m for m in meshes.values() if m.label not in used_labels]
        if candidates:
            mesh = candidates[rng.integers(len(candidates))]
            used_labels.add(mesh.label)
            scale = uniform(rng, mesh.scale)
            return ObjectParams(
                name=name,
                label=mesh.label,
                shape="mesh",
                half_size=(mesh.half_size * scale).tolist(),
                look=Look(rgba=[1.0, 1.0, 1.0, 1.0]),
                mass=mass,
                friction=friction,
                pos=[0.0, 0.0],
                yaw=float(rng.uniform(-np.pi, np.pi)),
                mesh=mesh.name,
                scale=scale,
            )
    for _ in range(50):
        color, rgba = _color(rng, cfg)
        width = uniform(rng, oc.width)
        height = uniform(rng, oc.height)
        shape = "cylinder" if rng.random() < 0.35 else "box"
        if shape == "cylinder":
            half_size = [width / 2, height / 2]
            noun = "puck" if height < 0.7 * width else "cylinder"
        else:
            length = width * float(rng.choice([1.0, rng.uniform(1.0, 2.2)]))
            half_size = [length / 2, width / 2, height / 2]
            ratio = length / width
            noun = (
                "cube"
                if ratio < 1.15 and abs(height - width) < 0.2 * width
                else ("bar" if ratio > 1.7 else "block")
            )
        label = f"{color} {noun}"
        if label not in used_labels:
            used_labels.add(label)
            return ObjectParams(
                name=name,
                label=label,
                shape=shape,
                half_size=half_size,
                look=_look(rng, rgba, oc.texture_prob, (2.0, 6.0)),
                mass=mass,
                friction=friction,
                pos=[0.0, 0.0],
                yaw=float(rng.uniform(-np.pi, np.pi)),
                color=color,
            )
    raise RuntimeError("Could not draw a unique object label")


def _sample_container(
    rng: np.random.Generator, cfg: GenerationConfig, name: str, used_labels: set[str]
) -> ContainerParams:
    cc = cfg.container
    wall_height = uniform(rng, cc.wall_height)
    nouns = TRAY_NOUNS if wall_height < 0.02 else CONTAINER_NOUNS
    color, rgba = _color(rng, cfg)
    label = f"{color} {nouns[rng.integers(len(nouns))]}"
    used_labels.add(label)
    return ContainerParams(
        name=name,
        label=label,
        inner_half=uniform(rng, cc.inner_size) / 2,
        wall_height=wall_height,
        thickness=uniform(rng, cc.wall_thickness),
        look=_look(rng, rgba, cc.texture_prob, (4.0, 10.0)),
        pos=[0.0, 0.0],
        yaw=float(rng.uniform(-np.pi, np.pi)),
        color=color,
    )


def _random_rgb(rng: np.random.Generator, low: float = 0.0, high: float = 1.0) -> list[float]:
    return rng.uniform(low, high, 3).tolist()


def _surface_rgb(rng: np.random.Generator, cfg: GenerationConfig) -> np.ndarray:
    """A surface color: mostly muted, sometimes vivid."""
    vivid = rng.random() < cfg.background.vivid_prob
    saturation = rng.uniform(0.4, 0.9) if vivid else rng.uniform(0.0, 0.35)
    return _hsv_to_rgb(rng.random(), saturation, rng.uniform(0.2, 0.95))


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    k = (np.array([5.0, 3.0, 1.0]) + h * 6) % 6
    return v - v * s * np.clip(np.minimum(k, 4 - k), 0, 1)


def _background_look(rng: np.random.Generator, cfg: GenerationConfig, repeat: Range) -> Look:
    rgb = _surface_rgb(rng, cfg)
    if rng.random() < cfg.background.texture_prob:
        # The second color contrasts in brightness, and sometimes in hue, so the pattern shows.
        second = rgb * rng.uniform(0.35, 0.7) if rng.random() < 0.7 else _surface_rgb(rng, cfg)
        texture = {
            "kind": TEXTURE_KINDS[rng.integers(len(TEXTURE_KINDS))],
            "seed": int(rng.integers(2**31)),
            "rgb1": rgb.tolist(),
            "rgb2": np.clip(second, 0, 1).tolist(),
            "repeat": uniform(rng, repeat),
        }
        return Look(
            rgba=[1.0, 1.0, 1.0, 1.0], texture=texture, specular=float(rng.uniform(0, 0.3)), shininess=0.1
        )
    return Look(rgba=[*rgb.tolist(), 1.0], specular=float(rng.uniform(0, 0.3)), shininess=0.1)


def _sample_background(rng: np.random.Generator, cfg: GenerationConfig) -> BackgroundParams:
    bc = cfg.background
    hx, hy = uniform(rng, bc.table_half_size), uniform(rng, bc.table_half_size)
    # The table spans at least x in [-0.1, 0.45] and y in [-0.35, 0.35]: under the base and the workspace.
    cx = float(rng.uniform(0.45 - hx, hx - 0.1))
    cy = float(rng.uniform(0.35 - hy, hy - 0.35))
    sky_low = rng.uniform(0.1, 0.8)
    return BackgroundParams(
        table=_background_look(rng, cfg, (1.0, 6.0)),
        table_half=[hx, hy],
        table_center=[float(cx), cy],
        floor=_background_look(rng, cfg, (2.0, 10.0)),
        wall=_background_look(rng, cfg, (1.0, 6.0)),
        walls=bool(rng.random() < bc.walls_prob),
        sky=[_random_rgb(rng, sky_low, 1.0), _random_rgb(rng, 0.0, sky_low)],
        headlight=uniform(rng, cfg.lighting.headlight),
        ambient=uniform(rng, cfg.lighting.ambient),
    )


def _sample_lights(rng: np.random.Generator, cfg: GenerationConfig) -> list[LightParams]:
    lc = cfg.lighting
    lights = []
    for _ in range(int(rng.integers(lc.count[0], lc.count[1] + 1))):
        azimuth = rng.uniform(-np.pi, np.pi)
        elevation = np.deg2rad(uniform(rng, lc.elevation))
        distance = uniform(rng, lc.distance)
        target = np.array([0.2, 0.0, 0.0]) + rng.uniform(-0.1, 0.1, 3) * [1, 1, 0]
        pos = target + distance * np.array(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
        )
        direction = (target - pos) / np.linalg.norm(target - pos)
        intensity = uniform(rng, lc.diffuse)
        tint = 1 + rng.uniform(-1, 1) * lc.tint * np.array([1.0, 0.2, -1.0])
        lights.append(
            LightParams(
                pos=pos.tolist(),
                dir=direction.tolist(),
                diffuse=np.clip(intensity * tint, 0, 1).tolist(),
                specular=float(rng.uniform(0.0, 0.4)),
                castshadow=bool(rng.random() < lc.shadow_prob),
                directional=bool(rng.random() < 0.5),
            )
        )
    return lights


def look_at_quat(pos: np.ndarray, target: np.ndarray, roll: float = 0.0) -> list[float]:
    """Quaternion of a MuJoCo camera at ``pos`` looking at ``target`` (cameras look along their -z)."""
    forward = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Roll around the viewing axis.
    c, s = np.cos(roll), np.sin(roll)
    right, up = c * right + s * up, -s * right + c * up
    mat = np.column_stack([right, up, -forward])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat.tolist()


def _sample_front_camera(rng: np.random.Generator, cfg: GenerationConfig) -> CameraParams:
    fc = cfg.front_camera
    target = np.asarray(fc.lookat) + rng.uniform(-1, 1, 3) * fc.lookat_jitter * np.array([1, 1, 0.3])
    azimuth = np.deg2rad(uniform(rng, fc.azimuth))
    elevation = np.deg2rad(uniform(rng, fc.elevation))
    distance = uniform(rng, fc.distance)
    # Azimuth 0 sits in front of the robot (+x), looking back at it.
    pos = target + distance * np.array(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
    )
    roll = np.deg2rad(rng.uniform(-fc.roll, fc.roll))
    return CameraParams(pos=pos.tolist(), quat=look_at_quat(pos, target, roll), fovy=uniform(rng, fc.fovy))
