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

"""A tabletop scene described by plain parameters, compiled into a MuJoCo model with ``MjSpec``.

``randomization.py`` samples ``SceneParams``; this module only applies them, so a scene can be stored as
JSON next to its episode and rebuilt exactly.
"""

import warnings
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .assets import MeshAsset, skybox_image, surface_path
from .so101 import JOINTS, arm_spec


@dataclass
class Look:
    rgba: list[float]
    # A procedural texture, {"kind": "grain" or "noise", "seed", "rgb1", "rgb2", "repeat"}, or a photographed
    # surface, {"kind": "image", "name", "size"} with ``size`` the meters one image spans.
    texture: dict | None = None
    specular: float = 0.3
    shininess: float = 0.3


@dataclass
class ObjectParams:
    name: str
    label: str
    # "box" (half sizes x, y, z), "cylinder" (radius, half height) or "mesh".
    shape: str
    half_size: list[float]
    look: Look
    mass: float
    friction: float
    pos: list[float]
    yaw: float
    color: str | None = None
    mesh: str | None = None
    scale: float = 1.0


@dataclass
class ContainerParams:
    name: str
    label: str
    inner_half: float
    wall_height: float
    thickness: float
    look: Look
    pos: list[float]
    yaw: float
    color: str | None = None


@dataclass
class LightParams:
    pos: list[float]
    dir: list[float]
    diffuse: list[float]
    specular: float
    castshadow: bool
    directional: bool


@dataclass
class CameraParams:
    pos: list[float]
    quat: list[float]
    fovy: float


@dataclass
class BackgroundParams:
    table: Look
    table_half: list[float]
    table_center: list[float]
    floor: Look
    wall: Look
    walls: bool
    # A gradient for plain rooms; an indoor panorama, turned and exposed, replaces it and the walls.
    sky: list[list[float]]
    hdri: str | None
    hdri_yaw: float
    hdri_exposure: float
    headlight: float
    ambient: float


@dataclass
class DynamicsParams:
    kp_scale: list[float]
    damping_scale: list[float]
    frictionloss_scale: list[float]
    arm_rgba: list[float]


@dataclass
class SceneParams:
    objects: list[ObjectParams]
    containers: list[ContainerParams]
    lights: list[LightParams]
    background: BackgroundParams
    front_camera: CameraParams
    # Offsets from Menagerie's wrist camera: position, rotation vector, and the field of view.
    wrist_offset: list[float]
    wrist_rotvec: list[float]
    wrist_fovy: float
    dynamics: DynamicsParams


@dataclass
class SceneInfo:
    """What the rest of the generator needs to know about a compiled scene."""

    params: SceneParams
    object_bodies: dict[str, int] = field(default_factory=dict)
    container_bodies: dict[str, int] = field(default_factory=dict)
    # Geoms of each object or container, rendered or not, for segmentation and contacts.
    geoms: dict[str, list[int]] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    table_geom: int = -1


def texture_image(texture: dict, size: int = 256) -> np.ndarray:
    """A seeded procedural pattern blending ``rgb1`` and ``rgb2``, as HxWx3 uint8."""
    rng = np.random.default_rng(texture["seed"])
    kind = texture["kind"]
    u, v = np.meshgrid(np.linspace(0, 1, size, endpoint=False), np.linspace(0, 1, size, endpoint=False))
    if kind == "grain":
        # Wood-like: wavy rings along one axis.
        freq = rng.uniform(8, 25)
        warp = _value_noise(rng, size, 4) * rng.uniform(0.5, 2.0)
        t = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (u + 0.1 * warp))
    elif kind == "noise":
        t = _value_noise(rng, size, int(rng.integers(4, 32)))
    else:
        raise ValueError(f"Unknown texture kind {kind!r}")
    t = np.clip(t, 0, 1)[..., None]
    rgb1, rgb2 = np.asarray(texture["rgb1"]), np.asarray(texture["rgb2"])
    return (np.clip((1 - t) * rgb1 + t * rgb2, 0, 1) * 255).astype(np.uint8)


def _value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Smooth noise in [0, 1]: a coarse random grid upsampled bilinearly."""
    grid = rng.random((cells + 1, cells + 1))
    x = np.linspace(0, cells, size, endpoint=False)
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    top = grid[i][:, i] * (1 - f)[None, :] + grid[i][:, i + 1] * f[None, :]
    bottom = grid[i + 1][:, i] * (1 - f)[None, :] + grid[i + 1][:, i + 1] * f[None, :]
    return top * (1 - f)[:, None] + bottom * f[:, None]


def build_scene(
    params: SceneParams, meshes: dict[str, MeshAsset], height: int, width: int
) -> tuple[mujoco.MjModel, SceneInfo]:
    spec = mujoco.MjSpec()
    arm = arm_spec()
    # Attaching keeps the parent's physics options: use the arm's tuned ones.
    for name in dir(arm.option):
        if not name.startswith("_"):
            setattr(spec.option, name, getattr(arm.option, name))
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 2048
    # The workspace extent, not the room: it scales the near clipping plane the wrist camera needs.
    spec.stat.extent = 0.6
    spec.stat.center = [0.2, 0.0, 0.1]

    bg = params.background
    spec.visual.headlight.diffuse = [bg.headlight] * 3
    spec.visual.headlight.ambient = [bg.ambient] * 3
    spec.visual.headlight.specular = [0.0] * 3
    if bg.hdri is not None:
        sky = skybox_image(bg.hdri, bg.hdri_yaw, bg.hdri_exposure)
        texture = spec.add_texture(
            name="sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            width=sky.shape[1],
            height=sky.shape[0],
            nchannel=3,
        )
        texture.data = sky.tobytes()
    else:
        spec.add_texture(
            name="sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=bg.sky[0],
            rgb2=bg.sky[1],
            width=256,
            height=1536,
        )
    world = spec.worldbody
    tx, ty = bg.table_center
    hx, hy = bg.table_half
    table = world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, 0.015],
        pos=[tx, ty, -0.015],
        material=_material(spec, "table", bg.table),
    )
    table.friction = [1.0, 0.005, 0.0001]
    decor = {"contype": 0, "conaffinity": 0, "type": mujoco.mjtGeom.mjGEOM_BOX}
    # A panorama shows its own floor and walls.
    if bg.hdri is None:
        world.add_geom(
            name="floor",
            size=[3, 3, 0.01],
            pos=[0, 0, -0.76],
            material=_material(spec, "floor", bg.floor),
            **decor,
        )
    if bg.walls:
        wall = _material(spec, "wall", bg.wall)
        world.add_geom(size=[0.01, 3, 1.5], pos=[-1.2, 0, 0.0], material=wall, **decor)
        world.add_geom(size=[3, 0.01, 1.5], pos=[0, 1.6, 0.0], material=wall, **decor)
        world.add_geom(size=[3, 0.01, 1.5], pos=[0, -1.6, 0.0], material=wall, **decor)
        world.add_geom(size=[0.01, 3, 1.5], pos=[2.0, 0, 0.0], material=wall, **decor)

    for i, light in enumerate(params.lights):
        world.add_light(
            name=f"light_{i}",
            pos=light.pos,
            dir=light.dir,
            diffuse=light.diffuse,
            specular=[light.specular] * 3,
            castshadow=light.castshadow,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            if light.directional
            else mujoco.mjtLightType.mjLIGHT_SPOT,
            cutoff=60,
        )
    cam = params.front_camera
    world.add_camera(name="front", pos=cam.pos, quat=cam.quat, fovy=cam.fovy)

    _apply_arm_params(arm, params)
    with warnings.catch_warnings():
        # Joint names must stay unprefixed: they are the dataset's feature names.
        warnings.simplefilter("ignore")
        spec.attach(arm, frame=world.add_frame(), prefix="")

    for obj in params.objects:
        _add_object(spec, obj, meshes)
    for container in params.containers:
        _add_container(spec, container)

    model = spec.compile()
    _apply_dynamics(model, params.dynamics)

    info = SceneInfo(params=params, table_geom=model.geom("table").id)
    for obj in params.objects:
        body = model.body(obj.name).id
        info.object_bodies[obj.name] = body
        info.geoms[obj.name] = _body_geoms(model, body)
        info.labels[obj.name] = obj.label
    for container in params.containers:
        body = model.body(container.name).id
        info.container_bodies[container.name] = body
        info.geoms[container.name] = _body_geoms(model, body)
        info.labels[container.name] = container.label
    return model, info


def _material(spec: mujoco.MjSpec, name: str, look: Look) -> str:
    mat = spec.add_material(name=name, rgba=look.rgba, specular=look.specular, shininess=look.shininess)
    if look.texture is not None and look.texture["kind"] == "image":
        spec.add_texture(
            name=f"{name}_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(surface_path(look.texture["name"])),
        )
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
        # With uniform mapping, texrepeat counts repeats per meter: the photograph keeps its real scale.
        mat.texrepeat = [1 / look.texture["size"]] * 2
        mat.texuniform = True
    elif look.texture is not None:
        image = texture_image(look.texture)
        tex = spec.add_texture(
            name=f"{name}_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=image.shape[1],
            height=image.shape[0],
            nchannel=3,
        )
        tex.data = image.tobytes()
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
        repeat = look.texture.get("repeat", 1.0)
        mat.texrepeat = [repeat, repeat]
        mat.texuniform = True
    return name


def _apply_arm_params(arm: mujoco.MjSpec, params: SceneParams) -> None:
    for material in arm.materials:
        if not material.name.startswith("sts3215"):
            material.rgba = params.dynamics.arm_rgba
    wrist = arm.camera("wrist_cam")
    wrist.pos = np.asarray(wrist.pos) + np.asarray(params.wrist_offset)
    delta = np.zeros(4)
    mujoco.mju_axisAngle2Quat(delta, *_axis_angle(params.wrist_rotvec))
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(wrist.quat, dtype=np.float64), delta)
    wrist.quat = quat
    wrist.fovy = params.wrist_fovy


def _axis_angle(rotvec: list[float]) -> tuple[np.ndarray, float]:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return rotvec / angle, angle


def _apply_dynamics(model: mujoco.MjModel, dynamics: DynamicsParams) -> None:
    for i, joint in enumerate(JOINTS):
        act = model.actuator(joint).id
        dof = model.joint(joint).dofadr[0]
        # Position actuators: force = kp * (ctrl - q) - kv * qdot.
        model.actuator_gainprm[act, 0] *= dynamics.kp_scale[i]
        model.actuator_biasprm[act, 1] *= dynamics.kp_scale[i]
        model.dof_damping[dof] *= dynamics.damping_scale[i]
        model.dof_frictionloss[dof] *= dynamics.frictionloss_scale[i]


def _add_object(spec: mujoco.MjSpec, obj: ObjectParams, meshes: dict[str, MeshAsset]) -> None:
    half_height = obj.half_size[-1]
    body = spec.worldbody.add_body(
        name=obj.name,
        pos=[obj.pos[0], obj.pos[1], half_height + 0.002],
        quat=_yaw_quat(obj.yaw),
    )
    body.add_freejoint()
    material = _material(spec, f"{obj.name}_mat", obj.look)
    common = {
        "material": material,
        "condim": 4,
        "friction": [obj.friction, 0.01, 0.001],
        "solref": [0.01, 1],
    }
    if obj.shape == "box":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=obj.half_size, mass=obj.mass, **common)
    elif obj.shape == "cylinder":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=obj.half_size, mass=obj.mass, **common)
    elif obj.shape == "mesh":
        meshes[obj.mesh].add_to(spec, body, obj, common)
    else:
        raise ValueError(f"Unknown object shape {obj.shape!r}")


def _add_container(spec: mujoco.MjSpec, container: ContainerParams) -> None:
    body = spec.worldbody.add_body(
        name=container.name, pos=[*container.pos, 0.0], quat=_yaw_quat(container.yaw)
    )
    material = _material(spec, f"{container.name}_mat", container.look)
    h, t, w = container.wall_height, container.thickness, container.inner_half
    outer = w + t
    walls = [
        ([outer, outer, t / 2], [0, 0, t / 2]),
        ([t / 2, outer, h / 2], [w + t / 2, 0, h / 2]),
        ([t / 2, outer, h / 2], [-w - t / 2, 0, h / 2]),
        ([outer, t / 2, h / 2], [0, w + t / 2, h / 2]),
        ([outer, t / 2, h / 2], [0, -w - t / 2, h / 2]),
    ]
    for size, pos in walls:
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=material)


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def _body_geoms(model: mujoco.MjModel, body: int) -> list[int]:
    start, count = model.body_geomadr[body], model.body_geomnum[body]
    return list(range(start, start + count))
