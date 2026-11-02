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

import json
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

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
    # Initial supporting surface above the table; runtime recovery refreshes this from physics.
    support_height: float = 0.0


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
    # Imported assets use a conservative rectangular usable floor. inner_half is its short half-width.
    style: str = "procedural"

    @property
    def inner_half_xy(self) -> np.ndarray:
        aspect = 1.0 if self.style == "procedural" else container_asset(self.style)["aspect"]
        return self.inner_half * np.array([aspect, 1.0])

    @property
    def outer_half_xy(self) -> np.ndarray:
        if self.style == "procedural":
            return np.full(2, self.inner_half + self.thickness)
        return self.inner_half * np.array(container_asset(self.style)["outer_half"])


CONTAINER_STYLES = (
    "procedural",
    "kitchenware_tray",
    "crate",
    "metal_tray_small",
    "metal_tray_medium",
    "metal_tray_long",
    "wooden_tray",
    "white_dish",
)


@lru_cache
def container_asset(style: str) -> dict:
    if style not in CONTAINER_STYLES or style == "procedural":
        raise ValueError(f"Unknown container style {style!r}")
    return json.loads((Path(__file__).parent / "container_assets" / style / "metadata.json").read_text())


def container_asset_directory(style: str) -> Path:
    """Require the one-time conversion without downloading during episode generation."""
    asset = container_asset(style)
    root = Path(__file__).parent / "container_assets" / style
    if (
        not (root / "visual.obj").is_file()
        or not (root / asset["albedo"]).is_file()
        or len(list(root.glob("collision_*.obj"))) != asset["collision_parts"]
    ):
        raise FileNotFoundError(
            f"Container assets for {style!r} need preparation. "
            "See examples/so101_datagen/container_assets/README.md and run "
            f"examples/so101_datagen/prepare_containers.py --styles {style}."
        )
    return root


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
    mode: str = "typical"


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
    elif look.texture is not None and look.texture["kind"] == "container_asset":
        style = look.texture["name"]
        asset = container_asset(style)
        texture = spec.add_texture(
            name=f"{name}_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(container_asset_directory(style) / asset["albedo"]),
        )
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        # Preserve the source mesh UVs, including the crate's authored 3x repeat.
        mat.texuniform = False
        mat.texrepeat = [1, 1]
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


# Rolling resistance of mesh objects, in meters (contact torque per unit normal force).
ROLLING_FRICTION = 0.003


def _add_object(spec: mujoco.MjSpec, obj: ObjectParams, meshes: dict[str, MeshAsset]) -> None:
    half_height = obj.half_size[-1]
    body = spec.worldbody.add_body(
        name=obj.name,
        pos=[obj.pos[0], obj.pos[1], obj.support_height + half_height + 0.002],
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
        # Scanned fruit and vegetables are round: without rolling friction they roll away for good.
        common.update(condim=6, friction=[obj.friction, 0.02, ROLLING_FRICTION])
        meshes[obj.mesh].add_to(spec, body, obj, common)
    else:
        raise ValueError(f"Unknown object shape {obj.shape!r}")


def _add_container(spec: mujoco.MjSpec, container: ContainerParams) -> None:
    body = spec.worldbody.add_body(
        name=container.name, pos=[*container.pos, 0.0], quat=_yaw_quat(container.yaw)
    )
    material = _material(spec, f"{container.name}_mat", container.look)
    if container.style != "procedural":
        asset = container_asset(container.style)
        root = container_asset_directory(container.style)
        scale = [container.inner_half, container.inner_half, container.wall_height / asset["height"]]
        for i, path in enumerate([root / "visual.obj", *sorted(root.glob("collision_*.obj"))]):
            name = f"{container.name}_mesh_{i}"
            spec.add_mesh(name=name, file=str(path.resolve()), scale=scale)
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                material=material,
                contype=0 if i == 0 else 1,
                conaffinity=0 if i == 0 else 1,
                group=2 if i == 0 else 3,
            )
        # A solid support slab avoids gaps between decomposed floor pieces and stabilizes thin floors.
        floor = container.wall_height * asset["floor"] / asset["height"]
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[*container.inner_half_xy, floor / 2],
            pos=[0, 0, floor / 2],
            solref=[0.004, 1.0],
            priority=1,
            group=3,
        )
        return
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
