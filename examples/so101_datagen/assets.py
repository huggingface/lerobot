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

"""External assets, fetched once into the LeRobot cache.

- Mesh objects: a curated subset of Google Scanned Objects in kevinzakka/mujoco_scanned_objects's MJCF,
  the models listed in ``gso_subset.json`` at a pinned commit. Google Scanned Objects are CC-BY-4.0
  (Google LLC).
- Everyday objects (fruit, vegetables, cans, cups, cutlery...): a curated subset of RoboCasa's Objaverse
  objects (robocasa/robocasa-assets, CC-BY-4.0), listed in ``robocasa_subset.json``. Only those are read
  from the pinned archive, by ranged requests, so the 2 GB archive is never downloaded whole.
- Environments: indoor HDRI panoramas (rooms, workshops, offices) shown as the skybox, and photographed
  surfaces for the table, from Poly Haven (CC0), listed with their checksums in ``polyhaven_subset.json``.
"""

import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mujoco
import numpy as np

if TYPE_CHECKING:
    from .scene import ObjectParams

GSO_REPO = "https://github.com/kevinzakka/mujoco_scanned_objects.git"
GSO_SHA = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"
SUBSET_FILE = Path(__file__).parent / "gso_subset.json"
ROBOCASA_REPO = "robocasa/robocasa-assets"
ROBOCASA_REVISION = "1b92c3d02ca4354984fec961357db0bff7b32166"
ROBOCASA_ARCHIVE = "objaverse.zip"
ROBOCASA_FILE = Path(__file__).parent / "robocasa_subset.json"
POLYHAVEN_FILE = Path(__file__).parent / "polyhaven_subset.json"
# Poly Haven refuses requests without a user agent of their own.
USER_AGENT = "lerobot-so101-datagen"


@dataclass
class MeshAsset:
    """One scanned object: its files, label and extent in its own frame (meters, unscaled)."""

    name: str
    label: str
    root: Path
    lower: np.ndarray
    upper: np.ndarray
    collision_parts: int
    # Uniform scales that keep it graspable.
    scale: tuple[float, float]

    @property
    def half_size(self) -> np.ndarray:
        return (self.upper - self.lower) / 2

    @property
    def center(self) -> np.ndarray:
        return (self.upper + self.lower) / 2

    def add_to(self, spec: mujoco.MjSpec, body: mujoco.MjsBody, obj: "ObjectParams", common: dict) -> None:
        """Add the textured visual mesh and the convex collision parts, centered on the body frame."""
        s = obj.scale
        offset = (-self.center * s).tolist()
        tex = spec.add_texture(
            name=f"{obj.name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(self.root / "texture.png")
        )
        mat = spec.add_material(name=f"{obj.name}_gso", specular=0.3, shininess=0.3)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        spec.add_mesh(name=f"{obj.name}_visual", file=str(self.root / "model.obj"), scale=[s, s, s])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{obj.name}_visual",
            material=mat.name,
            pos=offset,
            contype=0,
            conaffinity=0,
            group=2,
            mass=0,
        )
        collision = {key: value for key, value in common.items() if key != "material"}
        for i in range(self.collision_parts):
            spec.add_mesh(
                name=f"{obj.name}_col{i}", file=str(self.root / f"model_collision_{i}.obj"), scale=[s, s, s]
            )
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{obj.name}_col{i}",
                pos=offset,
                group=3,
                mass=obj.mass / self.collision_parts,
                **collision,
            )


def gso_dir(names: list[str]) -> Path:
    """A sparse checkout of the scanned objects holding ``names``, fetched at the pinned commit if needed.

    ``LEROBOT_GSO_DIR`` points at an existing full checkout instead.
    """
    if os.environ.get("LEROBOT_GSO_DIR"):
        return Path(os.environ["LEROBOT_GSO_DIR"]).expanduser()
    root = Path.home() / ".cache" / "lerobot" / "mujoco_scanned_objects" / GSO_SHA
    missing = [name for name in names if not (root / "models" / name / "model.xml").is_file()]
    if missing:
        git = ("git", "-C", str(root))
        if not (root / ".git").is_dir():
            root.mkdir(parents=True, exist_ok=True)
            subprocess.run((*git, "init", "-q"), check=True)  # nosec B603
            subprocess.run((*git, "remote", "add", "origin", GSO_REPO), check=True)  # nosec B603
            subprocess.run((*git, "sparse-checkout", "set", "--no-cone"), check=True)  # nosec B603
        subprocess.run(  # nosec B603
            (*git, "sparse-checkout", "set", "--no-cone", *[f"models/{name}/" for name in names]), check=True
        )
        subprocess.run(  # nosec B603
            (*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", GSO_SHA), check=True
        )
        subprocess.run((*git, "checkout", "-q", GSO_SHA), check=True)  # nosec B603
    return root


@dataclass
class MjcfAsset(MeshAsset):
    """One object in RoboCasa's MJCF layout: visual meshes with their materials, and convex collision parts.

    Its meshes, textures and materials are copied into the scene under the object's name, and its geoms
    into the object's body, so the object stays one body; MuJoCo applies the meshes' own scale and
    orientation. ``lower`` and ``upper`` come from the subset file, measured once at curation.
    """

    xml: Path = Path()

    def add_to(self, spec: mujoco.MjSpec, body: mujoco.MjsBody, obj: "ObjectParams", common: dict) -> None:
        source = mujoco.MjSpec.from_file(str(self.xml))
        prefix, s = f"{obj.name}_", obj.scale
        for texture in source.textures:
            spec.add_texture(
                name=prefix + texture.name, type=texture.type, file=str(self.root / texture.file)
            )
        rgb = mujoco.mjtTextureRole.mjTEXROLE_RGB
        for material in source.materials:
            copy = spec.add_material(
                name=prefix + material.name,
                rgba=material.rgba,
                specular=material.specular,
                shininess=material.shininess,
            )
            if material.textures[rgb]:
                copy.textures[rgb] = prefix + material.textures[rgb]
        files = {}
        for mesh in source.meshes:
            files[mesh.name] = mesh.file
            spec.add_mesh(
                name=prefix + mesh.name,
                file=str(self.root / mesh.file),
                scale=(np.asarray(mesh.scale) * s).tolist(),
                refquat=mesh.refquat,
            )
        offset = -self.center * s
        collision = {key: value for key, value in common.items() if key != "material"}
        geoms = [g for g in source.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]
        parts = {id(g) for g in geoms if files[g.meshname].startswith("collision")}
        for geom in geoms:
            placement = {
                "meshname": prefix + geom.meshname,
                "pos": (offset + np.asarray(geom.pos) * s).tolist(),
            }
            if id(geom) in parts:
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    group=3,
                    mass=obj.mass / len(parts),
                    **placement,
                    **collision,
                )
            else:
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    material=prefix + geom.material if geom.material else "",
                    contype=0,
                    conaffinity=0,
                    group=2,
                    mass=0,
                    **placement,
                )


def robocasa_dir(names: list[str]) -> Path:
    """The RoboCasa objects ``names`` (archive paths such as ``objaverse/apple/apple_3``), extracted if needed.

    Reads the pinned archive from the local Hugging Face cache when it is there, else only the members
    needed, by ranged requests. ``LEROBOT_ROBOCASA_DIR`` points at an existing extraction instead.
    """
    if os.environ.get("LEROBOT_ROBOCASA_DIR"):
        return Path(os.environ["LEROBOT_ROBOCASA_DIR"]).expanduser()
    root = Path.home() / ".cache" / "lerobot" / "robocasa" / ROBOCASA_REVISION
    missing = [name for name in names if not (root / name / "model.xml").is_file()]
    if missing:
        from huggingface_hub import HfFileSystem, try_to_load_from_cache

        local = try_to_load_from_cache(
            ROBOCASA_REPO, ROBOCASA_ARCHIVE, revision=ROBOCASA_REVISION, repo_type="dataset"
        )
        if isinstance(local, str):
            archive = open(local, "rb")  # noqa: SIM115
        else:
            path = f"datasets/{ROBOCASA_REPO}@{ROBOCASA_REVISION}/{ROBOCASA_ARCHIVE}"
            archive = HfFileSystem().open(path, "rb", block_size=2**20)
        root.mkdir(parents=True, exist_ok=True)
        with archive, zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            for name in missing:
                # Extracted beside the target and renamed into place, like every other fetch.
                with tempfile.TemporaryDirectory(dir=root) as tmp:
                    for member in members:
                        if member.startswith(name + "/") and not member.endswith("/"):
                            zf.extract(member, tmp)
                    (root / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.rmtree(root / name, ignore_errors=True)
                    os.replace(Path(tmp) / name, root / name)
    return root


def load_meshes(subset_file: Path = SUBSET_FILE, robocasa_file: Path = ROBOCASA_FILE) -> dict[str, MeshAsset]:
    """The curated mesh objects, scanned and everyday ones, fetched if needed, by model name."""
    meshes = _load_gso(subset_file)
    if robocasa_file.is_file():
        subset = json.loads(robocasa_file.read_text())
        root = robocasa_dir(sorted(subset))
        for name, entry in subset.items():
            meshes[name] = MjcfAsset(
                name=name,
                label=entry["label"],
                root=root / name,
                lower=np.asarray(entry["lower"]),
                upper=np.asarray(entry["upper"]),
                collision_parts=0,
                scale=tuple(entry["scale"]),
                xml=root / name / "model.xml",
            )
    return meshes


def _load_gso(subset_file: Path) -> dict[str, MeshAsset]:
    subset = json.loads(subset_file.read_text())
    if not subset:
        return {}
    root = gso_dir(sorted(subset))
    meshes = {}
    for name, entry in subset.items():
        model_dir = root / "models" / name
        lower, upper = _obj_bounds(model_dir / "model.obj")
        parts = len(list(model_dir.glob("model_collision_*.obj")))
        meshes[name] = MeshAsset(
            name=name,
            label=entry["label"],
            root=model_dir,
            lower=lower,
            upper=upper,
            collision_parts=parts,
            scale=tuple(entry.get("scale", (1.0, 1.0))),
        )
    return meshes


def _obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    points = np.asarray(vertices, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


@cache
def polyhaven_subset() -> dict:
    return json.loads(POLYHAVEN_FILE.read_text())


def polyhaven_dir() -> Path:
    """Where the environments are cached; ``LEROBOT_POLYHAVEN_DIR`` points elsewhere."""
    if os.environ.get("LEROBOT_POLYHAVEN_DIR"):
        return Path(os.environ["LEROBOT_POLYHAVEN_DIR"]).expanduser()
    return Path.home() / ".cache" / "lerobot" / "polyhaven"


def fetch_environments() -> None:
    """Download and convert every listed HDRI and surface that isn't cached yet."""
    for name in polyhaven_subset()["hdris"]:
        panorama_path(name)
    for name in polyhaven_subset()["surfaces"]:
        surface_path(name)


def panorama_path(name: str) -> Path:
    """The HDRI ``name`` tonemapped to an 8-bit equirectangular PNG, fetched and converted if needed."""
    out = polyhaven_dir() / "panoramas" / f"{name}.png"
    if not out.is_file():
        entry = polyhaven_subset()["hdris"][name]
        hdr = _download(entry["url"], polyhaven_dir() / "hdris" / Path(entry["url"]).name, entry["md5"])
        radiance = cv2.imread(str(hdr), cv2.IMREAD_UNCHANGED)
        _write_atomic(out, cv2.imencode(".png", _tonemap(radiance))[1].tobytes())
        hdr.unlink()
    return out


def surface_path(name: str) -> Path:
    """The color photograph of surface ``name`` as a PNG, which MuJoCo can load."""
    out = polyhaven_dir() / "surfaces" / f"{name}.png"
    if not out.is_file():
        entry = polyhaven_subset()["surfaces"][name]
        jpg = _download(entry["url"], polyhaven_dir() / "surfaces" / Path(entry["url"]).name, entry["md5"])
        _write_atomic(out, cv2.imencode(".png", cv2.imread(str(jpg)))[1].tobytes())
        jpg.unlink()
    return out


def skybox_image(name: str, yaw: float, exposure: float, face: int = 256) -> np.ndarray:
    """A panorama as MuJoCo skybox data, turned by ``yaw`` degrees: six faces stacked, RGB uint8."""
    panorama = cv2.cvtColor(cv2.imread(str(panorama_path(name))), cv2.COLOR_BGR2RGB)
    h, w = panorama.shape[:2]
    t = (np.arange(face) + 0.5) / face * 2 - 1
    a, b = np.meshgrid(t, -t)
    faces = []
    for x, y, z in _cube_faces(a, b):
        lon = np.arctan2(y, x) + np.deg2rad(yaw)
        lat = np.arctan2(z, np.hypot(x, y))
        map_x = ((0.5 - lon / (2 * np.pi)) % 1.0 * w).astype(np.float32)
        map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)
        faces.append(cv2.remap(panorama, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
    return np.clip(np.concatenate(faces) * exposure, 0, 255).astype(np.uint8)


def _cube_faces(a: np.ndarray, b: np.ndarray) -> list[tuple]:
    """World directions of the pixels of each skybox face, in MuJoCo's order (right, left, up, down,
    front, back), for ``a`` growing along columns and ``b`` up the rows. Fitted against rendered rays."""
    one = np.ones_like(a)
    return [
        (one, a, b),
        (-one, -a, b),
        (a, b, one),
        (a, -b, -one),
        (a, -one, b),
        (-a, one, b),
    ]


def _tonemap(radiance: np.ndarray) -> np.ndarray:
    """HDR radiance (BGR float) to 8-bit sRGB: exposed so the median is mid-gray, then Reinhard."""
    luminance = radiance @ np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
    exposed = radiance * (0.25 / max(float(np.median(luminance)), 1e-6))
    mapped = exposed / (1 + exposed)
    return (np.clip(mapped, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def _download(url: str, path: Path, md5: str) -> Path:
    if path.is_file():
        return path
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
        data = response.read()
    if hashlib.md5(data).hexdigest() != md5:  # nosec B324
        raise ValueError(f"Checksum mismatch for {url}")
    _write_atomic(path, data)
    return path


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file and a rename, so concurrent workers never read half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    os.replace(tmp.name, path)
