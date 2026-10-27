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

"""Mesh objects: a curated subset of Google Scanned Objects, in kevinzakka/mujoco_scanned_objects's MJCF.

Only the models listed in ``gso_subset.json`` are fetched, at a pinned commit, into the LeRobot cache.
Google Scanned Objects are CC-BY-4.0 (Google LLC).
"""

import json
import os
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .scene import ObjectParams

GSO_REPO = "https://github.com/kevinzakka/mujoco_scanned_objects.git"
GSO_SHA = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"
SUBSET_FILE = Path(__file__).parent / "gso_subset.json"


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


def load_meshes(subset_file: Path = SUBSET_FILE) -> dict[str, MeshAsset]:
    """The curated scanned objects, fetched if needed, by model name."""
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
