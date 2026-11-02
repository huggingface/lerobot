# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""One-time conversion of pinned SimReady USD packages into ignored MuJoCo assets.

Run in a separate environment with usd-core, coacd, scipy, numpy, pillow and huggingface-hub.
See container_assets/README.md. Generation itself needs none of the conversion dependencies.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import coacd
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from pxr import Usd, UsdGeom
from scipy.spatial import ConvexHull, QhullError

ASSETS = Path(__file__).parent / "container_assets"
# Conservative source-space inner half sizes; source meshes use meters.
INNER = {
    "kitchenware_tray": (0.108, 0.170),
    "crate": (0.270, 0.170),
    "metal_tray_small": (0.084, 0.046),
    "metal_tray_medium": (0.095, 0.060),
    "metal_tray_long": (0.173, 0.086),
    "wooden_tray": (0.120, 0.150),
    "white_dish": (0.185, 0.115),
}
LEGACY = {"kitchenware_tray", "crate"}


def write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
    precision: int = 8,
) -> None:
    with path.open("w") as stream:
        stream.write("# Adapted from SimReady assets, CC-BY-4.0; see ../README.md\n")
        for vertex in vertices:
            stream.write("v " + " ".join(f"{x:.{precision}g}" for x in vertex) + "\n")
        if uv is not None:
            for point in uv:
                stream.write("vt " + " ".join(f"{x:.8g}" for x in point) + "\n")
        for i, face in enumerate(faces):
            stream.write(
                "f "
                + " ".join(
                    str(int(x) + 1) + (f"/{int(uv_faces[i, j]) + 1}" if uv is not None else "")
                    for j, x in enumerate(face)
                )
                + "\n"
            )


def hull(vertices: np.ndarray) -> ConvexHull | None:
    try:
        result = ConvexHull(vertices)
        return result if result.volume >= 1e-11 else None
    except QhullError:
        return None


def clip(vertices: np.ndarray, axis: int, sign: int, bound: float) -> np.ndarray | None:
    distance = sign * vertices[:, axis] - bound
    inside = distance <= 1e-9
    if not inside.any():
        return None
    if inside.all():
        return vertices
    poly = hull(vertices)
    if poly is None:
        return None
    points = list(vertices[inside])
    edges = {
        tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) for face in poly.simplices for i in range(3)
    }
    for i, j in edges:
        if inside[i] != inside[j]:
            points.append(
                vertices[i] + distance[i] / (distance[i] - distance[j]) * (vertices[j] - vertices[i])
            )
    result = np.unique(np.round(points, 10), axis=0)
    return result if len(result) >= 4 and hull(result) is not None else None


def exterior_parts(parts: list, half: list[float]) -> list:
    result = []
    for vertices, _ in parts:
        # Match OBJ serialization before clipping, as in the reviewed conversion.
        vertices = np.array([[float(f"{x:.8g}") for x in row] for row in vertices])
        if any(
            vertices[:, axis].min() >= half[axis] or vertices[:, axis].max() <= -half[axis] for axis in (0, 1)
        ):
            clipped = [vertices]
        else:
            clipped = [clip(vertices, axis, sign, -half[axis]) for axis in (0, 1) for sign in (-1, 1)]
        for vertices in clipped:
            if vertices is None:
                continue
            poly = hull(vertices)
            if poly is None:
                continue
            faces = poly.simplices.copy()
            for j, face in enumerate(faces):
                if (
                    np.dot(
                        np.cross(
                            vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]]
                        ),
                        poly.equations[j, :3],
                    )
                    < 0
                ):
                    faces[j] = face[::-1]
            result.append((vertices, faces))
    return result


def convert(style: str, package: Path, destination: Path) -> None:
    meta = json.loads((ASSETS / style / "metadata.json").read_text())
    root = package / "simready_usd"
    stage = Usd.Stage.Open(str(next(root.glob("*.usd"))))
    default = stage.GetDefaultPrim()
    if default.GetVariantSet("PhysX").GetVariantNames():
        default.GetVariantSet("PhysX").SetVariantSelection("Disabled")
        default.GetVariantSet("MuJoCo").SetVariantSelection("Enabled")
    visuals, colliders = [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ transform)[:, :3]
        faces, uv_faces, cursor = [], [], 0
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        for count in mesh.GetFaceVertexCountsAttr().Get():
            for i in range(1, count - 1):
                ids = np.array([cursor, cursor + i, cursor + i + 1])
                faces.append(indices[ids])
                uv_faces.append(ids)
            cursor += count
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        uv = np.asarray(st.ComputeFlattened()) * meta.get("uv_scale", 1) if st else None
        entry = (vertices, np.asarray(faces, dtype=np.int32), uv, np.asarray(uv_faces))
        if UsdGeom.Imageable(prim).GetPurposeAttr().Get() == "guide":
            colliders.append(entry)
        else:
            visuals.append(entry)
    if len(visuals) != 1:
        raise ValueError(f"Expected one visual mesh for {style}, got {len(visuals)}")
    vertices, faces, uv, uv_faces = visuals[0]
    center = (vertices.min(0) + vertices.max(0)) / 2
    bottom = vertices[:, 2].min()
    inner = INNER[style]

    def normalized(points: np.ndarray) -> np.ndarray:
        points = points.copy()
        if style not in LEGACY:
            points[:, :2] -= center[:2]
        points[:, 2] -= bottom
        if inner[0] < inner[1]:
            points = points[:, [1, 0, 2]] * np.array([1, -1, 1])
        return points / min(inner)

    vertices = normalized(vertices)
    destination.mkdir(parents=True, exist_ok=True)
    write_obj(
        destination / "visual.obj", vertices, faces, uv, uv_faces, precision=7 if style in LEGACY else 8
    )
    if colliders:
        parts = [(normalized(entry[0]), entry[1]) for entry in colliders]
    else:
        parts = coacd.run_coacd(
            coacd.Mesh(vertices, faces),
            threshold=0.025,
            max_convex_hull=48 if style in LEGACY else 40,
            preprocess_resolution=60,
            resolution=3000,
            mcts_iterations=80,
            seed=0,
        )
    if style not in LEGACY:
        parts = exterior_parts(parts, [meta["aspect"], 1.0])
    if len(parts) != meta["collision_parts"]:
        raise ValueError(f"{style}: expected {meta['collision_parts']} collision parts, got {len(parts)}")
    for i, (points, triangles) in enumerate(parts):
        write_obj(
            destination / f"collision_{i:03d}.obj", points, triangles, precision=7 if style in LEGACY else 9
        )
    texture = package / meta["source_albedo"]
    source_hash = meta.get("source_albedo_sha256", meta["albedo_sha256"])
    if hashlib.sha256(texture.read_bytes()).hexdigest() != source_hash:
        raise ValueError(f"Source texture checksum mismatch for {style}")
    if style in LEGACY:
        shutil.copyfile(texture, destination / "albedo.png")
    else:
        image = Image.open(texture).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image.save(destination / "albedo.png")
    if hashlib.sha256((destination / "albedo.png").read_bytes()).hexdigest() != meta["albedo_sha256"]:
        raise ValueError(f"Converted texture checksum mismatch for {style}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styles", nargs="+", choices=list(INNER), default=list(INNER))
    parser.add_argument(
        "--source-root", type=Path, help="Optional offline directory containing source asset packages"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    coacd.set_log_level("warn")
    for style in args.styles:
        destination = ASSETS / style
        meta = json.loads((destination / "metadata.json").read_text())
        if (
            not args.force
            and (destination / "visual.obj").exists()
            and (destination / "albedo.png").exists()
            and len(list(destination.glob("collision_*.obj"))) == meta["collision_parts"]
        ):
            print(f"{style}: using existing converted assets", flush=True)
            continue
        if args.source_root is None:
            snapshot = Path(
                snapshot_download(
                    meta["source_repo"],
                    repo_type="dataset",
                    revision=meta["revision"],
                    allow_patterns=[f"{meta['source_asset']}/simready_usd/*"],
                )
            )
            package = snapshot / meta["source_asset"]
        else:
            package = args.source_root / meta["source_asset"]
        # Publish only complete, validated conversions; failed builds preserve previous local assets.
        with tempfile.TemporaryDirectory(prefix="so101-container-") as temporary:
            built = Path(temporary)
            convert(style, package, built)
            for path in destination.glob("collision_*.obj"):
                path.unlink()
            for path in built.iterdir():
                shutil.copyfile(path, destination / path.name)
        print(f"{style}: converted once; future runs reuse local files", flush=True)


if __name__ == "__main__":
    main()
