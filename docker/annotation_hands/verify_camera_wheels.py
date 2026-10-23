"""Verify CPU-built camera operators against the immutable worker provenance."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


def checksum(path):
    """Hash an artifact without loading the wheel into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory: Path, version: dict):
    """Require the pinned source build, both operators, all offered GPUs and exact bytes."""
    report = json.loads((directory / "build.json").read_text())
    expected = version["camera_operator_build"]
    if (
        report.get("status") != "success"
        or report.get("cpu_imports_passed") is not True
        or report.get("gpu_inference") is not False
        or report.get("source_repo") != expected["source_repo"]
        or report.get("source_revision") != expected["source_revision"]
        or report.get("image_version_sha256") != expected["source_image_version_sha256"]
        or report.get("cuda_arch_list") != "8.0;9.0;12.0+PTX"
    ):
        raise ValueError("Camera wheels do not match the pinned CPU source build")
    if checksum(directory / "build.json") != version["source_files"]["camera-wheels/build.json"]:
        raise ValueError("Camera build provenance checksum mismatch")
    modules, files = set(), set()
    for entry in report["wheels"]:
        relative = PurePosixPath(entry["path"])
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "wheels"
            or "-cp310-" not in relative.name
            or not relative.name.endswith("linux_x86_64.whl")
        ):
            raise ValueError("Expected a Python 3.10 Linux camera wheel")
        path = directory / str(relative)
        if path.stat().st_size != entry["bytes"] or checksum(path) != entry["sha256"]:
            raise ValueError("Camera wheel checksum mismatch")
        if entry["sha256"] != version["source_files"]["camera-wheels/" + str(relative)]:
            raise ValueError("Camera wheel differs from the worker image manifest")
        with zipfile.ZipFile(path) as wheel:
            operators = {name for name in wheel.namelist() if name.endswith(".so")}
            if operators != set(entry["architectures"]):
                raise ValueError("Camera wheel native module catalog changed")
            for name in operators:
                if not wheel.read(name).startswith(b"\x7fELF"):
                    raise ValueError("Camera operator is not an ELF binary")
                if not all(arch in entry["architectures"][name] for arch in ("sm_80", "sm_90", "sm_120")):
                    raise ValueError("Camera operator is missing an offered HF GPU architecture")
                modules.add(Path(name).name.split(".")[0])
        files.add(path.name)
    if modules != set(version["compiled_camera_extensions"]) or len(files) != 2:
        raise ValueError("Expected only LieTorch and DROID camera operator wheels")
    if files != {path.name for path in (directory / "wheels").glob("*.whl")}:
        raise ValueError("Unverified camera wheel present")
    print("Verified CPU camera wheels: Python 3.10, pinned source, SM80/SM90/SM120", flush=True)


if __name__ == "__main__":
    verify(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()))
