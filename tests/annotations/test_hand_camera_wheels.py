# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Fail-closed deployment provenance checks; synthetic binaries never execute."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

source = Path(__file__).parents[2] / "docker/annotation_hands/verify_camera_wheels.py"
spec = importlib.util.spec_from_file_location("verify_camera_wheels", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def build(tmp_path):
    directory = tmp_path / "camera-wheels"
    (directory / "wheels").mkdir(parents=True)
    report = {
        "status": "success",
        "cpu_imports_passed": True,
        "gpu_inference": False,
        "source_repo": "test/private-source",
        "source_revision": "a" * 40,
        "image_version_sha256": "b" * 64,
        "cuda_arch_list": "8.0;9.0;12.0+PTX",
        "wheels": [],
    }
    version = {
        "camera_operator_build": {
            "source_repo": report["source_repo"],
            "source_revision": report["source_revision"],
            "source_image_version_sha256": report["image_version_sha256"],
        },
        "compiled_camera_extensions": ["lietorch_backends", "droid_backends"],
        "source_files": {},
    }
    for name in version["compiled_camera_extensions"]:
        path = directory / "wheels" / f"{name}-0.0-cp310-cp310-linux_x86_64.whl"
        operator = name + ".cpython-310-x86_64-linux-gnu.so"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(operator, b"\x7fELFsynthetic fixture, never imported")
        digest = module.checksum(path)
        version["source_files"]["camera-wheels/wheels/" + path.name] = digest
        report["wheels"].append(
            {
                "path": "wheels/" + path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "architectures": {operator: "sm_80 sm_90 sm_120"},
            }
        )

    def seal():
        p = directory / "build.json"
        p.write_text(json.dumps(report))
        version["source_files"]["camera-wheels/build.json"] = module.checksum(p)

    seal()
    return directory, report, version, seal


def test_complete_deployment_receipt_accepts_only_pinned_operator_catalog(build):
    directory, report, version, seal = build
    module.verify(directory, version)
    report["source_revision"] = "c" * 40
    seal()
    with pytest.raises(ValueError, match="pinned CPU"):
        module.verify(directory, version)


def test_tampered_wheel_or_extra_package_is_rejected(build):
    directory, report, version, seal = build
    extra = directory / "wheels/extra-cp310-cp310-linux_x86_64.whl"
    extra.write_bytes(b"unverified package")
    with pytest.raises(ValueError, match="Unverified"):
        module.verify(directory, version)
    extra.unlink()
    wheel = directory / report["wheels"][0]["path"]
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module.verify(directory, version)


def test_missing_hf_gpu_architecture_is_rejected_even_with_matching_file_hashes(build):
    directory, report, version, seal = build
    operators = report["wheels"][0]["architectures"]
    operators[next(iter(operators))] = "sm_80 sm_120"
    seal()
    with pytest.raises(ValueError, match="architecture"):
        module.verify(directory, version)


def test_provenance_cannot_silently_change_after_quote(build):
    directory, report, version, seal = build
    (directory / "build.json").write_text(json.dumps({**report, "elapsed_s": 0}))
    with pytest.raises(ValueError, match="provenance checksum"):
        module.verify(directory, version)
