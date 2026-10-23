"""Supplemental Python vendors cannot replace the frozen native implementation."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docker/annotation_hands/install_startup_sources.py"
spec = importlib.util.spec_from_file_location("hand_startup_sources", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
PREFIX = "release-195fps-verified/DROID-W/thirdparty/gaussian_splatting/"


def bundle(tmp_path, name=PREFIX + "utils/graphics_utils.py"):
    archive = tmp_path / "camera-startup-source.tar.gz"
    files = {name: b"# frozen vendor source\n", PREFIX + "LICENSE.md": b"Preserved notice\n"}
    entries = []
    with tarfile.open(archive, "w:gz") as target:
        for path, data in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
            entries.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    manifest = tmp_path / "camera-startup-sources.json"
    manifest.write_text(
        json.dumps(
            {
                "files": entries,
                "archive_sha256": installer.checksum(archive),
                "python_sources_only": True,
                "compiled_binaries": False,
                "model_weights": False,
            }
        )
    )
    version = tmp_path / "image-version.json"
    version.write_text(
        json.dumps(
            {
                "source_files": {
                    archive.name: installer.checksum(archive),
                    manifest.name: installer.checksum(manifest),
                }
            }
        )
    )
    return archive, manifest, version


def test_additive_vendors_preserve_frozen_native_files(tmp_path):
    source = bundle(tmp_path)
    destination = tmp_path / "runtime"
    native = destination / "release-195fps-verified/DROID-W/src/lib/droid.cpp"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"frozen operator source")
    installer.install(*source, destination)
    assert native.read_bytes() == b"frozen operator source"
    assert (destination / (PREFIX + "utils/graphics_utils.py")).read_bytes() == b"# frozen vendor source\n"
    with pytest.raises(FileExistsError):
        installer.install(*source, destination)


@pytest.mark.parametrize("name", ["../escape.py", PREFIX + "operator.so", PREFIX + "weights.pt"])
def test_startup_sources_reject_escape_operators_and_weights(tmp_path, name):
    source = bundle(tmp_path, name)
    destination = tmp_path / "runtime"
    with pytest.raises(ValueError, match="additive Python"):
        installer.install(*source, destination)
    assert not destination.exists()
