"""Check the frozen checkpoint guard before spending GPU startup time."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docker/annotation_hands/verify_runtime_imports.py"
spec = importlib.util.spec_from_file_location("hand_startup_paths", SCRIPT)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


def layout(tmp_path):
    camera = tmp_path / "opt/hand-camera"
    wrapper = camera / "DROID-on"
    frozen = camera / "release-195fps-verified/DROID-W"
    wrapper.mkdir(parents=True)
    frozen.mkdir(parents=True)
    assets = tmp_path / "workspace/hand-assets/camera"
    (frozen / "pretrained").symlink_to(assets)
    return wrapper, frozen, assets


def test_wrapper_alias_retains_frozen_guard_without_weights(tmp_path):
    wrapper, frozen, assets = layout(tmp_path)
    (wrapper / "pretrained").symlink_to(frozen / "pretrained")
    assert not assets.exists()
    startup.verify_checkpoint_alias(wrapper)


def test_missing_wrapper_alias_rejects_image(tmp_path):
    wrapper, _, _ = layout(tmp_path)
    with pytest.raises(RuntimeError, match="wrapper does not resolve"):
        startup.verify_checkpoint_alias(wrapper)


def test_wrong_asset_mount_rejects_image(tmp_path):
    wrapper, frozen, _ = layout(tmp_path)
    (frozen / "pretrained").unlink()
    (frozen / "pretrained").symlink_to(tmp_path / "different-weights")
    (wrapper / "pretrained").symlink_to(frozen / "pretrained")
    with pytest.raises(RuntimeError, match="Job asset mount"):
        startup.verify_checkpoint_alias(wrapper)


@pytest.mark.parametrize("header_count", [13, 14])
def test_frozen_solver_rejects_cudss_07_csr_signature(tmp_path, header_count):
    source = tmp_path / "cudss_solver.py"
    source.write_text("def create(self):\n self.call('cudssMatrixCreateCsr', " + ",".join(["0"] * 14) + ")\n")
    header = tmp_path / "cudss.h"
    names = [f"int p{i}" for i in range(header_count)]
    if header_count == 14:
        names[8] = "cudssDataType_t offsetType"
    header.write_text("cudssStatus_t cudssMatrixCreateCsr(" + ",".join(names) + ");")
    if header_count == 13:
        with pytest.raises(RuntimeError, match="CSR ABI"):
            startup.verify_solver_signature(source, header)
    else:
        startup.verify_solver_signature(source, header)
