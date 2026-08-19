import importlib.machinery
import sys
import types
import zipfile
from pathlib import Path

import pytest
import yaml

from lerobot.envs import libero_safety_assets as target


def _install_fake_libero(monkeypatch, tmp_path, assets_path=None):
    """Register a fake `libero`/`libero.libero` in sys.modules with a real __spec__ (so
    importlib.util.find_spec works) and a get_libero_path("assets") stub, without needing the
    real LIBERO-Safety fork installed."""
    pkg_root = tmp_path / "fake_libero_safety" / "libero" / "libero"
    pkg_root.mkdir(parents=True, exist_ok=True)

    fake_top = types.ModuleType("libero")
    fake_top.__path__ = [str(pkg_root.parent)]
    fake_top.__spec__ = importlib.machinery.ModuleSpec("libero", loader=None, is_package=True)
    fake_top.__spec__.submodule_search_locations = fake_top.__path__

    fake_libero = types.ModuleType("libero.libero")
    fake_libero.__path__ = [str(pkg_root)]
    fake_libero.__spec__ = importlib.machinery.ModuleSpec("libero.libero", loader=None, is_package=True)
    fake_libero.__spec__.submodule_search_locations = fake_libero.__path__

    assets_holder = {"path": str(assets_path or (pkg_root / "assets"))}
    fake_libero.get_libero_path = lambda key: assets_holder["path"]

    monkeypatch.setitem(sys.modules, "libero", fake_top)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero)
    return pkg_root, assets_holder


@pytest.fixture(autouse=True)
def _isolated_dirs(monkeypatch, tmp_path):
    """Every test gets its own HF_LEROBOT_HOME / ~/.libero so nothing touches the real cache."""
    monkeypatch.setattr(target, "HF_LEROBOT_HOME", tmp_path / "hf_lerobot_home")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "dotlibero"))
    return tmp_path


def test_ensure_config_writes_default_when_missing(monkeypatch, tmp_path):
    pkg_root, _ = _install_fake_libero(monkeypatch, tmp_path)

    target.ensure_libero_safety_config()

    config_file = tmp_path / "dotlibero" / "config.yaml"
    assert config_file.exists()
    cfg = yaml.safe_load(config_file.read_text())
    assert cfg["benchmark_root"] == str(pkg_root)
    assert cfg["bddl_files"] == str(pkg_root / "bddl_files")
    assert cfg["init_states"] == str(pkg_root / "init_files")
    assert cfg["assets"] == str(pkg_root / "assets")


def test_ensure_config_is_noop_when_already_present(monkeypatch, tmp_path):
    _install_fake_libero(monkeypatch, tmp_path)
    config_file = tmp_path / "dotlibero" / "config.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("sentinel: true\n")

    target.ensure_libero_safety_config()

    assert yaml.safe_load(config_file.read_text()) == {"sentinel": True}


def test_ensure_config_noop_when_libero_not_installed(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "libero", None)  # simulate "not installed"
    monkeypatch.delitem(sys.modules, "libero.libero", raising=False)

    target.ensure_libero_safety_config()

    assert not (tmp_path / "dotlibero" / "config.yaml").exists()


def _make_fake_assets_zip(tmp_path: Path) -> Path:
    """A zip whose real `assets/` dir is nested, matching the upstream archive layout."""
    src_dir = tmp_path / "zip_src"
    nested = src_dir / "release_v1" / "assets"
    nested.mkdir(parents=True)
    (nested / "human_hand.obj").write_text("fake mesh data")
    zip_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(nested / "human_hand.obj", "release_v1/assets/human_hand.obj")
    return zip_path


def test_ensure_assets_downloads_extracts_and_links_on_first_call(monkeypatch, tmp_path):
    _, assets_holder = _install_fake_libero(monkeypatch, tmp_path)
    zip_path = _make_fake_assets_zip(tmp_path)
    fake_snapshot_dir = zip_path.parent

    calls = []
    monkeypatch.setattr(
        target, "snapshot_download", lambda repo_id, repo_type: calls.append(1) or str(fake_snapshot_dir)
    )

    result = target.ensure_libero_safety_assets("org/repo")

    assert len(calls) == 1
    assert (result / "human_hand.obj").read_text() == "fake mesh data"
    assert (result / target._ASSET_READY_MARKER).exists()
    linked = Path(assets_holder["path"])
    assert linked.is_symlink()
    assert linked.resolve() == result.resolve()


def test_ensure_assets_second_call_skips_network(monkeypatch, tmp_path):
    _install_fake_libero(monkeypatch, tmp_path)
    zip_path = _make_fake_assets_zip(tmp_path)
    fake_snapshot_dir = zip_path.parent

    calls = []
    monkeypatch.setattr(
        target, "snapshot_download", lambda repo_id, repo_type: calls.append(1) or str(fake_snapshot_dir)
    )

    first = target.ensure_libero_safety_assets("org/repo")
    second = target.ensure_libero_safety_assets("org/repo")

    assert len(calls) == 1, "second call must be a pure filesystem check, no network call"
    assert first == second


def test_ensure_assets_raises_clear_error_when_no_zip_in_repo(monkeypatch, tmp_path):
    _install_fake_libero(monkeypatch, tmp_path)
    empty_snapshot = tmp_path / "empty_snapshot"
    empty_snapshot.mkdir()
    monkeypatch.setattr(target, "snapshot_download", lambda repo_id, repo_type: str(empty_snapshot))

    with pytest.raises(FileNotFoundError, match="No 'assets.zip' found"):
        target.ensure_libero_safety_assets("org/repo")


def test_link_assets_respects_existing_populated_non_symlink_dir(monkeypatch, tmp_path):
    baked = tmp_path / "baked_assets"
    baked.mkdir()
    (baked / "already_here.obj").write_text("baked")
    _install_fake_libero(monkeypatch, tmp_path, assets_path=baked)

    some_other_dir = tmp_path / "downloaded_elsewhere"
    some_other_dir.mkdir()

    target._link_assets_into_libero_safety(some_other_dir)

    assert (baked / "already_here.obj").exists()
    assert not baked.is_symlink()


def test_extract_assets_zip_rejects_path_traversal(tmp_path):
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../etc/evil", "pwned")

    with pytest.raises(ValueError, match="unsafe path"):
        target._extract_assets_zip(evil_zip, tmp_path / "target")
