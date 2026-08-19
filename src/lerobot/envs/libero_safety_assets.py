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
"""Auto-provisioning for LIBERO-Safety: a non-interactive ``~/.libero/config.yaml`` and the
object/scene assets (e.g. the human-hand model used by ``human_safety``) that ship separately
from the fork's git repo. Lets ``--env.type=libero_safety`` work with no manual
config/PYTHONPATH/asset setup beyond having the LIBERO-Safety fork itself installed (see
``docker/Dockerfile.benchmark.libero_safety`` / ``docs/source/libero_safety.mdx``).

Deliberately does not import ``libero``/``libero.libero`` at module scope: importing
``libero.libero`` for the first time runs its interactive ``input()`` config prompt if
``~/.libero/config.yaml`` doesn't exist yet (see upstream ``libero/libero/__init__.py``), and
``ensure_libero_safety_config`` must be able to write that file *before* anything imports it.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import zipfile
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from lerobot.utils.constants import HF_LEROBOT_HOME

LIBERO_SAFETY_ASSETS_REPO_ID = "LIBERO-Safety/libero_safety_assets"

_ASSET_READY_MARKER = ".lerobot_download_complete"


def _libero_config_path() -> Path:
    root = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    return root / "config.yaml"


def _find_installed_libero_package_root() -> Path | None:
    """Locate the installed ``libero.libero`` package directory without importing it.

    ``find_spec`` only executes a parent package's ``__init__.py`` if needed to resolve its
    ``__path__``; the top-level ``libero`` package (both stock hf-libero and the LIBERO-Safety
    fork) has none — it's an implicit namespace package — so this is safe to call before a
    config file exists.
    """
    try:
        spec = importlib.util.find_spec("libero.libero")
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def ensure_libero_safety_config(assets_dir: Path | None = None) -> None:
    """Write a default, non-interactive ``~/.libero/config.yaml`` if one doesn't exist yet.

    Mirrors upstream ``libero.libero.get_default_path_dict()`` so the resulting file is
    indistinguishable from one a user created by hand, except it never blocks on ``input()``.
    A no-op if the config already exists (respects any existing manual setup) or if `libero`
    isn't installed at all (the later `from libero.libero import ...` will raise its own clear
    `ModuleNotFoundError` instead).
    """
    config_file = _libero_config_path()
    if config_file.exists():
        return

    root = _find_installed_libero_package_root()
    if root is None:
        return

    config = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root / ".." / "datasets"),
        "assets": str(assets_dir) if assets_dir is not None else str(root / "assets"),
    }
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(yaml.safe_dump(config))
    print(f"LIBERO-Safety: wrote default config to {config_file}")


def _assets_cache_dir(repo_id: str) -> Path:
    """Lerobot-managed extraction target, separate from the Hub's own snapshot cache so
    "are the assets ready" is a plain filesystem check instead of a network round-trip."""
    return HF_LEROBOT_HOME / repo_id / "assets"


def _extract_assets_zip(zip_path: Path, target: Path) -> None:
    extract_tmp = target.parent / "_extract_tmp"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                member_path = (extract_tmp / member).resolve()
                if not member_path.is_relative_to(extract_tmp.resolve()):
                    raise ValueError(
                        f"'assets.zip' contains an unsafe path outside the extraction dir: {member}"
                    )
            zf.extractall(extract_tmp)  # nosec B202 — member paths validated above

        # The zip nests the real `assets/` folder a level or two down rather than being flat
        # (verified against the real archive by docker/Dockerfile.benchmark.libero_safety) —
        # find it by name instead of assuming a fixed depth.
        candidates = sorted(
            (p for p in extract_tmp.rglob("assets") if p.is_dir()), key=lambda p: len(p.parts)
        )
        if not candidates:
            raise FileNotFoundError(
                f"'assets.zip' didn't contain an 'assets/' directory (extracted to {extract_tmp} for inspection)."
            )
        found = candidates[0]

        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(found), str(target))
    finally:
        shutil.rmtree(extract_tmp, ignore_errors=True)


def ensure_libero_safety_assets(repo_id: str = LIBERO_SAFETY_ASSETS_REPO_ID) -> Path:
    """Ensure LIBERO-Safety's object/scene assets are present locally and linked into the
    installed package's expected ``assets`` path, downloading from the Hub only once.

    Second and later calls (same `repo_id`) are a pure filesystem check — no network call —
    as long as the extracted directory and its `.lerobot_download_complete` marker survive.
    """
    target = _assets_cache_dir(repo_id)
    marker = target / _ASSET_READY_MARKER

    if marker.exists() and target.is_dir() and any(target.iterdir()):
        print(f"LIBERO-Safety assets already present at {target} — skipping download.")
    else:
        print(
            f"LIBERO-Safety assets not found locally; downloading '{repo_id}' from the Hub (first run only)..."
        )
        snapshot_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
        zip_candidates = list(snapshot_dir.rglob("assets.zip"))
        if not zip_candidates:
            raise FileNotFoundError(
                f"No 'assets.zip' found in Hub dataset '{repo_id}' (looked under {snapshot_dir}). "
                "Pass a different --env.assets_repo_id if you're using a fork/mirror with a different layout."
            )
        _extract_assets_zip(zip_candidates[0], target)
        marker.write_text("ok\n")
        print(f"LIBERO-Safety assets ready at {target}.")

    _link_assets_into_libero_safety(target)
    return target


def _link_assets_into_libero_safety(assets_dir: Path) -> None:
    """Point the installed LIBERO-Safety package's `assets` path at `assets_dir` so no manual
    asset setup is required beyond having the fork itself installed."""
    from libero.libero import get_libero_path  # safe: config is already written by this point

    libero_assets_path = Path(get_libero_path("assets"))

    if libero_assets_path.is_symlink():
        if libero_assets_path.resolve() == assets_dir.resolve():
            return
        libero_assets_path.unlink()
    elif libero_assets_path.exists():
        if any(libero_assets_path.iterdir()):
            # Respect an existing manual setup (e.g. a Docker image that already bakes real
            # assets in place) rather than silently overwriting it.
            print(f"LIBERO assets path {libero_assets_path} already populated — leaving it as-is.")
            return
        libero_assets_path.rmdir()

    libero_assets_path.parent.mkdir(parents=True, exist_ok=True)
    libero_assets_path.symlink_to(assets_dir, target_is_directory=True)
    print(f"LIBERO-Safety: linked assets path {libero_assets_path} -> {assets_dir}")
