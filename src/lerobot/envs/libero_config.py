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
"""Make the LIBERO simulator importable without an interactive terminal.

`libero/libero/__init__.py` (from `hf-libero` and LIBERO-plus) calls `input()` at import time when its
config file (`$LIBERO_CONFIG_PATH/config.yaml`, default `~/.libero/config.yaml`) is missing. On a fresh
machine without a TTY (HF Jobs, CI, SLURM, Docker) that raises `EOFError` before any LeRobot code runs.

This module deliberately does not import `libero`, so it can run before that import and be tested
without the Linux-only simulator installed.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def libero_config_file() -> Path:
    """Return the config file LIBERO reads, honouring `LIBERO_CONFIG_PATH` like LIBERO does."""
    config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    return Path(config_dir) / "config.yaml"


def _find_libero_benchmark_root() -> Path | None:
    """Locate the `libero/libero` package directory without importing (and so executing) it."""
    try:
        spec = importlib.util.find_spec("libero")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    # `libero` may be a namespace package (LIBERO-plus), so check every search location.
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "libero"
        if (candidate / "__init__.py").is_file():
            return candidate
    return None


def default_libero_paths(benchmark_root: Path) -> dict[str, str]:
    """The paths LIBERO itself writes when its first-run prompt is answered with "N".

    Mirrors `libero.libero.get_default_path_dict()`. `assets` is listed for completeness: `hf-libero`
    resolves assets through `get_assets_path()` (bundled or downloaded from `lerobot/libero-assets`)
    rather than this entry.
    """
    return {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }


def ensure_libero_config() -> Path | None:
    """Write LIBERO's default config file if it is missing, so importing LIBERO never prompts.

    An existing config file is never touched. The written values are the ones LIBERO picks when its
    first-run prompt is declined; the only thing the prompt offers on top is a custom `datasets`
    folder, which LIBERO uses for its own demonstration downloads and LeRobot never reads.

    The file is written atomically, so concurrent processes (e.g. async vector-env workers) importing
    LIBERO at the same time never observe a half-written config.

    Returns:
        The path of the config file that was written, or None when nothing was written (a config
        already exists, or LIBERO is not installed, in which case the import fails as usual).
    """
    config_file = libero_config_file()
    if config_file.exists():
        return None
    benchmark_root = _find_libero_benchmark_root()
    if benchmark_root is None:
        return None

    config_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=config_file.parent, prefix=".config.", suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(default_libero_paths(benchmark_root), f)
        if config_file.exists():
            # Another process won the race; keep its file.
            return None
        os.replace(tmp_name, config_file)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    logger.info("Wrote default LIBERO config to %s (LIBERO would otherwise prompt for it).", config_file)
    return config_file
