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
config file (`$LIBERO_CONFIG_PATH/config.yaml`, default `~/.libero/config.yaml`) is missing, asking
whether to use a custom dataset folder. On a fresh machine without a terminal (HF Jobs, CI, SLURM,
Docker) that raises `EOFError` before any LeRobot code runs.

This module deliberately does not import `libero`, so it can run before that import and be tested
without the Linux-only simulator installed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import sys
import uuid
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TextIO

import yaml

logger = logging.getLogger(__name__)


def libero_config_file() -> Path:
    """Return the config file LIBERO reads, honouring `LIBERO_CONFIG_PATH` like LIBERO does."""
    config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    return Path(config_dir) / "config.yaml"


def _find_libero_spec() -> ModuleSpec | None:
    """Spec of the top-level `libero` package, found without importing it."""
    try:
        return importlib.util.find_spec("libero")
    except (ImportError, ValueError):
        return None


def _find_libero_benchmark_root() -> Path | None:
    """Locate the `libero/libero` package directory without importing (and so executing) it."""
    spec = _find_libero_spec()
    if spec is None or not spec.submodule_search_locations:
        return None
    # `libero` may be a namespace package (LIBERO-plus), so check every search location.
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "libero"
        if (candidate / "__init__.py").is_file():
            return candidate
    return None


def _has_interactive_terminal() -> bool:
    """Whether a user can see and answer LIBERO's `input()` prompt.

    Both stdin and stdout must be terminals: `input()` writes its prompt to stdout, so with stdout
    redirected to a log file the prompt is invisible and the process would wait on it silently.
    `sys.stdin` is None under pythonw and in some daemons, and multiprocessing workers (e.g. async
    vector-env workers) get a closed or `/dev/null` stdin, so all of these count as non-interactive.
    """
    streams: tuple[TextIO | None, ...] = (sys.stdin, sys.stdout)
    for stream in streams:
        if stream is None or getattr(stream, "closed", False):
            return False
        try:
            if not stream.isatty():
                return False
        except (AttributeError, OSError, ValueError):
            return False
    return True


def default_libero_paths(benchmark_root: Path) -> dict[str, str]:
    """The paths LIBERO itself writes when its first-run prompt is answered with "N".

    Mirrors `libero.libero.get_default_path_dict()`, with the paths normalized (LIBERO writes e.g.
    `<root>/./bddl_files` and `<root>/../datasets`). `assets` is listed for completeness: `hf-libero`
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


def _write_new_file(path: Path, content: str) -> None:
    """Create `path` and write `content`, raising `FileExistsError` if it already exists.

    Mode 0o666 is filtered by the umask, giving the permissions LIBERO's own `open(config_file, "w")`
    gives the file (0644 under the usual 022 umask), not `tempfile`'s owner-only 0600.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _publish_if_absent(path: Path, content: str) -> bool:
    """Atomically create `path` with `content` unless it already exists.

    The content is written to a temporary file first and hard-linked into place: `os.link` fails if the
    destination exists, so a config created concurrently (by another process, or by a user answering
    LIBERO's prompt) is never overwritten, and readers never see a partially written file. On
    filesystems without hard links this falls back to an exclusive create, which still never
    overwrites but is not atomic for concurrent readers.

    Returns:
        Whether this call created `path`.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        _write_new_file(tmp, content)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        except OSError:
            # No hard links on this filesystem (e.g. some FUSE, FAT or SMB mounts).
            try:
                _write_new_file(path, content)
            except FileExistsError:
                return False
        return True
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _warn_if_stale(config_file: Path) -> None:
    """Warn when an existing config points at a LIBERO install that no longer exists.

    This happens after switching LIBERO packages (e.g. from `hf-libero` to LIBERO-plus and uninstalling
    the first): LIBERO keeps reading the old paths. The file is left as is.
    """
    try:
        config = yaml.safe_load(config_file.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return  # LIBERO reports an unreadable config itself when it reads it.
    if not isinstance(config, dict):
        return
    configured_root = config.get("benchmark_root")
    if not isinstance(configured_root, str) or Path(configured_root).exists():
        return
    current_root = _find_libero_benchmark_root()
    logger.warning(
        "LIBERO config %s sets benchmark_root to %s, which does not exist; it was probably written by "
        "another LIBERO install. The installed LIBERO package would use %s. LeRobot does not modify "
        "existing configs: update the file, or delete it to have the defaults written again.",
        config_file,
        configured_root,
        current_root if current_root is not None else "<no LIBERO package found>",
    )


def ensure_libero_config() -> Path | None:
    """Write LIBERO's default config file if it is missing, so importing LIBERO never prompts headless.

    Only acts without an interactive terminal: in a terminal LIBERO's own first-run prompt is kept, so
    users can still pick a custom `datasets` folder (the only thing the prompt offers on top of the
    defaults written here; LIBERO uses it for its own demonstration downloads and LeRobot never reads
    it). An existing config file is never modified; if its `benchmark_root` no longer exists a warning
    is logged.

    The file is created atomically and only if absent, so concurrent processes (e.g. async vector-env
    workers) importing LIBERO at the same time never observe a half-written config or overwrite one.
    If the config directory is not writable, a warning is logged and LIBERO's own behaviour applies.

    Returns:
        The path of the config file that was written, or None when nothing was written (a config
        already exists, a terminal is attached, the config directory is not writable, or LIBERO is not
        installed, in which case the import fails as usual).
    """
    config_file = libero_config_file()
    if config_file.exists():
        _warn_if_stale(config_file)
        return None
    if _has_interactive_terminal():
        return None
    benchmark_root = _find_libero_benchmark_root()
    if benchmark_root is None:
        return None

    content = yaml.safe_dump(default_libero_paths(benchmark_root))
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        created = _publish_if_absent(config_file, content)
    except OSError as e:
        logger.warning(
            "Could not write LIBERO's default config to %s (%s), so importing LIBERO will likely fail. "
            "Set LIBERO_CONFIG_PATH to a writable directory.",
            config_file,
            e,
        )
        return None
    if not created:
        return None  # Another process created it first; keep its file.
    logger.info("Wrote default LIBERO config to %s (LIBERO would otherwise prompt for it).", config_file)
    return config_file
