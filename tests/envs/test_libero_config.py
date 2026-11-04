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
"""`ensure_libero_config` against a stand-in `libero` package, so it runs without the Linux-only simulator."""

import builtins
import errno
import importlib
import io
import json
import logging
import os
import stat
import subprocess
import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from typing import Any, NoReturn

import pytest
import yaml

from lerobot.envs import libero_config
from lerobot.envs.libero_config import default_libero_paths, ensure_libero_config, libero_config_file

# Reproduces the import-time behaviour of hf-libero's `libero/libero/__init__.py`: prompt on stdin
# when the config file is missing.
_PROMPTING_LIBERO_INIT = textwrap.dedent(
    """
    import os
    libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_file = os.path.join(libero_config_path, "config.yaml")
    if not os.path.exists(config_file):
        input("Do you want to specify a custom path for the dataset folder? (Y/N): ")
    """
)


def _libero_modules() -> list[str]:
    return [name for name in sys.modules if name == "libero" or name.startswith("libero.")]


def _drop_libero_modules() -> None:
    for name in _libero_modules():
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _reset_stale_warning_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts as a fresh run, and the sentinel a test sets doesn't leak into later tests."""
    monkeypatch.setenv(libero_config._STALE_WARNING_ENV, "")
    monkeypatch.delenv(libero_config._STALE_WARNING_ENV)


@pytest.fixture
def fake_libero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A `libero` package first on sys.path whose `libero.libero` prompts like hf-libero, and a temp config dir.

    Runs as if without a terminal, the case `ensure_libero_config` acts on.
    """
    site = tmp_path / "site"
    pkg = site / "libero" / "libero"
    pkg.mkdir(parents=True)
    (site / "libero" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text(_PROMPTING_LIBERO_INIT)
    # A real `libero` may be installed (Linux CI with the libero extra) or already imported. Saved and
    # restored by hand rather than with monkeypatch, so no stand-in module outlives the test.
    saved_modules = {name: sys.modules.pop(name) for name in _libero_modules()}
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_home"))
    monkeypatch.setattr(libero_config, "_has_interactive_terminal", lambda: False)
    try:
        yield pkg
    finally:
        _drop_libero_modules()
        sys.modules.update(saved_modules)


def _no_stdin(*args: object, **kwargs: object) -> NoReturn:
    raise EOFError("stdin is not interactive")


def _deny_stat_under(monkeypatch: pytest.MonkeyPatch, denied: Path) -> None:
    """Make `os.stat` (used by `Path.exists`, `Path.is_dir` and `os.path.exists`) fail with EACCES below
    `denied`, as for a path under another user's home. Works even when the tests run as root."""
    real_stat = os.stat

    def stat_or_deny(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if isinstance(path, (str, os.PathLike)) and os.fspath(path).startswith(str(denied)):
            raise PermissionError(errno.EACCES, "Permission denied", os.fspath(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_or_deny)


def test_config_file_honours_libero_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    assert libero_config_file() == tmp_path / "config.yaml"


def test_writes_libero_defaults_when_missing(fake_libero: Path) -> None:
    written = ensure_libero_config()

    assert written == libero_config_file()
    config = yaml.safe_load(written.read_text())
    assert config == default_libero_paths(fake_libero)
    # The keys LeRobot (init_states) and LIBERO's benchmark (bddl_files) read, joined like LIBERO joins them.
    assert config["benchmark_root"] == str(fake_libero)
    assert config["init_states"] == os.path.join(fake_libero, "./init_files")
    assert config["bddl_files"] == os.path.join(fake_libero, "./bddl_files")
    assert config["datasets"] == os.path.join(fake_libero, "../datasets")
    assert not [p for p in written.parent.iterdir() if p != written], "temporary file left behind"


def test_written_config_follows_umask(fake_libero: Path) -> None:
    """LIBERO writes its config with a plain open(), so the file gets 0o666 filtered by the umask."""
    old_umask = os.umask(0o027)
    try:
        written = ensure_libero_config()
    finally:
        os.umask(old_umask)

    assert written is not None
    assert written.stat().st_mode & 0o777 == 0o640


def test_libero_import_no_longer_prompts(fake_libero: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The headless failure: without the config, importing LIBERO calls input() and dies with EOFError."""
    monkeypatch.setattr(builtins, "input", _no_stdin)
    with pytest.raises(EOFError):
        importlib.import_module("libero.libero")
    # The failed import leaves the stand-in top-level `libero` imported; drop it for a fresh import.
    _drop_libero_modules()

    ensure_libero_config()
    importlib.import_module("libero.libero")  # no prompt


def test_interactive_terminal_keeps_libero_prompt(fake_libero: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In a terminal LIBERO's own prompt still runs, so users can choose a custom dataset folder."""
    monkeypatch.setattr(libero_config, "_has_interactive_terminal", lambda: True)
    prompts: list[str] = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": prompts.append(prompt) or "n")

    assert ensure_libero_config() is None
    assert not libero_config_file().exists()
    importlib.import_module("libero.libero")
    assert prompts


class _Stream(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _closed_stream() -> io.StringIO:
    stream = _Stream(tty=True)
    stream.close()
    return stream


@pytest.mark.parametrize(
    ("stdin", "stdout", "expected"),
    [
        (_Stream(tty=True), _Stream(tty=True), True),
        (_Stream(tty=False), _Stream(tty=True), False),  # piped stdin, CI, SLURM, /dev/null in workers
        (_Stream(tty=True), _Stream(tty=False), False),  # output redirected: the prompt would be invisible
        (None, _Stream(tty=True), False),  # pythonw, some daemons
        (_closed_stream(), _Stream(tty=True), False),  # closed by a multiprocessing parent
    ],
    ids=["terminal", "piped-stdin", "redirected-stdout", "no-stdin", "closed-stdin"],
)
def test_has_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch, stdin: io.StringIO | None, stdout: io.StringIO, expected: bool
) -> None:
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    assert libero_config._has_interactive_terminal() is expected


def test_jupyter_kernel_counts_as_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """ipykernel's stdin is not a TTY, but input() reaches the notebook user, so LIBERO's prompt is kept."""
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    monkeypatch.setattr(sys, "stdout", _Stream(tty=False))
    assert libero_config._has_interactive_terminal() is True


def test_existing_config_is_left_untouched(fake_libero: Path) -> None:
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    config_file.write_text("datasets: /my/custom/datasets\n")

    assert ensure_libero_config() is None
    assert config_file.read_text() == "datasets: /my/custom/datasets\n"


def test_config_created_concurrently_is_not_overwritten(
    fake_libero: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config appearing between the existence check and the write (another process) wins."""
    config_file = libero_config_file()
    publish: Callable[[Path, str], bool] = libero_config._publish_if_absent

    def publish_after_another_process(path: Path, content: str) -> bool:
        path.write_text("datasets: /from/another/process\n")
        return publish(path, content)

    monkeypatch.setattr(libero_config, "_publish_if_absent", publish_after_another_process)

    assert ensure_libero_config() is None
    assert config_file.read_text() == "datasets: /from/another/process\n"
    assert [p.name for p in config_file.parent.iterdir()] == ["config.yaml"], "temporary file left behind"


@pytest.mark.parametrize("config_exists", [False, True])
def test_publish_falls_back_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_exists: bool
) -> None:
    def no_hard_links(src: Path, dst: Path) -> NoReturn:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(libero_config.os, "link", no_hard_links)
    config_file = tmp_path / "config.yaml"
    if config_exists:
        config_file.write_text("theirs\n")

    created = libero_config._publish_if_absent(config_file, "ours\n")

    assert created is not config_exists
    assert config_file.read_text() == ("theirs\n" if config_exists else "ours\n")
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"], "temporary file left behind"


def test_write_new_file_closes_fd_when_fdopen_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[int] = []
    real_open = os.open

    def recording_open(*args: Any, **kwargs: Any) -> int:
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def failing_fdopen(*args: object, **kwargs: object) -> NoReturn:
        raise OSError(errno.ENOMEM, "fdopen failed")

    target = tmp_path / "config.yaml"
    with monkeypatch.context() as m, pytest.raises(OSError, match="fdopen failed"):
        m.setattr(libero_config.os, "open", recording_open)
        m.setattr(libero_config.os, "fdopen", failing_fdopen)
        libero_config._write_new_file(target, "content\n")

    assert len(opened) == 1
    with pytest.raises(OSError) as exc_info:
        os.fstat(opened[0])
    assert exc_info.value.errno == errno.EBADF, "file descriptor leaked"
    assert not target.exists()


def test_unwritable_config_dir_warns_instead_of_failing(
    fake_libero: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A path below a regular file cannot be created, even as root.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(blocker / "libero_home"))

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert "LIBERO_CONFIG_PATH" in caplog.text
    assert str(blocker / "libero_home" / "config.yaml") in caplog.text


def test_inaccessible_config_dir_warns_instead_of_failing(
    fake_libero: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Python 3.12's Path.exists raises PermissionError (e.g. a config dir under /root read by another
    user); that must not break `import lerobot.envs.libero`."""
    config_dir = libero_config_file().parent
    config_dir.mkdir(parents=True)
    _deny_stat_under(monkeypatch, config_dir)

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert "LIBERO_CONFIG_PATH" in caplog.text


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions enforced (not root)"
)
def test_unsearchable_config_dir_warns_instead_of_failing(
    fake_libero: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    locked = tmp_path / "locked"
    (locked / "libero_home").mkdir(parents=True)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(locked / "libero_home"))
    locked.chmod(0)
    try:
        with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
            assert ensure_libero_config() is None
    finally:
        locked.chmod(stat.S_IRWXU)

    assert "LIBERO_CONFIG_PATH" in caplog.text


def test_dangling_symlink_config_warns_and_is_left_untouched(
    fake_libero: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """LIBERO sees no config and prompts, yet nothing can be created in its place: say so."""
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    missing_target = tmp_path / "elsewhere" / "config.yaml"
    config_file.symlink_to(missing_target)

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert str(config_file) in caplog.text
    assert str(missing_target) in caplog.text
    assert config_file.is_symlink()
    assert os.readlink(config_file) == str(missing_target)
    assert not missing_target.exists()
    assert [p.name for p in config_file.parent.iterdir()] == ["config.yaml"], "temporary file left behind"


def _write_stale_config(tmp_path: Path) -> tuple[Path, Path, str]:
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    stale_root = tmp_path / "uninstalled" / "libero" / "libero"
    original = yaml.safe_dump(default_libero_paths(stale_root))
    config_file.write_text(original)
    return config_file, stale_root, original


def test_stale_config_warns_and_is_left_untouched(
    fake_libero: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_file, stale_root, original = _write_stale_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert config_file.read_text() == original
    assert str(config_file) in caplog.text
    assert str(stale_root) in caplog.text
    assert str(fake_libero) in caplog.text  # where the installed LIBERO lives


def test_stale_warning_is_logged_once_per_run(
    fake_libero: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Async vector-env workers import LIBERO again; they inherit the sentinel and stay quiet."""
    config_file, _, _ = _write_stale_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        ensure_libero_config()
        ensure_libero_config()  # as in a worker process, which inherits the environment

    assert len(caplog.records) == 1
    assert os.environ[libero_config._STALE_WARNING_ENV] == str(config_file)


def test_inaccessible_benchmark_root_skips_stale_warning(
    fake_libero: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A root-written config pointing into /root/... read by another user: can't tell, so no warning."""
    _, stale_root, _ = _write_stale_config(tmp_path)
    _deny_stat_under(monkeypatch, stale_root.parent.parent)

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert caplog.text == ""


@pytest.mark.parametrize(
    "config",
    [
        "benchmark_root: {root}\n",  # still valid
        "init_states: /somewhere/init_files\n",  # hand-written, no benchmark_root (e.g. our Docker images)
        "not: [valid yaml\n",
    ],
    ids=["valid-root", "no-root", "invalid-yaml"],
)
def test_no_stale_warning_otherwise(fake_libero: Path, caplog: pytest.LogCaptureFixture, config: str) -> None:
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    config_file.write_text(config.format(root=fake_libero))

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert caplog.text == ""


def test_namespace_libero_package_is_supported(fake_libero: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LIBERO-plus installs `libero` as a namespace package (no top-level __init__.py)."""
    (fake_libero.parent / "__init__.py").unlink()
    site = fake_libero.parent.parent

    # A regular `libero` package anywhere on sys.path (hf-libero in full CI) beats namespace portions,
    # so look the stand-in up in its own directory only.
    def find_stand_in() -> ModuleSpec | None:
        return PathFinder.find_spec("libero", [str(site)])

    monkeypatch.setattr(libero_config, "_find_libero_spec", find_stand_in)

    written = ensure_libero_config()

    assert written is not None
    assert yaml.safe_load(written.read_text())["init_states"] == os.path.join(fake_libero, "./init_files")


def test_noop_when_libero_is_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_home"))
    monkeypatch.setattr(libero_config, "_has_interactive_terminal", lambda: False)
    monkeypatch.setattr(libero_config, "_find_libero_spec", lambda: None)

    assert ensure_libero_config() is None
    assert not (tmp_path / "libero_home").exists()


@pytest.mark.skipif(
    sys.platform != "linux" or libero_config._find_libero_benchmark_root() is None,
    reason="needs the real LIBERO package (hf-libero, Linux only)",
)
def test_defaults_match_installed_libero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Our defaults are exactly what the installed LIBERO writes itself when its prompt is declined."""
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(libero_config, "_has_interactive_terminal", lambda: False)
    written = ensure_libero_config()
    assert written is not None

    # A subprocess keeps the real `libero` (whose module globals capture the config path) out of this
    # process. Its stdin is /dev/null, so a prompt would fail the import with EOFError.
    script = "import json, libero.libero as L; print(json.dumps(L.get_default_path_dict()))"
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    libero_defaults = json.loads(result.stdout.strip().splitlines()[-1])

    assert yaml.safe_load(written.read_text()) == libero_defaults
