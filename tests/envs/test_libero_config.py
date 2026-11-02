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
import subprocess
import sys
import textwrap
from importlib.machinery import PathFinder

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


@pytest.fixture
def fake_libero(tmp_path, monkeypatch):
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


def _no_stdin(*args, **kwargs):
    raise EOFError("stdin is not interactive")


def test_config_file_honours_libero_config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    assert libero_config_file() == tmp_path / "config.yaml"


def test_writes_libero_defaults_when_missing(fake_libero):
    written = ensure_libero_config()

    assert written == libero_config_file()
    config = yaml.safe_load(written.read_text())
    assert config == default_libero_paths(fake_libero)
    # The keys LeRobot (init_states) and LIBERO's benchmark (bddl_files) read.
    assert config["init_states"] == str(fake_libero / "init_files")
    assert config["bddl_files"] == str(fake_libero / "bddl_files")
    assert config["datasets"] == str(fake_libero.parent / "datasets")
    assert not [p for p in written.parent.iterdir() if p != written], "temporary file left behind"


def test_written_config_follows_umask(fake_libero):
    """LIBERO writes its config with a plain open(), so the file gets 0o666 filtered by the umask."""
    old_umask = os.umask(0o027)
    try:
        written = ensure_libero_config()
    finally:
        os.umask(old_umask)

    assert written is not None
    assert written.stat().st_mode & 0o777 == 0o640


def test_libero_import_no_longer_prompts(fake_libero, monkeypatch):
    """The headless failure: without the config, importing LIBERO calls input() and dies with EOFError."""
    monkeypatch.setattr(builtins, "input", _no_stdin)
    with pytest.raises(EOFError):
        importlib.import_module("libero.libero")
    # The failed import leaves the stand-in top-level `libero` imported; drop it for a fresh import.
    _drop_libero_modules()

    ensure_libero_config()
    importlib.import_module("libero.libero")  # no prompt


def test_interactive_terminal_keeps_libero_prompt(fake_libero, monkeypatch):
    """In a terminal LIBERO's own prompt still runs, so users can choose a custom dataset folder."""
    monkeypatch.setattr(libero_config, "_has_interactive_terminal", lambda: True)
    prompts = []
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
def test_has_interactive_terminal(monkeypatch, stdin, stdout, expected):
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    assert libero_config._has_interactive_terminal() is expected


def test_existing_config_is_left_untouched(fake_libero):
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    config_file.write_text("datasets: /my/custom/datasets\n")

    assert ensure_libero_config() is None
    assert config_file.read_text() == "datasets: /my/custom/datasets\n"


def test_config_created_concurrently_is_not_overwritten(fake_libero, monkeypatch):
    """A config appearing between the existence check and the write (another process) wins."""
    config_file = libero_config_file()
    publish = libero_config._publish_if_absent

    def publish_after_another_process(path, content):
        path.write_text("datasets: /from/another/process\n")
        return publish(path, content)

    monkeypatch.setattr(libero_config, "_publish_if_absent", publish_after_another_process)

    assert ensure_libero_config() is None
    assert config_file.read_text() == "datasets: /from/another/process\n"
    assert [p.name for p in config_file.parent.iterdir()] == ["config.yaml"], "temporary file left behind"


@pytest.mark.parametrize("config_exists", [False, True])
def test_publish_falls_back_without_hard_links(tmp_path, monkeypatch, config_exists):
    def no_hard_links(src, dst):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(libero_config.os, "link", no_hard_links)
    config_file = tmp_path / "config.yaml"
    if config_exists:
        config_file.write_text("theirs\n")

    created = libero_config._publish_if_absent(config_file, "ours\n")

    assert created is not config_exists
    assert config_file.read_text() == ("theirs\n" if config_exists else "ours\n")
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"], "temporary file left behind"


def test_unwritable_config_dir_warns_instead_of_failing(fake_libero, tmp_path, monkeypatch, caplog):
    # A path below a regular file cannot be created, even as root.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(blocker / "libero_home"))

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert "LIBERO_CONFIG_PATH" in caplog.text
    assert str(blocker / "libero_home" / "config.yaml") in caplog.text


def test_stale_config_warns_and_is_left_untouched(fake_libero, tmp_path, caplog):
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    stale_root = tmp_path / "uninstalled" / "libero" / "libero"
    original = yaml.safe_dump(default_libero_paths(stale_root))
    config_file.write_text(original)

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert config_file.read_text() == original
    assert str(config_file) in caplog.text
    assert str(stale_root) in caplog.text
    assert str(fake_libero) in caplog.text  # where the installed LIBERO lives


@pytest.mark.parametrize(
    "config",
    [
        "benchmark_root: {root}\n",  # still valid
        "init_states: /somewhere/init_files\n",  # hand-written, no benchmark_root (e.g. our Docker images)
        "not: [valid yaml\n",
    ],
    ids=["valid-root", "no-root", "invalid-yaml"],
)
def test_no_stale_warning_otherwise(fake_libero, caplog, config):
    config_file = libero_config_file()
    config_file.parent.mkdir(parents=True)
    config_file.write_text(config.format(root=fake_libero))

    with caplog.at_level(logging.WARNING, logger=libero_config.__name__):
        assert ensure_libero_config() is None

    assert caplog.text == ""


def test_namespace_libero_package_is_supported(fake_libero, monkeypatch):
    """LIBERO-plus installs `libero` as a namespace package (no top-level __init__.py)."""
    (fake_libero.parent / "__init__.py").unlink()
    site = fake_libero.parent.parent
    # A regular `libero` package anywhere on sys.path (hf-libero in full CI) beats namespace portions,
    # so look the stand-in up in its own directory only.
    monkeypatch.setattr(
        libero_config, "_find_libero_spec", lambda: PathFinder.find_spec("libero", [str(site)])
    )

    written = ensure_libero_config()

    assert written is not None
    assert yaml.safe_load(written.read_text())["init_states"] == str(fake_libero / "init_files")


def test_noop_when_libero_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_home"))
    monkeypatch.setattr(libero_config, "_has_interactive_terminal", lambda: False)
    monkeypatch.setattr(libero_config, "_find_libero_spec", lambda: None)

    assert ensure_libero_config() is None
    assert not (tmp_path / "libero_home").exists()


@pytest.mark.skipif(
    sys.platform != "linux" or libero_config._find_libero_benchmark_root() is None,
    reason="needs the real LIBERO package (hf-libero, Linux only)",
)
def test_defaults_match_installed_libero(tmp_path, monkeypatch):
    """Our defaults equal what the installed LIBERO writes itself when its prompt is declined."""
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

    # LIBERO joins e.g. "<root>/./bddl_files" and "<root>/../datasets"; compare normalized paths.
    expected = {key: os.path.normpath(value) for key, value in libero_defaults.items()}
    assert yaml.safe_load(written.read_text()) == expected
