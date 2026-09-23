import importlib
import sys
import textwrap
from pathlib import Path

import pytest

import lerobot
from lerobot.utils.import_utils import _datasets_available, _lazy_imports

NEEDS_DATASETS = pytest.mark.skipif(not _datasets_available, reason="lerobot.rollout needs the dataset extra")


def _modules_using_lazy_getattr() -> list:
    src = Path(lerobot.__file__).parent
    names = []
    for path in sorted(src.rglob("*.py")):
        if "\n    __getattr__ = lazy_getattr(__name__)\n" in path.read_text(encoding="utf-8"):
            name = ".".join(("lerobot", *path.relative_to(src).with_suffix("").parts)).removesuffix(
                ".__init__"
            )
            names.append(
                pytest.param(name, marks=NEEDS_DATASETS) if name.startswith("lerobot.rollout") else name
            )
    return names


LAZY_MODULES = _modules_using_lazy_getattr()


@pytest.fixture
def make_module(tmp_path, monkeypatch):
    """Write `lazy_pkg/mod.py` with the given lazy block, next to a `heavy` module it imports from."""
    pkg = tmp_path / "lazy_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "heavy.py").write_text("class Thing: ...\n\n\ndef helper():\n    return 'helper'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(block: str) -> None:
        header = "from lerobot.utils.import_utils import LAZY_IMPORTS, lazy_getattr\n\n"
        (pkg / "mod.py").write_text(header + textwrap.dedent(block))

    yield write
    for name in [m for m in sys.modules if m == "lazy_pkg" or m.startswith("lazy_pkg.")]:
        del sys.modules[name]


def test_names_load_on_first_use(make_module):
    make_module("""
        if LAZY_IMPORTS:
            from lazy_pkg.heavy import Thing, helper
            Alias = Thing
        else:
            __getattr__ = lazy_getattr(__name__)
    """)
    mod = importlib.import_module("lazy_pkg.mod")
    assert "lazy_pkg.heavy" not in sys.modules

    from lazy_pkg.mod import helper

    assert helper() == "helper"
    assert mod.Alias is sys.modules["lazy_pkg.heavy"].Thing
    with pytest.raises(AttributeError):
        _ = mod.missing


def test_dunder_lookups_do_not_read_the_source(make_module, monkeypatch):
    make_module("""
        if LAZY_IMPORTS:
            from lazy_pkg.heavy import helper
        else:
            __getattr__ = lazy_getattr(__name__)
    """)
    mod = importlib.import_module("lazy_pkg.mod")

    def no_source(obj):
        raise OSError("no source")

    monkeypatch.setattr("inspect.getsource", no_source)
    assert not hasattr(mod, "__path__")
    with pytest.raises(AttributeError, match="need its source file"):
        _ = mod.helper


def test_unsupported_statements_are_rejected(make_module):
    make_module("""
        if LAZY_IMPORTS:
            try:
                from lazy_pkg.heavy import helper
            except ImportError:
                pass
        else:
            __getattr__ = lazy_getattr(__name__)
    """)
    mod = importlib.import_module("lazy_pkg.mod")
    with pytest.raises(TypeError, match="can only hold imports"):
        _ = mod.helper


def test_reload_sees_the_new_block(make_module):
    block = """
        if LAZY_IMPORTS:
            from lazy_pkg.heavy import {name}
        else:
            __getattr__ = lazy_getattr(__name__)
    """
    make_module(block.format(name="helper"))
    mod = importlib.import_module("lazy_pkg.mod")
    assert mod.helper() == "helper"

    make_module(block.format(name="Thing"))
    mod = importlib.reload(mod)
    assert mod.Thing is sys.modules["lazy_pkg.heavy"].Thing


@pytest.mark.parametrize("module_name", LAZY_MODULES)
def test_every_lazy_name_in_lerobot_resolves(module_name):
    module = importlib.import_module(module_name)
    names = _lazy_imports(module_name)
    assert names
    for name in names:
        assert getattr(module, name) is not None
