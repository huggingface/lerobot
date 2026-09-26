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
import ast
import importlib
import sys
from pathlib import Path

import pytest

from lerobot.utils.import_utils import _datasets_available

NEEDS_DATASETS = [
    pytest.mark.dataset,
    pytest.mark.skipif(not _datasets_available, reason="needs the dataset extra"),
]

LAZY_MODULES = [
    pytest.param("lerobot.annotations.steerable_pipeline", marks=NEEDS_DATASETS),
    "lerobot.configs",
    "lerobot.envs",
    "lerobot.lerobot_types",
    "lerobot.policies",
    "lerobot.policies.rtc",
    pytest.param("lerobot.rollout", marks=NEEDS_DATASETS),
    "lerobot.inference",
]


@pytest.fixture
def make_module(tmp_path, monkeypatch):
    """Write `lazy_pkg/mod.py` with the given lazy exports, next to modules it can export from."""
    pkg = tmp_path / "lazy_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "heavy.py").write_text("def helper():\n    return 'helper'\n")
    (pkg / "broken.py").write_text("import json\n\nvalue = json.no_such_function\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(exports: dict[str, str]) -> Path:
        (pkg / "mod.py").write_text(
            "from lerobot.utils.import_utils import lazy_exports\n\n"
            f"__getattr__, __dir__ = lazy_exports(__name__, {exports!r})\n"
        )
        return pkg

    yield write
    for name in [m for m in sys.modules if m == "lazy_pkg" or m.startswith("lazy_pkg.")]:
        del sys.modules[name]


def test_a_failing_import_keeps_its_cause(make_module):
    make_module({"value": ".broken.value"})
    with pytest.raises(ImportError, match="no_such_function") as error:
        from lazy_pkg.mod import value  # noqa: F401
    assert isinstance(error.value.__cause__, AttributeError)


def test_a_package_returns_submodules_not_in_its_table(make_module):
    pkg = make_module({})
    (pkg / "__init__.py").write_text(
        "from lerobot.utils.import_utils import lazy_exports\n\n"
        "__getattr__, __dir__ = lazy_exports(__name__, {'helper': '.heavy.helper'})\n"
    )
    lazy_pkg = importlib.import_module("lazy_pkg")
    mod = importlib.import_module("lazy_pkg.mod")
    assert "lazy_pkg.heavy" not in sys.modules
    assert "heavy" not in dir(lazy_pkg)

    assert lazy_pkg.heavy is sys.modules["lazy_pkg.heavy"]
    for missing in ["missing", "__missing__"]:
        with pytest.raises(AttributeError):
            getattr(lazy_pkg, missing)
    # Only a package has submodules.
    with pytest.raises(AttributeError):
        _ = mod.heavy
    # A submodule that needs a missing extra reads as a missing name, with the import error as its cause.
    (pkg / "extra.py").write_text("import missing_extra_package\n")
    importlib.invalidate_caches()
    assert getattr(lazy_pkg, "extra", None) is None
    with pytest.raises(AttributeError) as error:
        _ = lazy_pkg.extra
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    with pytest.raises(ModuleNotFoundError, match="missing_extra_package"):
        from lazy_pkg import extra  # noqa: F401


def test_resolved_names_do_not_become_module_globals(make_module):
    # So module code behaves the same whatever was used first.
    make_module({"helper": ".heavy.helper"})
    mod = importlib.import_module("lazy_pkg.mod")
    assert mod.helper() == "helper"
    assert "helper" not in vars(mod)


def _resolve_type_checking_block(tree: ast.Module, package: str) -> tuple[dict, dict]:
    """The modules and exported objects the `if TYPE_CHECKING:` block names, resolved without running it."""
    block = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "TYPE_CHECKING")
    modules, exports = {}, {}
    for stmt in block.body:
        if isinstance(stmt, ast.Import):
            modules |= {
                alias.asname or alias.name: importlib.import_module(alias.name) for alias in stmt.names
            }
        elif isinstance(stmt, ast.ImportFrom):
            source = importlib.import_module("." * stmt.level + (stmt.module or ""), package)
            exports |= {alias.asname or alias.name: getattr(source, alias.name) for alias in stmt.names}
        else:  # `Name: TypeAlias = module.attribute`
            owner, attribute = ast.unparse(stmt.value).rsplit(".", 1)
            exports[stmt.target.id] = getattr(modules[owner], attribute)
    return modules, exports


def _names_read_at_runtime(tree: ast.Module) -> set[str]:
    """Names read at run time, outside `if TYPE_CHECKING:` blocks, annotations and a function's own imports."""
    skipped = [
        stmt
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
        for stmt in node.body
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign | ast.arg):
            skipped.append(node.annotation)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            skipped.append(node.returns)
    ignored = {id(n) for root in skipped if root is not None for n in ast.walk(root)}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            nested = {
                id(n)
                for inner in ast.walk(function)
                if inner is not function
                and isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                for n in ast.walk(inner)
            }
            # An import binds the name in its own function from that point on, and nested functions see it,
            # unless it only runs for type checking.
            imports = [
                ((alias.asname or alias.name).split(".")[0], (node.lineno, node.col_offset))
                for node in ast.walk(function)
                if isinstance(node, ast.Import | ast.ImportFrom)
                and id(node) not in nested
                and id(node) not in ignored
                for alias in node.names
            ]
            ignored |= {
                id(n)
                for n in ast.walk(function)
                if isinstance(n, ast.Name)
                and any(name == n.id and at < (n.lineno, n.col_offset) for name, at in imports)
            }
    return {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in ignored
    }


@pytest.mark.parametrize("module_name", LAZY_MODULES)
def test_lazy_exports_match_the_type_checking_imports(module_name):
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    modules, exports = _resolve_type_checking_block(tree, module.__package__)
    assert exports
    assert set(dir(module)) == set(vars(module)) | set(exports)
    for name, obj in exports.items():
        assert getattr(module, name) is obj, name
    # `from module import *` reads only __all__: a lazy name missing there is dropped, a stale one fails.
    assert set(exports) <= set(module.__all__)
    assert [name for name in module.__all__ if not hasattr(module, name)] == []
    assert not {*modules, *exports} & _names_read_at_runtime(tree)


def test_names_imported_for_type_checking_are_not_read_at_runtime():
    # lerobot.inference.factory imports the inference engines, which need torch, only where it creates them.
    source = Path(importlib.import_module("lerobot").__path__[0]) / "inference" / "factory.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    block = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "TYPE_CHECKING")
    names = {(alias.asname or alias.name).split(".")[0] for stmt in block.body for alias in stmt.names}
    assert not names & _names_read_at_runtime(tree)
