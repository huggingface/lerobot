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
import compileall
import importlib
import sys
from pathlib import Path

import pytest

from lerobot.utils.import_utils import _datasets_available

NEEDS_DATASETS = pytest.mark.skipif(not _datasets_available, reason="lerobot.rollout needs the dataset extra")

LAZY_MODULES = [
    "lerobot.configs",
    "lerobot.envs",
    "lerobot.lerobot_types",
    "lerobot.policies",
    "lerobot.policies.rtc",
    "lerobot.utils",
    pytest.param("lerobot.rollout", marks=NEEDS_DATASETS),
    pytest.param("lerobot.rollout.inference", marks=NEEDS_DATASETS),
]


@pytest.fixture
def make_module(tmp_path, monkeypatch):
    """Write `lazy_pkg/mod.py` with the given lazy exports, next to modules it can export from."""
    pkg = tmp_path / "lazy_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "heavy.py").write_text("class Thing: ...\n\n\ndef helper():\n    return 'helper'\n")
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


def test_names_load_on_first_use(make_module):
    make_module({"helper": ".heavy.helper", "Alias": "lazy_pkg.heavy.Thing"})
    mod = importlib.import_module("lazy_pkg.mod")
    assert "lazy_pkg.heavy" not in sys.modules
    assert {"helper", "Alias"} <= set(dir(mod))

    from lazy_pkg.mod import helper

    assert helper() == "helper"
    assert mod.Alias is sys.modules["lazy_pkg.heavy"].Thing
    assert "helper" not in vars(mod)
    with pytest.raises(AttributeError):
        _ = mod.missing


def test_a_failing_import_keeps_its_cause(make_module):
    make_module({"value": ".broken.value"})
    with pytest.raises(ImportError, match="no_such_function") as error:
        from lazy_pkg.mod import value  # noqa: F401
    assert isinstance(error.value.__cause__, AttributeError)


def test_works_without_source_files(make_module):
    pkg = make_module({"helper": ".heavy.helper"})
    assert compileall.compile_dir(pkg, legacy=True, quiet=1)
    for source in pkg.glob("*.py"):
        source.unlink()
    importlib.invalidate_caches()

    from lazy_pkg.mod import helper

    assert helper() == "helper"


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
    """Names the module reads when it runs, leaving out `if TYPE_CHECKING:` blocks and annotations."""
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
    assert set(dir(module)) - set(vars(module)) == set(exports)
    for name, obj in exports.items():
        assert getattr(module, name) is obj, name
    # Resolved names never become globals, so module code behaves the same whatever was used first.
    assert not set(exports) & set(vars(module))
    assert not {*modules, *exports} & _names_read_at_runtime(tree)
