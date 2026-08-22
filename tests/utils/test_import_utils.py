import importlib.util
import sys

from lerobot.utils.import_utils import _try_import


def test_try_import_real_module():
    assert _try_import("json") is True


def test_try_import_missing_module():
    assert _try_import("definitely_not_a_real_module_xyz") is False


def test_present_but_broken_package(tmp_path, caplog):
    pkg = tmp_path / "broken_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('raise ImportError("dependency pins violated")\n')
    sys.path.insert(0, str(tmp_path))
    try:
        # find_spec sees it — this is exactly the state the old check trusted
        assert importlib.util.find_spec("broken_pkg") is not None
        assert _try_import("broken_pkg") is False
        assert "broken_pkg" in caplog.text
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("broken_pkg", None)
