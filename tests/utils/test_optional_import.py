import re
import types

import pytest

from lerobot.utils.optional import optional_import


def test_optional_import_success_attr():
    sqrt = optional_import("math", "test-extra", attr="sqrt")
    assert isinstance(sqrt, types.BuiltinFunctionType)
    assert sqrt(9) == 3


def test_optional_import_missing_module_message():
    missing_module = "definitely_not_installed_mod_abcdef12345"
    with pytest.raises(ImportError) as exc:
        optional_import(missing_module, "lekiwi")
    # Ensure the error message guides to the correct extra install hint
    assert re.search(r"pip install lerobot\[lekiwi\]", str(exc.value))
