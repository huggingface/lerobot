"""Build DROID-W's required LieTorch backend, omitting example-only operators."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import setuptools


def main():
    """Keep vendor sources/flags intact and filter only the unused extension."""
    original = setuptools.setup

    def selected_setup(*args, **kwargs):
        extensions = kwargs["ext_modules"]
        if {value.name for value in extensions} != {"lietorch_backends", "lietorch_extras"}:
            raise RuntimeError("The frozen LieTorch extension catalog changed")
        kwargs["ext_modules"] = [value for value in extensions if value.name == "lietorch_backends"]
        print("Building lietorch_backends; lietorch_extras is used only by vendor examples", flush=True)
        return original(*args, **kwargs)

    sys.argv = ["setup.py", "bdist_wheel", "--dist-dir", "/tmp/lietorch-wheels"]  # nosec B108 - isolated root-owned Docker build
    with patch.object(setuptools, "setup", selected_setup):
        runpy.run_path(str(Path.cwd() / "setup.py"), run_name="__main__")


if __name__ == "__main__":
    main()
