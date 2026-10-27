#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import functools
import re
import traceback
from pathlib import Path

import draccus.wrappers.docstring as _draccus_docstring
import pytest

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.import_utils import is_package_available
from tests.utils import DEVICE

# On every `draccus.parse()`, draccus rebuilds each dataclass field's help text by
# re-reading and re-parsing the class source (draccus.wrappers.docstring). For a config
# as large as TrainPipelineConfig this costs ~2.5s per parse — negligible for the single
# parse a CLI does, but tests parse configs hundreds of times. The source can't change
# within a run, so memoize it for the whole test session.
_draccus_docstring.get_attribute_docstring = functools.cache(_draccus_docstring.get_attribute_docstring)

# Import fixture modules as plugins.
# Fixtures that depend on optional packages are only registered when those packages are available,
# so that tests can be collected and run even with a minimal install.
pytest_plugins = [
    "tests.fixtures.cadence",
    "tests.fixtures.optimizers",
]

if is_package_available("datasets"):
    pytest_plugins += [
        "tests.fixtures.dataset_factories",
        "tests.fixtures.files",
        "tests.fixtures.hub",
    ]


# Marks selecting the tests of each Fast Tests tier (`-m dataset`, see .github/workflows/fast_tests.yml);
# every other test is marked `base`. Marks only select tests: skipping when an extra is missing is still
# done by `pytest.importorskip` and friends.
TIER_MARKS = ("dataset", "hardware", "viz")


# The GPU Tests job (`-m gpu`, see .github/workflows/full_tests.yml) only runs tests marked `gpu`. The mark is added
# to policy tests, `require_cuda` tests and tests skipped without CUDA; other tests that use CUDA mark themselves.
# An unmarked test that skips for lack of CUDA fails instead, so it cannot silently miss the GPU job.
POLICY_TESTS = Path(__file__).parent / "policies"
NEEDS_GPU_REASON = re.compile(r"cuda|gpu|accelerator", re.IGNORECASE)


def _needs_gpu(item) -> bool:
    if item.path.is_relative_to(POLICY_TESTS) or getattr(
        getattr(item, "function", None), "requires_cuda", False
    ):
        return True
    return any(NEEDS_GPU_REASON.search(mark.kwargs.get("reason", "")) for mark in item.iter_markers("skipif"))


def pytest_configure(config):
    config.addinivalue_line("markers", "base: Fast Tests base tier (added to tests without a tier mark)")
    for mark in TIER_MARKS:
        config.addinivalue_line("markers", f"{mark}: Fast Tests tier that installs the `{mark}` extra")
    config.addinivalue_line(
        "markers", "gpu: run by the GPU Tests job (added to policy tests and `require_cuda` tests)"
    )


@pytest.hookimpl(tryfirst=True)  # before `-m` deselects by mark
def pytest_collection_modifyitems(config, items):
    for item in items:
        if not any(mark.name in TIER_MARKS for mark in item.iter_markers()):
            item.add_marker(pytest.mark.base)
        if _needs_gpu(item):
            item.add_marker(pytest.mark.gpu)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and isinstance(report.longrepr, tuple) and item.get_closest_marker("gpu") is None:
        reason = report.longrepr[2]
        if NEEDS_GPU_REASON.search(reason):
            report.outcome = "failed"
            report.longrepr = (
                f"{item.nodeid} skips without CUDA ({reason}) but is not marked `gpu`: add `@pytest.mark.gpu`"
            )


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


def _is_serial_exception(exc: Exception) -> bool:
    """Check if an exception is a SerialException without requiring pyserial."""
    if not is_package_available("pyserial", import_name="serial"):
        return False
    from serial import SerialException

    return isinstance(exc, SerialException)


def _check_component_availability(component_type, available_components, make_component):
    """Generic helper to check if a hardware component is available"""
    if component_type not in available_components:
        raise ValueError(
            f"The {component_type} type is not valid. Expected one of these '{available_components}'"
        )

    try:
        component = make_component(component_type)
        component.connect()
        del component
        return True

    except Exception as e:
        print(f"\nA {component_type} is not available.")

        if isinstance(e, ModuleNotFoundError):
            print(f"\nInstall module '{e.name}'")
        elif _is_serial_exception(e):
            print("\nNo physical device detected.")
        elif isinstance(e, ValueError) and "camera_index" in str(e):
            print("\nNo physical camera detected.")
        else:
            traceback.print_exc()

        return False


@pytest.fixture
def patch_builtins_input(monkeypatch):
    def print_text(text=None):
        if text is not None:
            print(text)

    monkeypatch.setattr("builtins.input", print_text)


@pytest.fixture
def policy_feature_factory():
    """PolicyFeature factory"""

    def _pf(ft: FeatureType, shape: tuple[int, ...]) -> PolicyFeature:
        return PolicyFeature(type=ft, shape=shape)

    return _pf


def assert_contract_is_typed(features: dict[PipelineFeatureType, dict[str, PolicyFeature]]) -> None:
    assert isinstance(features, dict)
    assert all(isinstance(k, PipelineFeatureType) for k in features)
    assert all(isinstance(v, dict) for v in features.values())
    assert all(all(isinstance(nk, str) for nk in v) for v in features.values())
    assert all(all(isinstance(nv, PolicyFeature) for nv in v.values()) for v in features.values())
