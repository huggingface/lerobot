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

import traceback

import pytest

# serial may not be installed in minimal environments used for quick test runs
try:
    from serial import SerialException
except Exception:  # pragma: no cover - best-effort fallback for environments without pyserial
    class SerialException(Exception):
        pass

# Avoid importing heavy test helpers (which depend on torch etc.) at module import time.
# Provide a lightweight DEVICE fallback used only for informational messages.
import os
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "unknown")

# Try to import lightweight config types used by some helpers; if unavailable,
# provide minimal fallbacks to allow test collection in environments without
# the full package installed.
try:
    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
except Exception:  # pragma: no cover - fallback for minimal test environments
    class FeatureType:
        pass

    class PipelineFeatureType:
        pass

    class PolicyFeature:
        def __init__(self, *args, **kwargs):
            pass

# Do not load heavy fixture plugins in lightweight test runs here.
# If you need full test runs, run pytest in a dev environment with project dependencies installed.
pytest_plugins = []


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


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
        elif isinstance(e, SerialException):
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
