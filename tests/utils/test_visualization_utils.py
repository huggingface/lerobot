#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.processor import TransitionKey
from lerobot.utils.constants import OBS_STATE


@pytest.fixture
def mock_rerun(monkeypatch):
    """
    Provide a mock `rerun` module so tests don't depend on the real library.
    Also reload the module-under-test so it binds to this mock `rr`.
    """
    calls = []

    class DummyScalar:
        def __init__(self, value):
            self.value = float(value)

    class DummyImage:
        def __init__(self, arr):
            self.arr = arr

    def dummy_log(key, obj=None, **kwargs):
        # Accept either positional `obj` or keyword `entity` and record remaining kwargs.
        if obj is None and "entity" in kwargs:
            obj = kwargs.pop("entity")
        calls.append((key, obj, kwargs))

    dummy_rr = SimpleNamespace(
        Scalars=DummyScalar,
        Image=DummyImage,
        log=dummy_log,
        init=lambda *a, **k: None,
        spawn=lambda *a, **k: None,
    )

    # Inject fake module into sys.modules
    monkeypatch.setitem(sys.modules, "rerun", dummy_rr)

    # Now import and reload the module under test, to bind to our rerun mock
    import lerobot.utils.visualization_utils as vu

    importlib.reload(vu)

    # Expose both the reloaded module and the call recorder
    yield vu, calls


def _keys(calls):
    """Helper to extract just the keys logged to rr.log"""
    return [k for (k, _obj, _kw) in calls]


def _obj_for(calls, key):
    """Find the first object logged under a given key."""
    for k, obj, _kw in calls:
        if k == key:
            return obj
    raise KeyError(f"Key {key} not found in calls: {calls}")


def _kwargs_for(calls, key):
    for k, _obj, kw in calls:
        if k == key:
            return kw
    raise KeyError(f"Key {key} not found in calls: {calls}")


def test_log_rerun_data_envtransition_scalars_and_image(mock_rerun):
    vu, calls = mock_rerun

    # Build EnvTransition dict
    obs = {
        f"{OBS_STATE}.temperature": np.float32(25.0),
        # CHW image should be converted to HWC for rr.Image
        "observation.camera": np.zeros((3, 10, 20), dtype=np.uint8),
    }
    act = {
        "action.throttle": 0.7,
        # 1D array should log individual Scalars with suffix _i
        "action.vector": np.array([1.0, 2.0], dtype=np.float32),
    }
    transition = {
        TransitionKey.OBSERVATION: obs,
        TransitionKey.ACTION: act,
    }

    # Extract observation and action data from transition like in the real call sites
    obs_data = transition.get(TransitionKey.OBSERVATION, {})
    action_data = transition.get(TransitionKey.ACTION, {})
    vu.log_rerun_data(observation=obs_data, action=action_data)

    # We expect:
    # - observation.state.temperature -> Scalars
    # - observation.camera -> Image (HWC) with static=True
    # - action.throttle -> Scalars
    # - action.vector_0, action.vector_1 -> Scalars
    expected_keys = {
        f"{OBS_STATE}.temperature",
        "observation.camera",
        "action.throttle",
        "action.vector_0",
        "action.vector_1",
    }
    assert set(_keys(calls)) == expected_keys

    # Check scalar types and values
    temp_obj = _obj_for(calls, f"{OBS_STATE}.temperature")
    assert type(temp_obj).__name__ == "DummyScalar"
    assert temp_obj.value == pytest.approx(25.0)

    throttle_obj = _obj_for(calls, "action.throttle")
    assert type(throttle_obj).__name__ == "DummyScalar"
    assert throttle_obj.value == pytest.approx(0.7)

    v0 = _obj_for(calls, "action.vector_0")
    v1 = _obj_for(calls, "action.vector_1")
    assert type(v0).__name__ == "DummyScalar"
    assert type(v1).__name__ == "DummyScalar"
    assert v0.value == pytest.approx(1.0)
    assert v1.value == pytest.approx(2.0)

    # Check image handling: CHW -> HWC
    img_obj = _obj_for(calls, "observation.camera")
    assert type(img_obj).__name__ == "DummyImage"
    assert img_obj.arr.shape == (10, 20, 3)  # transposed
    assert _kwargs_for(calls, "observation.camera").get("static", False) is True  # static=True for images


def test_log_rerun_data_plain_list_ordering_and_prefixes(mock_rerun):
    vu, calls = mock_rerun

    # First dict without prefixes treated as observation
    # Second dict without prefixes treated as action
    obs_plain = {
        "temp": 1.5,
        # Already HWC image => should stay as-is
        "img": np.zeros((5, 6, 3), dtype=np.uint8),
        "none": None,  # should be skipped
    }
    act_plain = {
        "throttle": 0.3,
        "vec": np.array([9, 8, 7], dtype=np.float32),
    }

    # Extract observation and action data from list like the old function logic did
    # First dict was treated as observation, second as action
    vu.log_rerun_data(observation=obs_plain, action=act_plain)

    # Expected keys with auto-prefixes
    expected = {
        "observation.temp",
        "observation.img",
        "action.throttle",
        "action.vec_0",
        "action.vec_1",
        "action.vec_2",
    }
    logged = set(_keys(calls))
    assert logged == expected

    # Scalars
    t = _obj_for(calls, "observation.temp")
    assert type(t).__name__ == "DummyScalar"
    assert t.value == pytest.approx(1.5)

    throttle = _obj_for(calls, "action.throttle")
    assert type(throttle).__name__ == "DummyScalar"
    assert throttle.value == pytest.approx(0.3)

    # Image stays HWC
    img = _obj_for(calls, "observation.img")
    assert type(img).__name__ == "DummyImage"
    assert img.arr.shape == (5, 6, 3)
    assert _kwargs_for(calls, "observation.img").get("static", False) is True

    # Vectors
    for i, val in enumerate([9, 8, 7]):
        o = _obj_for(calls, f"action.vec_{i}")
        assert type(o).__name__ == "DummyScalar"
        assert o.value == pytest.approx(val)


def test_log_rerun_data_kwargs_only(mock_rerun):
    vu, calls = mock_rerun

    vu.log_rerun_data(
        observation={"observation.temp": 10.0, "observation.gray": np.zeros((8, 8, 1), dtype=np.uint8)},
        action={"action.a": 1.0},
    )

    keys = set(_keys(calls))
    assert "observation.temp" in keys
    assert "observation.gray" in keys
    assert "action.a" in keys

    temp = _obj_for(calls, "observation.temp")
    assert type(temp).__name__ == "DummyScalar"
    assert temp.value == pytest.approx(10.0)

    img = _obj_for(calls, "observation.gray")
    assert type(img).__name__ == "DummyImage"
    assert img.arr.shape == (8, 8, 1)  # remains HWC
    assert _kwargs_for(calls, "observation.gray").get("static", False) is True

    a = _obj_for(calls, "action.a")
    assert type(a).__name__ == "DummyScalar"
    assert a.value == pytest.approx(1.0)
