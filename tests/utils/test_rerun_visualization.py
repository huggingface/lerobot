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

pytest.importorskip("rerun", reason="rerun-sdk is required (install lerobot[viz])")

from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


@pytest.fixture
def mock_rerun(monkeypatch):
    """
    Provide a mock `rerun` module (and `rerun.blueprint` submodule) so tests don't
    depend on the real library. Also reload the module-under-test so it binds to
    this mock `rr`.
    """
    calls = []
    blueprints = []

    class DummyScalar:
        def __init__(self, value):
            # Scalars may be built from a single float or from a 1D array batch.
            self.value = value

    class DummyImage:
        def __init__(self, arr):
            self.arr = arr

        def compress(self, *a, **k):
            return self

    class DummyDepthImage:
        def __init__(self, arr, meter=None, colormap=None):
            self.arr = arr
            self.meter = meter
            self.colormap = colormap

    def dummy_log(key, obj=None, **kwargs):
        # Accept either positional `obj` or keyword `entity` and record remaining kwargs.
        if obj is None and "entity" in kwargs:
            obj = kwargs.pop("entity")
        calls.append((key, obj, kwargs))

    def dummy_send_blueprint(blueprint, *a, **k):
        blueprints.append(blueprint)

    # Mock the `rerun.blueprint` submodule used to build the layout.
    dummy_rrb = SimpleNamespace(
        Spatial2DView=lambda origin=None, name=None: SimpleNamespace(
            kind="Spatial2DView", origin=origin, name=name
        ),
        TimeSeriesView=lambda name=None, contents=None: SimpleNamespace(
            kind="TimeSeriesView", name=name, contents=contents
        ),
        Grid=lambda *views: SimpleNamespace(kind="Grid", views=list(views)),
        Blueprint=lambda root: SimpleNamespace(kind="Blueprint", root=root),
    )

    dummy_rr = SimpleNamespace(
        __name__="rerun",
        __package__="rerun",
        __spec__=SimpleNamespace(name="rerun", submodule_search_locations=None),
        Scalars=DummyScalar,
        Image=DummyImage,
        DepthImage=DummyDepthImage,
        components=SimpleNamespace(Colormap=SimpleNamespace(Viridis="viridis")),
        log=dummy_log,
        send_blueprint=dummy_send_blueprint,
        init=lambda *a, **k: None,
        spawn=lambda *a, **k: None,
        blueprint=dummy_rrb,
    )

    # Inject fake modules into sys.modules (both `rerun` and `rerun.blueprint`).
    monkeypatch.setitem(sys.modules, "rerun", dummy_rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", dummy_rrb)

    # Now import and reload the module under test, to bind to our rerun mock
    import lerobot.utils.rerun_visualization as rv

    importlib.reload(rv)

    # Expose the reloaded module, the call recorder and the captured blueprints
    yield rv, calls, blueprints


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


def _views_by_kind(blueprint, kind):
    """Return the views of a given kind from the (single) blueprint's grid."""
    return [v for v in blueprint.root.views if v.kind == kind]


def test_log_rerun_data_envtransition_scalars_and_image(mock_rerun):
    rv, calls, blueprints = mock_rerun

    # Build EnvTransition dict
    obs = {
        f"{OBS_STATE}.temperature": np.float32(25.0),
        # CHW image should be converted to HWC for rr.Image
        "observation.camera": np.zeros((3, 10, 20), dtype=np.uint8),
    }
    act = {
        "action.throttle": 0.7,
        # 1D array should be logged as a single Scalars batch under one entity path
        "action.vector": np.array([1.0, 2.0], dtype=np.float32),
    }
    transition = {
        TransitionKey.OBSERVATION: obs,
        TransitionKey.ACTION: act,
    }

    # Extract observation and action data from transition like in the real call sites
    obs_data = transition.get(TransitionKey.OBSERVATION, {})
    action_data = transition.get(TransitionKey.ACTION, {})
    rv.log_rerun_data(observation=obs_data, action=action_data)

    # We expect:
    # - observation.state.temperature -> Scalars
    # - observation.camera -> Image (HWC) with static=True
    # - action.throttle -> Scalars
    # - action.vector -> single Scalars batch (no per-element suffix)
    expected_keys = {
        f"{OBS_STATE}.temperature",
        "observation.camera",
        "action.throttle",
        "action.vector",
    }
    assert set(_keys(calls)) == expected_keys

    # Check scalar types and values
    temp_obj = _obj_for(calls, f"{OBS_STATE}.temperature")
    assert type(temp_obj).__name__ == "DummyScalar"
    assert float(temp_obj.value) == pytest.approx(25.0)

    throttle_obj = _obj_for(calls, "action.throttle")
    assert type(throttle_obj).__name__ == "DummyScalar"
    assert float(throttle_obj.value) == pytest.approx(0.7)

    # 1D vector logged as a single batched Scalars under one entity path
    vec = _obj_for(calls, "action.vector")
    assert type(vec).__name__ == "DummyScalar"
    np.testing.assert_allclose(np.asarray(vec.value), [1.0, 2.0])

    # Check image handling: CHW -> HWC
    img_obj = _obj_for(calls, "observation.camera")
    assert type(img_obj).__name__ == "DummyImage"
    assert img_obj.arr.shape == (10, 20, 3)  # transposed
    assert _kwargs_for(calls, "observation.camera").get("static", False) is True  # static=True for images

    # A blueprint should have been built and sent exactly once, and cached on the function.
    assert len(blueprints) == 1
    assert rv.log_rerun_data.blueprint is blueprints[0]

    bp = blueprints[0]
    # One spatial view per image path
    spatial_views = _views_by_kind(bp, "Spatial2DView")
    assert {v.origin for v in spatial_views} == {"observation.camera"}

    # One time-series view each for observation and action scalars
    ts_views = {v.name: v for v in _views_by_kind(bp, "TimeSeriesView")}
    assert set(ts_views) == {"observation", "action"}
    assert ts_views["observation"].contents == [f"{OBS_STATE}.temperature"]
    assert ts_views["action"].contents == ["action.throttle", "action.vector"]


def test_log_rerun_data_plain_list_ordering_and_prefixes(mock_rerun):
    rv, calls, blueprints = mock_rerun

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
    rv.log_rerun_data(observation=obs_plain, action=act_plain)

    # Expected keys with auto-prefixes. The 1D vector is a single batched Scalars.
    expected = {
        "observation.temp",
        "observation.img",
        "action.throttle",
        "action.vec",
    }
    logged = set(_keys(calls))
    assert logged == expected

    # Scalars
    t = _obj_for(calls, "observation.temp")
    assert type(t).__name__ == "DummyScalar"
    assert float(t.value) == pytest.approx(1.5)

    throttle = _obj_for(calls, "action.throttle")
    assert type(throttle).__name__ == "DummyScalar"
    assert float(throttle.value) == pytest.approx(0.3)

    # Image stays HWC
    img = _obj_for(calls, "observation.img")
    assert type(img).__name__ == "DummyImage"
    assert img.arr.shape == (5, 6, 3)
    assert _kwargs_for(calls, "observation.img").get("static", False) is True

    # Vector logged as a single batched Scalars under one entity path
    vec = _obj_for(calls, "action.vec")
    assert type(vec).__name__ == "DummyScalar"
    np.testing.assert_allclose(np.asarray(vec.value), [9, 8, 7])

    # Blueprint sent once with the expected view layout
    assert len(blueprints) == 1
    bp = blueprints[0]
    spatial_views = _views_by_kind(bp, "Spatial2DView")
    assert {v.origin for v in spatial_views} == {"observation.img"}
    ts_views = {v.name: v for v in _views_by_kind(bp, "TimeSeriesView")}
    assert ts_views["observation"].contents == ["observation.temp"]
    assert ts_views["action"].contents == ["action.throttle", "action.vec"]


def test_log_rerun_data_kwargs_only(mock_rerun):
    rv, calls, blueprints = mock_rerun

    rv.log_rerun_data(
        observation={"observation.temp": 10.0, "observation.gray": np.zeros((8, 8, 1), dtype=np.uint8)},
        action={"action.a": 1.0},
    )

    keys = set(_keys(calls))
    assert "observation.temp" in keys
    assert "observation.gray" in keys
    assert "action.a" in keys

    temp = _obj_for(calls, "observation.temp")
    assert type(temp).__name__ == "DummyScalar"
    assert float(temp.value) == pytest.approx(10.0)

    img = _obj_for(calls, "observation.gray")
    assert type(img).__name__ == "DummyDepthImage"  # single-channel -> DepthImage
    assert img.arr.shape == (8, 8, 1)  # remains HWC
    assert _kwargs_for(calls, "observation.gray").get("static", False) is True

    a = _obj_for(calls, "action.a")
    assert type(a).__name__ == "DummyScalar"
    assert float(a.value) == pytest.approx(1.0)

    # Blueprint sent once, with a spatial view for the image and time-series views for scalars
    assert len(blueprints) == 1
    bp = blueprints[0]
    assert {v.origin for v in _views_by_kind(bp, "Spatial2DView")} == {"observation.gray"}
    ts_views = {v.name: v for v in _views_by_kind(bp, "TimeSeriesView")}
    assert ts_views["observation"].contents == ["observation.temp"]
    assert ts_views["action"].contents == ["action.a"]


def test_log_rerun_data_blueprint_sent_only_once(mock_rerun):
    """The blueprint is built from the first call and not resent on subsequent calls."""
    rv, calls, blueprints = mock_rerun

    rv.log_rerun_data(observation={"temp": 1.0}, action={"a": 2.0})
    assert len(blueprints) == 1
    first_blueprint = rv.log_rerun_data.blueprint

    rv.log_rerun_data(observation={"temp": 3.0}, action={"a": 4.0})
    # Still only one blueprint, and the cached one is unchanged.
    assert len(blueprints) == 1
    assert rv.log_rerun_data.blueprint is first_blueprint
