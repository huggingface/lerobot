# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Golden-vector parity tests for the ported RunningStats.

The ported implementation (``preprocessing/norm_stats.py``) is asserted against
the upstream reference implementation (``lingbotvla.utils.normalize``) with a
fixed seed, on the same batches, in the same order. Any numeric drift is a
porting bug; the tolerances below are pure float64 reassociation margins.

The upstream module is imported from a sibling checkout of
``Robbyant/lingbot-vla-v2``; the test skips when that checkout is unavailable
(CI without the upstream repo still runs the self-consistency tests).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Bypass ``lerobot.policies.lingbot_vla_v2.__init__`` (pulls transformers): load
# preprocessing/norm_stats.py directly by file path.
_NORM_STATS_PY = (
    Path(__file__).resolve().parents[3]
    / "src/lerobot/policies/lingbot_vla_v2/preprocessing/norm_stats.py"
)
_spec = importlib.util.spec_from_file_location("lingbot_vla_v2_norm_stats", _NORM_STATS_PY)
_norm_stats = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _norm_stats  # dataclass() looks up cls.__module__ in sys.modules
_spec.loader.exec_module(_norm_stats)

NormStats = _norm_stats.NormStats
RunningStats = _norm_stats.RunningStats
RunningStatsState = _norm_stats.RunningStatsState
deserialize_json = _norm_stats.deserialize_json
load = _norm_stats.load
save = _norm_stats.save

UPSTREAM_ROOT_CANDIDATES = [
    Path("/home/nvidia/platform/20-training/upstream/lingbot-vla-v2-upstream"),
    Path.home() / "lingbot-vla-v2-upstream",
]


def _import_upstream():
    for root in UPSTREAM_ROOT_CANDIDATES:
        if root.exists():
            sys.path.insert(0, str(root))
            try:
                from lingbotvla.utils.normalize import RunningStats as UpstreamRS

                return UpstreamRS
            except ImportError:
                continue
    return None


UPSTREAM = pytest.mark.skipif(_import_upstream() is None, reason="upstream lingbot-vla-v2 repo not importable")


def _make_batches(seed: int = 42, n_batches: int = 6, batch_rows: int = 37, dims: int = 7) -> list[np.ndarray]:
    """Deterministic, deliberately skewed data: mixed scales, occasional outliers,
    a near-constant dim — the shapes that stress running-stat estimators."""
    rng = np.random.default_rng(seed)
    batches = []
    for b in range(n_batches):
        base = rng.normal(loc=b * 0.3, scale=np.linspace(0.05, 4.0, dims), size=(batch_rows, dims))
        # A near-constant column (dim 1) and one with an occasional large outlier (dim 3).
        base[:, 1] = 1.5 + rng.normal(0, 1e-5, batch_rows)
        if b % 2 == 0:
            base[3, 3] = 30.0 * (b + 1)
        batches.append(base.astype(np.float64))
    return batches


def _feed(cls, batches) -> RunningStats:
    s = cls()
    for b in batches:
        s.update(b)
    return s


def _assert_stats_close(a, b, rtol=1e-10, atol=1e-12):
    for field in ("mean", "std", "q01", "q99", "q02", "q98", "min", "max"):
        va, vb = np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
        assert va.shape == vb.shape, f"{field}: shape {va.shape} != {vb.shape}"
        np.testing.assert_allclose(va, vb, rtol=rtol, atol=atol, err_msg=field)


# ---------------------------------------------------------------- self tests


def test_running_stats_self_consistency_against_numpy():
    batches = _make_batches()
    flat = np.concatenate(batches, axis=0)
    stats = _feed(RunningStats, batches).get_statistics()
    np.testing.assert_allclose(stats.mean, flat.mean(axis=0), rtol=1e-12, atol=1e-15)
    # std is sqrt(E[x²]−E[x]²): on the near-constant column (std≈1e-5) this
    # suffers catastrophic cancellation — absolute error stays ~1e-11 (float64
    # ULP of the ~1.5 offset), which only looks big relatively. Gate rel-err on
    # the genuinely varying dims, abs-err on the degenerate one.
    varying = flat.std(axis=0) > 1e-3
    np.testing.assert_allclose(stats.std[varying], flat.std(axis=0)[varying], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(stats.std[~varying], flat.std(axis=0)[~varying], rtol=0, atol=1e-10)
    np.testing.assert_allclose(stats.min, flat.min(axis=0))
    np.testing.assert_allclose(stats.max, flat.max(axis=0))
    # Quantiles come from a 5000-bin online histogram whose bins REBIN every
    # time a batch extends the observed min/max (upstream _adjust_histograms
    # redistributes counts into neighbouring bins). Each rebin smears counts
    # by ~1 old-bin, so accuracy degrades with the number of range extensions
    # — allow ~15 bins, not ~2.5. Exact-value parity with the upstream
    # implementation is covered separately by test_parity_quantiles.
    bin_width = (flat.max(axis=0) - flat.min(axis=0)) / 5000
    for field, q in (("q01", 0.01), ("q02", 0.02), ("q98", 0.98), ("q99", 0.99)):
        exact = np.quantile(flat, q, axis=0)
        for d in range(flat.shape[1]):
            # Degenerate (near-constant) dims: histogram bins are ~1e-8 wide and
            # rebinning leaves relative errors at the bin scale — gate on an
            # absolute tolerance anchored to the dim's own tiny spread instead.
            tol = max(float(bin_width[d]) * 15, float(flat[:, d].std()) * 0.5)
            np.testing.assert_allclose(
                getattr(stats, field)[d], exact[d], rtol=0, atol=tol,
                err_msg=f"histogram quantile {field} dim {d}",
            )


def test_running_stats_state_roundtrip():
    batches = _make_batches()
    s = _feed(RunningStats, batches)
    restored = RunningStats.from_state(s.get_state())
    extra = _make_batches(seed=7, n_batches=2)
    for b in extra:
        s.update(b)
        restored.update(b)
    a, r = s.get_statistics(), restored.get_statistics()
    _assert_stats_close(a, r, rtol=0, atol=0)


def test_running_stats_merge_parity():
    batches = _make_batches(n_batches=8)
    flat = np.concatenate(batches, axis=0)
    varying = flat.std(axis=0) > 1e-3
    full = _feed(RunningStats, batches)
    left = _feed(RunningStats, batches[:4])
    right = _feed(RunningStats, batches[4:])
    merged = RunningStats.merge([left, right])
    a, m = full.get_statistics(), merged.get_statistics()
    np.testing.assert_allclose(a.mean, m.mean, rtol=1e-12, atol=1e-15)
    # See test_running_stats_self_consistency_against_numpy: near-constant dims
    # cancel in E[x²]−E[x]²; abs error ~1e-11 is float64 noise, not drift.
    np.testing.assert_allclose(a.std[varying], m.std[varying], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.std[~varying], m.std[~varying], rtol=0, atol=1e-10)
    np.testing.assert_allclose(a.min, m.min)
    np.testing.assert_allclose(a.max, m.max)


def test_save_load_roundtrip(tmp_path):
    batches = _make_batches()
    stats = {"action.arm.position": _feed(RunningStats, batches).get_statistics()}
    out = tmp_path / "norm_stats.json"
    save(out, stats, count=int(sum(b.shape[0] for b in batches)))
    loaded = load(out)
    assert set(loaded) == set(stats)
    _assert_stats_close(stats["action.arm.position"], loaded["action.arm.position"], rtol=0, atol=0)
    # The serialized layout matches the upstream file contract.
    import json

    payload = json.loads(out.read_text())
    assert set(payload) == {"norm_stats", "count"}
    assert set(payload["norm_stats"]["action.arm.position"]) == {
        "mean", "std", "q01", "q99", "q02", "q98", "min", "max",
    }


def test_get_statistics_chunk_size_reshape():
    batches = _make_batches(dims=4)
    s = _feed(RunningStats, batches)
    stats_flat = s.get_statistics()
    s2 = _feed(RunningStats, batches)
    stats_chunked = s2.get_statistics(chunk_size=2)
    assert stats_chunked.mean.shape == (2, 2)
    np.testing.assert_allclose(stats_chunked.mean.reshape(-1), stats_flat.mean)


# ------------------------------------------------------- upstream parity tests


@UPSTREAM
def test_parity_mean_std_min_max():
    upstream_running_stats = _import_upstream()
    batches = _make_batches()
    a = _feed(RunningStats, batches).get_statistics()
    b = _feed(upstream_running_stats, batches).get_statistics()
    _assert_stats_close(a, b)


@UPSTREAM
def test_parity_quantiles():
    upstream_running_stats = _import_upstream()
    batches = _make_batches()
    a = _feed(RunningStats, batches).get_statistics()
    b = _feed(upstream_running_stats, batches).get_statistics()
    for field in ("q01", "q99", "q02", "q98"):
        np.testing.assert_allclose(
            np.asarray(getattr(a, field)), np.asarray(getattr(b, field)),
            rtol=0, atol=1e-12, err_msg=field,
        )


@UPSTREAM
def test_parity_merge():
    upstream_running_stats = _import_upstream()
    batches = _make_batches(n_batches=8)
    ours = RunningStats.merge([_feed(RunningStats, batches[:4]), _feed(RunningStats, batches[4:])])
    theirs = upstream_running_stats.merge(
        [_feed(upstream_running_stats, batches[:4]), _feed(upstream_running_stats, batches[4:])]
    )
    a, b = ours.get_statistics(), theirs.get_statistics()
    _assert_stats_close(a, b)


@UPSTREAM
def test_parity_chunk_reshape():
    upstream_running_stats = _import_upstream()
    batches = _make_batches(dims=4)
    a = _feed(RunningStats, batches).get_statistics(chunk_size=2)
    b = _feed(upstream_running_stats, batches).get_statistics(chunk_size=2)
    _assert_stats_close(a, b)
