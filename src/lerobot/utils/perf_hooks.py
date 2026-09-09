"""Measurement hooks for the exp0908 performance campaign. Off unless env vars say so.

PERF_OUT=<dir>          per-rank step timing series + memory, written at exit as
                        perf.rank<N>.json (cuda events around the update, wall per step,
                        max_memory_allocated, RSS).
PROF_STEPS=<a>-<b>      torch.profiler window over 1-based steps a..b inclusive; traces go
                        to PERF_OUT/trace/rank<N>.json.
"""

from __future__ import annotations

import json
import os
import resource
import time

import torch

_out = os.environ.get("PERF_OUT")
_window = os.environ.get("PROF_STEPS")
_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
_series: list[dict] = []
_prof = None
_t_import = time.perf_counter()
_marks: dict[str, float] = {"import": _t_import}


def enabled() -> bool:
    return _out is not None


def mark(name: str) -> None:
    """Startup milestones, wall clock since module import."""
    if enabled():
        _marks[name] = time.perf_counter()


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _bounds():
    a, b = _window.split("-")
    return int(a), int(b)


def step_begin(step: int) -> None:
    global _prof
    if not enabled():
        return
    if _window and step == _bounds()[0]:
        os.makedirs(os.path.join(_out, "trace"), exist_ok=True)
        _prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=False, with_stack=False,
        )
        _prof.__enter__()
    if _prof is not None:
        torch.profiler.record_function(f"## step {step}").__enter__()


def step_end(step: int, wall_s: float, data_s: float, update_s: float) -> None:
    global _prof
    if not enabled():
        return
    _series.append({
        "step": step, "wall_s": wall_s, "data_s": data_s, "update_s": update_s,
        "gpu_alloc_gb": torch.cuda.memory_allocated() / 2**30,
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_reserved_gb": torch.cuda.memory_reserved() / 2**30,
        "rss_gb": _rss_gb(),
    })
    if _prof is not None and step == _bounds()[1]:
        torch.cuda.synchronize()
        _prof.__exit__(None, None, None)
        _prof.export_chrome_trace(os.path.join(_out, "trace", f"rank{_rank}.json"))
        _prof = None


def flush() -> None:
    if not enabled():
        return
    os.makedirs(_out, exist_ok=True)
    t = time.perf_counter()
    with open(os.path.join(_out, f"perf.rank{_rank}.json"), "w") as f:
        json.dump({"rank": _rank,
                   "marks_s": {k: v - _t_import for k, v in _marks.items()},
                   "end_s": t - _t_import,
                   "steps": _series}, f)
