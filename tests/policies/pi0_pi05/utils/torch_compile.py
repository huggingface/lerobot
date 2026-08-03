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

import time
from collections.abc import Callable

import torch
from torch._dynamo.utils import counters, guard_failures
from torch.profiler import ProfilerActivity

FORWARD_RTOL = 1e-5
FORWARD_ATOL = 5e-2
SAMPLE_RTOL = 1e-5
SAMPLE_ATOL = 1e-2
COMPILE_MODE = "max-autotune"
STEADY_STATE_WARMUPS = 3
STEADY_STATE_REPEATS = 3


def make_compile_config(config_cls, *, compile_model):
    return config_cls(device="cuda", compile_model=compile_model, compile_mode=COMPILE_MODE)


def counter_total(name):
    return sum(counters.get(name, {}).values())


def compile_snapshot():
    return {
        "graph_breaks": counter_total("graph_break"),
        "recompiles": counter_total("recompiles"),
        "recompile_limits": counter_total("recompile_limit"),
        "unique_graphs": counters["stats"].get("unique_graphs", 0),
    }


def reset_compile_state():
    torch._dynamo.reset()
    counters.clear()
    guard_failures.clear()


def clone_cuda_graph_output(output):
    if torch.is_tensor(output):
        return output.clone()
    if isinstance(output, tuple):
        return tuple(clone_cuda_graph_output(item) for item in output)
    if isinstance(output, list):
        return [clone_cuda_graph_output(item) for item in output]
    if isinstance(output, dict):
        return {key: clone_cuda_graph_output(value) for key, value in output.items()}
    return output


def run_model_step(fn: Callable, kwargs: dict):
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    return fn(**kwargs)


def assert_explain_has_no_graph_breaks(fn: Callable, kwargs: dict, label: str):
    reset_compile_state()
    explanation = torch._dynamo.explain(fn)(**kwargs)

    assert explanation.graph_count > 0, f"{label} was not captured by Dynamo"
    assert explanation.graph_break_count == 0, (
        f"{label} has {explanation.graph_break_count} graph break(s): {explanation.break_reasons}"
    )
    assert not explanation.break_reasons, f"{label} graph break reasons: {explanation.break_reasons}"

    print(
        f"{label} capture: graphs={explanation.graph_count}, "
        f"graph_breaks={explanation.graph_break_count}, ops={explanation.op_count}, "
        f"guards={len(explanation.out_guards or [])}"
    )
    return explanation


@torch.no_grad()
def assert_compiled_output_matches_eager(eager_model, compiled_model, forward_kwargs, sample_kwargs):
    eager_forward = eager_model.forward(**forward_kwargs)
    compiled_forward = compiled_model.forward(**forward_kwargs)
    torch.testing.assert_close(compiled_forward, eager_forward, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)

    eager_actions = eager_model.sample_actions(**sample_kwargs)
    compiled_actions = compiled_model.sample_actions(**sample_kwargs)
    torch.testing.assert_close(compiled_actions, eager_actions, rtol=SAMPLE_RTOL, atol=SAMPLE_ATOL)


@torch.no_grad()
def assert_cache_stability(fn: Callable, kwargs: dict, label: str):
    reset_compile_state()

    first_output = clone_cuda_graph_output(run_model_step(fn, kwargs))
    first_snapshot = compile_snapshot()
    second_output = clone_cuda_graph_output(run_model_step(fn, kwargs))
    second_snapshot = compile_snapshot()
    third_output = clone_cuda_graph_output(run_model_step(fn, kwargs))
    third_snapshot = compile_snapshot()

    torch.testing.assert_close(second_output, first_output, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    torch.testing.assert_close(third_output, first_output, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    assert first_snapshot["unique_graphs"] > 0, f"{label} did not compile any graph"
    assert third_snapshot["graph_breaks"] == 0, f"{label} graph breaks: {third_snapshot}"
    assert third_snapshot["recompiles"] == 0, f"{label} recompiled: {third_snapshot}"
    assert third_snapshot["recompile_limits"] == 0, f"{label} hit recompile limit: {third_snapshot}"
    assert second_snapshot["unique_graphs"] == first_snapshot["unique_graphs"], (
        f"{label} compiled new graph on second call: first={first_snapshot}, second={second_snapshot}"
    )
    assert third_snapshot["unique_graphs"] == first_snapshot["unique_graphs"], (
        f"{label} compiled new graph on third call: first={first_snapshot}, third={third_snapshot}"
    )
    assert not guard_failures, f"{label} guard failures: {dict(guard_failures)}"

    print(f"{label} cache: first={first_snapshot}, third={third_snapshot}")


@torch.no_grad()
def benchmark_runtime(eager_fn: Callable, compiled_fn: Callable, kwargs: dict, label: str):
    run_warmups(eager_fn, kwargs)
    run_warmups(compiled_fn, kwargs)
    torch.cuda.synchronize()

    eager_metrics = profile_callable(eager_fn, kwargs)
    compiled_metrics = profile_callable(compiled_fn, kwargs)
    speedup = eager_metrics["cuda_event_ms"] / compiled_metrics["cuda_event_ms"]

    print(
        f"{label} runtime: eager_cuda={eager_metrics['cuda_event_ms']:.3f} ms, "
        f"compiled_cuda={compiled_metrics['cuda_event_ms']:.3f} ms, speedup={speedup:.3f}x, "
        f"host_wall_ms eager/compiled={eager_metrics['host_wall_ms']:.3f}/"
        f"{compiled_metrics['host_wall_ms']:.3f}, "
        f"cpu_self_time_ms eager/compiled={eager_metrics['cpu_self_time_ms']:.3f}/"
        f"{compiled_metrics['cpu_self_time_ms']:.3f}, "
        f"cuda_launches eager/compiled={eager_metrics['cuda_launch_count']}/"
        f"{compiled_metrics['cuda_launch_count']}, "
        f"profiler_events eager/compiled={eager_metrics['profiler_event_count']}/"
        f"{compiled_metrics['profiler_event_count']}, "
        f"peak_mem_mib eager/compiled={eager_metrics['peak_mem_mib']:.1f}/"
        f"{compiled_metrics['peak_mem_mib']:.1f}"
    )

    assert eager_metrics["cuda_event_ms"] > 0
    assert compiled_metrics["cuda_event_ms"] > 0
    assert eager_metrics["profiler_event_count"] > 0
    assert compiled_metrics["profiler_event_count"] > 0
    return eager_metrics, compiled_metrics


def run_warmups(fn: Callable, kwargs: dict):
    for _ in range(STEADY_STATE_WARMUPS):
        run_model_step(fn, kwargs)
    torch.cuda.synchronize()


def profile_callable(fn: Callable, kwargs: dict):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start_event.record()
    for _ in range(STEADY_STATE_REPEATS):
        run_model_step(fn, kwargs)
    end_event.record()
    torch.cuda.synchronize()
    cuda_event_ms = start_event.elapsed_time(end_event) / STEADY_STATE_REPEATS
    host_wall_ms = (time.perf_counter() - host_start) * 1000 / STEADY_STATE_REPEATS
    peak_mem_mib = torch.cuda.max_memory_allocated() / 1024**2

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU],
    ) as profiler:
        run_model_step(fn, kwargs)
        torch.cuda.synchronize()

    key_averages = profiler.key_averages()
    cpu_self_time_ms = sum(event.self_cpu_time_total for event in key_averages) / 1000
    cuda_launch_count = sum(
        event.count
        for event in key_averages
        if event.key in {"cudaLaunchKernel", "cudaGraphLaunch", "cudaLaunchKernelExC"}
    )
    profiler_event_count = sum(event.count for event in key_averages)

    return {
        "cuda_event_ms": cuda_event_ms,
        "host_wall_ms": host_wall_ms,
        "cpu_self_time_ms": cpu_self_time_ms,
        "cuda_launch_count": cuda_launch_count,
        "profiler_event_count": profiler_event_count,
        "peak_mem_mib": peak_mem_mib,
    }
