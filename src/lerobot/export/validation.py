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
"""Numerical parity validation between a PyTorch wrapper and an exported ONNX model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as functional

from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _compare_one(
    wrapper_cpu: nn.Module,
    cpu_inputs: tuple[Tensor, ...],
    ort_session,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    """Single PyTorch vs ONNX Runtime comparison on one batch of inputs."""
    with torch.no_grad():
        pt_output = wrapper_cpu(*cpu_inputs)

    ort_inputs = {ort_session.get_inputs()[i].name: cpu_inputs[i].numpy() for i in range(len(cpu_inputs))}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_output = torch.from_numpy(ort_outputs[0])

    max_abs_error: float = (pt_output.float() - ort_output.float()).abs().max().item()
    cos_sim: float = functional.cosine_similarity(
        pt_output.float().flatten().unsqueeze(0),
        ort_output.float().flatten().unsqueeze(0),
    ).item()
    passed: bool = bool(torch.allclose(pt_output.float(), ort_output.float(), rtol=rtol, atol=atol))
    return {"max_abs_error": max_abs_error, "cos_sim": cos_sim, "allclose": passed}


def _make_random_like(
    sample_inputs: tuple[Tensor, ...],
    generator: torch.Generator,
) -> tuple[Tensor, ...]:
    """Build random-input tensors matching the shapes/dtypes of ``sample_inputs``.

    Floating-point tensors get standard-normal random values; non-float tensors
    (e.g. int64 timestep indices) are passed through unchanged because randomly
    permuting them would break the model contract.
    """
    out: list[Tensor] = []
    for x in sample_inputs:
        if x.is_floating_point():
            out.append(torch.randn(x.shape, generator=generator, dtype=x.dtype))
        else:
            out.append(x.clone())
    return tuple(out)


def validate_onnx(
    wrapper: nn.Module,
    sample_inputs: tuple[Tensor, ...],
    onnx_path: Path | str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    num_random_trials: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare PyTorch wrapper output against ONNX Runtime output.

    Always runs one baseline comparison with the provided ``sample_inputs``.
    When ``num_random_trials > 0``, runs N additional comparisons with random
    Gaussian inputs (same shapes/dtypes as ``sample_inputs``; integer tensors
    pass through unchanged) and aggregates the worst-case results.

    Args:
        wrapper:           The PyTorch wrapper module used for the ONNX export.
        sample_inputs:     Baseline input tensors (same as used during export).
        onnx_path:         Path to the ``.onnx`` file.
        rtol:              Relative tolerance for ``torch.allclose``.
        atol:              Absolute tolerance for ``torch.allclose``.
        num_random_trials: Number of additional random-input comparisons.
        seed:              Seed for the random-input generator.

    Returns:
        Aggregated comparison dict with keys:
        - ``"max_abs_error"`` (float, worst across trials)
        - ``"cos_sim"`` (float, min across trials)
        - ``"allclose"`` (bool, True only if every trial passed)
        - ``"trials"`` (list[dict], one entry per trial, baseline first)
    """
    require_package("onnxruntime", extra="export", import_name="onnxruntime")
    import onnxruntime as ort

    wrapper.eval()
    # Move inputs to CPU for comparison (ONNX Runtime runs on CPU by default).
    cpu_inputs = tuple(x.cpu() for x in sample_inputs)
    wrapper_cpu = wrapper.cpu()

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress INFO/WARNING logs from ORT
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)

    trials: list[dict[str, float | bool]] = []

    # 1. Baseline comparison.
    baseline = _compare_one(wrapper_cpu, cpu_inputs, sess, rtol, atol)
    baseline["trial"] = "baseline"
    trials.append(baseline)

    # 2. Optional random-input trials.
    if num_random_trials > 0:
        generator = torch.Generator().manual_seed(seed)
        for i in range(num_random_trials):
            random_inputs = _make_random_like(cpu_inputs, generator)
            result = _compare_one(wrapper_cpu, random_inputs, sess, rtol, atol)
            result["trial"] = f"random_{i}"
            trials.append(result)

    # 3. Aggregate worst-case across all trials.
    worst_max_abs_error = max(t["max_abs_error"] for t in trials)
    min_cos_sim = min(t["cos_sim"] for t in trials)
    all_passed = all(t["allclose"] for t in trials)

    level = logging.INFO if all_passed else logging.WARNING
    summary = f"baseline + {num_random_trials} random trial(s)" if num_random_trials > 0 else "baseline"
    logger.log(
        level,
        f"Validation {'PASSED' if all_passed else 'FAILED'} ({summary}): "
        f"worst max_abs_error={worst_max_abs_error:.2e}, "
        f"min cos_sim={min_cos_sim:.6f}, "
        f"allclose(rtol={rtol}, atol={atol})={all_passed}",
    )
    if not all_passed:
        per_trial = ", ".join(f"{t['trial']}={t['max_abs_error']:.2e}" for t in trials if not t["allclose"])
        logger.warning(
            "Output mismatch exceeds tolerance in: %s. "
            "Consider increasing rtol/atol or checking for non-deterministic ops.",
            per_trial,
        )

    return {
        "max_abs_error": worst_max_abs_error,
        "cos_sim": min_cos_sim,
        "allclose": all_passed,
        "trials": trials,
    }
