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
"""Build a TensorRT engine from an ONNX model using ``trtexec``."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# SM_87 = Jetson Orin Nano / Orin NX (no FP8 tensor cores; INT8 silently degrades)
_JETSON_ORIN_SM = 87


def get_gpu_compute_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability of the current CUDA device, or ``None``."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor)


def _get_sm_version() -> int | None:
    cc = get_gpu_compute_capability()
    if cc is None:
        return None
    return cc[0] * 10 + cc[1]


def _onnx_md5(onnx_path: Path) -> str:
    return hashlib.md5(onnx_path.read_bytes()).hexdigest()[:8]


def _cached_engine_path(onnx_path: Path, precision: str, sm: int, output_dir: Path) -> Path:
    stem = Path(onnx_path).stem
    md5 = _onnx_md5(Path(onnx_path))
    cache_dir = output_dir / ".trt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{stem}_{precision}_sm{sm}_{md5}.engine"


def export_to_tensorrt(
    onnx_path: Path | str,
    output_path: Path | str,
    precision: str = "fp16",
    workspace_gb: int = 4,
    force_rebuild: bool = False,
    calibration_data: Path | str | None = None,
) -> Path:
    """Build a TensorRT engine from an ONNX file using ``trtexec``.

    Args:
        onnx_path:        Path to the input ``.onnx`` file.
        output_path:      Directory where the ``.engine`` file (and cache) will be written.
        precision:        ``"fp32"``, ``"fp16"``, or ``"int8"``.
        workspace_gb:     TensorRT workspace memory limit in GB.
        force_rebuild:    Skip the engine cache and always rebuild.
        calibration_data: Path to calibration data directory (required for ``"int8"``).

    Returns:
        Path to the ``.engine`` file.

    Raises:
        RuntimeError:  If CUDA is not available.
        ValueError:    If an unsupported precision is requested for the detected GPU
                       (e.g. INT8/FP8 on SM_87 which is Jetson Orin Nano/NX).
        FileNotFoundError: If ``trtexec`` is not found on PATH.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_path)

    sm = _get_sm_version()
    if sm is None:
        raise RuntimeError(
            "CUDA is not available. TensorRT export requires a CUDA-capable GPU."
        )

    # ── precision guard ──────────────────────────────────────────────────────
    if precision == "fp8":
        raise ValueError(
            f"precision='fp8' is not supported on SM_{sm}. "
            "SM_87 (Jetson Orin Nano/NX) has no FP8 tensor cores. Use fp16."
        )
    if precision == "int8":
        if sm == _JETSON_ORIN_SM:
            raise ValueError(
                f"precision='int8' is not supported on SM_{sm} (Jetson Orin Nano/NX): "
                "the device has no INT8 tensor cores and TRT will silently fall back or fail. "
                "Use fp16 instead."
            )
        if calibration_data is None:
            raise ValueError(
                "precision='int8' requires calibration data. "
                "Provide a path via --calibration-data=<path>."
            )
    if precision == "fp16" and sm < 80:
        logger.warning(
            f"FP16 TensorRT has limited acceleration on SM_{sm} (SM_80+ recommended for FP16). "
            "Proceeding anyway — consider precision='fp32' if you see accuracy issues."
        )

    # ── engine cache ─────────────────────────────────────────────────────────
    engine_path = _cached_engine_path(onnx_path, precision, sm, output_dir)
    if engine_path.exists() and not force_rebuild:
        logger.info(f"Using cached TensorRT engine: {engine_path}")
        return engine_path

    # ── check trtexec ────────────────────────────────────────────────────────
    if shutil.which("trtexec") is None:
        raise FileNotFoundError(
            "'trtexec' not found on PATH. "
            "Install TensorRT and ensure 'trtexec' is available "
            "(usually in /usr/src/tensorrt/bin/ or TensorRT/bin/)."
        )

    # ── build command ─────────────────────────────────────────────────────────
    cmd: list[str] = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_gb * 1024}MiB",
        "--noTF32",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.extend(["--int8", f"--calib={calibration_data}"])
    # else: fp32 — no extra flag needed

    logger.info(f"Building TensorRT engine: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit code {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    logger.info(f"TensorRT engine written to: {engine_path}")
    return engine_path
