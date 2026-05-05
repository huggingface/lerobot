#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor

from . import (
    backends as _backends,  # noqa: F401
    runners as _runners,  # noqa: F401
)
from ._package_utils import (
    build_hardware_config,
    generate_example_batch,
    save_policy_config,
)
from .backends import BACKENDS
from .manifest import Manifest, Metadata, ModelConfig, PolicyInfo, PolicySource
from .runners.base import RUNNERS, Runner

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)

DEFAULT_ONNX_OPSET: int = 17
# ONNX opset 17 matches ORT 1.16+ and supports the operators used by ACT/PI05 exports.

__all__ = ["export_policy"]


def _policy_class_path(policy: PreTrainedPolicy) -> str:
    policy_cls = type(policy)
    return f"{policy_cls.__module__}.{policy_cls.__qualname__}"


def export_policy(
    policy: PreTrainedPolicy,
    output_dir: str | Path,
    *,
    backend: str = "onnx",
    example_batch: dict[str, Tensor] | None = None,
    opset_version: int = DEFAULT_ONNX_OPSET,
    include_normalization: bool = True,
) -> Path:
    """Export a trained policy to a self-contained ``policy_package`` directory.

    Traces the policy's inference graph, serialises model artifacts via the
    chosen backend, bundles normalisation statistics and (for PI05) the
    tokenizer, then writes a ``manifest.json`` that fully describes the
    package for runtime loading.

    Args:
        policy: Trained policy instance implementing the
            :class:`~lerobot.export.protocols.Exportable` protocol.
        output_dir: Destination directory.  Created (including parents) if it
            does not exist.
        backend: Serialisation backend.  ``"onnx"`` (default) or
            ``"openvino"`` (runtime-only; serialises as ONNX).
        example_batch: Optional representative input batch used for tracing.
            When ``None`` a synthetic batch is generated automatically.
        opset_version: ONNX opset version passed to ``torch.onnx.export``.
        include_normalization: When ``True``, save normalisation statistics and
            add normalise/denormalise processor specs to the manifest.

    Returns:
        The resolved ``output_dir`` path.

    Raises:
        ValueError: If the backend is unknown or runtime-only, or if no runner
            matches the policy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    assets_dir = output_dir / "assets"
    artifacts_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    if example_batch is None:
        example_batch = generate_example_batch(policy)

    runner_cls = _select_runner(policy)
    modules, runner_cfg = runner_cls.export(policy, example_batch)
    serialization_backend = "onnx" if backend == "openvino" else backend
    backend_impl = BACKENDS.get(serialization_backend)
    if backend_impl is None:
        raise ValueError(f"Unknown backend: {backend!r}. Known: {sorted(BACKENDS) + ['openvino']}")
    if backend_impl.runtime_only:
        raise ValueError(f"Backend {serialization_backend!r} is runtime-only and cannot serialize a model.")
    artifacts = backend_impl.serialize(modules, artifacts_dir, opset_version=opset_version)
    export_assets = policy.export_assets(output_dir)
    stats_artifact = policy.export_stats(output_dir, include_normalization=include_normalization)
    preprocessors, postprocessors = policy.export_processor_specs(
        include_normalization=include_normalization and bool(stats_artifact),
        stats_artifact=stats_artifact,
        assets=export_assets,
    )

    save_policy_config(policy, assets_dir / "config.json")
    runner_block = {"type": runner_cls.type, **runner_cfg}
    manifest = Manifest(
        policy=PolicyInfo(
            name=getattr(policy, "name", policy.__class__.__name__.lower()),
            source=PolicySource(
                repo_id=getattr(policy.config, "repo_id", None),
                revision=getattr(policy.config, "revision", None),
                class_path=_policy_class_path(policy),
            ),
        ),
        model=ModelConfig(
            n_obs_steps=getattr(policy.config, "n_obs_steps", 1),
            runner=runner_block,
            artifacts=artifacts,
            preprocessors=preprocessors or None,
            postprocessors=postprocessors or None,
        ),
        hardware=build_hardware_config(policy),
        metadata=Metadata(created_at=datetime.now(UTC).isoformat(), created_by="lerobot.export"),
    )
    manifest.save(output_dir / "manifest.json")
    return output_dir


def _select_runner(policy: PreTrainedPolicy) -> type[Runner]:
    for runner_cls in RUNNERS:
        if runner_cls.matches(policy):
            return runner_cls
    known = ", ".join(r.type for r in RUNNERS)
    raise ValueError(
        f"No runner matches {type(policy).__name__}. Known runner types: {known}. "
        "Implement the Exportable protocol from lerobot.export.protocols "
        "(get_inference_type, get_export_config, get_export_modules, prepare_inputs)."
    )
