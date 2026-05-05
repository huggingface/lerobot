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
"""Single-pass runner backed by a single forward pass.

Used by feedforward chunk-emitting policies such as ACT. The policy exposes a
single ``"model"`` stage and emits a contiguous ``[chunk_size, action_dim]``
action tensor per call while exporting the converged manifest type
``"single_pass"``.

Example::

    from lerobot.export import load_exported_policy

    policy = load_exported_policy("act_package", backend="onnx")
    actions = policy.predict_action_chunk(observation)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..interfaces import _RuntimeSession
from ..protocols import Exportable, is_exportable
from .base import ExportModule, build_dynamic_axes, get_output_by_names, register_runner

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

__all__ = ["SinglePassRunner", "policy_as_exportable"]

DEFAULT_ACTION_CHUNK_SIZE: int = 50
# Matches the common action chunk length used by ACT-style exported policies.


@register_runner
class SinglePassRunner:
    """Runner for feedforward chunk-emitting policies (e.g. ACT).

    Implements the ``"single_pass"`` manifest runner type.  A single
    forward pass through the ``"model"`` stage produces the full action chunk
    ``[chunk_size, action_dim]``.
    """

    type: ClassVar[str] = "single_pass"
    inference_type: ClassVar[str] = "single_pass"

    def __init__(self, manifest: dict[str, Any], artifacts_dir: Path, runtime_session: _RuntimeSession):
        """Initialise from a loaded manifest and backend session.

        Args:
            manifest: Parsed manifest dict.
            artifacts_dir: Directory containing the artifact files.
            runtime_session: Open runtime session for inference.
        """
        self._manifest = manifest
        self._runtime_session = runtime_session
        del artifacts_dir

    @classmethod
    def matches(cls, policy: object) -> bool:
        """Return ``True`` for policies that declare ``"single_pass"`` inference type.

        Args:
            policy: Policy instance to test.

        Returns:
            ``True`` when the policy implements :class:`Exportable` and its
            ``get_inference_type()`` returns ``"single_pass"``.
        """
        return is_exportable(policy) and policy.get_inference_type() == cls.inference_type

    @classmethod
    def export(
        cls,
        policy: object,
        example_batch: dict[str, Tensor],
    ) -> tuple[list[ExportModule], dict[str, Any]]:
        """Produce a single ``"model"`` export module and runner config.

        Args:
            policy: Policy implementing the Exportable protocol.
            example_batch: Representative input batch for tracing.

        Returns:
            A 2-tuple of ``([model_module], runner_cfg)`` where ``runner_cfg``
            contains ``chunk_size``, ``n_action_steps``, and ``action_dim``.
        """
        exportable = policy_as_exportable(policy)
        export_config = exportable.get_export_config()
        modules = exportable.get_export_modules()
        stage = exportable.prepare_inputs(example_batch)["model"]
        export_module = ExportModule(
            name="model",
            wrapper=modules["model"],
            example_inputs=stage.tensors,
            input_names=stage.input_names,
            output_names=["action"],
            dynamic_axes=build_dynamic_axes(stage.input_names, ["action"]),
        )
        runner_cfg = {
            "chunk_size": export_config.chunk_size,
            "n_action_steps": export_config.n_action_steps or export_config.chunk_size,
            "action_dim": export_config.action_dim,
        }
        return [export_module], runner_cfg

    @classmethod
    def load(
        cls,
        manifest: dict[str, Any],
        artifacts_dir: Path,
        runtime_session: _RuntimeSession,
    ) -> SinglePassRunner:
        """Instantiate from a loaded manifest and backend session.

        Args:
            manifest: Parsed manifest dict.
            artifacts_dir: Directory containing the artifact files.
            runtime_session: Open runtime session for inference.

        Returns:
            A ready-to-use :class:`SinglePassRunner`.
        """
        return cls(manifest, artifacts_dir, runtime_session)

    def run(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """Run a single forward pass and return the action chunk.

        Runs the ``"model"`` stage and squeezes a leading batch dimension of 1.

        Args:
            batch: Dict mapping observation key names to numpy arrays.

        Returns:
            Action array of shape ``(chunk_size, action_dim)``.
        """
        obs = {k: v.astype(np.float32) for k, v in batch.items()}

        outputs = self._runtime_session.run("model", obs)

        action = get_output_by_names(
            outputs,
            primary_name="action",
            fallback_names=[],
            context="SinglePassRunner",
        )

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        """No-op: single-pass runner has no stateful context to clear."""
        return None


def policy_as_exportable(policy: object) -> Exportable:
    """Cast a policy to :class:`~lerobot.export.protocols.Exportable`, raising on failure.

    Args:
        policy: Policy instance to cast.

    Returns:
        The same object typed as :class:`~lerobot.export.protocols.Exportable`.

    Raises:
        TypeError: If the policy does not satisfy the Exportable protocol.
    """
    if not is_exportable(policy):
        raise TypeError(f"{type(policy).__name__} does not implement Exportable Protocol.")
    return policy
