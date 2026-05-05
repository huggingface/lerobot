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
"""KV-cache flow-matching runner: encode once, then iterate flow-matching denoise.

Used by VLA policies that combine a prefix encoder with cached KV attention
(e.g. PI05) and a flow-matching action head integrated with Euler steps. The
exported package contains two stages: ``encoder`` runs once per observation to
produce ``past_*`` KV tensors and a ``prefix_pad_mask``; ``denoise`` then runs
``num_inference_steps`` Euler integration steps with the cached prefix and an
evolving ``x_t``.

Future work: GR00T-style fused flow-matching policies (no prefix encoder, no
KV cache) are a planned sibling runner that would share an Euler-integration
mixin with this class.

Example::

    from lerobot.export import load_exported_policy

    policy = load_exported_policy("pi05_package", backend="onnx")
    actions = policy.predict_action_chunk(observation, num_steps=DEFAULT_DENOISE_STEPS)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..interfaces import _RuntimeSession
from ..protocols import is_exportable
from .base import ExportModule, build_dynamic_axes, get_output_by_names, register_runner
from .single_pass import DEFAULT_ACTION_CHUNK_SIZE, policy_as_exportable

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

__all__ = ["KVCacheFlowRunner"]

DEFAULT_DENOISE_STEPS: int = 10
# Ten denoise steps preserve the current flow-matching export/runtime behavior.


@register_runner
class KVCacheFlowRunner:
    """Runner for KV-cache flow-matching policies (e.g. PI05).

    Implements the ``"kv_cache_flow"`` manifest runner type.  The encoder
    stage runs once per observation to produce prefix KV tensors; the denoise
    stage then iterates ``num_inference_steps`` times using Euler integration
    to produce the action chunk.
    """

    type: ClassVar[str] = "kv_cache_flow"

    def __init__(self, manifest: dict[str, Any], artifacts_dir: Path, runtime_session: _RuntimeSession):
        """Initialise from a loaded manifest and backend session.

        Args:
            manifest: Parsed manifest dict.
            artifacts_dir: Directory containing the artifact files.
            runtime_session: Open runtime session exposing ``"encoder"`` and
                ``"denoise"`` stages.
        """
        self._manifest = manifest
        self._runtime_session = runtime_session
        del artifacts_dir

        runner_cfg = manifest["model"]["runner"]
        self._num_steps: int = runner_cfg.get("num_inference_steps", DEFAULT_DENOISE_STEPS)
        self._action_dim: int = runner_cfg["action_dim"]
        self._output_action_dim: int = runner_cfg.get("output_action_dim", self._action_dim)
        self._chunk_size: int = runner_cfg.get("chunk_size", DEFAULT_ACTION_CHUNK_SIZE)
        self._state_dim: int | None = runner_cfg.get("state_dim")
        self._input_mapping: dict[str, str] = runner_cfg.get("input_mapping", {})

    @classmethod
    def matches(cls, policy: object) -> bool:
        """Return ``True`` for policies that declare ``"kv_cache_flow"`` inference type.

        Args:
            policy: Policy instance to test.

        Returns:
            ``True`` when the policy implements :class:`Exportable` and its
            ``get_inference_type()`` returns ``"kv_cache_flow"``.
        """
        return is_exportable(policy) and policy.get_inference_type() == cls.type

    @classmethod
    def export(
        cls,
        policy: object,
        example_batch: dict[str, Tensor],
    ) -> tuple[list[ExportModule], dict[str, Any]]:
        """Produce ``"encoder"`` and ``"denoise"`` export modules and runner config.

        Runs the encoder once with the example batch to determine the prefix
        length, then constructs example inputs for the denoise stage.

        Args:
            policy: Policy implementing the Exportable protocol.
            example_batch: Representative input batch for tracing.

        Returns:
            A 2-tuple of ``([encoder_module, denoise_module], runner_cfg)``
            where ``runner_cfg`` contains KV-cache dimensions and scheduler
            parameters.
        """
        exportable = policy_as_exportable(policy)
        policy_obj: Any = exportable
        export_config = exportable.get_export_config()
        modules = exportable.get_export_modules()
        inputs_by_stage = exportable.prepare_inputs(example_batch)
        encoder_stage = inputs_by_stage["encoder"]
        encoder_wrapper = modules["encoder"]

        import torch

        with torch.no_grad():
            encoder_outputs = encoder_wrapper(*encoder_stage.tensors)
            prefix_len = encoder_outputs[0].shape[1]

        device = next(policy_obj.parameters()).device
        denoise_stage = exportable.prepare_runtime_inputs(
            "denoise",
            {"prefix_len": prefix_len, "device": device},
        )

        encoder_module = ExportModule(
            name="encoder",
            wrapper=encoder_wrapper,
            example_inputs=encoder_stage.tensors,
            input_names=encoder_stage.input_names,
            output_names=encoder_stage.output_names,
            dynamic_axes=build_dynamic_axes(encoder_stage.input_names, encoder_stage.output_names),
        )
        denoise_module = ExportModule(
            name="denoise",
            wrapper=modules["denoise"],
            example_inputs=denoise_stage.tensors,
            input_names=denoise_stage.input_names,
            output_names=denoise_stage.output_names,
            dynamic_axes=build_dynamic_axes(denoise_stage.input_names, denoise_stage.output_names),
            hints={
                "onnx_fixups": ["scatter_gather_dtypes"],
                # Reserved for deferred non-ONNX backend IO-spec follow-up work.
                "runtime_io_spec_extras": {"stage": "denoise"},
            },
        )

        runner_cfg = {
            "num_inference_steps": export_config.num_steps,
            "scheduler": "euler",
            "action_dim": export_config.action_dim,
            "output_action_dim": policy_obj.config.output_features["action"].shape[0],
            "chunk_size": export_config.chunk_size,
            "n_action_steps": policy_obj.config.n_action_steps,
            "num_layers": export_config.num_layers,
            "num_kv_heads": export_config.num_kv_heads,
            "head_dim": export_config.head_dim,
            "input_mapping": encoder_stage.metadata["input_mapping"],
            "state_dim": export_config.state_dim,
        }
        return [encoder_module, denoise_module], runner_cfg

    @classmethod
    def load(
        cls,
        manifest: dict[str, Any],
        artifacts_dir: Path,
        runtime_session: _RuntimeSession,
    ) -> KVCacheFlowRunner:
        """Instantiate from a loaded manifest and backend session.

        Args:
            manifest: Parsed manifest dict.
            artifacts_dir: Directory containing the artifact files.
            runtime_session: Open runtime session exposing ``"encoder"`` and
                ``"denoise"`` stages.

        Returns:
            A ready-to-use :class:`KVCacheFlowRunner`.
        """
        return cls(manifest, artifacts_dir, runtime_session)

    def run(
        self,
        batch: dict[str, np.ndarray],
        num_steps: int | None = None,
        noise: np.ndarray | None = None,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run encoder + iterative denoise loop and return the action chunk.

        Encodes the observation once, then runs ``num_steps`` Euler integration
        steps through the denoise stage to produce the action chunk.

        Args:
            batch: Dict mapping observation key names to numpy arrays.
            num_steps: Number of denoising steps.  Defaults to the value
                recorded in the manifest (``num_inference_steps``).
            noise: Optional initial noise tensor of shape
                ``(batch, chunk_size, action_dim)``.  When ``None`` noise is
                sampled from the standard normal distribution.
            generator: Optional numpy random generator used to sample noise
                when ``noise`` is ``None``.

        Returns:
            Action array of shape ``(chunk_size, output_action_dim)``.
        """
        num_steps = num_steps or self._num_steps

        obs = dict(batch)

        if self._input_mapping:
            mapped: dict[str, np.ndarray] = {}
            for obs_key, value in obs.items():
                onnx_key = self._input_mapping.get(obs_key, obs_key)
                mapped[onnx_key] = value
            obs = mapped

        obs = {
            key: _coerce_runtime_input(key, value) for key, value in obs.items() if hasattr(value, "astype")
        }

        first_obs = next(iter(obs.values()))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1

        num_images = sum(1 for k in obs if k.startswith("image_"))
        for i in range(num_images):
            mask_key = f"img_mask_{i}"
            if mask_key not in obs:
                obs[mask_key] = np.ones((batch_size,), dtype=np.float32)

        encoder_outputs = self._runtime_session.run("encoder", obs)

        prefix_pad_mask = encoder_outputs.get("prefix_pad_mask")
        if prefix_pad_mask is None:
            prefix_pad_mask = next(iter(encoder_outputs.values()))

        kv_cache = {k: v for k, v in encoder_outputs.items() if k.startswith("past_")}

        action_shape = (batch_size, self._chunk_size, self._action_dim)
        if noise is not None:
            x_t = noise.astype(np.float32)
        elif generator is not None:
            x_t = generator.standard_normal(action_shape).astype(np.float32)
        else:
            x_t = np.random.randn(*action_shape).astype(np.float32)

        dt = -1.0 / num_steps

        for step in range(num_steps):
            t = 1.0 + step * dt
            timestep = np.full((batch_size,), t, dtype=np.float32)

            denoise_inputs: dict[str, np.ndarray] = {
                "x_t": x_t,
                "timestep": timestep,
                "prefix_pad_mask": prefix_pad_mask,
                **kv_cache,
            }

            if "state" in obs:
                denoise_inputs["state"] = obs["state"]

            outputs = self._runtime_session.run("denoise", denoise_inputs)

            v_t = get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity"],
                context="KVCacheFlowRunner.denoise",
            )

            x_t = x_t + dt * v_t

        action = x_t

        if action.shape[-1] > self._output_action_dim:
            action = action[..., : self._output_action_dim]

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        """No-op: KV-cache runner has no persistent state between episodes."""
        return None


def _coerce_runtime_input(key: str, value: np.ndarray) -> np.ndarray:
    if key == "lang_tokens":
        return value.astype(np.int64)
    return value.astype(np.float32)
