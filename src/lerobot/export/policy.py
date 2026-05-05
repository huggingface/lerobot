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

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import backends as _backends  # noqa: F401
from .backends import BACKENDS
from .manifest import Manifest
from .normalize import Normalizer
from .processors import ExportProcessorPipeline, build_processor_pipeline
from .runners.base import RUNNERS, Runner

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ExportedPolicy"]


class ExportedPolicy:
    """Runtime wrapper around an exported policy package.

    Combines a :class:`~lerobot.export.runners.base.Runner` with the package
    :class:`~lerobot.export.manifest.Manifest` and provides a simple
    ``select_action`` / ``predict_action_chunk`` interface that mirrors the
    training-time ``PreTrainedPolicy`` API.
    """

    def __init__(
        self,
        runner: Runner,
        manifest: Manifest,
        preprocessor: ExportProcessorPipeline | None = None,
        postprocessor: ExportProcessorPipeline | None = None,
    ):
        """Initialise from a pre-built runner and manifest.

        Args:
            runner: Concrete runner instance responsible for inference.
            manifest: Parsed manifest describing the exported package.
        """
        self._runner = runner
        self._manifest = manifest
        self._preprocessor = preprocessor or ExportProcessorPipeline()
        self._postprocessor = postprocessor or ExportProcessorPipeline()
        self._action_queue: deque[NDArray[np.floating]] = deque()

    @classmethod
    def load(
        cls,
        package_path: str | Path,
        backend: str | None = None,
        device: str = "cpu",
    ) -> ExportedPolicy:
        """Load an exported policy package from disk.

        Reads ``manifest.json``, selects the appropriate runner and backend,
        opens the artifact sessions, and returns a ready-to-use
        :class:`ExportedPolicy`.

        Args:
            package_path: Path to the root of the exported policy package
                directory (must contain ``manifest.json`` and ``artifacts/``).
            backend: Override the backend to use for inference (e.g. ``"onnx"``,
                ``"openvino"``). When ``None`` the backend is auto-detected from
                the manifest and artifact file extensions.
            device: Target device string passed to the backend session
                (e.g. ``"cpu"``, ``"cuda:0"``).

        Returns:
            A fully initialised :class:`ExportedPolicy` ready for inference.

        Raises:
            ValueError: If the runner type or backend cannot be resolved.
        """
        package_path = Path(package_path)
        manifest = Manifest.load(package_path / "manifest.json")
        manifest_dict = manifest.to_dict()
        runner_type = manifest.model.runner["type"]
        runner_cls = next((runner for runner in RUNNERS if runner.type == runner_type), None)
        if runner_cls is None:
            raise ValueError(f"Unknown runner type in manifest: {runner_type!r}")

        artifacts_dir = package_path / "artifacts"
        backend_name = backend or _detect_backend_name(manifest_dict, artifacts_dir)
        backend_impl = BACKENDS.get(backend_name)
        if backend_impl is None:
            raise ValueError(f"Unknown backend: {backend_name!r}. Known: {sorted(BACKENDS)}")
        sessions = backend_impl.open(artifacts_dir, manifest_dict, device=device)
        runner = runner_cls.load(manifest_dict, artifacts_dir, sessions)
        normalizer = Normalizer.from_specs(
            manifest.model.preprocessors,
            manifest.model.postprocessors,
            package_path,
        )
        preprocessor, relative_processor = build_processor_pipeline(
            manifest.model.preprocessors,
            package_path=package_path,
            normalizer=normalizer,
        )
        postprocessor, _ = build_processor_pipeline(
            manifest.model.postprocessors,
            package_path=package_path,
            normalizer=normalizer,
            relative_processor=relative_processor,
        )
        return cls(runner, manifest, preprocessor, postprocessor)

    @property
    def manifest(self) -> Manifest:
        """The parsed manifest for this exported package."""
        return self._manifest

    def reset(self) -> None:
        """Clear the internal action queue and reset the runner state.

        Call this between episodes to ensure no stale actions are replayed.
        """
        self._action_queue.clear()
        self._preprocessor.reset()
        self._runner.reset()
        self._postprocessor.reset()

    def predict_action_chunk(
        self,
        observation: dict[str, NDArray[np.floating]],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Run a single forward pass and return the full action chunk.

        Args:
            observation: Dict mapping observation key names to numpy arrays.
                Keys and shapes must match those declared in the manifest.
            **kwargs: Additional keyword arguments forwarded to the runner
                (e.g. ``num_steps`` for KV-cache runners).

        Returns:
            Action array of shape ``(chunk_size, action_dim)`` or
            ``(batch, chunk_size, action_dim)`` depending on the runner.
        """
        processed = self._preprocessor(dict(observation))
        action = self._runner.run(processed, **kwargs)
        outputs = self._postprocessor({"action": action})
        return outputs["action"]

    def select_action(
        self,
        observation: dict[str, NDArray[np.floating]],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Return the next single action, buffering the rest of the chunk.

        On the first call (or after :meth:`reset`) a full action chunk is
        predicted and stored in an internal queue.  Subsequent calls dequeue
        one action at a time until the queue is exhausted, at which point a
        new chunk is predicted.

        Args:
            observation: Dict mapping observation key names to numpy arrays.
            **kwargs: Forwarded to :meth:`predict_action_chunk`.

        Returns:
            A 1-D numpy array of shape ``(action_dim,)`` representing the
            next action to execute.
        """
        if not self._action_queue:
            chunk = self.predict_action_chunk(observation, **kwargs)
            if chunk.ndim == 1:
                return chunk
            if chunk.ndim == 3:
                chunk = chunk[0]
            n_action_steps = self.manifest.model.runner.get("n_action_steps", len(chunk))
            for idx in range(min(n_action_steps, len(chunk))):
                self._action_queue.append(chunk[idx])
        return self._action_queue.popleft()


def _detect_backend_name(manifest: dict[str, Any], artifacts_dir: Path) -> str:
    """Infer the backend to use from the artifact file extensions on disk.

    The manifest does not record which backend produced (or should consume) the
    artifacts; selection is driven by file suffixes against the registered
    backends. Callers can bypass this inference by passing ``backend=...``
    explicitly to :func:`load_exported_policy`.
    """
    artifact_names = {Path(path).name for path in manifest["model"]["artifacts"].values()}
    candidates = [
        backend_name
        for backend_name, backend_impl in BACKENDS.items()
        if not backend_impl.runtime_only
        and any((artifacts_dir / name).suffix == backend_impl.extension for name in artifact_names)
    ]
    if len(candidates) == 1:
        return candidates[0]
    suffixes = sorted({(artifacts_dir / name).suffix for name in artifact_names})
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple backends match artifacts {sorted(artifact_names)} "
            f"(suffixes: {suffixes}): {candidates}. "
            "Pass backend=... explicitly to load_exported_policy()."
        )
    raise ValueError(
        f"Cannot detect backend for artifacts {sorted(artifact_names)} "
        f"(suffixes: {suffixes}). Registered backends: {sorted(BACKENDS)}. "
        "Pass backend=... explicitly to load_exported_policy()."
    )
