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

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.export import export_policy, load_exported_policy
from lerobot.export.backends import BACKENDS, register_backend
from lerobot.export.backends.onnx import ONNXBackend
from lerobot.export.runners.base import RUNNERS, ExportModule, build_dynamic_axes, register_runner
from tests.export.conftest import create_act_policy_and_batch


@pytest.fixture
def restore_registries() -> None:
    runner_snapshot = list(RUNNERS)
    backend_snapshot = dict(BACKENDS)
    yield
    RUNNERS[:] = runner_snapshot
    BACKENDS.clear()
    BACKENDS.update(backend_snapshot)


def test_builtin_backends_registered() -> None:
    assert "onnx" in BACKENDS


def test_openvino_backend_is_runtime_only() -> None:
    assert BACKENDS["openvino"].runtime_only is True


def test_export_accepts_openvino_backend_alias(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("openvino")
    policy, batch = create_act_policy_and_batch()
    out = export_policy(policy, tmp_path / "openvino_export", backend="openvino", example_batch=batch)
    assert (out / "manifest.json").exists()
    assert any(out.glob("artifacts/*.onnx"))


def test_toy_runner_and_backend_work_without_core_edits(tmp_path: Path, restore_registries: None) -> None:
    class ToyConfig:
        n_obs_steps = 1
        repo_id = None
        revision = None
        max_action_dim = 1

    class AddOne(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    class ToyPolicy(nn.Module):
        name = "toy_policy"

        def __init__(self):
            super().__init__()
            self.config = ToyConfig()
            self.model = AddOne().eval()
            self.config.stats = None

        def export_assets(self, output_dir: Path) -> dict[str, str]:
            del output_dir
            return {}

        def export_stats(self, output_dir: Path, *, include_normalization: bool) -> str | None:
            del output_dir, include_normalization
            return None

        def export_processor_specs(
            self,
            *,
            include_normalization: bool,
            stats_artifact: str | None,
            assets: dict[str, str] | None = None,
        ) -> tuple[list, list]:
            del include_normalization, stats_artifact, assets
            return [], []

    class ToyRuntimeSession:
        def __init__(
            self, modules: dict[str, torch.jit.ScriptModule], io_specs: dict[str, dict[str, list[str]]]
        ):
            self._modules = modules
            self._io_specs = io_specs

        def run(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            io_spec = self._io_specs[name]
            ordered = [torch.from_numpy(inputs[input_name]) for input_name in io_spec["input_names"]]
            output = self._modules[name](*ordered)
            if not isinstance(output, tuple):
                output = (output,)
            return {
                output_name: value.detach().cpu().numpy()
                for output_name, value in zip(io_spec["output_names"], output, strict=True)
            }

    @register_backend
    class ToyBackend:
        name = "toy_backend"
        extension = ".pt"

        def serialize(
            self,
            modules: list[ExportModule],
            artifacts_dir: Path,
            **kwargs: Any,
        ) -> dict[str, str]:
            del kwargs
            artifacts: dict[str, str] = {}
            for module in modules:
                model_path = artifacts_dir / f"{module.name}{self.extension}"
                io_path = artifacts_dir / f"{module.name}_io.json"
                torch.jit.trace(module.wrapper, module.example_inputs).save(str(model_path))
                io_path.write_text(
                    json.dumps({"input_names": module.input_names, "output_names": module.output_names})
                )
                artifacts[module.name] = f"artifacts/{model_path.name}"
            return artifacts

        def open(
            self, artifacts_dir: Path, manifest: dict[str, Any], *, device: str = "cpu"
        ) -> ToyRuntimeSession:
            del device
            modules = {}
            io_specs = {}
            for name, relative_path in manifest["model"]["artifacts"].items():
                model_path = artifacts_dir / Path(relative_path).name
                modules[name] = torch.jit.load(str(model_path))
                io_specs[name] = json.loads((artifacts_dir / f"{name}_io.json").read_text())
            return ToyRuntimeSession(modules, io_specs)

    @register_runner
    class ToyRunner:
        type = "toy_runner"

        def __init__(self, runtime_session: ToyRuntimeSession):
            self._runtime_session = runtime_session

        @classmethod
        def matches(cls, policy: object) -> bool:
            return isinstance(policy, ToyPolicy)

        @classmethod
        def export(
            cls,
            policy: object,
            example_batch: dict[str, torch.Tensor],
        ) -> tuple[list[ExportModule], dict[str, Any]]:
            toy_policy = policy
            assert isinstance(toy_policy, ToyPolicy)
            module = ExportModule(
                name="toy",
                wrapper=toy_policy.model,
                example_inputs=(example_batch["x"],),
                input_names=["x"],
                output_names=["action"],
                dynamic_axes=build_dynamic_axes(["x"], ["action"]),
            )
            return [module], {"n_action_steps": 1}

        @classmethod
        def load(
            cls,
            manifest: dict[str, Any],
            artifacts_dir: Path,
            runtime_session: ToyRuntimeSession,
        ) -> ToyRunner:
            del manifest, artifacts_dir
            return cls(runtime_session)

        def run(self, batch: dict[str, np.ndarray]) -> np.ndarray:
            return self._runtime_session.run("toy", batch)["action"]

        def reset(self) -> None:
            return None

    policy = ToyPolicy().eval()
    policy.config.stats = {
        "x": {
            "mean": np.zeros((1,), dtype=np.float32),
            "std": np.ones((1,), dtype=np.float32),
            "min": -np.ones((1,), dtype=np.float32),
            "max": np.ones((1,), dtype=np.float32),
        },
        "action": {
            "mean": np.zeros((1,), dtype=np.float32),
            "std": np.ones((1,), dtype=np.float32),
            "min": -np.ones((1,), dtype=np.float32),
            "max": np.ones((1,), dtype=np.float32),
        },
    }
    batch = {"x": torch.tensor([[2.0]], dtype=torch.float32)}
    package_path = export_policy(policy, tmp_path / "toy_package", backend="toy_backend", example_batch=batch)

    assert (package_path / "artifacts" / "toy.pt").exists()

    runtime = load_exported_policy(package_path, backend="toy_backend", device="cpu")
    output = runtime.predict_action_chunk({"x": np.array([[2.0]], dtype=np.float32)})

    np.testing.assert_allclose(output, np.array([[3.0]], dtype=np.float32))


def test_onnx_backend_round_trip_identity_module(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    backend = ONNXBackend()
    module = ExportModule(
        name="toy",
        wrapper=nn.Identity().eval(),
        example_inputs=(torch.tensor([[1.0, 2.0]], dtype=torch.float32),),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=build_dynamic_axes(["x"], ["y"]),
    )

    artifacts = backend.serialize([module], tmp_path, opset_version=17)
    session = backend.open(tmp_path, {"model": {"artifacts": artifacts}}, device="cpu")
    outputs = session.run("toy", {"x": np.array([[4.0, 5.0]], dtype=np.float32)})

    np.testing.assert_allclose(outputs["y"], np.array([[4.0, 5.0]], dtype=np.float32))


def test_detect_backend_name_raises_for_unknown_suffix(tmp_path: Path) -> None:
    from lerobot.export.policy import _detect_backend_name

    artifact_path = tmp_path / "model.unknown_ext"
    artifact_path.write_bytes(b"")
    manifest = {"model": {"artifacts": {"model": "model.unknown_ext"}}}

    with pytest.raises(ValueError, match="Cannot detect backend"):
        _detect_backend_name(manifest, tmp_path)


def test_onnx_backend_serialize_raises_for_unknown_fixup(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    backend = ONNXBackend()
    module = ExportModule(
        name="toy",
        wrapper=nn.Identity().eval(),
        example_inputs=(torch.tensor([[1.0, 2.0]], dtype=torch.float32),),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=build_dynamic_axes(["x"], ["y"]),
        hints={"onnx_fixups": ["nonexistent_fixup"]},
    )

    with pytest.raises(ValueError, match="Unknown onnx fixup 'nonexistent_fixup'"):
        backend.serialize([module], tmp_path, opset_version=17)


def importlib_available(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def test_dropping_a_runner_file_auto_registers_without_init_edit(
    tmp_path: Path,
    restore_registries: None,
) -> None:
    """File-drop guarantee: a new module discoverable on the runners package path registers itself.

    Uses a temporary directory appended to ``runners.__path__`` rather than
    writing into the installed source tree, so the test is isolated and does
    not rely on the working copy being writable.
    """
    import importlib
    import sys

    import lerobot.export.runners as runners_pkg

    new_module_path = tmp_path / "dropin_runner_for_test.py"
    new_module_path.write_text(
        "from typing import ClassVar\n"
        "from lerobot.export.runners.base import register_runner\n"
        "\n"
        "@register_runner\n"
        "class DropinTestRunner:\n"
        "    type: ClassVar[str] = 'dropin_test_marker'\n"
        "\n"
        "    @classmethod\n"
        "    def matches(cls, policy: object) -> bool:\n"
        "        return False\n"
    )
    original_path = list(runners_pkg.__path__)
    runners_pkg.__path__.append(str(tmp_path))
    try:
        for mod_name in list(sys.modules):
            if mod_name.startswith("lerobot.export.runners"):
                del sys.modules[mod_name]
        reloaded = importlib.import_module("lerobot.export.runners")
        reloaded.__path__.append(str(tmp_path))
        import pkgutil

        for module_info in pkgutil.iter_modules(reloaded.__path__):
            if module_info.name == "dropin_runner_for_test":
                importlib.import_module(f"lerobot.export.runners.{module_info.name}")
        types = {r.type for r in reloaded.RUNNERS}
        assert "dropin_test_marker" in types, (
            f"Auto-discovery failed: dropin_test_marker not in {sorted(types)}"
        )
    finally:
        runners_pkg.__path__[:] = original_path
        for mod_name in list(sys.modules):
            if mod_name.startswith("lerobot.export.runners"):
                del sys.modules[mod_name]
        importlib.import_module("lerobot.export.runners")


def test_dropping_a_backend_file_auto_registers_without_init_edit(
    tmp_path: Path,
    restore_registries: None,
) -> None:
    """File-drop guarantee for backends/ - same temp-path approach as runners test above."""
    import importlib
    import sys

    import lerobot.export.backends as backends_pkg

    new_module_path = tmp_path / "dropin_backend_for_test.py"
    new_module_path.write_text(
        "from typing import ClassVar\n"
        "from lerobot.export.backends.base import register_backend\n"
        "\n"
        "@register_backend\n"
        "class DropinTestBackend:\n"
        "    name: ClassVar[str] = 'dropin_backend_marker'\n"
        "    extension: ClassVar[str] = '.dropin'\n"
        "    runtime_only: ClassVar[bool] = True\n"
        "\n"
        "    def serialize(self, modules, artifacts_dir, **kwargs):\n"
        "        return {}\n"
        "\n"
        "    def open(self, artifacts_dir, manifest, *, device='cpu'):\n"
        "        raise NotImplementedError\n"
    )
    original_path = list(backends_pkg.__path__)
    backends_pkg.__path__.append(str(tmp_path))
    try:
        for mod_name in list(sys.modules):
            if mod_name.startswith("lerobot.export.backends"):
                del sys.modules[mod_name]
        reloaded = importlib.import_module("lerobot.export.backends")
        reloaded.__path__.append(str(tmp_path))
        import pkgutil

        for module_info in pkgutil.iter_modules(reloaded.__path__):
            if module_info.name == "dropin_backend_for_test":
                importlib.import_module(f"lerobot.export.backends.{module_info.name}")
        assert "dropin_backend_marker" in reloaded.BACKENDS, (
            f"Auto-discovery failed: dropin_backend_marker not in {sorted(reloaded.BACKENDS)}"
        )
    finally:
        backends_pkg.__path__[:] = original_path
        for mod_name in list(sys.modules):
            if mod_name.startswith("lerobot.export.backends"):
                del sys.modules[mod_name]
        importlib.import_module("lerobot.export.backends")
