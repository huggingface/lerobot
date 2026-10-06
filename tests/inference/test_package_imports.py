# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Public inference APIs must not introduce policy cycles or rollout dependencies."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("first_package", ["policies", "inference", "remote_inference"])
def test_public_inference_imports_without_rollout_or_optional_transports(first_package: str) -> None:
    # A fresh interpreter exercises each import order and prevents optional
    # packages installed in the development environment from hiding coupling.
    code = f"""
        import importlib
        import sys

        for name in ("datasets", "grpc", "zenoh", "msgpack"):
            sys.modules[name] = None
        importlib.import_module("lerobot.{first_package}")

        from lerobot.policies import PreTrainedPolicy
        from lerobot.inference import (
            ChunkPolicySpec, InferenceEngine, PolicyRunner, RTCInferenceConfig,
            RTCInferenceEngine, RemoteInferenceConfig, SyncInferenceEngine,
        )
        if {first_package!r} != "remote_inference":
            assert "lerobot.remote_inference" not in sys.modules
        import lerobot.inference as inference
        import lerobot.remote_inference as remote
        from lerobot.remote_inference import ProtocolError, RemoteClient, RemoteInferenceEngine, PolicyServer

        assert issubclass(RemoteInferenceEngine, InferenceEngine)
        for package in (inference, remote):
            assert "__getattr__" not in vars(package)
            assert "_LAZY_EXPORTS" not in vars(package)
            assert all(name in vars(package) for name in package.__all__)

        assert issubclass(RTCInferenceEngine, InferenceEngine)
        assert issubclass(SyncInferenceEngine, InferenceEngine)
        assert "lerobot.rollout" not in sys.modules
        assert "lerobot.transport.services_pb2_grpc" not in sys.modules
        assert all(sys.modules.get(name) is None for name in ("datasets", "grpc", "zenoh", "msgpack"))
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
