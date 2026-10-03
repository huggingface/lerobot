# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""A development install must not hide coupling between optional transports."""

import subprocess
import sys
import textwrap


def _without_grpc(code: str) -> subprocess.CompletedProcess[str]:
    # A fresh interpreter also avoids previously cached optional-dependency flags.
    return subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['grpc'] = None\n" + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_remote_modules_import_without_grpc() -> None:
    result = _without_grpc("""
        from lerobot.transport.zenoh import ZenohConfig, ZenohTransport
        from lerobot.remote_inference.client import RemoteClient
        from lerobot.remote_inference.configs import ServerConfig
        from lerobot.scripts.lerobot_policy_server import main

        assert 'lerobot.transport.services_pb2' not in sys.modules
        assert 'lerobot.transport.services_pb2_grpc' not in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_rl_transport_retains_its_optional_dependency_hint() -> None:
    result = _without_grpc("""
        try:
            from lerobot.transport import utils
        except ImportError as exc:
            assert "'grpcio' is required" in str(exc), str(exc)
            assert 'lerobot[grpcio-dep]' in str(exc), str(exc)
        else:
            raise AssertionError('RL transport should require its own gRPC extra')
    """)
    assert result.returncode == 0, result.stderr
