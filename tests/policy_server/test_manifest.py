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

"""Unit tests for the policy-server manifest (defaults + __post_init__ validation)."""

from __future__ import annotations

import textwrap

import draccus
import pytest

from lerobot.policy_server.manifest import (
    SERVING_MODE_AUTO,
    ModelSpec,
    PolicyServerManifest,
    ZenohSpec,
)


def _manifest(**overrides) -> PolicyServerManifest:
    kwargs: dict = {"model": ModelSpec(repo_or_path="mock/model")}
    kwargs.update(overrides)
    return PolicyServerManifest(**kwargs)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_parse():
    # The default zenoh mode is "client", which requires a router address;
    # peer mode is the minimal valid transport for a defaults-only manifest.
    manifest = _manifest(zenoh=ZenohSpec(mode="peer"))
    assert manifest.model.repo_or_path == "mock/model"
    assert manifest.model.revision == "main"
    assert manifest.model.device == "cuda"
    assert manifest.serving_mode == SERVING_MODE_AUTO
    assert manifest.max_sessions == 5
    assert manifest.warmup_inferences == 2
    assert manifest.trained_fps == 30.0
    assert manifest.strict_fps is False
    assert manifest.pin_task is False
    assert manifest.session_idle_timeout_s == 300.0
    assert manifest.health_port == 9100
    assert manifest.debug.capture_dir is None
    assert manifest.rtc.enabled is True
    # Bare ZenohSpec defaults (validated only when embedded in a manifest).
    assert ZenohSpec().mode == "client"
    assert ZenohSpec().connect_endpoints == []


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


def test_missing_repo_or_path_raises():
    with pytest.raises(ValueError, match="repo_or_path"):
        PolicyServerManifest()


def test_bad_serving_mode_raises():
    with pytest.raises(ValueError, match="serving_mode"):
        _manifest(serving_mode="multiplexed")


@pytest.mark.parametrize("max_sessions", [0, -3])
def test_max_sessions_below_one_raises(max_sessions):
    with pytest.raises(ValueError, match="max_sessions"):
        _manifest(max_sessions=max_sessions)


def test_zenoh_client_mode_requires_connect_endpoints():
    with pytest.raises(ValueError, match="connect_endpoints"):
        _manifest(zenoh=ZenohSpec(mode="client", connect_endpoints=[]))


def test_zenoh_client_mode_with_endpoints_ok():
    manifest = _manifest(zenoh=ZenohSpec(mode="client", connect_endpoints=["tcp/router:7447"]))
    assert manifest.zenoh.connect_endpoints == ["tcp/router:7447"]


def test_zenoh_peer_mode_without_endpoints_ok():
    manifest = _manifest(zenoh=ZenohSpec(mode="peer"))
    assert manifest.zenoh.mode == "peer"
    assert manifest.zenoh.connect_endpoints == []


def test_partial_tls_triple_raises():
    with pytest.raises(ValueError, match="TLS"):
        _manifest(zenoh=ZenohSpec(mode="peer", tls_root_ca_certificate="/certs/ca.pem"))


def test_full_tls_triple_ok():
    manifest = _manifest(
        zenoh=ZenohSpec(
            mode="peer",
            tls_root_ca_certificate="/certs/ca.pem",
            tls_connect_certificate="/certs/cert.pem",
            tls_connect_private_key="/certs/key.pem",
        )
    )
    assert manifest.zenoh.tls_connect_private_key == "/certs/key.pem"


# ---------------------------------------------------------------------------
# Draccus round-trip (YAML manifest → dataclass)
# ---------------------------------------------------------------------------


def test_draccus_yaml_round_trip(tmp_path):
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            model:
              repo_or_path: mock/model
              revision: v2.0
              device: cpu
            zenoh:
              mode: peer
              listen_endpoints:
                - tcp/127.0.0.1:7447
            default_task: pick the cube
            pin_task: true
            serving_mode: exclusive
            max_sessions: 1
            trained_fps: 25.0
            strict_fps: true
            health_port: 0
            """
        )
    )

    manifest = draccus.parse(PolicyServerManifest, config_path=str(yaml_path), args=[])

    assert isinstance(manifest, PolicyServerManifest)
    assert manifest.model.repo_or_path == "mock/model"
    assert manifest.model.revision == "v2.0"
    assert manifest.model.device == "cpu"
    assert manifest.zenoh.mode == "peer"
    assert manifest.zenoh.listen_endpoints == ["tcp/127.0.0.1:7447"]
    assert manifest.default_task == "pick the cube"
    assert manifest.pin_task is True
    assert manifest.serving_mode == "exclusive"
    assert manifest.max_sessions == 1
    assert manifest.trained_fps == 25.0
    assert manifest.strict_fps is True
    assert manifest.health_port == 0
    # Untouched fields keep their defaults.
    assert manifest.warmup_inferences == 2
    assert manifest.session_idle_timeout_s == 300.0


def test_draccus_cli_override_on_top_of_yaml(tmp_path):
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            model:
              repo_or_path: mock/model
              device: cpu
            zenoh:
              mode: peer
            """
        )
    )

    manifest = draccus.parse(
        PolicyServerManifest,
        config_path=str(yaml_path),
        args=["--max_sessions", "3", "--model.revision", "exp-1"],
    )

    assert manifest.max_sessions == 3
    assert manifest.model.revision == "exp-1"
    assert manifest.model.repo_or_path == "mock/model"
