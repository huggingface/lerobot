# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Real direct-TCP integration tests: require loopback sockets, no router/models."""

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import pytest

pytest.importorskip("zenoh")
import zenoh

from lerobot.transport.zenoh import BoundedSubscriber, TransportError, ZenohConfig, ZenohTransport


@pytest.fixture
def transports():
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{port.getsockname()[1]}"
    with (
        ZenohTransport(ZenohConfig(listen_endpoints=[endpoint], max_payload_bytes=1024)) as server,
        ZenohTransport(ZenohConfig(connect_endpoints=[endpoint], max_payload_bytes=1024)) as client,
    ):
        yield server, client


def test_explicit_config_and_disabled_discovery():
    config = ZenohConfig(listen_endpoints=["tcp/127.0.0.1:7447"]).build()
    assert json.loads(config.get_json("scouting/multicast/enabled")) is False
    assert json.loads(config.get_json("scouting/gossip/enabled")) is False
    assert json.loads(config.get_json("transport/shared_memory/enabled")) is False
    with pytest.raises(ValueError, match="Explicit"):
        ZenohConfig().build()
    with pytest.raises(ValueError, match="cannot listen"):
        ZenohConfig(
            mode="client", connect_endpoints=["tcp/127.0.0.1:7447"], listen_endpoints=["tcp/0.0.0.0:0"]
        ).build()


def test_bounded_handoff_keeps_accepted_work():
    channel = BoundedSubscriber(1)
    assert channel._offer(b"first")
    assert not channel._offer(b"second")
    assert channel.dropped == 1
    assert channel.get(0) == b"first"
    with pytest.raises(Empty):
        channel.get(0)


def test_query_reply_retained_outside_callback(transports):
    server, client = transports
    endpoint = server.declare_queryable("test/control")
    with ThreadPoolExecutor() as pool:
        reply = pool.submit(client.query, "test/control", b"reset", 2)
        pending = endpoint.get(2)
        assert pending.payload == b"reset"
        # The callback already returned; only the worker can acknowledge completion.
        assert not reply.done()
        assert pending.reply(b"applied")
        assert not pending.reply(b"duplicate")
        assert reply.result(2) == [b"applied"]


def test_direct_publish_and_presence_loss(transports):
    server, client = transports
    subscriber = client.subscribe("test/actions")
    presence = client.subscribe_liveliness("test/alive")
    token = server.declare_token("test/alive")
    assert presence.get(2).alive
    # Presence and pub/sub declarations can propagate independently.
    server.wait_for_subscriber("test/actions", 2)
    server.publish("test/actions", b"chunk")
    assert subscriber.get(2) == b"chunk"
    token.undeclare()
    assert not presence.get(2).alive
    with pytest.raises(TransportError, match="limit"):
        server.publish("test/actions", b"x" * 1025)


def test_query_deadline_and_late_reply(transports):
    server, client = transports
    endpoint = server.declare_queryable("test/slow", reply_timeout=0.04)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.query, "test/slow", b"work", 0.1)
        pending = endpoint.get(2)
        time.sleep(0.12)
        assert pending.reply(b"too late") is False
        try:
            result = future.result(2)
        except TimeoutError:
            pass  # local deadline may fire before the binding's expiry event
        else:
            assert result == []


def test_retained_queries_count_towards_capacity(transports):
    server, client = transports
    endpoint = server.declare_queryable("test/bounded", capacity=1)
    with ThreadPoolExecutor() as pool:
        first = pool.submit(client.query, "test/bounded", b"one", 2)
        pending = endpoint.get(2)
        second = pool.submit(client.query, "test/bounded", b"two", 0.2)
        assert second.result(2) == []
        assert endpoint.dropped == 1
        assert pending.reply(b"done")
        assert first.result(2) == [b"done"]


@pytest.fixture
def tls_router(tmp_path):
    """Run the documented mTLS/ACL example against an operator-provided real router."""
    import os
    import shutil
    import subprocess
    from pathlib import Path

    router = os.environ.get("LEROBOT_ZENOHD")
    if not router:
        pytest.skip("Set LEROBOT_ZENOHD to a supported zenohd binary for router/security validation")
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl is needed to create ephemeral test certificates")
    certs = tmp_path / "certs"
    certs.mkdir()

    def command(*args):
        subprocess.run([openssl, *map(str, args)], check=True, capture_output=True)

    command(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-sha256",
        "-days",
        "1",
        "-subj",
        "/CN=Test CA",
        "-keyout",
        certs / "ca-key.pem",
        "-out",
        certs / "ca.pem",
    )
    extension = certs / "extensions.cnf"
    extension.write_text("subjectAltName=DNS:localhost\n")
    for role in ("router", "policy-server", "robot-client", "outsider"):
        command(
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            f"/CN={role}",
            "-keyout",
            certs / f"{role}-key.pem",
            "-out",
            certs / f"{role}.csr",
        )
        command(
            "x509",
            "-req",
            "-in",
            certs / f"{role}.csr",
            "-CA",
            certs / "ca.pem",
            "-CAkey",
            certs / "ca-key.pem",
            "-CAcreateserial",
            "-days",
            "1",
            "-sha256",
            "-extfile",
            extension,
            "-out",
            certs / f"{role}.pem",
        )
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        port_number = port.getsockname()[1]
    endpoint = f"tls/localhost:{port_number}"
    examples = Path(__file__).parents[2] / "examples/remote_inference/zenoh"
    for name in ("router", "server", "robot"):
        contents = (examples / f"{name}.json5").read_text().replace("tls/localhost:7447", endpoint)
        (tmp_path / f"{name}.json5").write_text(contents.replace("./certs/", f"{certs}/"))
    (tmp_path / "outsider.json5").write_text(
        (tmp_path / "robot.json5").read_text().replace("robot-client", "outsider")
    )
    with (tmp_path / "router.log").open("w+") as log:
        process = subprocess.Popen([router, "-c", str(tmp_path / "router.json5")], stdout=log, stderr=log)
        try:
            ready = time.monotonic() + 5
            while time.monotonic() < ready:
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail(log.read())
                try:
                    with socket.create_connection(("localhost", port_number), timeout=0.05):
                        break
                except OSError:
                    time.sleep(0.02)
            else:
                pytest.fail("Router did not listen before startup deadline")
            yield endpoint, tmp_path, process
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_documented_mtls_router_acl(tls_router):
    endpoint, configs, router = tls_router
    prefix = "lerobot/inference/v1/deployments/manipulation"
    session = prefix + "/instances/test-instance/sessions/test-session"
    with (
        ZenohTransport(
            ZenohConfig(
                mode="client", connect_endpoints=[endpoint], config_file=str(configs / "server.json5")
            )
        ) as server,
        ZenohTransport(
            ZenohConfig(mode="client", connect_endpoints=[endpoint], config_file=str(configs / "robot.json5"))
        ) as robot,
        ZenohTransport(
            ZenohConfig(
                mode="client", connect_endpoints=[endpoint], config_file=str(configs / "outsider.json5")
            )
        ) as outsider,
        ZenohTransport(
            ZenohConfig(mode="client", connect_endpoints=[endpoint], config_file=str(configs / "robot.json5"))
        ) as robot_injector,
    ):
        descriptor = server.declare_queryable(prefix + "/describe")
        controls = server.declare_queryable(session + "/control")
        actions = robot.subscribe(session + "/act")
        observations = server.subscribe(session + "/obs")
        language = server.subscribe(session + "/language/request")
        answers = robot.subscribe(session + "/language/result")
        presence = robot.subscribe_liveliness(prefix + "/instances/test-instance/alive")
        token = server.declare_token(prefix + "/instances/test-instance/alive")
        assert presence.get(2).alive
        with ThreadPoolExecutor() as pool:
            future = pool.submit(robot.query, prefix + "/describe", b"describe", 2)
            request = descriptor.get(2)
            request.reply(b"descriptor")
            assert future.result(2) == [b"descriptor"]
            future = pool.submit(robot.query, session + "/control", b"reset", 2)
            request = controls.get(2)
            request.reply(b"applied")
            assert future.result(2) == [b"applied"]
        server.wait_for_subscriber(session + "/act", 2)
        robot.wait_for_subscriber(session + "/obs", 2)
        robot.wait_for_subscriber(session + "/language/request", 2)
        server.wait_for_subscriber(session + "/language/result", 2)
        robot.publish(session + "/obs", b"observation")
        assert observations.get(2) == b"observation"
        robot.publish(session + "/language/request", b"question")
        assert language.get(2) == b"question"
        server.publish(session + "/language/result", b"answer")
        assert answers.get(2) == b"answer"
        # Valid certificates do not authorize a robot or an unknown role to forge server actions.
        # A separate session routes through ACL. Same-session put/sub is local and
        # cannot be controlled by a router (nor defend a compromised robot process).
        robot_injector.publish(session + "/act", b"forged-by-robot")
        outsider.publish(session + "/act", b"forged-by-outsider")
        with pytest.raises(Empty):
            actions.get(0.1)
        server.publish(session + "/act", b"valid-action")
        assert actions.get(2) == b"valid-action"
        stolen = outsider.subscribe(session + "/obs")
        robot.publish(session + "/obs", b"private-observation")
        assert observations.get(2) == b"private-observation"
        with pytest.raises(Empty):
            stolen.get(0.1)
        with pytest.raises(TimeoutError):
            outsider.query(prefix + "/describe", b"unauthorized", 0.2)
        token.undeclare()
        assert not presence.get(2).alive
        token = server.declare_token(prefix + "/instances/test-instance/alive")
        assert presence.get(2).alive
        router.terminate()
        router.wait(timeout=5)
        assert not presence.get(3).alive


def test_mtls_router_rejects_missing_client_certificate(tls_router):
    endpoint, configs, _ = tls_router
    config_file = configs / "anonymous.json5"
    config_file.write_text(
        json.dumps(
            {
                "transport": {
                    "link": {
                        "protocols": ["tls"],
                        "tls": {
                            "root_ca_certificate": str(configs / "certs/ca.pem"),
                        },
                    }
                },
            }
        )
    )
    with (
        pytest.raises(zenoh.ZError),
        ZenohTransport(
            ZenohConfig(
                mode="client", connect_endpoints=[endpoint], config_file=str(config_file), open_timeout_s=0.3
            )
        ),
    ):
        pytest.fail("Router admitted an anonymous client despite mTLS")
