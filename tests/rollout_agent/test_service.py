# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import requests

from lerobot.rollout.agent.service import AgentHarness, make_server


def test_local_api_authentication_and_tool_discovery(tmp_path):
    harness = AgentHarness({"root": str(tmp_path), "workspace": str(Path(__file__).parents[2])})
    server = make_server(harness, 0, "test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert requests.get(url + "/tools", timeout=2).status_code == 401
        response = requests.get(url + "/tools", headers={"Authorization": "Bearer test-token"}, timeout=2)
        names = {tool["name"] for tool in response.json()}
        assert {"start_training", "move_ee", "build_dagger_dataset", "create_code_candidate"} <= names
        response = requests.post(
            url + "/call", headers={"Authorization": "Bearer test-token"}, json={"name": "status"}, timeout=2
        )
        assert response.json()["result"]["session"]["status"] == "idle"
    finally:
        harness.close()
        server.shutdown()
        server.server_close()


def test_stop_while_loading_never_starts_motion(tmp_path, monkeypatch):
    harness = AgentHarness(
        {"root": str(tmp_path), "workspace": str(Path(__file__).parents[2]), "robot_args": []}
    )
    harness.policies["test"] = {"checkpoint": "fake"}
    started, release = threading.Event(), threading.Event()
    robot_io = MagicMock()
    robot_io.limits.return_value = {"joint.pos": (-10, 10)}

    def construct(*args, **kwargs):
        Path(kwargs["root"]).mkdir(parents=True)
        started.set()
        assert release.wait(3)
        return robot_io

    monkeypatch.setattr("lerobot.rollout.agent.service.RealRobotIO", construct)
    harness.call("start_rollout", {"policy_id": "test", "instruction": "pick cup"})
    assert started.wait(1)
    stop = threading.Thread(target=lambda: harness.call("stop_rollout"))
    stop.start()
    deadline = time.monotonic() + 1
    while not harness._session_stop.is_set():
        assert time.monotonic() < deadline
        time.sleep(0.001)
    release.set()
    stop.join(3)
    assert not stop.is_alive()
    robot_io.execute.assert_not_called()
    robot_io.close.assert_called_once()
