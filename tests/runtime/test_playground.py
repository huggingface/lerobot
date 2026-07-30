#!/usr/bin/env python

import sys
import threading
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.runtime.playground import (
    LazyQwenPlanner,
    PlaygroundController,
    replace_observation_image_from_url,
    start_playground_server,
)


def test_controller_exposes_runtime_state_and_messages():
    state = SimpleNamespace(
        lock=threading.RLock(),
        get=lambda key, default=None: {
            "mode": "paused",
            "task": "close the fridge",
            "language_context": {"subtask": "reach for the handle"},
            "action_queue": [1, 2],
            "actions_dispatched": 7,
            "revision": 3,
        }.get(key, default),
    )
    controller = PlaygroundController(
        policy_path="lerobot/test-policy",
        blog_url="/blog/",
        planner=LazyQwenPlanner(),
    )
    controller.attach(SimpleNamespace(state=state), SimpleNamespace())
    controller.add_message("user", "What is open?")

    snapshot = controller.snapshot()

    assert snapshot["connected"] is True
    assert snapshot["state"]["task"] == "close the fridge"
    assert snapshot["state"]["queued_actions"] == 2
    assert snapshot["messages"][0]["text"] == "What is open?"
    assert snapshot["blog_url"] == "/blog/"
    assert snapshot["capabilities"]["planner"] == {
        "available": True,
        "model": "Qwen/Qwen3.5-2B",
        "loaded": False,
    }


def test_command_queue_round_trip():
    controller = PlaygroundController(policy_path="lerobot/test-policy")
    command = controller.enqueue("pause")

    assert controller.next_command() is command
    controller.finish(command, result={"mode": "paused"})

    assert command.completed.is_set()
    assert command.result == {"mode": "paused"}
    assert command.error is None


def test_server_serves_playground_and_state():
    controller = PlaygroundController(policy_path="lerobot/test-policy")
    server = start_playground_server(0, lambda: None, controller)
    assert server is not None
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:  # noqa: S310
            page = response.read().decode()
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2) as response:  # noqa: S310
            state = response.read().decode()
    finally:
        server.shutdown()

    assert "<title>LeRobot Playground</title>" in page
    assert '"policy_path":"lerobot/test-policy"' in state


def test_qwen_planner_loads_lazily_and_generates_subtask(monkeypatch):
    calls = {}

    class Inputs(dict):
        def to(self, device):
            calls["input_device"] = str(device)
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path):
            calls["processor_path"] = path
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return Inputs(input_ids=torch.zeros((1, 3), dtype=torch.long))

        def decode(self, tokens, **kwargs):
            calls["decoded_tokens"] = tokens
            calls["decode_kwargs"] = kwargs
            return "grasp the refrigerator handle"

    class Model:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = kwargs
            return cls()

        def to(self, device):
            self.device = torch.device(device)
            return self

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return torch.tensor([[0, 0, 0, 7]])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForMultimodalLM=Model, AutoProcessor=Processor),
    )
    planner = LazyQwenPlanner(device="cpu")

    answer = planner.plan(
        "close the fridge",
        {"observation.images.main": torch.zeros((1, 3, 8, 8))},
        system_prompt="Return the next subtask.",
    )

    assert answer == "grasp the refrigerator handle"
    assert planner.loaded is True
    assert calls["model_path"] == "Qwen/Qwen3.5-2B"
    assert calls["messages"][1]["content"][0]["url"].startswith("data:image/png;base64,")
    assert calls["generate_kwargs"]["max_new_tokens"] == 80


def test_remote_image_replaces_first_observation_image(monkeypatch):
    image = Image.new("RGB", (16, 8), (255, 0, 0))
    buffer = __import__("io").BytesIO()
    image.save(buffer, format="PNG")

    class Response:
        headers = {"Content-Type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return buffer.getvalue()

    monkeypatch.setattr(
        "lerobot.runtime.playground.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    monkeypatch.setattr(
        "lerobot.runtime.playground.build_opener",
        lambda *_args, **_kwargs: SimpleNamespace(open=lambda *_args, **_kwargs: Response()),
    )
    observation = {
        "observation.images.main": torch.zeros((1, 3, 4, 6)),
        "observation.state": torch.zeros((1, 7)),
    }

    result = replace_observation_image_from_url(observation, "https://example.com/frame.png")

    assert result is not observation
    assert result["observation.images.main"].shape == (1, 3, 4, 6)
    assert torch.allclose(result["observation.images.main"][:, 0], torch.ones((1, 4, 6)))
    assert torch.count_nonzero(result["observation.images.main"][:, 1:]) == 0
    assert result["observation.state"] is observation["observation.state"]


@pytest.mark.parametrize(
    "resolved_address",
    ["127.0.0.1", "10.0.0.2", "169.254.169.254", "::1"],
)
def test_remote_image_rejects_private_targets(monkeypatch, resolved_address):
    monkeypatch.setattr(
        "lerobot.runtime.playground.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, (resolved_address, 80))],
    )

    with pytest.raises(ValueError, match="public address"):
        replace_observation_image_from_url(
            {"observation.images.main": torch.zeros((1, 3, 4, 6))},
            "http://example.test/frame.png",
        )


def test_remote_image_supports_channel_last_observation(monkeypatch):
    image = Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8))
    buffer = __import__("io").BytesIO()
    image.save(buffer, format="PNG")

    class Response:
        headers = {"Content-Type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return buffer.getvalue()

    monkeypatch.setattr(
        "lerobot.runtime.playground.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    monkeypatch.setattr(
        "lerobot.runtime.playground.build_opener",
        lambda *_args, **_kwargs: SimpleNamespace(open=lambda *_args, **_kwargs: Response()),
    )

    result = replace_observation_image_from_url(
        {"observation.image": torch.zeros((8, 10, 3))},
        "https://example.com/frame.png",
    )

    assert result["observation.image"].shape == (8, 10, 3)
