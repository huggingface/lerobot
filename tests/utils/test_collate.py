#!/usr/bin/env python

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402


def test_lerobot_collate_preserves_messages_and_drops_raw_language():
    batch = [
        {
            "index": torch.tensor(0),
            "messages": [{"role": "assistant", "content": "a"}],
            "message_streams": ["low_level"],
            "target_message_indices": [0],
            "language_persistent": [{"content": "raw"}],
            "language_events": [],
        },
        {
            "index": torch.tensor(1),
            "messages": [{"role": "assistant", "content": "b"}],
            "message_streams": ["low_level"],
            "target_message_indices": [0],
            "language_persistent": [{"content": "raw"}],
            "language_events": [],
        },
    ]

    out = lerobot_collate_fn(batch)

    assert out["index"].tolist() == [0, 1]
    assert out["messages"][0][0]["content"] == "a"
    assert out["messages"][1][0]["content"] == "b"
    assert out["message_streams"] == [["low_level"], ["low_level"]]
    assert out["target_message_indices"] == [[0], [0]]
    assert "language_persistent" not in out
    assert "language_events" not in out
