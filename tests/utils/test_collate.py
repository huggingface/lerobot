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


def test_lerobot_collate_passes_through_standard_batch():
    """On a non-language batch, the collate must match ``default_collate``.

    Guards against silent regressions: ``lerobot_train.py`` only opts into
    ``lerobot_collate_fn`` when the dataset declares language columns, but
    if a future change ever wires it in unconditionally we want the
    behavior to remain a transparent pass-through for ordinary tensor
    batches.
    """
    from torch.utils.data._utils.collate import default_collate

    batch = [
        {
            "observation.image": torch.zeros(3, 4, 4),
            "action": torch.tensor([0.0, 1.0]),
            "index": torch.tensor(0),
        },
        {
            "observation.image": torch.ones(3, 4, 4),
            "action": torch.tensor([2.0, 3.0]),
            "index": torch.tensor(1),
        },
    ]

    custom = lerobot_collate_fn(batch)
    expected = default_collate(batch)

    assert custom.keys() == expected.keys()
    for key in expected:
        assert torch.equal(custom[key], expected[key]), f"key={key} diverged"


def test_lerobot_collate_drops_none_samples():
    """Recipes that yielded no target message return ``None`` — those samples
    must be filtered out, and an entirely-``None`` batch must collapse to ``None``.
    """
    batch = [None, {"index": torch.tensor(0)}, None]
    out = lerobot_collate_fn(batch)
    assert out is not None
    assert out["index"].tolist() == [0]

    assert lerobot_collate_fn([None, None]) is None
