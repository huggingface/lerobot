from __future__ import annotations

import sys
from dataclasses import dataclass

from lerobot.configs.parser import wrap


@dataclass
class DummyConfig:
    a: int = 0


@wrap()
def _dummy_entrypoint(cfg: DummyConfig) -> int:
    # Return the parsed value for assertion
    return cfg.a


def test_wrap_resolves_forward_refs_and_parses_args(monkeypatch):
    # Simulate CLI args with a value for DummyConfig.a
    monkeypatch.setattr(sys, "argv", ["prog", "--a=5"])

    # Should not raise and should parse into the dataclass despite forward-referenced annotations
    result = _dummy_entrypoint()  # type: ignore[call-arg]
    assert result == 5
