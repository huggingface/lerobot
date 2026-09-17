#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import pytest

pytest.importorskip("gym_hil")
pytest.importorskip("placo")

from lerobot.scripts.lerobot_train_rlt import train  # noqa: E402


def test_train_rlt_help_resolves_config_dataclass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lerobot-train-rlt", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        train()

    assert exc_info.value.code == 0
