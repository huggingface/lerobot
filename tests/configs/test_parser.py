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

import sys
from dataclasses import dataclass

import pytest

from lerobot.configs.parser import wrap


def test_wrap_invalid_typed_value_exits_cleanly(capsys):
    @dataclass
    class Config:
        n_episodes: int = 1

    @wrap()
    def dummy_func(cfg: Config):
        return cfg

    sys.argv = ["dummy_script.py", "--n_episodes=abc"]

    with pytest.raises(SystemExit) as exc_info:
        dummy_func()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert "n_episodes" in captured.err
    assert "abc" in captured.err
    # should be a single concise error line, not a multi-frame dump
    assert len(captured.err.strip().splitlines()) == 1
