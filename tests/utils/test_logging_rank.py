#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from types import SimpleNamespace

import pytest

from lerobot.utils.utils import init_logging


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    httpx_level = logging.getLogger("httpx").level
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)
        logging.getLogger("httpx").setLevel(httpx_level)


@pytest.mark.parametrize("is_main", [False, True])
@pytest.mark.parametrize("file_level", ["DEBUG", "WARNING"])
def test_rank_file_threshold_and_console(is_main, file_level, tmp_path, capsys):
    path = tmp_path / "rank.log"
    init_logging(log_file=path, file_level=file_level, accelerator=SimpleNamespace(is_main_process=is_main))
    for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
        logging.log(level, "rank-message-%s", level)
    for handler in logging.getLogger().handlers:
        handler.flush()
    contents = path.read_text()
    console = capsys.readouterr().err
    for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
        message = f"rank-message-{level}"
        assert (message in contents) == (level >= getattr(logging, file_level))
        assert (message in console) == (is_main and level >= logging.INFO)


def test_non_main_without_file_stays_silent(capsys):
    init_logging(accelerator=SimpleNamespace(is_main_process=False))
    logging.warning("non-main-warning")
    logging.error("non-main-error")
    assert capsys.readouterr().err == ""
