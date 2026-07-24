#!/usr/bin/env python

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

"""Regression tests for `lerobot.utils.utils.init_logging`."""

import io
import logging

from lerobot.utils.utils import init_logging


def test_init_logging_exception_includes_traceback():
    """logging.exception must print a traceback after init_logging (issue #3978)."""
    stream = io.StringIO()
    # Re-init onto a captive stream: wire root logger like init_logging does.
    init_logging()
    root = logging.getLogger()
    # Replace console handlers with our StringIO so the test is hermetic.
    root.handlers.clear()
    handler = logging.StreamHandler(stream)
    # Re-apply the same formatter shape via init_logging on a temp file path then copy ,
    # simpler: call init_logging again is hard; mirror the product formatter by reusing init
    # through monkeypatch of StreamHandler.stdout is messy. Rebuild using init_logging + replace stream.

    init_logging()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.stream = stream

    try:
        raise ValueError("boom-for-logging-test")
    except ValueError:
        logging.exception("Unhandled exception in act_with_policy")

    out = stream.getvalue()
    assert "Unhandled exception in act_with_policy" in out
    assert "ValueError: boom-for-logging-test" in out
    assert "Traceback (most recent call last)" in out


def test_init_logging_plain_info_has_no_traceback():
    stream = io.StringIO()
    init_logging()
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.stream = stream
    logging.info("hello-no-exc")
    out = stream.getvalue()
    assert "hello-no-exc" in out
    assert "Traceback" not in out
