#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import subprocess
import sys


def test_streaming_core_imports_without_dataset_extra() -> None:
    code = """
import importlib.abc
import sys

class BlockDatasets(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "datasets" or fullname.startswith("datasets."):
            raise ModuleNotFoundError("blocked optional datasets dependency")
        return None

sys.meta_path.insert(0, BlockDatasets())
from lerobot.streaming.episode_video import EpisodeByteCache, ExactCoveragePool
from lerobot.streaming.mp4 import Mp4Index
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
