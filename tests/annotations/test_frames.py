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
"""Unit tests for :class:`VideoFrameProvider` method bindings.

These were prompted by a real regression: ``video_for_episode`` was once
indented one level too deep so it ended up nested *inside* a module-level
helper (after that function's ``return`` statement) — silently dead code
that meant production runs with ``use_video_url=False`` would
``AttributeError`` on ``self.frame_provider.video_for_episode(...)``. The
existing module tests didn't catch it because they exercise stub providers.

The tests below assert on the class itself (not on an instance), so a
future reindent regression flips them to red without needing a real
LeRobot dataset on disk.
"""

from __future__ import annotations

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.annotations.steerable_pipeline.frames import (  # noqa: E402
    VideoFrameProvider,
)


def test_video_for_episode_is_a_method_of_videoframeprovider():
    """``video_for_episode`` must be a bound method, not nested dead code."""
    assert callable(getattr(VideoFrameProvider, "video_for_episode", None))


def test_episode_clip_path_is_a_method_of_videoframeprovider():
    """``episode_clip_path`` is now a method (was a free function reaching
    into ``provider._meta`` from outside the class)."""
    assert callable(getattr(VideoFrameProvider, "episode_clip_path", None))


def test_videoframeprovider_has_a_lock_for_concurrent_use():
    """A ``ThreadPoolExecutor`` runs the plan / interjections / vqa phases
    concurrently; the cache + warn-flag accesses must be guarded.
    """
    import threading

    # Fresh-instance check via a minimal fake to avoid touching the hub.
    # The lock is declared with ``init=False`` and has a default factory,
    # so a constructed instance must own a real ``threading.Lock``.
    lock_field = next(
        (f for f in VideoFrameProvider.__dataclass_fields__.values() if f.name == "_lock"),
        None,
    )
    assert lock_field is not None
    assert lock_field.default_factory is threading.Lock
