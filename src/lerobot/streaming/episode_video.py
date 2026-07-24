# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


"""Compatibility imports for the split training-time episode streaming stack.

New internal code should import from `episode_cache`, `episode_pool`, `manifest`, or
`range_fetch` directly. This module keeps the original prototype import path working.
"""

from lerobot.streaming.episode_cache import EpisodeByteCache, open_video_decoder
from lerobot.streaming.episode_pool import ExactCoveragePool
from lerobot.streaming.manifest import EpisodeVideoManifest, EpisodeVideoSpan, VideoFileRecord
from lerobot.streaming.range_fetch import (
    NativeHTTPRangeFetcher,
    ThreadLocalRangeFetcher,
    _log_http_failure as _log_http_failure,
    make_range_fetcher,
)

__all__ = [
    "EpisodeByteCache",
    "EpisodeVideoManifest",
    "EpisodeVideoSpan",
    "ExactCoveragePool",
    "NativeHTTPRangeFetcher",
    "ThreadLocalRangeFetcher",
    "VideoFileRecord",
    "make_range_fetcher",
    "open_video_decoder",
]
