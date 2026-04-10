#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class LazyVectorEnv:
    """Defer vector-env construction until first usage.

    This is useful for benchmarks with many tasks: we can register one env object
    per task without eagerly allocating all simulator/rendering resources.
    """

    def __init__(self, env_cls: Callable[[Sequence[Callable[[], Any]]], Any], factory_fns: list[Callable]):
        self._env_cls = env_cls
        self._factory_fns = factory_fns
        self._env = None

    @property
    def env_cls(self) -> Callable[[Sequence[Callable[[], Any]]], Any]:
        return self._env_cls

    @property
    def factory_fns(self) -> list[Callable]:
        return self._factory_fns

    @property
    def num_factory_fns(self) -> int:
        return len(self._factory_fns)

    def materialize(self):
        if self._env is None:
            self._env = self._env_cls(self._factory_fns)
        return self._env

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getattr__(self, name):
        return getattr(self.materialize(), name)
