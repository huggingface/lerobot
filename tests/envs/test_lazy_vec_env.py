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

from lerobot.envs.lazy_vec_env import LazyVectorEnv


class _DummyVectorEnv:
    def __init__(self):
        self.marker = "ok"
        self.closed = False

    def close(self):
        self.closed = True


def test_lazy_vec_env_materializes_only_on_access():
    created = []

    def _make_env(fns):
        created.append(len(fns))
        return _DummyVectorEnv()

    lazy = LazyVectorEnv(_make_env, [lambda: None, lambda: None])
    assert created == []
    assert lazy.num_factory_fns == 2

    assert lazy.marker == "ok"
    assert created == [2]

    # Second access should re-use the same materialized env.
    assert lazy.marker == "ok"
    assert created == [2]


def test_lazy_vec_env_can_rematerialize_after_close():
    created = []

    def _make_env(fns):
        created.append(len(fns))
        return _DummyVectorEnv()

    lazy = LazyVectorEnv(_make_env, [lambda: None])
    lazy.materialize()
    assert created == [1]

    lazy.close()
    lazy.materialize()
    assert created == [1, 1]

