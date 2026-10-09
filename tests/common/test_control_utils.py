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

from types import SimpleNamespace

import pytest

from lerobot.common.control_utils import sanity_check_dataset_name


def test_local_repo_id_without_slash_is_valid():
    sanity_check_dataset_name("test_20260918_090318", None)


def test_hub_repo_id_is_valid():
    sanity_check_dataset_name("user/my_dataset", None)


def test_eval_prefix_requires_policy_for_local_and_hub_ids():
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("eval_my_dataset", None)
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("user/eval_my_dataset", None)


def test_policy_requires_eval_prefix_for_local_and_hub_ids():
    policy = SimpleNamespace(type="act")
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("my_dataset", policy)
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("user/my_dataset", policy)
    sanity_check_dataset_name("eval_my_dataset", policy)
    sanity_check_dataset_name("user/eval_my_dataset", policy)
