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

import importlib

__all__ = ["submit_annotate_to_hf", "submit_to_hf"]


def __getattr__(name):
    """CPU processing jobs need neither training nor VLM imports."""
    modules = {"submit_annotate_to_hf": "annotate", "submit_to_hf": "hf"}
    if name not in modules:
        raise AttributeError(name)
    from lerobot.utils.import_utils import require_package

    require_package("datasets", extra="dataset")
    return getattr(importlib.import_module(f"lerobot.jobs.{modules[name]}"), name)
