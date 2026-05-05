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

"""Runner registry package.

Auto-discovers every sibling module so that any new file dropped in this
directory that decorates a class with ``@register_runner`` is registered
without editing this file.
"""

import importlib
import pkgutil

from .base import RUNNERS, Runner, register_runner

# Import every sibling module so its @register_runner decorators run on import.
for _module_info in pkgutil.iter_modules(__path__):
    _name = _module_info.name
    if _name.startswith("_") or _name == "base":
        continue
    importlib.import_module(f"{__name__}.{_name}")

__all__ = ["RUNNERS", "Runner", "register_runner"]
