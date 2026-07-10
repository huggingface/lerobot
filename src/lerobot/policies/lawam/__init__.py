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

from typing import TYPE_CHECKING

from .configuration_lawam import LaWAMConfig
from .processor_lawam import make_lawam_pre_post_processors

if TYPE_CHECKING:
    from .modeling_lawam import LaWAMPolicy

__all__ = [
    "LaWAMConfig",
    "LaWAMPolicy",
    "make_lawam_pre_post_processors",
]


def __getattr__(name: str):
    if name == "LaWAMPolicy":
        from .modeling_lawam import LaWAMPolicy

        return LaWAMPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
