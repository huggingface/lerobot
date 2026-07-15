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

"""PI052 bridge to the generic language-conditioned runtime.

The runtime, REPL, and CLI are policy-agnostic and live in
:mod:`lerobot.runtime`. PI052 supplies only :class:`PI052PolicyAdapter`;
the ``lerobot-rollout --language`` entry point wires it into
:func:`lerobot.runtime.cli.run`.
"""

from .pi052_adapter import PI052PolicyAdapter

__all__ = ["PI052PolicyAdapter"]
