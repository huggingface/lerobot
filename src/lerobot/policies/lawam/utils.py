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

from __future__ import annotations


def sync_managed_modules_training_mode(*modules, mode: bool):
    """Synchronize nested train/eval modes without pulling in LaWAM trainer utilities.

    Frozen submodules stay in eval mode during training, matching the relevant
    behavior of LaWAM's original helper while keeping this policy self-contained.
    """

    def _has_any_parameters(module) -> bool:
        return any(True for _ in module.parameters())

    def _has_trainable_parameters(module) -> bool:
        return any(param.requires_grad for param in module.parameters())

    def _sync(module, inherited_mode: bool) -> None:
        if _has_any_parameters(module):
            current_mode = bool(mode and _has_trainable_parameters(module))
        else:
            current_mode = inherited_mode
        module.train(current_mode)
        for child in module.children():
            _sync(child, current_mode)

    for module in modules:
        if module is not None:
            _sync(module, bool(mode))
    return modules
