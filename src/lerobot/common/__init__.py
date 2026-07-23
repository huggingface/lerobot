# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Cross-cutting modules that bridge multiple lerobot packages.

Unlike ``lerobot.utils`` (which must remain dependency-free), modules here
are allowed to import from ``lerobot.policies``, ``lerobot.processor``,
``lerobot.configs``, etc.  They are deliberately NOT re-exported from the
top-level ``lerobot`` package.

Available modules (import directly)::

    from lerobot.common.control_utils import predict_action, ...
    from lerobot.common.train_utils import save_checkpoint, ...
    from lerobot.common.wandb_utils import WandBLogger, ...
"""

__all__: list[str] = []
