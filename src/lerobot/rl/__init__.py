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
Reinforcement learning modules.

Requires: ``pip install 'lerobot[hilserl]'``

Available modules (import directly)::

    from lerobot.rl.actor import ...
    from lerobot.rl.learner import ...
    from lerobot.rl.learner_service import ...
    from lerobot.rl.buffer import ...
    from lerobot.rl.eval_policy import ...
    from lerobot.rl.gym_manipulator import ...
"""

from lerobot.utils.import_utils import require_package

require_package("grpcio", extra="hilserl", import_name="grpc")

__all__: list[str] = []
