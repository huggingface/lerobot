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
Async inference server/client.

Requires: ``pip install 'lerobot[async]'``

Available modules (import directly)::

    from lerobot.async_inference.policy_server import ...
    from lerobot.async_inference.robot_client import ...
"""

from lerobot.utils.import_utils import require_package

require_package("grpcio", extra="async", import_name="grpc")

__all__: list[str] = []
