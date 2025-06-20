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

from dataclasses import dataclass

from lerobot.configs.policies.base import PolicyConfig


@dataclass
class RemotePolicyConfig(PolicyConfig):
    """
    Configuration for a policy that connects to a remote server to get actions.
    """

    # Inherits `name` from PolicyConfig, which will be 'remote'

    # The address of the policy server.
    server_address: str = "localhost"

    # The port of the policy server.
    server_port: int = 50051
