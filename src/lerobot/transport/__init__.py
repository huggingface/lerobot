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
Transport layer for async inference and low-latency IPC.

Includes:
1. gRPC transport layer for network / async inference.
2. Zero-Copy POSIX Shared Memory IPC for high-frequency sensor ingestion.
"""

from lerobot.transport.zerocopy_ipc import ZeroCopyDataset, is_zerocopy_available

__all__: list[str] = [
    "ZeroCopyDataset",
    "is_zerocopy_available",
]
