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

from lerobot.utils.import_utils import require_package

# LeRobotDataset (imported at module top in dataset.py) pulls in heavy dataset deps;
# guard the optional dependency here so importing this package fails loudly if it's missing.
require_package("datasets", extra="dataset")

from .hf import submit_to_hf

__all__ = ["submit_to_hf"]
