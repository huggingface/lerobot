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

from typing import Any

import pytest
import torch


@pytest.fixture
def tiny_classifier_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `Classifier` build on a tiny, randomly initialized HF ResNet instead of downloading one.

    Its overall stride is 8, so 32x32 images give the 4x4 feature map `SpatialLearnedEmbeddings` expects.
    """
    transformers = pytest.importorskip("transformers")

    def from_pretrained(*args: Any, **kwargs: Any) -> torch.nn.Module:
        # Same backbone weights every time, without resetting the global RNG of the test.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            config = transformers.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1])
            return transformers.ResNetModel(config)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", from_pretrained)
