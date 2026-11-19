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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.processor.hil_processor import RewardClassifierProcessorStep
from lerobot.processor.pipeline import DataProcessorPipeline


@pytest.mark.parametrize("override_path", [None, "local/other-classifier"])
def test_reward_classifier_checkpoint_survives_pipeline_save_load(
    tmp_path: Path, override_path: str | None
) -> None:
    classifier = MagicMock()
    classifier.predict_reward.return_value = 1.0
    with patch(
        "lerobot.rewards.classifier.modeling_classifier.Classifier.from_pretrained", return_value=classifier
    ) as load_classifier:
        step = RewardClassifierProcessorStep(pretrained_path="local/classifier", success_reward=2.0)
        pipeline = DataProcessorPipeline(
            [step], name="reward", to_transition=identity_transition, to_output=identity_transition
        )
        transition = create_transition(observation={"observation.images.camera": torch.zeros(1, 3, 8, 8)})
        expected = pipeline(transition)
        pipeline.save_pretrained(tmp_path)

        overrides = (
            {"reward_classifier_processor": {"pretrained_path": override_path}}
            if override_path is not None
            else None
        )
        restored = DataProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="reward.json",
            overrides=overrides,
            to_transition=identity_transition,
            to_output=identity_transition,
        )
        actual = restored(transition)

    assert load_classifier.call_count == 2
    load_classifier.assert_called_with(override_path or "local/classifier")
    assert actual[TransitionKey.REWARD] == expected[TransitionKey.REWARD] == 2.0
    assert actual[TransitionKey.DONE] is expected[TransitionKey.DONE] is True


def test_reward_processor_without_checkpoint_remains_inactive(tmp_path: Path) -> None:
    pipeline = DataProcessorPipeline(
        [RewardClassifierProcessorStep()],
        name="reward",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    pipeline.save_pretrained(tmp_path)
    restored = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="reward.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    transition = create_transition(
        observation={"observation.images.camera": torch.zeros(1, 3, 8, 8)}, reward=0.25, done=True
    )

    assert restored(transition) == transition
