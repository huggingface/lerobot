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
"""A policy as one step of a processor pipeline: preprocess, predict, postprocess."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, PolicyPrediction
from lerobot.utils.constants import ACTION, PREDICTION

from .converters import batch_to_transition, transition_to_batch
from .pipeline import DataProcessorPipeline, ProcessorStep

if TYPE_CHECKING:  # a runtime import would be circular: policies import this package
    from lerobot.policies.pretrained import PreTrainedPolicy


class PolicyProcessorStep(ProcessorStep):
    """Runs a policy's ``predict_action_chunk`` on the preprocessed batch.

    The policy returns ``action`` (its chunk, in model coordinates) and possibly a
    ``prediction``. The step keeps that chunk as ``prediction["action"]`` and starts
    ``action`` from a copy of it, which the postprocessor steps turn into the command to
    execute: the prediction is never modified after the policy returned it.

    Not serializable: pipelines saved with a checkpoint hold the processors around it.
    """

    def __init__(self, policy: PreTrainedPolicy) -> None:
        """Wrap ``policy``."""
        self.policy = policy

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Predict a chunk from the transition's observation and complementary data."""
        return batch_to_transition(self.predict(transition_to_batch(transition)))

    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        """The preprocessed batch with the policy's ``action`` and ``prediction`` added."""
        output = self.policy.predict_action_chunk(batch)
        action = output["action"]
        if not isinstance(action, torch.Tensor):
            raise TypeError(
                f"{type(self.policy).__name__}.predict_action_chunk must return an action tensor."
            )
        prediction = PolicyPrediction(**output.get("prediction") or {})
        prediction["action"] = action
        return {**batch, ACTION: action.clone(), PREDICTION: prediction}

    def reset(self) -> None:
        """Start a new episode."""
        self.policy.reset()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """The policy adds the action; observation features pass through."""
        return features


def make_inference_pipeline(
    preprocessor: DataProcessorPipeline,
    policy: PreTrainedPolicy,
    postprocessor: DataProcessorPipeline,
) -> DataProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """One pipeline from a raw batch to the command to execute: ``pre -> policy -> post``.

    Its output batch holds ``action`` (canonical, ready to execute) and ``prediction``
    (what the policy returned, ``prediction["action"]`` in model coordinates).
    ``reset()`` resets the processors and the policy.
    """
    return DataProcessorPipeline(
        steps=[*preprocessor.steps, PolicyProcessorStep(policy), *postprocessor.steps],
        name="inference",
        to_transition=preprocessor.to_transition,
        to_output=postprocessor.to_output,
    )
