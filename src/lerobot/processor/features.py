#!/usr/bin/env python

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

"""Cross-cutting pipeline features: steps that belong to no particular policy.

Some processing is orthogonal to the model. Converting actions to be relative to the current state
needs nothing but `observation.state` and `action`, so it should work with ACT without ACT knowing
what a relative action is. Before this existed, each policy that wanted the behaviour hand-placed the
steps and re-declared the config flags, which is why the same five-line block appeared in four
policies.

A feature declares what it builds and where the steps attach; `apply_policy_features` composes them
into whatever pipeline the policy's factory produced, which is why this reaches every policy —
including third-party plugins — without any of them referring to the feature.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from .pipeline import ProcessorStep
from .relative_action_processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig

    from .context import ProcessorBuildContext


@dataclass(frozen=True)
class AnchoredStep:
    """A step plus where it attaches, expressed relative to another step's class.

    Anchoring by class rather than by index keeps a feature working across policies that order their
    pipelines differently. Resolution requires *exactly one* match: a feature that cannot be placed
    unambiguously is an error, never a silent fallback to position 0.
    """

    step: ProcessorStep
    anchor: type[ProcessorStep]
    position: Literal["before", "after"]


class ProcessorFeature(ABC):
    """A policy-independent addition to a policy's pre/post-processor pipelines.

    Subclasses declare `owner`, which says where their parameters come from — and, by extension,
    whether they persist with the checkpoint:

    - ``"config"`` — training-semantic. The parameters live on `PreTrainedConfig` and are saved in
      ``config.json``, because training with the feature changes what the model learned, so eval
      **must** match. Relative actions are this kind.
    - ``"context"`` — deployment or embodiment. The parameters come from `ProcessorBuildContext` and
      are deliberately not persisted, because the same checkpoint runs on different hardware. The
      end-effector kinematics steps in ``lerobot.robots.so_follower.robot_kinematic_processor`` are
      this kind; they are not wired up as a feature yet (they carry a live solver object and no
      ``get_config``), but the distinction is what they will need.
    """

    name: ClassVar[str]
    owner: ClassVar[Literal["config", "context"]]

    @abstractmethod
    def enabled_for(self, config: PreTrainedConfig, context: ProcessorBuildContext) -> bool:
        """Whether this feature should be composed into the pipelines at all."""

    @abstractmethod
    def build(
        self, config: PreTrainedConfig, context: ProcessorBuildContext
    ) -> tuple[list[AnchoredStep], list[AnchoredStep]]:
        """Return the (preprocessor, postprocessor) steps to splice in, with their anchors."""


class RelativeActionsFeature(ProcessorFeature):
    """Express actions relative to the current state, for any policy.

    The pair is built together so `AbsoluteActionsProcessorStep` gets a live reference to the
    preprocessor's step. That reference is not serializable, which used to require re-linking the two
    after every checkpoint load; building both here means there is nothing to re-link.
    """

    name = "relative_actions"
    owner = "config"

    def enabled_for(self, config: PreTrainedConfig, context: ProcessorBuildContext) -> bool:
        return bool(config.use_relative_actions)

    def build(
        self, config: PreTrainedConfig, context: ProcessorBuildContext
    ) -> tuple[list[AnchoredStep], list[AnchoredStep]]:
        # Imported here: these live in the same package, and importing at module scope would make
        # `processor.factory` -> `processor.features` -> `processor.normalize_processor` a cycle.
        from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

        relative_step = RelativeActionsProcessorStep(
            enabled=True,
            exclude_joints=list(config.relative_exclude_joints or []),
            action_names=config.action_feature_names,
        )
        absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
        return (
            # Relative offsets must be computed on raw values, so this goes before normalization.
            [AnchoredStep(relative_step, NormalizerProcessorStep, "before")],
            # ...and reversed after unnormalization, back in the same raw units.
            [AnchoredStep(absolute_step, UnnormalizerProcessorStep, "after")],
        )


#: Features every policy gets when its config enables them. A policy that places these steps itself
#: is detected and left alone (see `apply_policy_features`).
DEFAULT_POLICY_FEATURES: tuple[ProcessorFeature, ...] = (RelativeActionsFeature(),)


def apply_policy_features(
    config: PreTrainedConfig,
    context: ProcessorBuildContext,
    preprocessor,
    postprocessor,
    features: Sequence[ProcessorFeature] | None = None,
) -> None:
    """Compose the enabled cross-cutting features into a freshly built pipeline pair, in place.

    Called by the policy processor dispatcher right after a policy's factory returns, so features
    reach every policy — including third-party plugins and the few policies that assemble their
    pipelines without the shared builders — without any of them mentioning the feature.

    A feature is skipped when the policy's factory already placed its steps, which is how a policy
    that wants bespoke placement (GR00T positions its own relative/absolute pair) keeps control
    without being special-cased by name here.

    Args:
        config: The policy config, source of parameters for `owner="config"` features.
        context: Per-run build inputs, source of parameters for `owner="context"` features.
        preprocessor: The freshly built preprocessor pipeline.
        postprocessor: The freshly built postprocessor pipeline.
        features: Features to consider. Defaults to `DEFAULT_POLICY_FEATURES`.
    """
    for feature in DEFAULT_POLICY_FEATURES if features is None else features:
        if not feature.enabled_for(config, context):
            continue

        pre_anchored, post_anchored = feature.build(config, context)
        already_placed = {type(entry.step) for entry in (*pre_anchored, *post_anchored)}
        if any(
            isinstance(step, tuple(already_placed)) for step in (*preprocessor.steps, *postprocessor.steps)
        ):
            logging.debug(
                "Skipping the '%s' feature: %s already places its steps itself.",
                feature.name,
                type(config).__name__,
            )
            continue

        preprocessor.steps = splice_anchored_steps(list(preprocessor.steps), pre_anchored)
        postprocessor.steps = splice_anchored_steps(list(postprocessor.steps), post_anchored)


def splice_anchored_steps(steps: list[ProcessorStep], anchored: list[AnchoredStep]) -> list[ProcessorStep]:
    """Insert anchored steps into `steps`, resolving each anchor to exactly one position.

    Args:
        steps: The pipeline's steps, in order.
        anchored: Steps to splice in.

    Returns:
        A new list with the anchored steps inserted.

    Raises:
        ValueError: If an anchor matches no step or more than one, either of which would make the
            placement a guess.
    """
    result = list(steps)
    for entry in anchored:
        matches = [index for index, step in enumerate(result) if isinstance(step, entry.anchor)]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot place {type(entry.step).__name__}: its anchor "
                f"{entry.anchor.__name__} matched {len(matches)} steps in "
                f"{[type(step).__name__ for step in result]}, but exactly one is required. "
                f"Place the step explicitly in the policy's processor factory instead."
            )
        index = matches[0] if entry.position == "before" else matches[0] + 1
        result.insert(index, entry.step)
    return result
