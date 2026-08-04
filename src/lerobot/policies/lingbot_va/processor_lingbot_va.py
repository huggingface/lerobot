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

"""Pre/post-processor pipelines for the LingBot-VA policy.

The preprocessor passes inputs through (IDENTITY) and the postprocessor maps the policy's
``[-1, 1]`` actions back to physical units with the built-in ``UnnormalizerProcessorStep``
(QUANTILES) using per-channel q01/q99 restored from the checkpoint.

With ``action_anchor="episode"`` the pipeline additionally expresses actions relative to a single
per-episode anchor (see ``LingBotEpisodeAnchorStep``).
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
    to_relative_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_lingbot_va import LingBotVAConfig


@ProcessorStepRegistry.register("lingbot_episode_anchor")
@dataclass
class LingBotEpisodeAnchorStep(RelativeActionsProcessorStep):
    """Express actions relative to one anchor pose held for a whole episode.

    Why per-episode and not per-chunk: LingBot-VA's action stream *is* its memory. Each chunk's
    predicted actions are fed back into the KV cache as clean context (``_compute_kv_cache``), and
    the cache is append-only, so a representation whose meaning depends on a reference that changes
    mid-episode makes the cached tokens disagree with the tokens being generated. Anchoring every
    chunk on its own start state -- what the base ``RelativeActionsProcessorStep`` does -- reset the
    stream to zero displacement at every boundary, a discontinuity no training sample contained, and
    the postprocessor then added the new anchor on top of a prediction that had already continued
    from the old one. A single anchor per episode is *anchor-stable*: context and prediction share
    one reference for the whole rollout. This is also what upstream does -- ``get_relative_pose``
    anchors on the first action of a segment, and the RoboTwin client captures ``inint_eef_pose``
    once per episode and adds it back to every chunk.

    Subclasses ``RelativeActionsProcessorStep`` because it is the same operation with a longer-lived
    reference: the exclude-joint mask, the cached-reference accessors and the paired
    ``AbsoluteActionsProcessorStep`` that reverses it are all inherited. In particular the pair is
    re-bound after deserialization by ``factory._reconnect_relative_absolute_steps``, which is
    isinstance-based and already runs on every pretrained load -- there is no lingbot-specific wiring.
    Only the reference's *lifetime* and where it comes from differ:

    * **Training** -- the dataset hands the anchor over as the first row of the action tensor.
      ``LingBotVAConfig.action_delta_indices`` prepends a sentinel delta far more negative than any
      episode, and ``DatasetReader._get_query_indices`` clamps query indices into the episode's
      range, so that row is the episode's first *action* (the commanded pose at episode start, which
      is upstream's anchor too). This step consumes and strips it. It rides on the ACTION key rather
      than an observation key because ``observation_delta_indices`` applies to every
      ``observation.*`` feature, cameras included -- an observation-side anchor would cost one extra
      video decode per camera per sample.
    * **Inference** -- no command exists yet at reset, so the anchor is latched from the first
      observation's ``observation.state`` and held until ``reset()``. The residual difference between
      the commanded pose used in training and the measured pose used here is the standing tracking
      error with the arm at rest; upstream's RoboTwin client has the same asymmetry.

    Runs *before* the normalizer so the anchor row is never quantile-normalized and the subtraction
    happens in physical units.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            # Inference: latch the anchor from the episode's first observation, then hold it. The
            # base class re-caches state on every call; that per-tick reference is exactly what
            # makes the KV cache incoherent, so latch once and keep it until reset().
            if self.get_cached_state() is None:
                observation = new_transition.get(TransitionKey.OBSERVATION) or {}
                state = observation.get(OBS_STATE)
                if state is None:
                    raise RuntimeError(
                        "action_anchor='episode' needs observation.state on the first observation of "
                        "an episode to latch the anchor, but none was provided."
                    )
                # Collapse a temporally stacked (B, T_obs, state_dim) state to the current frame.
                self.set_cached_state((state[:, 0] if state.ndim == 3 else state).clone())
            return new_transition

        # Training: row 0 of the action tensor is the episode's first action.
        if action.ndim < 2:
            raise ValueError(
                f"action_anchor='episode' expects a batched, time-stacked action tensor "
                f"(B, T, action_dim); got shape {tuple(action.shape)}."
            )
        self.set_cached_state(action[:, 0].clone())
        new_transition[TransitionKey.ACTION] = to_relative_actions(
            action[:, 1:], self.get_cached_state(), self._build_mask(action.shape[-1])
        )

        # Keep the padding flags aligned with the actions they describe.
        complementary = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        pad_key = f"{ACTION}_is_pad"
        if complementary and pad_key in complementary and complementary[pad_key] is not None:
            complementary = dict(complementary)
            complementary[pad_key] = complementary[pad_key][:, 1:]
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary

        return new_transition

    def reset(self) -> None:
        """Drop the latched anchor so the next episode latches its own."""
        self.set_cached_state(None)


def make_lingbot_va_pre_post_processors(
    config: LingBotVAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the pre/post processor pipelines for LingBot-VA."""

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # One shared instance: the anchor is latched (inference) or derived (training) during
    # preprocessing and read back during postprocessing. The reverse direction needs no lingbot
    # subclass -- AbsoluteActionsProcessorStep already does exactly "unnormalize then add the
    # paired step's cached reference", and factory._reconnect_relative_absolute_steps re-binds the
    # pair after deserialization because the anchor step IS-A RelativeActionsProcessorStep.
    anchor_enabled = config.action_anchor == "episode"
    anchor_step = LingBotEpisodeAnchorStep(
        enabled=anchor_enabled,
        exclude_joints=config.action_anchor_exclude_joints,
        action_names=config.action_feature_names,
    )

    # Anchoring sits before the normalizer: raw -> anchor -> normalize -> model -> unnormalize ->
    # unanchor, so the anchored actions and the stats live in the same space on both sides.
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        anchor_step,
        steps.normalize,
        steps.to_device,
    ]

    # Unnormalize actions back to physical units. Config-driven norm_map (was hardcoded QUANTILES)
    # so it stays symmetric with the preprocessor's NormalizerProcessorStep.
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(enabled=anchor_enabled, relative_step=anchor_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
