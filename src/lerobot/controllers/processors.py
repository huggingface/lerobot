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

"""Processor pipelines between a robot's state and a controller's network.

A controller's preprocessor applies ``scales * x + offsets`` to the robot state it reads (joint positions
around the default pose, scaled velocities, ...) and reorders joints from robot order into the order its
network was trained in. Its postprocessor maps the network's action back: reorder into robot order, then
``scales * a + offsets`` into absolute joint targets. Coefficients are always given in robot joint order.

The map is :class:`~lerobot.processor.JointAffineProcessorStep`, which writes ``observation.state`` and the
action. Each observation feature is renamed onto ``observation.state`` for its own step and renamed back, so
a feature keeps its key and its own coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from lerobot.processor import (
    JointAffineProcessorStep,
    JointReorderProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    observation_to_transition,
    policy_action_to_transition,
    transition_to_observation,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import DataProcessorPipeline, ProcessorStep
from lerobot.utils.constants import OBS_STATE

ArrayLike = float | Sequence[float] | np.ndarray
IndexOrder = Sequence[int] | np.ndarray


def _affine_lists(scales: ArrayLike, offsets: ArrayLike) -> tuple[list[float], list[float]]:
    """Broadcast a scalar coefficient onto the other and return JSON-safe lists of the same length."""
    scale = np.atleast_1d(np.asarray(scales, dtype=np.float64)).reshape(-1)
    offset = np.atleast_1d(np.asarray(offsets, dtype=np.float64)).reshape(-1)
    if scale.size == 1 and offset.size != 1:
        scale = np.broadcast_to(scale, offset.shape)
    elif offset.size == 1 and scale.size != 1:
        offset = np.broadcast_to(offset, scale.shape)
    if scale.shape != offset.shape:
        raise ValueError(f"scales and offsets must have the same length, got {scale.size} and {offset.size}.")
    return scale.tolist(), offset.tolist()


def _observation_affine(key: str, scales: ArrayLike, offsets: ArrayLike) -> list[ProcessorStep]:
    """One feature's ``scales * x + offsets``, parked on ``observation.state`` while the step runs."""
    scale, offset = _affine_lists(scales, offsets)
    affine: ProcessorStep = JointAffineProcessorStep(scales=scale, offsets=offset)
    if key == OBS_STATE:
        return [affine]
    return [
        RenameObservationsProcessorStep(rename_map={key: OBS_STATE}),
        affine,
        RenameObservationsProcessorStep(rename_map={OBS_STATE: key}),
    ]


def make_controller_processors(
    observation_affine: Mapping[str, tuple[ArrayLike, ArrayLike]],
    action_affine: tuple[ArrayLike, ArrayLike],
    observation_order: Mapping[str, IndexOrder] | None = None,
    action_order: IndexOrder | None = None,
) -> tuple[
    DataProcessorPipeline[dict[str, Any], dict[str, Any]], DataProcessorPipeline[torch.Tensor, torch.Tensor]
]:
    """Build a controller's preprocessor and postprocessor.

    Args:
        observation_affine: For each observation key to map, its ``(scales, offsets)`` in robot joint order.
            A scalar coefficient broadcasts to the other's length. The result is ``scales * x + offsets``.
        action_affine: The action's ``(scales, offsets)`` in robot joint order, applied after reordering:
            ``scales * a + offsets``, the default pose plus the scaled residual.
        observation_order: For each observation key to reorder after the affine map, the robot index of each
            entry the network reads.
        action_order: The network-action index of each robot joint the postprocessor returns.

    Returns:
        The preprocessor, from a dict of robot-state arrays to tensors in network order, and the
        postprocessor, from a network action tensor to absolute joint targets in robot order.
    """
    if OBS_STATE in observation_affine and len(observation_affine) > 1:
        raise ValueError(
            "observation_affine cannot map observation.state together with other keys. "
            "JointAffineProcessorStep maps one state vector, and the other keys borrow that name."
        )

    preprocessor: DataProcessorPipeline[dict[str, Any], dict[str, Any]] = DataProcessorPipeline(
        steps=[
            *_observation_steps(observation_affine),
            JointReorderProcessorStep(
                observation_indices={
                    key: [int(i) for i in order] for key, order in (observation_order or {}).items()
                }
            ),
        ],
        name="controller_preprocessor",
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    action_scales, action_offsets = _affine_lists(*action_affine)
    postprocessor: DataProcessorPipeline[torch.Tensor, torch.Tensor] = DataProcessorPipeline(
        steps=[
            JointReorderProcessorStep(
                action_indices=None if action_order is None else [int(i) for i in action_order]
            ),
            JointAffineProcessorStep(scales=action_scales, offsets=action_offsets),
        ],
        name="controller_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _observation_steps(observation_affine: Mapping[str, tuple[ArrayLike, ArrayLike]]) -> list[ProcessorStep]:
    steps: list[ProcessorStep] = []
    for key, (scales, offsets) in observation_affine.items():
        steps.extend(_observation_affine(key, scales, offsets))
    return steps
