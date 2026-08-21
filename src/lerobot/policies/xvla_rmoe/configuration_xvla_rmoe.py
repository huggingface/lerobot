#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

"""Configuration for XVLA-RMoE: X-VLA with Cross-Step Routing Memory MoE.

Ports the `smolvla_rmoe` idea (see `lerobot.policies.smolvla_rmoe`) onto X-VLA's own
Transformer/flow-matching implementation (`lerobot.policies.xvla`). All original X-VLA
fields are inherited unchanged; only new RMoE-specific fields are added here. Setting
`use_moe=False` (or `use_moe=True, use_routing_memory=False, use_timestep_router=False`
etc., see the module docstring in `modeling_xvla_rmoe.py` for the full set of baseline
modes) reproduces the corresponding ablation while sharing the exact same code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..xvla.configuration_xvla import XVLAConfig


@PreTrainedConfig.register_subclass("xvla_rmoe")
@dataclass
class XVLARMoEConfig(XVLAConfig):
    """X-VLA config extended with Cross-Step Routing Memory Mixture-of-Experts fields.

    Every field from `XVLAConfig` is inherited as-is (backbone, action space, training
    presets, ...). The fields below control the dense soft-MoE FFN replacement, the
    cross-step GRU routing memory, and truncated cross-step recurrent training.
    """

    # --- Dense soft MoE -----------------------------------------------------------
    # All experts run on every token and are combined by softmax routing weights (no
    # top-K / sparse routing, no conditional compute).
    use_moe: bool = True
    num_moe_experts: int = 4
    num_moe_layers: int = 4
    moe_layer_indices: list[int] | None = None  # None -> last `num_moe_layers` of `depth`

    routing_hidden_dim: int = 64
    routing_timestep_dim: int = 64
    chunk_pos_emb_dim: int = 32

    # --- Router conditioning toggles (used to reproduce baselines, see section 18 of
    # the design doc / `modeling_xvla_rmoe.py` docstring) -------------------------
    use_routing_memory: bool = True
    use_timestep_router: bool = True
    use_delta_t_conditioning: bool = True
    use_chunk_position_embedding: bool = True

    # Every expert starts as a near-bit-identical deep copy of the same original FFN,
    # and the router starts at zero. Left alone this is a permutation-symmetry fixed
    # point: identical experts fed identical routing weights get identical gradients
    # and never diverge, and the mixture output is then exactly invariant to the
    # routing weights, so the router (and the GRU feeding it) never receives a real
    # gradient either. A tiny independent noise draw per expert at init breaks this
    # without visibly changing step-0 behaviour. Do not set to 0 outside of
    # ablation/debugging (see `test_expert_symmetry_breaking_...` regression tests).
    expert_symmetry_breaking_std: float = 1e-5

    # --- Truncated cross-step recurrent training -----------------------------------
    # Vanilla flow-matching training only ever samples a single timestep per example,
    # so routing_state is always the zero state during training and the GRU/router
    # never receive gradient through a non-zero state. On a fraction of training calls
    # (`recurrent_training_probability`), unroll `recurrent_unroll_steps` consecutive
    # denoising timesteps sharing one noise sample and one action chunk, threading
    # routing_state through the GRU between them.
    use_recurrent_routing_training: bool = True
    recurrent_training_probability: float = 0.25
    recurrent_unroll_steps: int = 3
    recurrent_detach_state: bool = False
    recurrent_timestep_sampling: str = "inference_grid"  # only option for now
    recurrent_loss_reduction: str = "mean"  # "mean" or "sum" over the K unrolled steps

    # Evaluation/analysis-only: return per-layer, per-token routing weights from
    # `RoutingInfo` (never on the training hot path -- see `RoutingInfo` docstring).
    return_full_routing_weights: bool = False

    # LIBERO is single-arm, while X-VLA's ``ee6d`` action space reserves two
    # 10-D arm slots. The LIBERO processor fills only the first slot and pads
    # the second with zeros. Do not train against that synthetic second arm:
    # doing so halves the useful gripper supervision and adds losses for
    # channels that can never be executed by the environment.
    single_arm_ee6d_loss: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.num_moe_experts < 2:
            raise ValueError(f"`num_moe_experts` must be >= 2, got {self.num_moe_experts}.")
        if self.routing_hidden_dim <= 0:
            raise ValueError(f"`routing_hidden_dim` must be > 0, got {self.routing_hidden_dim}.")
        if self.routing_timestep_dim <= 0:
            raise ValueError(f"`routing_timestep_dim` must be > 0, got {self.routing_timestep_dim}.")
        if self.expert_symmetry_breaking_std < 0.0:
            raise ValueError(
                f"`expert_symmetry_breaking_std` must be >= 0, got {self.expert_symmetry_breaking_std}."
            )
        if not (0.0 <= self.recurrent_training_probability <= 1.0):
            raise ValueError(
                f"`recurrent_training_probability` must be in [0, 1], got {self.recurrent_training_probability}."
            )
        if self.recurrent_unroll_steps < 2:
            raise ValueError(f"`recurrent_unroll_steps` must be >= 2, got {self.recurrent_unroll_steps}.")
        if self.recurrent_unroll_steps > self.num_denoising_steps:
            raise ValueError(
                f"`recurrent_unroll_steps` ({self.recurrent_unroll_steps}) cannot exceed "
                f"`num_denoising_steps` ({self.num_denoising_steps})."
            )
        if self.recurrent_timestep_sampling not in {"inference_grid"}:
            raise ValueError(
                f"Unsupported `recurrent_timestep_sampling`: {self.recurrent_timestep_sampling!r}."
            )
        if self.recurrent_loss_reduction not in {"mean", "sum"}:
            raise ValueError(
                f"`recurrent_loss_reduction` must be 'mean' or 'sum', got {self.recurrent_loss_reduction!r}."
            )

        if self.moe_layer_indices is not None:
            if len(self.moe_layer_indices) == 0:
                raise ValueError("`moe_layer_indices` must be non-empty when explicitly provided.")
            bad = [i for i in self.moe_layer_indices if i < 0 or i >= self.depth]
            if bad:
                raise ValueError(f"`moe_layer_indices` contains out-of-range layer indices {bad}.")
            self.num_moe_layers = len(self.moe_layer_indices)
        else:
            if self.num_moe_layers <= 0 or self.num_moe_layers > self.depth:
                raise ValueError(
                    f"`num_moe_layers` must be in [1, depth={self.depth}], got {self.num_moe_layers}."
                )
            self.moe_layer_indices = list(range(self.depth - self.num_moe_layers, self.depth))

        if not self.use_moe:
            # No FFN is ever replaced by a MoEFFN, so nothing router/memory-related can
            # run -- keep the dependent flags internally consistent instead of forcing
            # every caller to remember to also flip them off.
            if self.use_routing_memory or self.use_recurrent_routing_training:
                logging.warning(
                    "`use_moe=False`: disabling `use_routing_memory` and "
                    "`use_recurrent_routing_training` (there is no MoE router to condition)."
                )
            self.use_routing_memory = False
            self.use_recurrent_routing_training = False
        elif not self.use_routing_memory and self.use_recurrent_routing_training:
            logging.warning(
                "`use_routing_memory=False`: disabling `use_recurrent_routing_training` "
                "(truncated recurrent training only trains the GRU routing memory)."
            )
            self.use_recurrent_routing_training = False
