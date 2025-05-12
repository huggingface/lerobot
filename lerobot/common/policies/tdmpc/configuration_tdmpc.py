#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
# and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("tdmpc")
@dataclass
class TDMPCConfig(PreTrainedConfig):
    """Configuration class for TDMPCPolicy.

    Defaults are configured for training with xarm_lift_medium_replay providing proprioceptive and single
    camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes`, `output_shapes`, and perhaps `max_random_shift_ratio`.

    Args:
        n_action_repeats: The number of times to repeat the action returned by the planning. (hint: Google
            action repeats in Q-learning or ask your favorite chatbot)
        horizon: Horizon for model predictive control.
        n_action_steps: Number of action steps to take from the plan given by model predictive control. This
            is an alternative to using action repeats. If this is set to more than 1, then we require
            `n_action_repeats == 1`, `use_mpc == True` and `n_action_steps <= horizon`. Note that this
            approach of using multiple steps from the plan is not in the original implementation.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range. Note that here this defaults to None meaning inputs are not normalized. This is to
            match the original implementation.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets. NOTE: Clipping
            to [-1, +1] is used during MPPI/CEM. Therefore, it is recommended that you stick with "min_max"
            normalization mode here.
        image_encoder_hidden_dim: Number of channels for the convolutional layers used for image encoding.
        state_encoder_hidden_dim: Hidden dimension for MLP used for state vector encoding.
        latent_dim: Observation's latent embedding dimension.
        q_ensemble_size: Number of Q function estimators to use in an ensemble for uncertainty estimation.
        mlp_dim: Hidden dimension of MLPs used for modelling the dynamics encoder, reward function, policy
            (π), Q ensemble, and V.
        discount: Discount factor (γ) to use for the reinforcement learning formalism.
        use_mpc: Whether to use model predictive control. The alternative is to just sample the policy model
            (π) for each step.
        cem_iterations: Number of iterations for the MPPI/CEM loop in MPC.
        max_std: Maximum standard deviation for actions sampled from the gaussian PDF in CEM.
        min_std: Minimum standard deviation for noise applied to actions sampled from the policy model (π).
            Doubles up as the minimum standard deviation for actions sampled from the gaussian PDF in CEM.
        n_gaussian_samples: Number of samples to draw from the gaussian distribution every CEM iteration. Must
            be non-zero.
        n_pi_samples: Number of samples to draw from the policy / world model rollout every CEM iteration. Can
            be zero.
        uncertainty_regularizer_coeff: Coefficient for the uncertainty regularization used when estimating
            trajectory values (this is the λ coefficient in eqn 4 of FOWM).
        n_elites: The number of elite samples to use for updating the gaussian parameters every CEM iteration.
        elite_weighting_temperature: The temperature to use for softmax weighting (by trajectory value) of the
            elites, when updating the gaussian parameters for CEM.
        gaussian_mean_momentum: Momentum (α) used for EMA updates of the mean parameter μ of the gaussian
            parameters optimized in CEM. Updates are calculated as μ⁻ ← αμ⁻ + (1-α)μ.
        max_random_shift_ratio: Maximum random shift (as a proportion of the image size) to apply to the
            image(s) (in units of pixels) for training-time augmentation. If set to 0, no such augmentation
            is applied. Note that the input images are assumed to be square for this augmentation.
        reward_coeff: Loss weighting coefficient for the reward regression loss.
        expectile_weight: Weighting (τ) used in expectile regression for the state value function (V).
            v_pred < v_target is weighted by τ and v_pred >= v_target is weighted by (1-τ). τ is expected to
            be in [0, 1]. Setting τ closer to 1 results in a more "optimistic" V. This is sensible to do
            because v_target is obtained by evaluating the learned state-action value functions (Q) with
            in-sample actions that may not be always optimal.
        value_coeff: Loss weighting coefficient for both the state-action value (Q) TD loss, and the state
            value (V) expectile regression loss.
        consistency_coeff: Loss weighting coefficient for the consistency loss.
        advantage_scaling: A factor by which the advantages are scaled prior to exponentiation for advantage
            weighted regression of the policy (π) estimator parameters. Note that the exponentiated advantages
            are clamped at 100.0.
        pi_coeff: Loss weighting coefficient for the action regression loss.
        temporal_decay_coeff: Exponential decay coefficient for decaying the loss coefficient for future time-
            steps. Hint: each loss computation involves `horizon` steps worth of actions starting from the
            current time step.
        target_model_momentum: Momentum (α) used for EMA updates of the target models. Updates are calculated
            as ϕ ← αϕ + (1-α)θ where ϕ are the parameters of the target model and θ are the parameters of the
            model being trained.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    n_action_repeats: int = 2
    horizon: int = 5
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture / modeling.
    # Neural networks.
    image_encoder_hidden_dim: int = 32
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 50
    q_ensemble_size: int = 5
    mlp_dim: int = 512
    # Reinforcement learning.
    discount: float = 0.9

    # Inference.
    use_mpc: bool = True
    cem_iterations: int = 6
    max_std: float = 2.0
    min_std: float = 0.05
    n_gaussian_samples: int = 512
    n_pi_samples: int = 51
    uncertainty_regularizer_coeff: float = 1.0
    n_elites: int = 50
    elite_weighting_temperature: float = 0.5
    gaussian_mean_momentum: float = 0.1

    # Training and loss computation.
    max_random_shift_ratio: float = 0.0476
    # Loss coefficients.
    reward_coeff: float = 0.5
    expectile_weight: float = 0.9
    value_coeff: float = 0.1
    consistency_coeff: float = 20.0
    advantage_scaling: float = 3.0
    pi_coeff: float = 0.5
    temporal_decay_coeff: float = 0.5
    # Target model.
    target_model_momentum: float = 0.995

    # Training presets
    optimizer_lr: float = 3e-4

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_gaussian_samples <= 0:
            raise ValueError(
                f"The number of gaussian samples for CEM should be non-zero. Got `{self.n_gaussian_samples=}`"
            )
        if self.normalization_mapping["ACTION"] is not NormalizationMode.MIN_MAX:
            raise ValueError(
                "TD-MPC assumes the action space dimensions to all be in [-1, 1]. Therefore it is strongly "
                f"advised that you stick with the default. See {self.__class__.__name__} docstring for more "
                "information."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.n_action_steps > 1:
            if self.n_action_repeats != 1:
                raise ValueError(
                    "If `n_action_steps > 1`, `n_action_repeats` must be left to its default value of 1."
                )
            if not self.use_mpc:
                raise ValueError("If `n_action_steps > 1`, `use_mpc` must be set to `True`.")
            if self.n_action_steps > self.horizon:
                raise ValueError("`n_action_steps` must be less than or equal to `horizon`.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # There should only be one image key.
        if len(self.image_features) > 1:
            raise ValueError(
                f"{self.__class__.__name__} handles at most one image for now. Got image keys {self.image_features}."
            )

        if len(self.image_features) > 0:
            image_ft = next(iter(self.image_features.values()))
            if image_ft.shape[-2] != image_ft.shape[-1]:
                # TODO(alexander-soare): This limitation is solely because of code in the random shift
                # augmentation. It should be able to be removed.
                raise ValueError(f"Only square images are handled now. Got image shape {image_ft.shape}.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(self.horizon + 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return list(range(self.horizon))
