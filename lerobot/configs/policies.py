from dataclasses import dataclass, field

import draccus


@dataclass
class PretrainedConfig(draccus.ChoiceRegistry):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1
    input_shapes: dict[str, list[int]] = field(default_factory=lambda: {})
    output_shapes: dict[str, list[int]] = field(default_factory=lambda: {})
    input_normalization_modes: dict[str, str] = field(default_factory=lambda: {})
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {})

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError
