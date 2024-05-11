"""A protocol that all policies should follow.

This provides a mechanism for type-hinting and isinstance checks without requiring the policies classes
subclass a base class.

The protocol structure, method signatures, and docstrings should be used by developers as a reference for
how to implement new policies.
"""

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class Policy(Protocol):
    """The required interface for implementing a policy.

    We also expect all policies to subclass torch.nn.Module and PyTorchModelHubMixin.
    """

    name: str

    def __init__(self, cfg, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        """
        Args:
            cfg: Policy configuration class instance or None, in which case the default instantiation of the
                 configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization.
        """

    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """

    def forward(self, batch: dict[str, Tensor]) -> dict:
        """Run the batch through the model and compute the loss for training or validation.

        Returns a dictionary with "loss" and potentially other information. Apart from "loss" which is a Tensor, all
        other items should be logging-friendly, native Python types.
        """

    def select_action(self, batch: dict[str, Tensor]):
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """


@runtime_checkable
class PolicyWithUpdate(Policy, Protocol):
    def update(self):
        """An update method that is to be called after a training optimization step.

        Implements an additional updates the model parameters may need (for example, doing an EMA step for a
        target model, or incrementing an internal buffer).
        """
