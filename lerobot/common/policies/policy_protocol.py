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
    """The required interface for implementing a policy."""

    name: str

    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """

    def forward(self, batch: dict[str, Tensor]) -> dict:
        """Run the batch through the model and compute the loss for training or validation.

        Returns a dictionary with "loss" and maybe other information.
        """

    def select_action(self, batch: dict[str, Tensor]):
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
