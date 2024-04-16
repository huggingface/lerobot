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

    def forward(self, batch: dict[str, Tensor], **kwargs):
        """Wired to `select_action`."""

    def select_action(self, batch: dict[str, Tensor], **kwargs):
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """

    def compute_loss(self, batch: dict[str, Tensor], **kwargs):
        """Runs the batch through the model and computes the loss for training or validation."""

    def update(self, batch, **kwargs):
        """Does compute_loss then an optimization step.

        TODO(alexander-soare): We will move the optimization step back into the training loop, so this will
        disappear.
        """
