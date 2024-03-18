from abc import ABC, abstractmethod
from collections import deque

import torch
from torch import Tensor, nn


class AbstractPolicy(nn.Module, ABC):
    """Base policy which all policies should be derived from.

    The forward method should generally not be overriden as it plays the role of handling multi-step policies. See its
    documentation for more information.
    """

    @abstractmethod
    def update(self, replay_buffer, step):
        """One step of the policy's learning algorithm."""

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        self.load_state_dict(d)

    @abstractmethod
    def select_action(self, observation) -> Tensor:
        """Select an action (or trajectory of actions) based on an observation during rollout.

        Should return a (batch_size, n_action_steps, *) tensor of actions.
        """

    def forward(self, *args, **kwargs):
        """Inference step that makes multi-step policies compatible with their single-step environments.

        WARNING: In general, this should not be overriden.

        Consider a "policy" that observes the environment then charts a course of N actions to take. To make this fit
        into the formalism of a TorchRL environment, we view it as being effectively a policy that (1) makes an
        observation and prepares a queue of actions, (2) consumes that queue when queried, regardless of the environment
        observation, (3) repopulates the action queue when empty. This method handles the aforementioned logic so that
        the subclass doesn't have to.

        This method effectively wraps the `select_action` method of the subclass. The following assumptions are made:
        1. The `select_action` method returns a Tensor of actions with shape (B, H, *) where B is the batch size, H is
           the action trajectory horizon and * is the action dimensions.
        2. Prior to the `select_action` method being called, theres is an `n_action_steps` instance attribute defined.
        """
        n_action_steps_attr = "n_action_steps"
        if not hasattr(self, n_action_steps_attr):
            raise RuntimeError(f"Underlying policy must have an `{n_action_steps_attr}` attribute")
        if not hasattr(self, "_action_queue"):
            self._action_queue = deque([], maxlen=getattr(self, n_action_steps_attr))
        if len(self._action_queue) == 0:
            # Each element in the queue has shape (B, *).
            self._action_queue.extend(self.select_action(*args, **kwargs).transpose(0, 1))

        return self._action_queue.popleft()
