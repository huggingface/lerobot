from abc import ABC, abstractmethod
from collections import deque

import torch
from torch import Tensor, nn


class AbstractPolicy(nn.Module, ABC):
    """Base policy which all policies should be derived from.

    The forward method should generally not be overriden as it plays the role of handling multi-step policies. See its
    documentation for more information.
    """

    def __init__(self, n_action_steps: int | None):
        """
        n_action_steps: Sets the cache size for storing action trajectories. If None, it is assumed that a single
            action is returned by `select_actions` and that doesn't have a horizon dimension. The `forward` method then
            adds that dimension.
        """
        super().__init__()
        self.n_action_steps = n_action_steps
        self.clear_action_queue()

    @abstractmethod
    def update(self, replay_buffer, step):
        """One step of the policy's learning algorithm."""

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        self.load_state_dict(d)

    @abstractmethod
    def select_actions(self, observation) -> Tensor:
        """Select an action (or trajectory of actions) based on an observation during rollout.

        If n_action_steps was provided at initialization, this should return a (batch_size, n_action_steps, *) tensor of
        actions. Otherwise if n_actions_steps is None, this should return a (batch_size, *) tensor of actions.
        """

    def clear_action_queue(self):
        """This should be called whenever the environment is reset."""
        if self.n_action_steps is not None:
            self._action_queue = deque([], maxlen=self.n_action_steps)

    def forward(self, *args, **kwargs) -> Tensor:
        """Inference step that makes multi-step policies compatible with their single-step environments.

        WARNING: In general, this should not be overriden.

        Consider a "policy" that observes the environment then charts a course of N actions to take. To make this fit
        into the formalism of a TorchRL environment, we view it as being effectively a policy that (1) makes an
        observation and prepares a queue of actions, (2) consumes that queue when queried, regardless of the environment
        observation, (3) repopulates the action queue when empty. This method handles the aforementioned logic so that
        the subclass doesn't have to.

        This method effectively wraps the `select_actions` method of the subclass. The following assumptions are made:
        1. The `select_actions` method returns a Tensor of actions with shape (B, H, *) where B is the batch size, H is
           the action trajectory horizon and * is the action dimensions.
        2. Prior to the `select_actions` method being called, theres is an `n_action_steps` instance attribute defined.
        """
        if self.n_action_steps is None:
            return self.select_actions(*args, **kwargs)
        if len(self._action_queue) == 0:
            # `select_actions` returns a (batch_size, n_action_steps, *) tensor, but the queue effectively has shape
            # (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(self.select_actions(*args, **kwargs).transpose(0, 1))
        return self._action_queue.popleft()
