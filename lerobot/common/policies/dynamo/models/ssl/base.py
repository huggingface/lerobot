import abc

import torch
import torch.nn as nn
from typing import Tuple, Dict


class AbstractSSL(nn.Module):
    """
    This class should contain everything inside the SSL method (e.g. key queue for MoCo, EMA for BYOL, etc.), loss function, and the optimizer.
    """

    @abc.abstractmethod
    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
    ):
        """
        Initializes the SSL method.
        Inputs:
            encoder: the encoder module
            projector: the projector module
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor],]:
        """
        Inputs:
            obs: the input observations
        Outputs:
            obs_enc: the encoded observations
            obs_proj: the projected observations
            loss: the total loss
            loss_components: the components of the total loss
        """
        raise NotImplementedError

    def step(self):
        """
        This function should be called at each training step to update the SSL method's internal state.
        e.g. step the optimizer, update the key queue for MoCo, update EMA for BYOL, etc.
        """
        raise NotImplementedError
