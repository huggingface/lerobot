import torch
from torch import Tensor


def pad_or_crop_horizon(actions: Tensor, horizon: int) -> Tensor:
    if actions.shape[1] >= horizon:
        return actions[:, :horizon]
    padding = actions[:, -1:].expand(-1, horizon - actions.shape[1], -1)
    return torch.cat((actions, padding), dim=1)
