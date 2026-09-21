import torch
from torch import Tensor


def pad_or_crop_horizon(actions: Tensor, horizon: int) -> Tensor:
    if actions.shape[1] >= horizon:
        return actions[:, :horizon]
    padding = actions[:, -1:].expand(-1, horizon - actions.shape[1], -1)
    return torch.cat((actions, padding), dim=1)


def pad_or_crop_mask(mask: Tensor, horizon: int) -> Tensor:
    """Crop/pad a `[B, T]` bool mask to `horizon` steps, mirroring `pad_or_crop_horizon`'s handling
    of the action tensor it accompanies. Any step added by padding (T < horizon) is marked `True`
    (excluded) rather than copying the last real step's mask value, since that last step's own
    validity says nothing about a position that doesn't exist in the original chunk.
    """
    if mask.shape[1] >= horizon:
        return mask[:, :horizon]
    extra = torch.ones(mask.shape[0], horizon - mask.shape[1], dtype=torch.bool, device=mask.device)
    return torch.cat((mask, extra), dim=1)


def masked_mse(pred: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Mean squared error over only the timesteps where `valid_mask` (`[B, T]`, True = include)
    holds, broadcast across `pred`/`target`'s trailing dim `[B, T, D]`. Exactly reproduces
    `F.mse_loss(pred, target)` (both default to a mean over every element) when `valid_mask` is all
    `True`. The denominator is clamped to at least 1 so an all-`False` mask returns a safe `0`
    instead of `0/0` -- the numerator is already exactly `0` in that case since every term is
    masked out, so the clamp only avoids the division-by-zero, it doesn't change the result.
    """
    sq_err = (pred - target).square()
    mask = valid_mask.unsqueeze(-1).to(sq_err.dtype)
    numerator = (sq_err * mask).sum()
    denominator = (mask.sum() * sq_err.shape[-1]).clamp_min(1.0)
    return numerator / denominator
