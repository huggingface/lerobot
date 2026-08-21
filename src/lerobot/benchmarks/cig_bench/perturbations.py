import torch


def axis_offsets(batch_size, magnitude, device=None, dtype=None):
    axes = torch.eye(3, device=device, dtype=dtype)
    return axes.repeat((batch_size + 2) // 3, 1)[:batch_size] * magnitude
