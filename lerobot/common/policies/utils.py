import torch
from torch import nn


def populate_queues(queues, batch):
    for key in batch:
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def normalize_inputs(batch, stats, normalize_input_modes):
    if normalize_input_modes is None:
        return batch
    for key, mode in normalize_input_modes.items():
        if mode == "mean_std":
            mean = stats[key]["mean"].unsqueeze(0)
            std = stats[key]["std"].unsqueeze(0)
            batch[key] = (batch[key] - mean) / (std + 1e-8)
        elif mode == "min_max":
            min = stats[key]["min"].unsqueeze(0)
            max = stats[key]["max"].unsqueeze(0)
            # normalize to [0,1]
            batch[key] = (batch[key] - min) / (max - min)
            # normalize to [-1, 1]
            batch[key] = batch[key] * 2 - 1
        else:
            raise ValueError(mode)
    return batch


def unnormalize_outputs(batch, stats, unnormalize_output_modes):
    if unnormalize_output_modes is None:
        return batch
    for key, mode in unnormalize_output_modes.items():
        if mode == "mean_std":
            mean = stats[key]["mean"].unsqueeze(0)
            std = stats[key]["std"].unsqueeze(0)
            batch[key] = batch[key] * std + mean
        elif mode == "min_max":
            min = stats[key]["min"].unsqueeze(0)
            max = stats[key]["max"].unsqueeze(0)
            batch[key] = (batch[key] + 1) / 2
            batch[key] = batch[key] * (max - min) + min
        else:
            raise ValueError(mode)
    return batch
