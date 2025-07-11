from typing import Dict, List

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def is_batch_need_padding(values: list[torch.Tensor], pad_dim: int = -1) -> int:
    return len(values[0].shape) > 0  # and len(set([v.shape[pad_dim] for v in values])) > 1


def pad_tensor(
    tensor: torch.Tensor, max_size: int, pad_dim: int = -1, pad_value: float = 0.0
) -> torch.Tensor:
    is_numpy = isinstance(tensor, np.ndarray)
    if is_numpy:
        tensor = torch.tensor(tensor)
    pad = max_size - tensor.shape[pad_dim]
    if pad > 0:
        pad_sizes = (0, pad)  # pad right
        tensor = torch.nn.functional.pad(tensor, pad_sizes, value=pad_value)
    return tensor.numpy() if is_numpy else tensor


def pad_list_of_tensors(
    tensors: List[torch.Tensor], pad_dim: int = -1, pad_value: float = 0.0
) -> List[torch.Tensor]:
    max_size = max([v.shape[pad_dim] for v in tensors])
    return [pad_tensor(tensor, max_size, pad_dim=pad_dim, pad_value=pad_value) for tensor in tensors]


def multidataset_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_dim: int = -1,
    pad_value: float = 0.0,
    keys_to_max_dim: dict = {},
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to pad tensors with multiple dimensions.

    Args:
        batch (List[Dict[str, torch.Tensor]]): List of dataset samples (each sample is a dictionary).

    Returns:
        Dict[str, torch.Tensor]: Batch with padded tensors.
    """
    batch_keys = batch[0].keys()
    collated_batch = [{} for _ in range(len(batch))]
    # FIXME(mshukor): pad to max shape per feature type
    for key in batch_keys:
        values = [sample[key] for sample in batch]
        if (
            key in keys_to_max_dim
            and isinstance(values[0], torch.Tensor)
            and is_batch_need_padding(values, pad_dim=pad_dim)
            and keys_to_max_dim[key] is not None
        ):
            max_size = keys_to_max_dim[key]
            for i in range(len(batch)):
                collated_batch[i][key] = pad_tensor(
                    batch[i][key], max_size, pad_dim=pad_dim, pad_value=pad_value
                )
        else:
            for i in range(len(batch)):
                collated_batch[i][key] = batch[i][key]
    collated_batch = default_collate(collated_batch)

    return collated_batch
