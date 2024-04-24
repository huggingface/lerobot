import torch
from torch import nn


def create_stats_buffers(shapes, modes, stats=None):
    """
    This function generates buffers to store the mean and standard deviation, or minimum and maximum values,
    used for normalizing tensors. The mode of normalization is determined by the `modes` dictionary, which can
    be either "mean_std" (for mean and standard deviation) or "min_max" (for minimum and maximum). These buffers
    are created as PyTorch nn.ParameterDict objects with nn.Parameters set to not require gradients, suitable
    for normalization purposes.

    If the provided `shapes` contain keys related to images, the shape is adjusted to be invariant to height
    and width, assuming a channel-first (c, h, w) format.

    Parameters:
        shapes (dict): A dictionary where keys represent tensor identifiers and values represent the shapes of those tensors.
        modes (dict): A dictionary specifying the normalization mode for each key in `shapes`. Valid modes are "mean_std" or "min_max".
        stats (dict, optional): A dictionary containing pre-defined statistics for normalization. It can contain 'mean' and 'std' for
            "mean_std" mode, or 'min' and 'max' for "min_max" mode. If provided, these statistics will overwrite the default buffers.
            It's expected for training the model for the first time. If not provided, the default buffers are supposed to be overriden
            by a call to `policy.load_state_dict(state_dict)`. It's useful for loading a pretrained model for finetuning or evaluation,
            without requiring to initialize the dataset used to train the model just to acess the `stats`.

    Returns:
        dict: A dictionary where keys match the `modes` and `shapes` keys, and values are nn.ParameterDict objects containing
              the appropriate buffers for normalization.
    """
    stats_buffers = {}

    for key, mode in modes.items():
        assert mode in ["mean_std", "min_max"]

        shape = shapes[key]

        # override shape to be invariant to height and width
        if "image" in key:
            # assume shape is channel first (b, c, h, w) or (b, t, c, h, w)
            shape[-1] = 1
            shape[-2] = 1

        buffer = {}
        if mode == "mean_std":
            mean = torch.zeros(shape, dtype=torch.float32)
            std = torch.ones(shape, dtype=torch.float32)
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif mode == "min_max":
            # TODO(rcadene): should we assume input is in [-1, 1] range?
            min = torch.ones(shape, dtype=torch.float32) * -1
            max = torch.ones(shape, dtype=torch.float32)
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )

        if stats is not None:
            if mode == "mean_std":
                buffer["mean"].data = stats[key]["mean"]
                buffer["std"].data = stats[key]["std"]
            elif mode == "min_max":
                buffer["min"].data = stats[key]["min"]
                buffer["max"].data = stats[key]["max"]

        stats_buffers[key] = buffer
    return stats_buffers


class Normalize(nn.Module):
    """
    A PyTorch module for normalizing data based on predefined statistics.

    The class is initialized with a set of shapes, modes, and optional pre-defined statistics. It creates buffers for normalization based
    on these inputs, which are then used to adjust data during the forward pass. The normalization process operates on a batch of data,
    with different keys in the batch being normalized according to the specified modes. The following normalization modes are supported:
    - "mean_std": Normalizes data using the mean and standard deviation.
    - "min_max": Normalizes data to a [0, 1] range and then to a [-1, 1] range.

    Parameters:
        shapes (dict): A dictionary where keys represent tensor identifiers and values represent the shapes of those tensors.
        modes (dict): A dictionary indicating the normalization mode for each tensor key. Valid modes are "mean_std" or "min_max".
        stats (dict, optional): A dictionary containing pre-defined statistics for normalization. It can contain 'mean' and 'std' for
            "mean_std" mode, or 'min' and 'max' for "min_max" mode. If provided, these statistics will overwrite the default buffers.
            It's expected for training the model for the first time. If not provided, the default buffers are supposed to be overriden
            by a call to `policy.load_state_dict(state_dict)`. It's useful for loading a pretrained model for finetuning or evaluation,
            without requiring to initialize the dataset used to train the model just to acess the `stats`.
    """

    def __init__(self, shapes, modes, stats=None):
        super().__init__()
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch):
        for key, mode in self.modes.items():
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"].unsqueeze(0)
                std = buffer["std"].unsqueeze(0)
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif mode == "min_max":
                min = buffer["min"].unsqueeze(0)
                max = buffer["max"].unsqueeze(0)
                # normalize to [0,1]
                batch[key] = (batch[key] - min) / (max - min)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(mode)
        return batch


class Unnormalize(nn.Module):
    """
    A PyTorch module for unnormalizing data based on predefined statistics.

    The class is initialized with a set of shapes, modes, and optional pre-defined statistics. It creates buffers for unnormalization based
    on these inputs, which are then used to adjust data during the forward pass. The unnormalization process operates on a batch of data,
    with different keys in the batch being normalized according to the specified modes. The following unnormalization modes are supported:
    - "mean_std": Unnormalizes data using the mean and standard deviation.
    - "min_max": Unnormalizes data to a [0, 1] range and then to a [-1, 1] range.

    Parameters:
        shapes (dict): A dictionary where keys represent tensor identifiers and values represent the shapes of those tensors.
        modes (dict): A dictionary indicating the unnormalization mode for each tensor key. Valid modes are "mean_std" or "min_max".
        stats (dict, optional): A dictionary containing pre-defined statistics for unnormalization. It can contain 'mean' and 'std' for
            "mean_std" mode, or 'min' and 'max' for "min_max" mode. If provided, these statistics will overwrite the default buffers.
            It's expected for training the model for the first time. If not provided, the default buffers are supposed to be overriden
            by a call to `policy.load_state_dict(state_dict)`. It's useful for loading a pretrained model for finetuning or evaluation,
            without requiring to initialize the dataset used to train the model just to acess the `stats`.
    """

    def __init__(self, shapes, modes, stats=None):
        super().__init__()
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch):
        for key, mode in self.modes.items():
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"].unsqueeze(0)
                std = buffer["std"].unsqueeze(0)
                batch[key] = batch[key] * std + mean
            elif mode == "min_max":
                min = buffer["min"].unsqueeze(0)
                max = buffer["max"].unsqueeze(0)
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                raise ValueError(mode)
        return batch
