import torch
import einops
import torchvision
import torch.nn as nn
from typing import List, Tuple


class MultiviewStack(nn.Module):
    def __init__(
            self,
            encoders: List[nn.Module],
            normalizations: List[Tuple[List, List]],
            output_dim: int,
    ):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.normalizations = []
        for mean, std in normalizations:
            self.normalizations.append(
                torchvision.transforms.Normalize(mean=mean, std=std)
            )

    def forward(self, x):
        orig_shape = x.shape  # NTVCHW or TVCHW
        x = einops.rearrange(x, "... V C H W -> (...) V C H W")
        outputs = []
        # TODO: vectorize this for-statement
        for i, encoder in enumerate(self.encoders):
            this_view = x[:, i]
            this_view = self.normalizations[i](this_view)
            outputs.append(encoder(this_view))
        out = torch.stack(outputs, dim=-1)
        out = out.reshape(*orig_shape[:-3], -1)
        return out
