"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numbers
import torch

def fuse_feats(feats_1, feats_2):
    if len(feats_1) != len(feats_2):
        raise ValueError()
    out_feats = []
    for i in range(len(feats_1)):
        out_feats.append(feats_1[i]+feats_2[i])
    return out_feats

def freeze_module(module):
    # state_dict = module.state_dict()
    # for k, v in state_dict.items():
    #     print(v.requires_grad)
    for p in module.parameters():
        p.requires_grad = False

def enable_module_trainable(model):
    for p in model.parameters():
        p.requires_grad = True
    # state_dict = module.state_dict()
    # for k, v in state_dict.items():
    #     v.requires_grad = True

def zero_module(module):
    """
    Zero out the parameters of a module.
    """
    for p in module.parameters():
        p.requires_grad = False
        p.zero_()
        p.requires_grad = True

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def ind2coords(array_shape, ind, device='cpu'):
    """array_shape (rows, cols)
    ind = row * cols + col
    """
    row = torch.div(ind, array_shape[1], rounding_mode='trunc') # ind / W
    col = ind % array_shape[1] # or numpy.mod(ind.astype('int'), array_shape[1])
    coords = torch.zeros((1, 1, len(ind), 2), dtype=torch.float32, device=device)
    coords[:, :, :, 1] = 2 * row.to(torch.float32) / array_shape[0] - 1
    coords[:, :, :, 0] = 2 * col.to(torch.float32) / array_shape[1] - 1
    return coords

class gauss_filter(nn.Module):
  
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(gauss_filter, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        self.kernel_size = kernel_size
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
            , indexing='ij'
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=(self.kernel_size[0]//2, self.kernel_size[1]//2))

