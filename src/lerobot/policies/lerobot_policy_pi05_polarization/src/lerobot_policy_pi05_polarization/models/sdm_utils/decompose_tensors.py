"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import torch.nn as nn
import math

def divide_tensor_spatial(x, block_size=256, method='tile_stride'):
    assert x.dim() == 4, "Input tensor must have 4 dimensions [B, C, H, W]"
    B, C, H, W = x.shape    
    assert H == W, "Height and Width must be equal"
    assert H % block_size == 0 and W % block_size ==0, "The tensor size cannot be divided by the block size"
    mosaic_scale = H // block_size
    
    if method == 'tile_stride':
        """ decomposing x into K x K of (Hc, Wc) non-overlapped blocks (grid)"""           
        
        K = mosaic_scale * mosaic_scale
        fold_params_grid = dict(kernel_size=(mosaic_scale, mosaic_scale), stride=(mosaic_scale, mosaic_scale), padding=(0,0), dilation=(1,1))
        unfold_grid = nn.Unfold(**fold_params_grid)   
        tensor_grids = unfold_grid(x)
        tensor_grids = tensor_grids.reshape(B, C, K, block_size, block_size).permute(0, 2, 1, 3, 4)
        return tensor_grids
    
    if method == 'tile_block':    
        tensor_blocks = x.view(B, C, mosaic_scale, block_size, mosaic_scale, block_size)
        tensor_blocks = tensor_blocks.permute(0, 2, 4, 1, 3, 5)
        tensor_blocks = tensor_blocks.contiguous().view(B, mosaic_scale**2, C, block_size, block_size)
        return tensor_blocks
    
    return -1

def merge_tensor_spatial(x, method='tile_stride'):
    
    K, N, feat_dim, Hm, Wm = x.shape
    mosaic_scale = int(math.sqrt(K))

    if method == 'tile_stride':
        x = x.reshape(K, N, feat_dim, -1)
        fold_params_grid = dict(kernel_size=(mosaic_scale, mosaic_scale), stride=(mosaic_scale, mosaic_scale), padding=(0,0), dilation=(1,1))
        fold_grid = nn.Fold(output_size=(Hm * mosaic_scale, Wm * mosaic_scale), **fold_params_grid) #  downsample based on the encoder     
        x = x.permute(1, 2, 0, 3).reshape(N, feat_dim * K, -1) 
        x = fold_grid(x)
        return x

    if method == 'tile_block':
        x = x.permute(1, 0, 2, 3, 4).reshape(N, mosaic_scale, mosaic_scale, feat_dim, Hm, Wm)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(N, feat_dim, mosaic_scale * Hm, mosaic_scale * Wm)
        return x
