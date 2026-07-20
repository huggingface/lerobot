
"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import *
from . import transformer
from . import convnext
from . import uper
from .decompose_tensors import *

class ImageFeatureExtractor(nn.Module):
    def __init__(self, input_nc):
        super(ImageFeatureExtractor, self).__init__()
        back = []

        ### ConvNexT backbone (from sctratch)
        out_channels = (96, 192, 384, 768)
        back.append(convnext.ConvNeXt(in_chans=input_nc, use_checkpoint=False))      
        self.backbone = nn.Sequential(*back)
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.backbone(x) 
        # output the feats corresponding to the whole image (not related to the resolution)
        return feats

class ImageFeatureFusion(nn.Module):
    def __init__(self, in_channels, use_efficient_attention=False):
        super(ImageFeatureFusion, self).__init__()
        self.fusion =  uper.UPerHead(in_channels = in_channels)
        
        attn = []
        self.num_comm_enc = [0,1,2,4]
   
        for i in range(len(in_channels)):
            if self.num_comm_enc[i] > 0:
                attn.append(transformer.CommunicationBlock(in_channels[i], num_enc_sab = self.num_comm_enc[i], dim_hidden=in_channels[i], ln=True, dim_feedforward = in_channels[i], use_efficient_attention=use_efficient_attention))
        self.comm = nn.Sequential(*attn)  
    
    def forward(self, glc, nImgArray):
        batch_size = len(nImgArray)
        sum_nimg = int(torch.sum(nImgArray))
        
        out_fuse = []
        attn_cnt = 0
        for k in range(len(glc)):
            if self.num_comm_enc[k] > 0:
                in_fuse = glc[k]
                _, C, H, W = in_fuse.shape 
                in_fuse = in_fuse.reshape(-1, sum_nimg, C, H, W).permute(0, 3, 4, 1, 2) 
                K = in_fuse.shape[0] - 1
                in_fuse = in_fuse.reshape(-1, sum_nimg, C) 
                feats = []
                ids = 0
                for b in range(batch_size):
                    feat = in_fuse[:, ids:ids+int(nImgArray[b]), :] 
                    feat = self.comm[attn_cnt](feat)
                    feats.append(feat)
                    ids = ids + int(nImgArray[b])
                feats = torch.cat(feats, dim=1) 
                feats = feats.reshape(K+1, H*W, sum_nimg, C).permute(0, 2, 3, 1) 
                feats = feats.reshape((K+1)*sum_nimg, C, H, W) 
                out_fuse.append(feats)
                attn_cnt += 1
            else:
                out_fuse.append(glc[k])            
        out = self.fusion(out_fuse) 
        return out

class ScaleInvariantSpatialLightImageEncoder(nn.Module): # image feature encoder at canonical resolution
    def __init__(self, input_nc, use_efficient_attention=False):
        super(ScaleInvariantSpatialLightImageEncoder, self).__init__()
        self.backbone = ImageFeatureExtractor(input_nc)
        self.fusion = ImageFeatureFusion(self.backbone.out_channels, use_efficient_attention=use_efficient_attention)
        self.feat_dim = 256

    def forward(self, x, nImgArray, canonical_resolution):
        N, C, H, W = x.shape        
        mosaic_scale = H // canonical_resolution
        K = mosaic_scale * mosaic_scale

        """ (1a) resizing x to (Hc, Wc)"""
        x_resized = F.interpolate(x, size= (canonical_resolution, canonical_resolution), mode='bilinear', align_corners=True)

        """ (1b) decomposing x into K x K of (Hc, Wc) non-overlapped blocks (stride)"""           
        x_grid = divide_tensor_spatial(x, block_size=canonical_resolution, method='tile_stride') # (B, K, C, canonical_resolution, canonical_resolution)
        x_grid = x_grid.permute(1,0,2,3,4).reshape(-1, C, canonical_resolution, canonical_resolution) # (K*B, C, canonical_resolutioin, canonical_resolution)
  
        """(2a) feature extraction """
        x = self.fusion(self.backbone(x_resized), nImgArray) # atten along the light-axis
        f_resized = x.reshape(1, N, self.feat_dim, canonical_resolution//4 * canonical_resolution//4) # (1, N, C, canonical_resolution//4 * canonical_resolution//4)
        del x_resized

        """(2b) feature extraction """
        x = self.fusion(self.backbone(x_grid), nImgArray) # (K * N, C, canonical_resolution//4, canonical_resolution//4)
        x = x.reshape(K, N, x.shape[1], canonical_resolution//4, canonical_resolution//4)
        glc_grid = merge_tensor_spatial(x, method='tile_stride')
        del x_grid
       
        """ (3) upsample """
        glc_resized = F.interpolate(f_resized.reshape(N, self.feat_dim, canonical_resolution//4, canonical_resolution//4) , size= (H//4, W//4), mode='bilinear', align_corners=True)
        del f_resized

        glc = glc_resized + glc_grid
        return glc

 
class GLC_Upsample(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Upsample, self).__init__()       
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab=num_enc_sab, dim_hidden=dim_hidden, ln=True, dim_feedforward=dim_feedforward, use_efficient_attention=False)
       
    def forward(self, x):
        x = self.comm(x)        
        return x

class GLC_Aggregation(nn.Module):
    def __init__(self, input_nc, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Aggregation, self).__init__()              
        self.aggregation = transformer.AggregationBlock(dim_input=input_nc, num_enc_sab=num_agg_transformer, num_outputs=1, dim_hidden=dim_aggout, dim_feedforward=dim_feedforward, num_heads=8, ln=True, attention_dropout=0.1, use_efficient_attention=use_efficient_attention)

    def forward(self, x):
        x = self.aggregation(x)      
        return x

class Regressor(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, use_efficient_attention=False, dim_feedforward=256, output='normal'):
        super(Regressor, self).__init__()     
        # Communication among different samples (Pixel-Sampling Transformer)
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=input_nc, ln=True, dim_feedforward = dim_feedforward, use_efficient_attention=use_efficient_attention)   
        self.prediction_normal = PredictionHead(input_nc, 3)
        self.target = output
        if output == 'brdf':   
            self.prediction_base = PredictionHead(input_nc, 3) # No urcainty
            self.prediction_rough = PredictionHead(input_nc, 1)
            self.prediction_metal = PredictionHead(input_nc, 1)

    def forward(self, x, num_sample_set):
        """Standard forward
        INPUT: img [Num_Pix, F]
        OUTPUT: [Num_Pix, 3]"""  
        if x.shape[0] % num_sample_set == 0:
            x_ = x.reshape(-1, num_sample_set, x.shape[1])
            x_ = self.comm(x_)            
            x = x_.reshape(-1, x.shape[1])
        else:
            ids = list(range(x.shape[0]))
            num_split = len(ids) // num_sample_set
            x_1 = x[:(num_split)*num_sample_set, :].reshape(-1, num_sample_set, x.shape[1])
            x_1 = self.comm(x_1).reshape(-1, x.shape[1])
            x_2 = x[(num_split)*num_sample_set:,:].reshape(1, -1, x.shape[1])
            x_2 = self.comm(x_2).reshape(-1, x.shape[1])
            x = torch.cat([x_1, x_2], dim=0)

        x_n = self.prediction_normal(x)        
        if self.target == 'brdf':
            x_brdf = (self.prediction_base(x), self.prediction_rough(x), self.prediction_metal(x))
        else:
            x_brdf = []
        return x_n, x_brdf
    
class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PredictionHead, self).__init__()
        modules_regression = []
        modules_regression.append(nn.Linear(dim_input, dim_input//2))
        modules_regression.append(nn.ReLU())
        modules_regression.append(nn.Linear(dim_input//2, dim_output))
        self.regression = nn.Sequential(*modules_regression)

    def forward(self, x):
        return self.regression(x)