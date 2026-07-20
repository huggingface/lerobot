"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import os
import torch
import numpy as np

def loadmodel(model, filename, strict=True):
    if os.path.exists(filename):
        params = torch.load('%s' % filename)
        model.load_state_dict(params,strict=strict)
        print('Loading pretrained model... %s ' % filename)
    else:
        print('Pretrained model not Found')
    return model

def mode_change(net, Training):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def print_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('# parameters: %d' % params)

def make_index_list(maxNumImages, numImageList):
    index = np.zeros((len(numImageList) * maxNumImages), np.int32) # (b*N,)
    for k in range(len(numImageList)):
        index[maxNumImages*k:maxNumImages*k+int(numImageList[k])] = 1
    return index
    
