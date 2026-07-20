"""
This code is based on SDM-UniPS [Scalable, Detailed and Mask-free Universal Photometric Stereo Network]
    by Satoshi IKEHATA
    https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdm_utils.decompose_tensors import divide_tensor_spatial, merge_tensor_spatial
from .sdm_utils import utils
from .sfpuel_utils import model as sfpuel

class SfPUEL(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        img_channel, polar_channel = in_channels
        attn_in_channel = img_channel + 255

        self.feature_extractor = sfpuel.FeatureExtractor(img_channel, polar_channel)
        self.image_attn_module = sfpuel.ImageAttnModule(attn_in_channel, dim_hidden_1=256, dim_hidden_2=384, dim_feedforward=1024, num_heads=8, ln=True, attn_dropout=0.1)
        self.pixel_attn_module = sfpuel.PixelAttnModule(384, dim_hidden=384, dim_feedforward=1024, num_heads=8, ln=True, attn_dropout=0.1)
        self.pred_head = sfpuel.OutputHead(384)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def preprocess(self, tensor, decoder_resolution):
        B, N, C, H, W = tensor.shape
        if H > decoder_resolution:
            tensor = tensor.reshape(-1, C, H, W)
            patches = divide_tensor_spatial(tensor, decoder_resolution, method='tile_stride')
            H, W = patches.shape[-2:]
            patches = patches.reshape(B, N, -1, C, H, W).permute(2, 0, 1, 3, 4, 5)
        else:
            patches = tensor[None]
        return patches
    
    def postprocess(self, patches_tensor):
        if patches_tensor.shape[0] > 1:
            merged_tensor = merge_tensor_spatial(patches_tensor, method='tile_stride')
        else:
            merged_tensor = patches_tensor[0]
        return merged_tensor
    
    def forward(self, inputs, decoder_resolution=512, canonical_resolution=256, pixel_samples=10000):
        polar, mask = inputs[0]['polar'], inputs[0]['mask']
        pprop = torch.cat([inputs[1][item] for item in inputs[1]], 1)
        B, Nmax, C, H, W = polar.shape
        ddevice = polar.device
        
        im_patches = self.preprocess(polar, decoder_resolution)
        if H > decoder_resolution:
            mask_patches = divide_tensor_spatial(mask, decoder_resolution, method='tile_stride')
            mask_patches = mask_patches.transpose(0, 1)
            pprop_patches = divide_tensor_spatial(pprop, decoder_resolution, method='tile_stride')
            stks_patches = stks_patches.transpose(0, 1)
        else:
            mask_patches = mask[None]
            pprop_patches = pprop[None]

        K, B, Nmax, C, H, W = im_patches.shape
        n_patches = torch.ones((K, B, 3, H, W),device=ddevice)
        prob_patches = torch.ones((K, B, 2, H, W),device=ddevice)
        num_im = torch.tensor([Nmax]*B, device=ddevice)
        for patch_id in range(K):
            im_patch = im_patches[patch_id]
            mask_patch = mask_patches[patch_id]
            pprop_patch = pprop_patches[patch_id]
            pprop_patch = pprop_patch * mask_patch
            im_patch_cm = im_patch.reshape(-1, C, H, W)
            mask_patch_cm = mask_patch.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W)
            im_masked = torch.cat([im_patch_cm * mask_patch_cm, mask_patch_cm], dim=1)
            ext_feat = self.feature_extractor(im_masked, pprop_patch, num_im, canonical_resolution)
            decoder_imgsize = (decoder_resolution, decoder_resolution)
            mask_patch_dec = mask_patch

            nout = torch.zeros((B, H * W, 3), device=ddevice)
            probout = torch.zeros((B, H * W, 2), device=ddevice)

            f_scale = decoder_resolution//canonical_resolution
            smoothing = utils.gauss_filter(ext_feat.shape[1], 10*f_scale + 1, 1).to(ddevice)
            ext_feat = smoothing(ext_feat)
            
            p = 0
            for b in range(B):
                target = range(p, p+int(num_im[b]))
                p = p+int(num_im[b])
                m_ = mask_patch_dec[b, :, :, :].reshape(-1, H*W).permute(1,0)
                ids = np.nonzero(m_>0)[:,0]
                ids = ids[np.random.permutation(len(ids))]
                if len(ids) > pixel_samples:
                    num_split = len(ids) // pixel_samples + 1
                    idset = np.array_split(ids, num_split)
                else:
                    idset = [ids]     
                
                im_flat = im_patch_cm[target, :, :, :].reshape(int(num_im[b]), C, H*W).permute(2,0,1)
                for ids in idset:
                    im_flat_spl = im_flat[ids, :, :]
                    coords = utils.ind2coords(decoder_imgsize, ids, ddevice).expand(int(num_im[b]),-1,-1,-1)
                    ext_feat_spl = F.grid_sample(ext_feat[target, :, :, :], coords, mode='bilinear', align_corners=False).reshape(len(target), -1, len(ids)).permute(2,0,1)
                    enh_feat = self.image_attn_module(ext_feat_spl, im_flat_spl)
                    glb_feat = self.pixel_attn_module(enh_feat, coords[:1])
                    
                    x_n, x_prob = self.pred_head(glb_feat)
                    probout[b, ids, :] = x_prob
                    nout[b, ids, :] = F.normalize(x_n, dim=1, p=2)

            n_patches[patch_id] = nout.permute(0, 2, 1).reshape(B, 3, H, W)
            prob_patches[patch_id] = probout.permute(0, 2, 1).reshape(B, 2, H, W)
        
        normal = self.postprocess(n_patches)
        material_est = self.postprocess(prob_patches)
        outputs = {
            'normal' : normal,
            'material_est': material_est
        }
        return outputs
    
def generate_model(in_channels):
    model = SfPUEL(in_channels)
    return model
