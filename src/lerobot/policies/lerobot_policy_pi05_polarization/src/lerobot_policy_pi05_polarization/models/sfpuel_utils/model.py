import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
#print(sys.path)
                

from models.sdm_utils import transformer
from models.sdm_utils.decompose_tensors import divide_tensor_spatial, merge_tensor_spatial
import models.sdm_utils.model as sdm
from . import convnext
from . import utils as util
from . import EfficientAttn as ea
# import convnext
# import utils as util
# import EfficientAttn as ea

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

class ZeroConvOut(nn.Module):
    def __init__(self, in_channels, out_channel=None):
        super().__init__()
        if isinstance(in_channels, (list,tuple)):
            self.zeros_convs = nn.ModuleList([nn.Conv2d(in_chan, in_chan, 1, padding=0) for in_chan in in_channels])
        elif isinstance(in_channels, int):
            if out_channel is None:
                out_channel = in_channels
            self.zeros_convs = nn.ModuleList([nn.Conv2d(in_channels, out_channel, 1, padding=0)])
        else:
            raise Exception(f'Unexpected type {type(in_channels)}')
    def forward(self, x):
        if isinstance(x, (list,tuple)):
            feats = []
            for idx, conv in enumerate(self.zeros_convs):
                feats.append(conv(x[idx]))
        else:
            feats = self.zeros_convs[0](x)
        return feats

class PolarEncoder(nn.Module):
    def __init__(self, in_channel, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            *[nn.Conv2d(in_channel, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, out_dim, 3, padding=1),
            nn.SiLU(),
            ])
        self.encoder_zero = nn.Sequential(*[ZeroConvOut(out_dim)])
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_zero(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        back = []
        out_channels = (96, 192, 384, 768)
        back.append(convnext.ConvNeXt(in_chans=input_nc, use_checkpoint=False))      
        self.backbone = nn.Sequential(*back)
        self.out_channels = out_channels
        
    def forward(self, x):
        feats = self.backbone(x) # [(B, 96, 32, 32), (B, 192, 16, 16), (B, 384, 8, 8), (B, 768, 4, 4)]
        return feats

class PolPSFusers(nn.Module):
    def __init__(self, out_channels, num_heads=4):
        super().__init__()
        fusers, mlps = [], []
        for ch in out_channels:
            cross_attn = ea.EfficientAdditiveAttnetion(ch, ch, num_heads)
            fusers.append(cross_attn)
            mlp = ea.EfficientAdditiveAttnetionMLP(ch)
            mlps.append(mlp)
        
        self.fusers = nn.ModuleList(fusers)
        self.mlps = nn.ModuleList(mlps)


    def forward(self, polar_feats, ps_feats, dolp=None):
        feats = []
        if dolp is not None:
            droprate = ea.get_dolp_droprate(dolp).repeat(4, 1, 1, 1)
            for idx, fuser in enumerate(self.fusers):
                ps_feat, polar_feat = ps_feats[idx], polar_feats[idx]
                hw = (ps_feat.shape[-2], ps_feat.shape[-1])
                droprate_ = F.interpolate(droprate, size=hw, mode='bilinear', align_corners=True).clamp(0, 1)
                ps_feat_ = ps_feat.reshape([*ps_feat.shape[:-2], -1]).transpose(1, 2).contiguous()
                polar_feat_ = polar_feat.reshape([*polar_feat.shape[:-2], -1]).transpose(1, 2).contiguous()
                droprate_ = droprate_.reshape([*droprate_.shape[:-2], -1]).transpose(1, 2).contiguous()
                fused_feat_ = fuser(polar_feat_, ps_feat_, droprate_)
                fused_feat = fused_feat_.transpose(1, 2).reshape(polar_feat.shape).contiguous() + polar_feat
                fused_feat = self.mlps[idx](fused_feat) + fused_feat
                feats.append(fused_feat)
        else:
            for idx, fuser in enumerate(self.fusers):
                ps_feat, polar_feat = ps_feats[idx], polar_feats[idx]
                hw = (ps_feat.shape[-2], ps_feat.shape[-1])
                ps_feat_ = ps_feat.reshape([*ps_feat.shape[:-2], -1]).transpose(1, 2).contiguous()
                polar_feat_ = polar_feat.reshape([*polar_feat.shape[:-2], -1]).transpose(1, 2).contiguous()
                fused_feat_ = fuser(polar_feat_, ps_feat_)
                fused_feat = fused_feat_.transpose(1, 2).reshape(polar_feat.shape).contiguous() + polar_feat
                fused_feat = self.mlps[idx](fused_feat) + fused_feat
                feats.append(fused_feat)
        return feats


class FeatureExtractor(nn.Module):
    def __init__(self, img_ch, stokes_ch):
        super().__init__()
        self.pspe = sdm.ImageFeatureExtractor(img_ch)
        self.polfe = ImageEncoder(img_ch)
        self.pol_ps_fuser = PolPSFusers(self.pspe.out_channels)
        self.pspe_fusion = sdm.ImageFeatureFusion(self.pspe.out_channels)
        self.polfe_fusion = sdm.ImageFeatureFusion(self.pspe.out_channels)
        self.feat_dim = 256
        self.polfe_fusion_zc = ZeroConvOut(self.feat_dim)
        self.polar_encoder = PolarEncoder(stokes_ch, self.polfe.out_channels[0])

    def zero_init(self):
        util.zero_module(self.polfe_fusion_zc)
        util.zero_module(self.polar_encoder.encoder_zero)

    def forward(self, x, polar, nImgArray, canonical_resolution):
        N, C, H, W = x.shape
        polar_ch = polar.shape[1]
        mosaic_scale = H // canonical_resolution
        K = mosaic_scale * mosaic_scale
        """ (1a) resizing x to (Hc, Wc)"""
        x_resized = F.interpolate(x, size=(canonical_resolution, canonical_resolution), mode='bilinear', align_corners=True)
        polar_resized = F.interpolate(polar, size=(canonical_resolution, canonical_resolution), mode='bilinear', align_corners=True)
        dolp_resized = polar_resized[:, -1:]
        """ (1b) decomposing x into K x K of (Hc, Wc) non-overlapped blocks (stride)"""           
        if H > canonical_resolution:
            x_grid = divide_tensor_spatial(x, block_size=canonical_resolution, method='tile_stride') 
            polar_grid = divide_tensor_spatial(polar, block_size=canonical_resolution, method='tile_stride') 
        else:
            x_grid = x.unsqueeze(1)
            polar_grid = polar.unsqueeze(1)
        dolp_grid = polar_grid[:, :, -1:]
        dolp_grid = dolp_grid.transpose(0, 1).reshape(-1, 1, canonical_resolution, canonical_resolution)
        x_grid = x_grid.permute(1,0,2,3,4).reshape(-1, C, canonical_resolution, canonical_resolution) 
        polar_grid = polar_grid.permute(1,0,2,3,4).reshape(-1, polar_ch, canonical_resolution, canonical_resolution) 
        """(2a) feature extraction """
        pspe_feats = self.pspe(x_resized)
        polar_resized_feats = self.polar_encoder(polar_resized)
        polfe_feats = self.polfe([x_resized, polar_resized_feats])
        pspe_feats = self.pol_ps_fuser(pspe_feats, polfe_feats, dolp_resized)
        pspe_fusion_feat = self.pspe_fusion(pspe_feats, nImgArray)
        polfe_fusion_feat = self.polfe_fusion(polfe_feats, nImgArray)
        polfe_fusion_zc_feat = self.polfe_fusion_zc(polfe_fusion_feat)
        pspe_fusion_feat = pspe_fusion_feat + polfe_fusion_zc_feat
        """ (3a) upsample """
        ext_resized = F.interpolate(pspe_fusion_feat, size=(H//4, W//4), mode='bilinear', align_corners=True)
        """(2b) feature extraction """
        pspe_feats = self.pspe(x_grid)
        polar_grid_feats = self.polar_encoder(polar_resized)
        polfe_feats = self.polfe([x_grid, polar_grid_feats])
        pspe_feats = self.pol_ps_fuser(pspe_feats, polfe_feats, dolp_grid)
        pspe_fusion_feat = self.pspe_fusion(pspe_feats, nImgArray)
        polfe_fusion_feat = self.polfe_fusion(polfe_feats, nImgArray)
        polfe_fusion_zc_feat = self.polfe_fusion_zc(polfe_fusion_feat)
        pspe_fusion_feat = pspe_fusion_feat + polfe_fusion_zc_feat
        pspe_fusion_feat = pspe_fusion_feat.reshape(K, N, pspe_fusion_feat.shape[1], canonical_resolution//4, canonical_resolution//4)
        ext_grid = merge_tensor_spatial(pspe_fusion_feat, method='tile_stride')
        ext_feat = ext_resized + ext_grid
        return ext_feat

class ImageAttnModule(nn.Module):
    def __init__(self, in_ch, dim_hidden_1=256, dim_hidden_2=384, dim_feedforward=1024, num_heads=8, ln=True, attn_dropout=0.1):
        super().__init__()
        self.attn_blk_1 = nn.Sequential(*[
            transformer.SAB(in_ch, dim_hidden_1, num_heads=num_heads, ln=ln, attention_dropout=attn_dropout, dim_feedforward=dim_feedforward),
            transformer.SAB(dim_hidden_1, dim_hidden_1, num_heads=num_heads, ln=ln, attention_dropout=attn_dropout, dim_feedforward=dim_feedforward),
        ])
        self.attn_blk_2 = nn.Sequential(*[
            transformer.SAB(in_ch, dim_hidden_2, num_heads=num_heads, ln=ln, attention_dropout=attn_dropout, dim_feedforward=dim_feedforward),
            transformer.SAB(dim_hidden_2, dim_hidden_2, num_heads=num_heads, ln=ln, attention_dropout=attn_dropout, dim_feedforward=dim_feedforward),
            transformer.SAB(dim_hidden_2, dim_hidden_2, num_heads=num_heads, ln=ln, attention_dropout=attn_dropout, dim_feedforward=dim_feedforward),
        ])

    def forward(self, feat, img):
        x = torch.cat([img, feat], dim=2)
        x = self.attn_blk_1(x)
        
        x = torch.cat([img, x], dim=2)
        x = self.attn_blk_2(x)
        return x

class PixelAttnModule(nn.Module):
    def __init__(self, in_ch, dim_hidden=384, dim_feedforward=1024, num_heads=8, ln=True, attn_dropout=0.1):
        super().__init__()
        self.emb_dim = dim_hidden
        self.geo_emb = nn.Parameter(torch.Tensor(1, dim_hidden, 256, 256)) 
        self.cross_attn_module = transformer.MultiHeadSelfAttentionBlock(dim_hidden, dim_hidden, num_heads, ln=False, attn_mode='Normal')
        self.pixel_attn_module = nn.Sequential(*[
            transformer.SAB(in_ch, dim_hidden, num_heads=num_heads, ln=ln, attention_dropout=attn_dropout, dim_feedforward=dim_feedforward, attn_mode='Efficient'),
            transformer.SAB(dim_hidden, dim_hidden, num_heads=num_heads, ln=ln, attention_dropout=attn_dropout, dim_feedforward=dim_feedforward, attn_mode='Efficient'),
        ])

    def forward(self, feat, coords):
        geo_query_emb = F.grid_sample(self.geo_emb, coords, mode='nearest', align_corners=True).reshape(1, self.emb_dim, -1).permute(2, 0 ,1)
        pix_feat = self.cross_attn_module(geo_query_emb, feat)        
        pix_feat = pix_feat.permute(1, 0, 2)
        glb_pix_feat = self.pixel_attn_module(pix_feat)
        return glb_pix_feat

class OutputHead(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        self.n_head = PredictionHead(input_nc, 3)
        self.m_head = PredictionHead(input_nc, 2)

    def forward(self, x):
        x = x.squeeze(dim=0)
        x_n = self.n_head(x)        
        x_m_prob = self.m_head(x)
        return x_n, x_m_prob


if __name__ == "__main__":
    model = FeatureExtractor(3, 5)
    #from torchsummary import summary
    #summary(model, [(3, 512, 512), (96, 128, 128)] ) # not working, missing arguments
    #print(model) # too long

    with torch.no_grad():
        image = torch.randn(1, 3, 512, 512)
        polar_features = torch.randn(1, 96, 128, 128)
        feats = model.polfe(
            [image, polar_features]
     )


    print(type(feats)) # <class 'tuple'>
    print(len(feats)) # 4
    for i,f in enumerate(feats):
        print(i, f.shape) # 0 torch.Size([1, 96, 128, 128])
                         # 1 torch.Size([1, 192, 64, 64])
                         # 2 torch.Size([1, 384, 32, 32])
                         # 3 torch.Size([1, 768, 16, 16])



    fusion = model.polfe_fusion(
        feats,
        nImgArray=torch.tensor([1])
    )

    print(fusion.shape) # torch.Size([1, 256, 128, 128])

    # import inspect
    # #print(model.polfe_fusion)
    # print(inspect.signature(model.polfe_fusion.forward))
    # print(inspect.getsource(model.polfe_fusion.forward))