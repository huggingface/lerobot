import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=not bn)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=not bn)
        self.activation = activation
        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.align_corners = align_corners
        self.expand = expand
        out_features = features // 2 if expand else features

        self.res1 = ResidualConvUnit(features, activation, bn)
        self.res2 = ResidualConvUnit(features, activation, bn)
        self.out_conv = nn.Conv2d(features, out_features, 1)
        self.size = size

    def forward(self, *xs, size=None):
        x = xs[0]
        if len(xs) == 2 and xs[1] is not None:
            res = self.res1(xs[1])
            # ensure same spatial size
            if res.shape[2:] != x.shape[2:]:
                res = F.interpolate(res, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)
            x = x + res
        else:
            x = self.res1(x)

        x = self.res2(x)

        # Upsample
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=self.align_corners)
        elif self.size is not None:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=self.align_corners)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)

        x = self.out_conv(x)
        return x


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), deconv=False, bn=use_bn, expand=False, size=size)


class DPTHead(nn.Module):
    """Fixed DPTHead — accepts in_channels as int or list, handles token->map reshape properly."""
    def __init__(
        self,
        in_channels: Union[int, List[int]],
        features=256,
        use_bn=False,
        grid_size=16,
        out_channels=[256, 512, 1024, 1024],
        lvl=-1,
    ):
        super().__init__()
        # normalize in_channels to list
        if isinstance(in_channels, (list, tuple)):
            in_ch_list = list(in_channels)
        else:
            in_ch_list = [int(in_channels)] * len(out_channels)
        if len(in_ch_list) != len(out_channels):
            raise ValueError("in_channels must be an int or a list with same length as out_channels")

        self.grid_size = grid_size
        self.lvl = lvl
        self.features = features
        self.out_channels = out_channels
        self.in_ch_list = in_ch_list

        # projects: use per-layer in-channel sizes
        self.projects = nn.ModuleList([
            nn.Conv2d(in_ch_list[i], out_channels[i], kernel_size=1)
            for i in range(len(out_channels))
        ])

        # resize layers: choose upsample/conv/identity based on lvl
        self.resize_layers = nn.ModuleList()
        for i, oc in enumerate(out_channels):
            # compute factor: positive -> downsample by factor, negative -> upsample by factor
            factor_exp = (i + lvl)
            if factor_exp < 0:
                scale = 2 ** (-factor_exp)
                self.resize_layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True))
            elif factor_exp > 0:
                stride = 2 ** factor_exp
                # use conv stride to downsample (kernel=3,pad=1)
                self.resize_layers.append(nn.Conv2d(oc, oc, kernel_size=3, stride=stride, padding=1))
            else:
                self.resize_layers.append(nn.Identity())

        # scratch to reduce channels to `features`
        self.scratch = nn.ModuleList([
            nn.Conv2d(out_channels[i], features, kernel_size=3, stride=1, padding=1, bias=False)
            for i in range(len(out_channels))
        ])

        # refine blocks
        self.refine = nn.ModuleList([_make_fusion_block(features, use_bn) for _ in range(len(out_channels) - 1)])

    def forward(self, out_features: List[torch.Tensor]):
        outs = []
        # Step1: shape check + reshape token->map + project + resize
        for i, x in enumerate(out_features):
            if x.ndim != 3:
                raise ValueError(f"Each input must be (B, N, C). Got ndim={x.ndim} for index {i}")
            B, N, C = x.shape
            # infer grid (allow user-provided grid_size if matches)
            g = int(math.sqrt(N))
            if g * g != N:
                # try using constructor grid_size if it matches
                if self.grid_size is not None and self.grid_size * self.grid_size == N:
                    g = self.grid_size
                else:
                    raise ValueError(f"Input #{i} token length N={N} not a perfect square; can't reshape to map. "
                                     "Either provide (B,C,H,W) maps or tokens with N=H*W.")
            # reshape to (B, C, H, W)
            x_map = x.permute(0, 2, 1).reshape(B, C, g, g)

            # check project expecting in_ch_list
            expected_in_ch = self.in_ch_list[i]
            if C != expected_in_ch:
                raise RuntimeError(f"Layer {i}: token channel C={C} doesn't match DPTHead configured in_channels={expected_in_ch}. "
                                   "Either construct DPTHead with correct in_channels list or provide tokens with matching channel dim.")
            x_proj = self.projects[i](x_map)
            x_resized = self.resize_layers[i](x_proj)
            outs.append(x_resized)

        # Step2: scratch -> unify channels
        outs = [self.scratch[i](outs[i]) for i in range(len(outs))]

        # Step3: hierarchical refine from deep->shallow
        for i in range(len(outs) - 1, 0, -1):
            # upsample deeper to current spatial size
            up = F.interpolate(outs[i], size=outs[i - 1].shape[2:], mode='bilinear', align_corners=True)
            outs[i - 1] = self.refine[i - 1](outs[i - 1], up)

        # return the shallowest refined map
        return outs[0]
