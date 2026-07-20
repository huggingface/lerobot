"""This file is modified version from mmsegmentation (https://github.com/open-mmlab/mmsegmentation)"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
                    nn.ReLU()
                    )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)

            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False)

            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels = (96, 192, 384, 768), channels = 256, pool_scales=(1, 2, 3, 6),):
        super(UPerHead, self).__init__()
        # PSP Module
        self.in_channels = in_channels
        self.channels = channels
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU())
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channels, kernel_size=1, padding=0),
            nn.ReLU())


            fpn_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU())

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU())


    def psp_forward(self, inputs):
        """Forward function of PSP module."""

        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        """Forward function.
        inputs = {x_96, x_192, x_384, x_768}
        """

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size = prev_shape,
                mode='bilinear',
                align_corners = False
                )
        
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        
        # append psp feature
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)

        return output