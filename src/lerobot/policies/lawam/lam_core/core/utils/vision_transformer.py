# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from .modules import Block
from .pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from .utils import trunc_normal_


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=5,
        tubelet_size=1,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        use_rope=False,
        handle_nonsquare_inputs=True,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.handle_nonsquare_inputs = handle_nonsquare_inputs

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Position embedding
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        # Attention Blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x, masks=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Tokenize input
        # Image
        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        # Video
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
            x = self.patch_embed(x)
            x += pos_embed
        else:
            x = self.patch_embed(x)

        # Mask away unwanted tokens (if masks provided)

        # Fwd prop
        outs = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, None, T=T, H_patches=H_patches, W_patches=W_patches, use_reentrant=False
                )
            else:
                x = blk(x, mask=masks, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already correct size, just return
            _, _, T, H, W = x.shape
            if H == self.img_height and W == self.img_width and T == self.num_frames:
                return pos_embed

            # Just chop off last N tokens of positional embedding
            elif H == self.img_height and W == self.img_width and T < self.num_frames:
                new_N = int((T // self.tubelet_size) * (H // self.patch_size) * (W // self.patch_size))
                return pos_embed[:, :new_N, :]

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T / N_t, H / N_h, W / N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already correct size, just return
            _, _, H, W = x.shape
            if H == self.img_height and W == self.img_width:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant_xformers(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# We do not use any of the following ViT definitions in V-JEPA 2, but retain them for
# compatibility reasons.
def vit_synthetic(patch_size=16, **kwargs):
    # For performance testing only
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1,
        depth=1,
        num_heads=1,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model




def vit_large_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant_xformers_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_gigantic(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1664,
        depth=48,
        num_heads=16,
        mpl_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_gigantic_xformers(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1664,
        depth=48,
        num_heads=26,
        mpl_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


VIT_EMBED_DIMS = {
    "vit_synthetic": 1,
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
    "vit_gigantic": 1664,
}
