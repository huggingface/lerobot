#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/siglip2/modular_siglip2.py.
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.configuration_utils import PretrainedConfig

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.models.siglip2.configuration_siglip2 import Siglip2Config, Siglip2TextConfig
from collections import defaultdict
from itertools import accumulate
from math import isqrt
from typing import Dict


logger = logging.get_logger(__name__)


import inspect
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal, logging
from transformers.integrations.flash_attention import flash_attention_forward as original_flash_attention_forward
flash_241 = is_flash_attn_greater_or_equal("2.4.1")
deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
    ):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.
    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_sliding_windows (`bool`, *optional*):
            Whether to activate sliding window attention.
    """
    
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = is_causal and query_length != 1
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}
    if flash_241:
        if deterministic is None:
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states[0],
        key_states[0],
        value_states[0],
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        max_seqlen_q=max_length_q,
        max_seqlen_k=max_length_k,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
        )


    return attn_output


from transformers.utils import is_flash_attn_greater_or_equal_2_10
_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

def flash_attention_forward_for_packing(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    seq_len_list: Optional[List[int]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)
    

    cu_seqlens = F.pad(torch.cumsum(torch.tensor(seq_len_list, device=query.device, dtype=torch.int32), dim=0), (1, 0))
    cu_seqlens = cu_seqlens.to(torch.int32)
    max_seq_len = max(seq_len_list)
    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seq_len,
        max_length_k=max_seq_len,
        **kwargs,
    )
    return attn_output.squeeze(0), None

# General docstring
_CONFIG_FOR_DOC = "Siglip2Config"



class Siglip2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Siglip2VisionModel`]. It is used to instantiate a
    Siglip2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip2
    [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        num_patches (`int`, *optional*, defaults to 256):
            The number of patches in the image with the size of (`patch_size`, `patch_size`).
            The image is resized to fill maximum of this number of patches, and to preserve
            the aspect ratio. In case the resulted number of patches is lower, the image is
            padded in "patch" dimension.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import Siglip2VisionConfig, Siglip2VisionModel

    >>> # Initializing a Siglip2VisionConfig with google/siglip2-base-patch16-naflex style configuration
    >>> configuration = Siglip2VisionConfig()

    >>> # Initializing a Siglip2VisionModel (with random weights) from the google/siglip2-base-patch16-naflex style configuration
    >>> model = Siglip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        num_patches=256,
        patch_size=14, # manully modified
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        window_size=14, # 
        full_attention_indexes=[7, 14, 21, 26],
        use_rope=True,
        use_windows_attn=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_patches = num_patches
        self.window_size = window_size
        self.full_attention_indexes = full_attention_indexes
        self.use_windows_attn = use_windows_attn
        self.use_rope = use_rope


@dataclass
class Siglip2VisionOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    spatial_shapes: Optional[torch.LongTensor] = None


def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 3D tensor image of shape (num_channels, image_height, image_width) into 2D tensor of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image

def convert_images_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 4D tensor image of shape (batch_size, num_channels, image_height, image_width) into 2D tensor of patches of shape
    (batch_size, num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    batch_size, num_channels, image_height, image_width = image.shape
    assert image_height % patch_size == 0 and image_width % patch_size == 0, f"image_height % patch_size == 0 and image_width % patch_size == 0"
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(batch_size, num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(0, 2, 4, 3, 5, 1) # (batch_size, num_patches_height, num_patches_width, patch_size, patch_size, num_channels)
    
    patched_image = patched_image.reshape(batch_size * num_patches_height * num_patches_width, -1)
    return patched_image


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.window_size = config.window_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)


    def split_patch_embeddings_to_windows_with_meta(self, patch_embeds, batch_hw, window_size):
        """
        Args:
            patch_embeds: Tensor, shape (1, sum(H_i*W_i), C)
            batch_hw:    List[(H_i, W_i)]
            window_size: int

        Returns:
            windows_tensor: Tensor, shape (total_windows, window_size*window_size, C)
            win_meta_list:      List[dict] with keys:
                - img_idx:   index in batch_hw
                - patch_hw:  original (H, W)
                - win_xy:    (h0, w0) 左上角相对于原图
                - win_hw:    原图内有效窗口大小 (h_eff, w_eff)
        """

        # 1. 计算每张图在 flat tensor 中的起始位置
        batch_hw = batch_hw.tolist()
        counts = [H * W for (H, W) in batch_hw]
        starts = [0] + list(accumulate(counts))[:-1]

        # 2. 按 (H,W) 分组，同一尺寸一起处理
        size2info = defaultdict(list)
        for img_idx, ((H, W), start) in enumerate(zip(batch_hw, starts)):
            size2info[(H, W)].append((img_idx, start))

        all_windows = []
        all_meta    = []
        # print(size2info)
        # 3. 对每个尺寸组做 batch unfold + pad
        for (H, W), info in size2info.items():
            H, W = int(H), int(W)
            B = len(info)
            C = patch_embeds.shape[-1]
            img_idxs, img_starts = zip(*info)

            # 3.1 取出并 reshape 成 (B, C, H, W)
            imgs = []
            for st in img_starts:
                flat = patch_embeds[0, st: st + H * W]            # (H*W, C)
                imgs.append(flat.transpose(0,1).reshape(C, H, W))
            batch_tensor = torch.stack(imgs, dim=0)            # (B, C, H, W)
            # 3.2 计算 pad 大小 (bottom, right)，保证能被 window_size 整除
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size

            # pad 格式： (left, right, top, bottom) for last two dims
            batch_padded = F.pad(batch_tensor, (0, pad_w, 0, pad_h))

            H_pad, W_pad = H + pad_h, W + pad_w
            n_h = H_pad // window_size
            n_w = W_pad // window_size
            n_windows = n_h * n_w

            # 3.3 batched unfold -> (B, C*ws*ws, n_windows)
            patches_unf = F.unfold(
                batch_padded,
                kernel_size=(window_size, window_size),
                stride=(window_size, window_size)
            )

            # 3.4 reshape到 (B*n_windows, ws*ws, C)
            patches = (
                patches_unf
                .view(B, C, window_size * window_size, n_windows)  # (B, C, ws*ws, n_win)
                .permute(0, 3, 2, 1)                               # (B, n_win, ws*ws, C)
                .reshape(-1, window_size * window_size, C)        # (B*n_win, ws*ws, C)
            )
            all_windows.append(patches)

            # 3.5 生成 meta：记录原图内有效窗口大小
            for b, img_idx in enumerate(img_idxs):
                for win_id in range(n_windows):
                    i, j = divmod(win_id, n_w)
                    h0, w0 = i * window_size, j * window_size
                    # 在原图内的实际结束坐标
                    h1 = min(h0 + window_size, H)
                    w1 = min(w0 + window_size, W)
                    all_meta.append({
                        'img_idx':  img_idx,
                        'patch_hw': (H, W),
                        'win_xy':   (h0, w0),
                        'win_hw':   (h1 - h0, w1 - w0),  # 有效区域大小
                    })

        # 4. 拼接并根据 img_idx + win_xy 排序，恢复输入顺序
        sorted_idx = sorted(
            range(len(all_meta)),
            key=lambda k: (
                all_meta[k]['img_idx'],
                all_meta[k]['win_xy'][0],
                all_meta[k]['win_xy'][1]
            )
        )
        all_windows = torch.cat(all_windows, dim=0)
        all_windows = all_windows[sorted_idx]
        win_meta_list      = [all_meta[i] for i in sorted_idx]

        windows_list = []
        for meta, win in zip(win_meta_list, all_windows):
            h_eff, w_eff = meta['win_hw']
            valid_num = h_eff * w_eff
            # 只保留真正来自原图的 patch tokens
            if valid_num  == window_size * window_size:
                windows_list.append(win) 
            else:
                win = win.view(window_size, window_size, -1)[:h_eff, :w_eff, :].reshape(h_eff * w_eff, -1)
                windows_list.append(win)  # shape (valid_num, C)

        # 如果你需要一个单一 tensor，可以再 cat 一次：
        all_tokens = torch.cat(windows_list, dim=0).unsqueeze(0)  # shape (sum(valid_num), C)
        
        
        # 1. 先重算每张图在原始 flat tensor 中的起始位置
        counts = [H * W for H, W in batch_hw]
        starts = [0] + list(accumulate(counts))[:-1]
        total_patches = sum(counts)

        # 2. 构造映射：mapping[orig_idx] = new_idx
        mapping = [None] * total_patches
        offset = 0  # all_tokens 维度上的游标

        for meta in win_meta_list:
            img_idx = meta['img_idx']
            H, W      = meta['patch_hw']
            h0, w0    = meta['win_xy']
            h_eff, w_eff = meta['win_hw']
            base = starts[img_idx]

            # 对该窗口内所有真正来自原图的 patch token 计算映射
            for u in range(h_eff):
                for v in range(w_eff):
                    # 原始 flat 坐标
                    orig_idx = base + (h0+u) * W + (w0) + v
                    # 在 all_tokens 里的位置：在该窗口区段里按 row-major 展平
                    p = u * w_eff + v
                    mapping[orig_idx] = offset + p

                # 窗口结束后，offset 推进该窗口的有效 token 数
            offset += h_eff * w_eff
        reverse_mapping = torch.tensor(mapping, dtype=torch.long)

        return all_tokens, win_meta_list, reverse_mapping

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = []

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings.append(resized_embeddings)

        return torch.cat(resulted_positional_embeddings, dim=0).unsqueeze(0)

    def get_spatial_shapes(self, bchw_list: List[torch.Tensor]) -> torch.Tensor:
        hw_list = []
        for shape in bchw_list:
            b, _, h, w = shape
            hw_list.extend([(h//self.patch_size, w//self.patch_size)] * b)  
        hw_tensor = torch.tensor(hw_list)  
        return hw_tensor

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, num_channels, height, width)
        """

        bchw_list = [each.shape for each in pixel_values]
        
        pixel_values = torch.cat([convert_images_to_patches(each, self.patch_size) for each in pixel_values], dim=0)

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        spatial_shapes = self.get_spatial_shapes(bchw_list)
        
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes
        )
        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings

        windows_tensor, win_meta_list, reverse_mapping = self.split_patch_embeddings_to_windows_with_meta(embeddings, spatial_shapes, self.window_size)
        
        return windows_tensor, win_meta_list, spatial_shapes, reverse_mapping


class Rope2DPosEmb(nn.Module):
    
    """
    copy from https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking/blob/main/modeling_kimi_vl.py#L324
    2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000, window_size=14):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.window_size = window_size

        self.freqs_cis = None

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, win_meta_list: List[Dict], device: torch.device) -> torch.Tensor:
        """
        Args:
            win_meta_list (List[Dict]): window meta list
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(device)
        
        # assert all xy <512
        assert all(win_meta['win_xy'][0] + win_meta['win_hw'][0] < 512 and win_meta['win_xy'][1] + win_meta['win_hw'][1] < 512 for win_meta in win_meta_list)
        freqs_cis = torch.cat([self.freqs_cis[win_meta['win_xy'][0]:win_meta['win_xy'][0] + win_meta['win_hw'][0], win_meta['win_xy'][1]: win_meta['win_xy'][1] + win_meta['win_hw'][1]].reshape(-1, self.dim // 2) for win_meta in win_meta_list], dim=0)
        freqs_cis = freqs_cis.unsqueeze(0)
        return freqs_cis
    

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Union[Siglip2VisionConfig, Siglip2TextConfig]):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_windows_attn = config.use_windows_attn
        self.use_rope = config.use_rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        rope_freqs_cis: Optional[torch.Tensor] = None,
        win_meta_list: Optional[List[Dict]] = None,
        windows_attn: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""


        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)


        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim) # .transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim) # .transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            queries, keys = apply_rope(queries, keys, rope_freqs_cis)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            if self.config._attn_implementation == "flash_attention_2":
                from transformers.modeling_utils import AttentionInterface
                AttentionInterface._global_mapping['flash_attention_2_packing'] = flash_attention_forward_for_packing
                setattr(AttentionInterface, 'flash_attention_2_packing', flash_attention_forward_for_packing)
                attention_interface = ALL_ATTENTION_FUNCTIONS['flash_attention_2_packing']
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if windows_attn and self.use_windows_attn:
            seq_len_list = [win_meta['win_hw'][0] * win_meta['win_hw'][1] for win_meta in win_meta_list]
        else:
            mapper = defaultdict(lambda: 0)
            for win_meta in win_meta_list:
                mapper[win_meta['img_idx']]  += win_meta['win_hw'][0] * win_meta['win_hw'][1]
            seq_len_list = [mapper[i] for i in range(len(mapper))]

        attention_mask = None
        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            seq_len_list=seq_len_list,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Union[Siglip2VisionConfig, Siglip2TextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        rope_freqs_cis: Optional[torch.Tensor] = None,
        win_meta_list: Optional[List[Dict]] = None,
        windows_attn: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            rope_freqs_cis=rope_freqs_cis,
            win_meta_list=win_meta_list,
            windows_attn=windows_attn,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2Config
    """

    def __init__(self, config: Siglip2Config):
        super().__init__()
        self.config = config

        self.rope_2d = Rope2DPosEmb(
            config.hidden_size // config.num_attention_heads, 512, 512, config.window_size
        )
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.full_attention_indexes = config.full_attention_indexes
        
 
    # Ignore copy
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        win_meta_list: Optional[List[Dict]] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        rope_freqs_cis = self.rope_2d.get_freqs_cis(win_meta_list=win_meta_list, device=inputs_embeds.device)

        hidden_states = inputs_embeds
        for win_idx, encoder_layer in enumerate(self.layers):
            
            if win_idx not in self.full_attention_indexes:
                windows_attn = True
            else:
                windows_attn = False
                
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    output_attentions,
                    rope_freqs_cis,
                    win_meta_list,
                    windows_attn
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    output_attentions=output_attentions,
                    rope_freqs_cis=rope_freqs_cis,
                    win_meta_list=win_meta_list,
                    windows_attn=windows_attn
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


def reconstruct_patch_embeddings(last_hidden_state: torch.Tensor, win_meta_list: list[dict], spatial_shapes: torch.Tensor) -> torch.Tensor:

    idx_map = build_idx_map(win_meta_list, spatial_shapes)
    last_hidden_state = last_hidden_state[:, idx_map, :]
    return last_hidden_state

SIGLIP2_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    @can_return_tuple
    @add_start_docstrings_to_model_forward(SIGLIP2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Siglip2VisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        windows_tensor, win_meta_list, spatial_shapes, reverse_mapping = self.embeddings(pixel_values)
        
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=windows_tensor,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            win_meta_list=win_meta_list,
            spatial_shapes=spatial_shapes,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state = last_hidden_state[:, reverse_mapping, :]
        return Siglip2VisionOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            spatial_shapes=spatial_shapes,
        )


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


SIGLIP2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Siglip2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SIGLIP2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Siglip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Siglip2Config
    base_model_prefix = "siglip2"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Siglip2TextEmbeddings",
        "Siglip2EncoderLayer",
        "Siglip2VisionEmbeddings",
        "Siglip2EncoderLayer",
        "Siglip2MultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Siglip2VisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, Siglip2Config)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, Siglip2Attention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, Siglip2MLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, Siglip2MultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, Siglip2Model):
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        elif isinstance(module, Siglip2ForImageClassification):
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


@add_start_docstrings(
    """The vision model from Siglip2 without any head or projection on top.""",
    SIGLIP2_START_DOCSTRING,
)
class Siglip2VisionModel(Siglip2PreTrainedModel):
    config_class = Siglip2VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)

        self.vision_model = Siglip2VisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @can_return_tuple
    @add_start_docstrings_to_model_forward(SIGLIP2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Siglip2VisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Siglip2VisionModel

        >>> model = Siglip2VisionModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


__all__ = [
    "Siglip2VisionModel",
]