import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from packaging.version import Version
# from xformers.ops import memory_efficient_attention


def find_next_divisible_by_8_numpy(n: np.ndarray) -> np.ndarray:
    """
    Finds the smallest integers greater than each element in a NumPy array 'n'
    that are divisible by 8. Assumes non-negative integers.

    Args:
        n: A NumPy array of integers.

    Returns:
        A NumPy array containing the smallest integers greater than each input element
        that are divisible by 8.
    """
    remainder = n % 8
    # Calculate the amount to add: 0 if already divisible, otherwise 8 - remainder
    # np.where is efficient for conditional operations on arrays
    amount_to_add = np.where(remainder == 0, 8, 8 - remainder)
    return n + amount_to_add


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.rand((bsize,), device=device).pow(1 / alpha)
    gamma2 = torch.rand((bsize,), device=device).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def prefix_query_segments(
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
    """Return prefix segment order after the image block.

    Task-specific query tokens are always placed after language tokens. Current
    task queries precede future task queries; future-depth remains the last
    query segment so the existing suffix-to-future-depth blocking can keep using
    the tail span.
    """
    segments = ["language"]
    if not use_depth_align:
        return tuple(segments)

    segments.append("current_depth")
    if use_future_video:
        if use_future_video_cls:
            segments.append("future_video_cls")
        if use_future_video_patch and not future_video_share_future_depth_query:
            segments.append("future_video")
    if use_future_depth:
        segments.append("future_depth")
    return tuple(segments)


def prefix_query_token_spans(
    prefix_len,
    num_task_tokens,
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
    """Return [start, end) spans for non-language task query segments."""
    counts = {
        "current_depth": num_task_tokens,
        "future_video_cls": 1,
        "future_video": num_task_tokens,
        "future_depth": num_task_tokens,
    }
    ordered = prefix_query_segments(
        use_depth_align=use_depth_align,
        use_future_depth=use_future_depth,
        use_future_video=use_future_video,
        use_future_video_cls=use_future_video_cls,
        use_future_video_patch=use_future_video_patch,
        future_video_share_future_depth_query=future_video_share_future_depth_query,
    )
    query_segments = [name for name in ordered if name != "language"]
    cursor = prefix_len - sum(counts[name] for name in query_segments)
    spans = {}
    for name in query_segments:
        count = counts[name]
        spans[name] = (cursor, cursor + count)
        cursor += count
    return spans


def fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch):
    """Return [start, end) of a tail query block inside the prefix.

    This legacy helper is still used for future-depth tail blocking in V2.
    New prefix layout code should prefer prefix_query_token_spans(), which also
    handles current-depth and separate future-video spans.
    """
    fv_len = (1 if use_cls else 0) + (num_task_tokens if use_patch else 0)
    return prefix_len - fv_len, prefix_len


def block_suffix_to_fv_(
    att_2d_masks, suffix_row_start, prefix_len, num_task_tokens, use_cls=False, use_patch=True, drop_mask=None
):
    """In-place mask out the suffix-to-future-video attention edge.

    `make_att_2d_masks`' cumsum scheme cannot express "a query cannot see a
    segment that precedes it", so we zero the rectangular [suffix rows, FV cols]
    block on the already-built 2D mask instead of touching mask_ar.

    att_2d_masks: bool[B, Q, K], True == visible. `suffix_row_start` is the first
    query row belonging to the suffix: prefix_len in the square training mask,
    0 in the suffix-only inference mask. Leaves FV -> img/lang rows untouched so
    the distillation query still reads the current observation.

    `drop_mask`: optional bool[B], True where this sample's suffix must NOT see
    FV. None == block every sample (hard mask). Used for per-sample stochastic
    masking (FV-attention dropout): keep = visible iff not dropped, applied via
    broadcast multiply so it stays a static graph under torch.compile.
    """
    fv_start, fv_end = fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch)
    if fv_end <= fv_start:
        return att_2d_masks
    if drop_mask is None:
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = False
    else:
        # keep[b] = True where the sample is NOT dropped -> AND keeps those rows
        # visible and zeros the dropped ones, with no data-dependent indexing.
        keep = (~drop_mask).view(-1, 1, 1)
        block = att_2d_masks[:, suffix_row_start:, fv_start:fv_end]
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = block & keep
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def our_eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Performs eager attention, optimized with torch.einsum.

    Args:
        query_states: Query tensor of shape [batch_size, seq_len, num_attention_heads, head_dim].
        key_states: Key tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        value_states: Value tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        attention_mask: Attention mask tensor, typically [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len, seq_len].

    Returns:
        Output tensor of shape [batch_size, seq_len, num_attention_heads * head_dim].
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads

    key_states = einops.repeat(key_states, "b l h d -> b l (h g) d", g=num_key_value_groups)
    value_states = einops.repeat(value_states, "b l h d -> b l (h g) d", g=num_key_value_groups)

    query_states_permuted = torch.einsum("blhd->bhld", query_states)
    key_states_permuted = torch.einsum("blhd->bhld", key_states)

    att_weights = torch.einsum("bhqd,bhkd->bhqk", query_states_permuted, key_states_permuted)
    att_weights *= head_dim**-0.5

    big_neg = -2.3819763e38
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

    probs = nn.functional.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    value_states_permuted = torch.einsum("blhd->bhld", value_states)  # [B, H, L_v, D]
    att_output = torch.einsum("bhqk,bhkv->bhqv", probs, value_states_permuted)  # [B, H, L_q, D]
    att_output = torch.einsum("bhld->blhd", att_output)  # [B, L, H, D]
    att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)

    return att_output


def our_sdpa_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """SDPA attention with the SAME (b, l, h, d) in / (b, l, h*d) out contract as
    ``our_eager_attention_forward``.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` — the same softmax attention
    (fidelity-preserving: identical math up to floating-point reassociation, no approximation),
    but it fuses the softmax and never materializes the ``[b, h, q, k]`` score matrix, so it is
    O(seq) in memory instead of O(seq^2) like the eager path. Torch-native (auto-selects the
    flash / memory-efficient / math backend), no compiled dependency. Grouped-query attention
    is handled by ``enable_gqa`` (num_kv_heads < num_att_heads). The default SDPA scale is
    ``1/sqrt(head_dim)``, matching the eager path.

    Args:
        query_states: ``[batch, seq, num_att_heads, head_dim]``.
        key_states / value_states: ``[batch, seq, num_kv_heads, head_dim]``.
        attention_mask: bool tensor, ``True`` = attend; ``[batch, seq, seq]`` or ``[batch, 1, seq, seq]``.
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_kv_heads = key_states.shape[2]

    # (b, l, h, d) -> (b, h, l, d)
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)

    mask = attention_mask
    if mask is not None:
        if mask.dim() == 3:  # (b, q, k) -> (b, 1, q, k) to broadcast over heads
            mask = mask.unsqueeze(1)
        if mask.dtype != torch.bool:
            mask = mask.bool()

    att_output = nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,  # bool: True keeps, False masks (matches the eager where-mask)
        enable_gqa=num_kv_heads != num_att_heads,
    )

    # (b, h, l, d) -> (b, l, h*d)
    att_output = att_output.transpose(1, 2).reshape(bsize, seq_len, num_att_heads * head_dim)
    return att_output


# @torch.jit.script
def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    original_dtype = x.dtype  # bf16
    d = x.shape[-1]
    d_half = d // 2
    device = x.device

    # Cast input to compute_dtype for all internal operations
    x_casted = x.to(dtype)
    positions_casted = positions.to(dtype)

    freq_exponents = (2.0 / d) * torch.arange(d_half, dtype=dtype, device=device)
    timescale = max_wavelength**freq_exponents
    radians = torch.einsum("bl,h->blh", positions_casted, 1.0 / timescale)  # fp32 -> bf16

    radians = radians[..., None, :]  # [B, L, 1, D_half]

    sin = torch.sin(radians)  # bf16
    cos = torch.cos(radians)  # bf16

    x1, x2 = x_casted.split(d_half, dim=-1)  # fp32

    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)  # fp32

    return res.to(original_dtype)  # bf16
