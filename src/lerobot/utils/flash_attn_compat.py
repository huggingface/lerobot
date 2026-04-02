"""FlashAttention-4 (CuTeDSL) imports and related utilities.

Provides a single import point for flash attention functions and padding/rotary
helpers used across policies. Requires the ``flash-attn-4`` package.
"""

import torch
import torch.nn.functional as F
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "index_first_axis",
    "unpad_input",
    "pad_input",
    "apply_rotary_emb",
]


# ---------------------------------------------------------------------------
# bert_padding utilities (pure-PyTorch, no compiled extensions needed)
# ---------------------------------------------------------------------------


def index_first_axis(hidden_states: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """Gather rows from *hidden_states* at the given flat *indices*."""
    return hidden_states[indices]


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor, int]:
    """Remove padding tokens and return packed tensors.

    Args:
        hidden_states: (batch_size, seq_len, ...)
        attention_mask: (batch_size, seq_len) with 1 for real tokens, 0 for padding.

    Returns:
        hidden_states_unpad: (total_nonpad, ...)
        indices: flat indices into (batch_size * seq_len) for re-padding
        cu_seqlens: cumulative sequence lengths of shape (batch_size + 1,)
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    batch_size, seq_len = attention_mask.shape
    hidden_states = hidden_states.reshape(batch_size * seq_len, *hidden_states.shape[2:])
    hidden_states_unpad = hidden_states[indices]
    return hidden_states_unpad, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seqlen: int,
) -> torch.Tensor:
    """Re-pad *hidden_states* into shape (batch_size, seqlen, ...)."""
    output = torch.zeros(
        batch_size * seqlen,
        *hidden_states.shape[1:],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output[indices] = hidden_states
    return output.reshape(batch_size, seqlen, *hidden_states.shape[1:])


# ---------------------------------------------------------------------------
# Rotary embedding helper
# ---------------------------------------------------------------------------


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embeddings (RoPE) to *x*.

    Args:
        x: (..., seq_len, dim)
        cos: (..., seq_len, rotary_dim / 2)
        sin: (..., seq_len, rotary_dim / 2)
        interleaved: if True, pairs are (x[0], x[1]), (x[2], x[3]), etc.
                     if False (default, GPT-NeoX style), splits first/second half.
    """
    d = cos.shape[-1]
    rot_dim = 2 * d

    if interleaved:
        x_rot = x[..., :rot_dim].reshape(*x.shape[:-1], d, 2)
        x1, x2 = x_rot.unbind(dim=-1)
    else:
        x1 = x[..., :d]
        x2 = x[..., d:rot_dim]

    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos

    if interleaved:
        out = torch.stack([o1, o2], dim=-1).reshape(*x.shape[:-1], rot_dim)
    else:
        out = torch.cat([o1, o2], dim=-1)

    if x.shape[-1] > rot_dim:
        out = torch.cat([out, x[..., rot_dim:]], dim=-1)

    return out
