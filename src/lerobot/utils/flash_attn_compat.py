"""Compatibility layer for FlashAttention (FA4 CuTeDSL / FA2 fallback).

Provides a unified import point for flash attention functions and related utilities
(bert_padding, rotary embeddings) that works with both the new FlashAttention-4
CuTeDSL package (flash-attn-4) and the legacy FlashAttention-2 package (flash-attn).

FA4 is preferred when available; FA2 is used as a fallback.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flash attention function imports — prefer FA4, fall back to FA2
# ---------------------------------------------------------------------------

_FLASH_ATTN_VERSION: str = "none"
_FLASH_ATTN_FA4: bool = False

flash_attn_func = None
flash_attn_varlen_func = None

try:
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func  # type: ignore[assignment]
    from flash_attn.cute import __version__ as _fa4_version

    _FLASH_ATTN_VERSION = _fa4_version
    _FLASH_ATTN_FA4 = True
    logger.debug("Using FlashAttention-4 (CuTeDSL) %s", _FLASH_ATTN_VERSION)
except ImportError:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func  # type: ignore[assignment]
        import flash_attn as _fa2_mod

        _FLASH_ATTN_VERSION = getattr(_fa2_mod, "__version__", "unknown")
        logger.debug("Using FlashAttention-2 %s", _FLASH_ATTN_VERSION)
    except ImportError:
        logger.debug("No FlashAttention package found")


def is_flash_attn_available() -> bool:
    """Return True if any FlashAttention backend (FA4 or FA2) is importable."""
    return flash_attn_func is not None


def is_flash_attn_fa4() -> bool:
    """Return True if FlashAttention-4 (CuTeDSL) is the active backend."""
    return _FLASH_ATTN_FA4


def get_flash_attn_version() -> str:
    return _FLASH_ATTN_VERSION


# ---------------------------------------------------------------------------
# bert_padding utilities (formerly flash_attn.bert_padding)
#
# Pure-PyTorch replacements so we don't depend on FA2's compiled extensions.
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
# Rotary embedding helper (formerly flash_attn.layers.rotary.apply_rotary_emb)
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
