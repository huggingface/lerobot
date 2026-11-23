# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/backbone.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Two-tower video backbone of the chunk-causal window.

The backbone has an encoder tower and a decoder tower. Each tower wraps a
transformer core (``praxis.layers.Transformer``).

The encoder tower reads one sequence per call:
``[registers | anchor | text | noised | clean]``. ``noised`` holds the K
noised chunks of the window, ``clean`` the clean copies of chunks 0 to K-2.
Noised chunk k attends the global tokens, the clean chunks before k and
itself (teacher forcing). The tower returns the conditioning tokens of the
noised chunks and the tap features for the action expert.

The decoder tower reads ``[registers | noised]`` at full latent resolution.
It uses the encoder output as per-token modulation and returns the video
velocity.

Each latent frame takes one block of ``FRAME_STRIDE`` positions. In a frame,
the position of a token is ``1 + row * ROW_STRIDE + column``. Registers have
position 0. See ``chunk_causal_positions`` for the encoder positions.

Parameter names are the checkpoint key suffixes. The checkpoint stores the
encoder under ``backbone.encoder.`` and the decoder under
``backbone.decoder.``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from lerobot.policies.praxis.praxis_core.layers import Attention, Transformer, modulate as layer_modulate

ROW_STRIDE = 360
FRAME_STRIDE = 129_600
DEFAULT_TAP_BLOCKS = (0, 2, 4, 6, 7, 9, 11, 13)

# Module keys of the input and output projections. They match the checkpoint.
TEXT = "TextEncodingTensor"
VIDEO = "VideoTensor"


@dataclass
class TapFeatures:
    """Encoder features of one chunk-causal call for the action expert.

    ``taps`` holds the block outputs ``(1, S, C)`` in tap order.
    ``embeddings`` is the input of block 0. ``rope_ids`` are the ``(1, S)``
    rotary positions of the encoder sequence. Spans are ``(start, length)``
    token ranges of the encoder sequence: ``global_spans`` maps
    ``register``, ``anchor`` and ``text`` to theirs, ``noised_span`` covers
    the noised chunks and ``clean_span`` the clean copies (None for a window
    of one chunk).
    """

    taps: list[Tensor]
    embeddings: Tensor
    frames_per_chunk: int
    tokens_per_frame: int
    rope_ids: Tensor
    global_spans: dict[str, tuple[int, int]]
    noised_span: tuple[int, int]
    clean_span: tuple[int, int] | None
    n_chunks: int


@dataclass
class BackboneOutput:
    """Video velocity ``(1, C, T, h, w)`` (None if not decoded) and taps."""

    velocity: Tensor | None
    taps: TapFeatures


class ZeroCenteredRMSNorm(nn.Module):
    """RMS norm with the scale ``1 + weight``, in the dtype of the input."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.dim,), weight=1 + self.weight, eps=self.eps)


class TimestepEmbedding(nn.Module):
    """Fixed random Fourier features of the flow timestep."""

    freqs: Tensor
    phases: Tensor

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.zeros(1, dim))
        self.register_buffer("phases", torch.zeros(1, dim))

    def forward(self, t: Tensor) -> Tensor:
        """Map ``t`` of shape ``(B,)`` to features of shape ``(B, dim)``."""
        with torch.no_grad():
            x = t[..., None].float() * self.freqs.float() + self.phases.float()
            x = x.to(t.dtype)
            x = (x * 2 * torch.pi).cos() * 0.5
        return x


class TextProjection(nn.Module):
    """MLP that maps text encoder states to backbone tokens."""

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(c_in, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            ZeroCenteredRMSNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # The copy gives the same memory layout as the original implementation.
        return self.proj(torch.cat([x], dim=1))


class VideoInputProjection(nn.Module):
    """Linear map of latent channels to tokens ``(1, T, h, w, dim)``."""

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(c_in, dim)

    def forward(self, v: Tensor) -> Tensor:
        v = v.movedim(1, -1)
        shape = v.shape
        out = self.proj(torch.cat([v.flatten(1, -2)], dim=1))
        return out.view(*shape[:-1], -1)


class EncoderOutputProjection(nn.Module):
    """Linear map of encoder tokens to the decoder conditioning."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.cat([x], dim=1))


class DecoderOutputProjection(nn.Module):
    """RMS norm and linear map of decoder tokens to latent channels."""

    def __init__(self, dim: int, c_out: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(ZeroCenteredRMSNorm(dim), nn.Linear(dim, c_out))

    def forward(self, x: Tensor) -> Tensor:
        y = self.proj(torch.cat([x], dim=1))
        # The split is an identity. Its backward (a cat) gives the linear
        # layer a contiguous output gradient, as in the original implementation, so the
        # weight gradient GEMM has the same operand layout and the same bits.
        return torch.split(y, [y.shape[1]], dim=1)[0]


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Return ``x * scale + shift`` in float32, cast to the dtype of ``x``.

    On CUDA the fused kernel of ``praxis.layers.modulate`` computes the
    result. For bf16 inputs the float32 product is exact, so the result is
    equal to the result of the eager ops. The backward of the kernel is the
    backward of the original training code.
    """
    return layer_modulate(x, shift, scale, dtype="mixed")


def video_positions(frames: int, height: int, width: int, start: int) -> Tensor:
    """Return the rotary positions of one video slot in frame-major order."""
    frame = torch.arange(frames).view(-1, 1, 1) * FRAME_STRIDE
    row = torch.arange(height).view(1, -1, 1) * ROW_STRIDE
    col = torch.arange(width).view(1, 1, -1)
    return (start + 1 + frame + row + col).reshape(-1)


def _full_blocks(core: Transformer) -> bool:
    """Full blocks in the teacher-forcing mask: the flash backend needs them."""
    return any(isinstance(m, Attention) and m.backend == "flash" for m in core.modules())


def chunk_causal_positions(
    num_registers: int,
    anchor_hw: tuple[int, int],
    num_text: int,
    noised_thw: tuple[int, int, int],
    num_clean_frames: int,
    shifts: tuple[int, int, int] = (0, 0, 0),
) -> Tensor:
    """Return the encoder rotary positions of the chunk-causal layout.

    The sequence is ``[registers | anchor | text | noised | clean]``. The
    registers have position 0. The anchor takes frame block 0, the text frame
    block 1 (positions ``FRAME_STRIDE + 1 ..``) and the noised frames start at
    frame block 2. Clean frame j has the positions of noised frame j.
    ``shifts`` adds ``(anchor, text, video)`` offsets: the random offsets of
    training. The clean frames move with the noised frames.
    """
    anchor_shift, text_shift, video_shift = shifts
    regs = torch.zeros(num_registers, dtype=torch.int64)
    anchor = video_positions(1, *anchor_hw, start=anchor_shift)
    text = torch.arange(1, num_text + 1) + FRAME_STRIDE + text_shift
    noised = video_positions(*noised_thw, start=2 * FRAME_STRIDE + video_shift)
    tokens_per_frame = noised_thw[1] * noised_thw[2]
    clean = noised[: num_clean_frames * tokens_per_frame]
    return torch.cat([regs, anchor, text, noised, clean])


def _chunk_rows(rows: Tensor, chunk_tokens: Sequence[int]) -> Tensor:
    """Expand row k of ``rows`` (1, K, C) over the ``chunk_tokens[k]`` tokens."""
    return torch.cat(
        [rows[:, k : k + 1].expand(-1, n, -1) for k, n in enumerate(chunk_tokens)],
        dim=1,
    )


def unpatch(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Move the channel patch of ``(B, C*ph*pw, T, h, w)`` into space."""
    b, c, t, h, w = x.shape
    ph, pw = patch
    x = x.view(b, c // (ph * pw), ph, pw, t, h, w)
    return x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, -1, t, h * ph, w * pw)


def patch(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Move ``(ph, pw)`` spatial patches of ``(B, C, T, H, W)`` into channels."""
    b, c, t, hh, ww = x.shape
    ph, pw = patch
    x = x.view(b, c, t, hh // ph, ph, ww // pw, pw)
    x = x.permute(0, 1, 4, 6, 2, 3, 5)
    return x.reshape(b, c * ph * pw, t, hh // ph, ww // pw)


def _blockwise_min_max(
    values: Tensor, valid: Tensor, num_blocks: int, block_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the per-block min, max and presence of ``values`` where valid."""
    big = torch.iinfo(torch.int64).max // 2
    pad = num_blocks * block_size - values.numel()
    values = F.pad(values, (0, pad), value=0)
    valid = F.pad(valid, (0, pad), value=False)
    v_min = torch.where(valid, values, big).view(num_blocks, block_size).amin(dim=1)
    v_max = torch.where(valid, values, -big).view(num_blocks, block_size).amax(dim=1)
    present = valid.view(num_blocks, block_size).any(dim=1)
    return v_min, v_max, present


# Teacher-forcing mask of the chunk-causal layout.
#
# Frame id of the clean tokens when the noised tokens must not read them. It is
# far above every noised frame id, so "earlier chunk" never holds for a noised
# query, and the clean-to-clean chunk order stays the same.
_FAR_FRAME = 1 << 20
# With full blocks, the per-token tensors that the mask function reads are
# padded to this static length (or the next power of two above the sequence
# length). The flash kernel cannot read a tensor of symbolic length.
_MASK_CAPTURE_CAPACITY = 1 << 18
_TF_KINDS = ("global", "noised", "clean")


def _tf_token_info(
    slots: Sequence[tuple[str, int, int]], chunk_size: int, clean_past: bool
) -> tuple[list[int], list[bool], list[bool]]:
    """Return the per-token frame id, clean flag and chunk-causal flag.

    Global tokens have the frame id -1. A noised token has the index of its
    frame. A clean token has the first frame id of its chunk, moved past all
    noised frames when ``clean_past`` is false.
    """
    offset = 0 if clean_past else _FAR_FRAME * chunk_size
    frame_ids: list[int] = []
    is_clean: list[bool] = []
    is_bc: list[bool] = []
    for kind, frames, tokens_per_frame in slots:
        if kind not in _TF_KINDS:
            raise ValueError(f"unknown slot kind {kind!r}")
        n = frames * tokens_per_frame
        if kind == "global":
            frame_ids += [-1] * n
        elif kind == "noised":
            for f in range(frames):
                frame_ids += [f] * tokens_per_frame
        else:
            for f in range(frames):
                frame_ids += [(f // chunk_size) * chunk_size + offset] * tokens_per_frame
        is_clean += [kind == "clean"] * n
        is_bc += [kind != "global"] * n
    return frame_ids, is_clean, is_bc


def _tf_row_indices(adjacency: Tensor, width: int) -> tuple[Tensor, Tensor]:
    """Return the per-row count and the left-packed column indices (int32)."""
    rows_n = adjacency.size(0)
    count = adjacency.sum(dim=1).to(torch.int32)
    idx = torch.zeros(rows_n, width, dtype=torch.int32)
    rows, cols = adjacency.nonzero(as_tuple=True)
    pos = (adjacency.int().cumsum(dim=1)[rows, cols] - 1).to(torch.int32)
    idx[rows, pos] = cols.to(torch.int32)
    return count, idx


def _assemble_tf_block_mask(
    block_attends: Tensor,
    total_len: int,
    block_size: int,
    device: torch.device,
    mask_mod: Callable[[Tensor | None, Tensor | None, Tensor, Tensor], Tensor],
    block_full: Tensor | None,
) -> BlockMask:
    """Build a BlockMask with partial and (optionally) full block lists.

    ``block_full`` is a subset of ``block_attends``: the tiles where every
    token pair attends. The kernel skips the mask function on them. The
    partial and the full lists share one width, because the kernels index
    both with the same strides.
    """
    q_blocks = block_attends.size(0)
    partial = block_attends if block_full is None else block_attends & ~block_full

    def width(adjacency: Tensor) -> int:
        return max(int(adjacency.sum(dim=1).max().item()) if q_blocks else 0, 1)

    if block_full is None:
        kv_width, q_width = width(partial), width(partial.T)
    else:
        kv_width = max(width(partial), width(block_full))
        q_width = max(width(partial.T), width(block_full.T))
    kv_count, kv_idx = _tf_row_indices(partial, kv_width)
    q_count, q_idx = _tf_row_indices(partial.T, q_width)

    def dev(t: Tensor) -> Tensor:
        return t.view(1, 1, q_blocks, *t.shape[1:]).to(device)

    full_kv_count = full_kv_idx = full_q_count = full_q_idx = None
    if block_full is not None:
        fkc, fki = _tf_row_indices(block_full, kv_width)
        fqc, fqi = _tf_row_indices(block_full.T, q_width)
        full_kv_count, full_kv_idx = dev(fkc), dev(fki)
        full_q_count, full_q_idx = dev(fqc), dev(fqi)
    return BlockMask(
        seq_lengths=(total_len, total_len),
        kv_num_blocks=dev(kv_count),
        kv_indices=dev(kv_idx),
        full_kv_num_blocks=full_kv_count,
        full_kv_indices=full_kv_idx,
        q_num_blocks=dev(q_count),
        q_indices=dev(q_idx),
        full_q_num_blocks=full_q_count,
        full_q_indices=full_q_idx,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=mask_mod,
    )


def teacher_forcing_mask(
    slots: Sequence[tuple[str, int, int]],
    chunk_size: int,
    clean_past: bool,
    block_size: int,
    device: torch.device,
    full_blocks: bool,
) -> BlockMask:
    """Build the teacher-forcing attention mask of the chunk-causal layout.

    ``slots`` holds ``(kind, frames, tokens_per_frame)`` per slot, in sequence
    order. ``kind`` is ``global``, ``noised`` or ``clean``. A global slot is
    ``("global", 1, tokens)``. Frame ``f`` belongs to chunk
    ``f // chunk_size``. The rules:

    * Every token attends all global tokens. Global tokens attend only global
      tokens.
    * A clean token attends the clean tokens of its own and earlier chunks.
    * A noised token attends the noised tokens of its own chunk and, with
      ``clean_past``, the clean tokens of earlier chunks.
    * A clean token never attends a noised token.

    With ``full_blocks`` the mask also lists the tiles where every pair
    attends as full blocks. The flash attention backend needs them.
    """
    frame_ids, is_clean, is_bc = _tf_token_info(slots, chunk_size, clean_past)
    total_len = len(frame_ids)
    bs = block_size
    cs = chunk_size

    sample_d = torch.zeros(total_len, dtype=torch.int32).to(device)
    frame_d = torch.tensor(frame_ids, dtype=torch.int32).to(device)
    clean_d = torch.tensor(is_clean, dtype=torch.bool).to(device)
    bc_d = torch.tensor(is_bc, dtype=torch.bool).to(device)
    # Floor division of the frame ids, done once here (the same values as
    # ``frame // chunk_size`` in the mask rule, also for the -1 of globals).
    chunk_d = torch.div(frame_d, cs, rounding_mode="floor")

    def mask_mod(b: Tensor | None, h: Tensor | None, qi: Tensor, ki: Tensor) -> Tensor:
        same_sample = sample_d[qi] == sample_d[ki]
        q_bc, k_bc = bc_d[qi], bc_d[ki]
        q_clean, k_clean = clean_d[qi], clean_d[ki]
        q_f, k_f = frame_d[qi], frame_d[ki]
        q_c, k_c = chunk_d[qi], chunk_d[ki]
        cc = q_clean & k_clean & (k_f <= q_f)
        nc = (~q_clean) & k_clean & (k_c < q_c)
        nn_ = (~q_clean) & (~k_clean) & (q_c == k_c)
        bc_rule = q_bc & k_bc & (cc | nc | nn_)
        return same_sample & ((~k_bc) | bc_rule)

    if full_blocks:
        capacity = max(_MASK_CAPTURE_CAPACITY, 1 << (max(total_len, 1) - 1).bit_length())

        def static(t: Tensor, fill: int | bool) -> Tensor:
            return F.pad(t, (0, capacity - t.numel()), value=fill)

        chunk_s = static(chunk_d, -1)
        sample_s = static(sample_d, -1)
        frame_s = static(frame_d, -1)
        clean_s = static(clean_d, False)
        bc_s = static(bc_d, False)

        def mask_mod(b: Tensor | None, h: Tensor | None, qi: Tensor, ki: Tensor) -> Tensor:
            same_sample = sample_s[qi] == sample_s[ki]
            q_bc, k_bc = bc_s[qi], bc_s[ki]
            q_clean, k_clean = clean_s[qi], clean_s[ki]
            q_f, k_f = frame_s[qi], frame_s[ki]
            q_c, k_c = chunk_s[qi], chunk_s[ki]
            cc = q_clean & k_clean & (k_f <= q_f)
            nc = (~q_clean) & k_clean & (k_c < q_c)
            nn_ = (~q_clean) & (~k_clean) & (q_c == k_c)
            bc_rule = q_bc & k_bc & (cc | nc | nn_)
            return same_sample & ((~k_bc) | bc_rule)

    # Host-side block adjacency from per-block interval summaries. It can
    # include too many blocks, but never too few.
    num_blocks = (total_len + bs - 1) // bs
    sample_h = torch.zeros(total_len, dtype=torch.int64)
    frame_h = torch.tensor(frame_ids, dtype=torch.int64)
    chunk_h = torch.where(frame_h >= 0, frame_h // cs, frame_h)
    clean_h = torch.tensor(is_clean, dtype=torch.bool)
    bc_h = torch.tensor(is_bc, dtype=torch.bool)
    in_range = torch.ones(total_len, dtype=torch.bool)

    s_min, s_max, pad_free = _blockwise_min_max(sample_h, in_range, num_blocks, bs)
    noised_valid = bc_h & ~clean_h
    clean_valid = bc_h & clean_h
    cf_min, cf_max, has_clean = _blockwise_min_max(frame_h, clean_valid, num_blocks, bs)
    cc_min, cc_max, _ = _blockwise_min_max(chunk_h, clean_valid, num_blocks, bs)
    _, _, has_noised = _blockwise_min_max(frame_h, noised_valid, num_blocks, bs)
    nc_min, nc_max, _ = _blockwise_min_max(chunk_h, noised_valid, num_blocks, bs)
    _, _, has_global = _blockwise_min_max(sample_h, in_range & ~bc_h, num_blocks, bs)

    q = slice(None), None
    k = None, slice(None)
    same_possible = (s_min[q] <= s_max[k]) & (s_min[k] <= s_max[q])
    cc_possible = has_clean[q] & has_clean[k] & (cf_min[k] <= cf_max[q])
    nc_possible = has_noised[q] & has_clean[k] & (cc_min[k] < nc_max[q])
    nn_possible = has_noised[q] & has_noised[k] & (nc_min[q] <= nc_max[k]) & (nc_min[k] <= nc_max[q])
    block_attends = (
        pad_free[q] & pad_free[k] & same_possible & (has_global[k] | cc_possible | nc_possible | nn_possible)
    )

    block_full = None
    if full_blocks:
        # A tile is full when the rules hold for every token pair: both blocks
        # hold only real tokens, and either the key block is all global, or
        # both blocks hold one token class and the rule holds over their whole
        # frame or chunk intervals. Mixed blocks stay partial.
        starts = torch.arange(num_blocks) * bs
        real = starts + bs <= total_len
        one_sample = s_min == s_max
        _, _, has_bc = _blockwise_min_max(sample_h, bc_h, num_blocks, bs)
        all_global = ~has_bc
        all_clean = has_clean & ~has_noised & has_bc & ~has_global
        all_noised = has_noised & ~has_clean & has_bc & ~has_global
        same_sample = one_sample[q] & one_sample[k] & (s_min[q] == s_min[k])
        cc_full = all_clean[q] & all_clean[k] & (cf_max[k] <= cf_min[q])
        nc_full = all_noised[q] & all_clean[k] & (cc_max[k] < nc_min[q])
        nn_full = (
            all_noised[q]
            & all_noised[k]
            & (nc_min[q] == nc_max[q])
            & (nc_min[k] == nc_max[k])
            & (nc_min[q] == nc_min[k])
        )
        block_full = (
            block_attends & real[q] & real[k] & same_sample & (all_global[k] | cc_full | nc_full | nn_full)
        )
    return _assemble_tf_block_mask(block_attends, total_len, bs, device, mask_mod, block_full)


class _TeacherForcingMaskCache:
    """Small LRU cache of teacher-forcing masks, keyed by the slot layout."""

    def __init__(self, block_size: int, max_entries: int = 64) -> None:
        self.block_size = block_size
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, BlockMask] = OrderedDict()

    def get(
        self,
        slots: Sequence[tuple[str, int, int]],
        chunk_size: int,
        clean_past: bool,
        device: torch.device,
        full_blocks: bool,
    ) -> BlockMask:
        key = (
            tuple(tuple(s) for s in slots),
            chunk_size,
            clean_past,
            self.block_size,
            str(device),
            full_blocks,
        )
        mask = self._entries.pop(key, None)
        if mask is None:
            mask = teacher_forcing_mask(slots, chunk_size, clean_past, self.block_size, device, full_blocks)
        self._entries[key] = mask
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return mask


class EncoderTower(nn.Module):
    """Encoder tower of the backbone.

    ``core`` is the transformer stack of the tower. Read
    ``positive_zero_residual_add`` from the checkpoint config. When it is
    true, the tower adds ``+0.0`` to the core output. This keeps the signs of
    zeros identical to the original implementation.
    """

    def __init__(
        self,
        core: Transformer,
        dim: int = 3072,
        text_dim: int = 2048,
        video_dim: int = 192,
        time_dim: int = 128,
        num_registers: int = 4,
        tap_blocks: tuple[int, ...] = DEFAULT_TAP_BLOCKS,
        positive_zero_residual_add: bool = True,
        mask_block_size: int = 128,
    ) -> None:
        super().__init__()
        self.registers = nn.Parameter(torch.zeros(1, num_registers, dim))
        self.t_embedding = TimestepEmbedding(time_dim)
        self.t_proj = nn.ModuleDict({VIDEO: nn.Linear(time_dim, 2 * dim)})
        self.input_projections = nn.ModuleDict(
            {
                TEXT: TextProjection(text_dim, dim),
                VIDEO: VideoInputProjection(video_dim, dim),
            }
        )
        self.output_projections = nn.ModuleDict({VIDEO: EncoderOutputProjection(dim)})
        self.core = core
        self.tap_blocks = tuple(tap_blocks)
        self.positive_zero_residual_add = positive_zero_residual_add
        self._tf_masks = _TeacherForcingMaskCache(mask_block_size)

    def _chunk_modulate(self, flat: Tensor, t_emb: Tensor, chunk_tokens: Sequence[int]) -> Tensor:
        """Modulate the tokens of each chunk with its timestep row."""
        shift, scale = self.t_proj[VIDEO](t_emb).chunk(2, dim=-1)
        return modulate(flat, _chunk_rows(shift, chunk_tokens), _chunk_rows(1 + scale, chunk_tokens))

    def forward_chunk_causal(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        rope_shifts: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[Tensor, TapFeatures]:
        """Encode a chunk-causal window of K chunks.

        The sequence is ``[registers | anchor | text | noised | clean]``.
        ``noised`` is ``(1, video_dim, K * F, h, w)``: the K noised chunks of F
        frames. ``clean`` is ``(1, video_dim, (K - 1) * F, h, w)``: the clean
        copy of chunks ``0 .. K - 2``, or None for K = 1. ``timesteps`` is
        ``(1, K)``: the timestep of each noised chunk. The clean copy has the
        timestep 0. Noised chunk k attends the global tokens, the clean
        chunks before k and itself. Clean chunk j attends the global tokens
        and the clean chunks ``0 .. j``. ``rope_shifts`` holds the rotary
        offsets ``(anchor, text, video)`` of training (see
        ``chunk_causal_positions``). Returns the conditioning tokens
        ``(1, K * F, h, w, dim)`` of the noised chunks and the taps.
        """
        kk = int(timesteps.shape[1])
        frames, height, width = (int(d) for d in noised.shape[2:])
        if frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        chunk = frames // kk
        n_clean_frames = 0 if clean is None else int(clean.shape[2])
        if n_clean_frames != (kk - 1) * chunk:
            raise ValueError(f"{kk} chunks need {(kk - 1) * chunk} clean frames, got {n_clean_frames}")
        tpf = height * width
        # One projection call per input, as in the original implementation.
        anchor_tok = self.input_projections[VIDEO](anchor).flatten(1, -2)
        text_tok = self.input_projections[TEXT](text)
        noised_tok = self.input_projections[VIDEO](noised)
        noised_shape = noised_tok.shape
        noised_tok = self._chunk_modulate(
            noised_tok.flatten(1, -2), self.t_embedding(timesteps), [chunk * tpf] * kk
        )
        parts = [self.registers, anchor_tok, text_tok, noised_tok]
        if clean is not None:
            clean_tok = self.input_projections[VIDEO](clean).flatten(1, -2)
            t_clean = self.t_embedding(torch.zeros_like(timesteps))
            parts.append(self._chunk_modulate(clean_tok, t_clean, [chunk * tpf] * (kk - 1)))
        x = torch.cat(parts, dim=1)

        n_reg = self.registers.size(1)
        n_anchor = anchor_tok.size(1)
        n_text = text_tok.size(1)
        n_noised = frames * tpf
        n_clean = n_clean_frames * tpf
        positions = chunk_causal_positions(
            n_reg,
            (int(anchor.shape[3]), int(anchor.shape[4])),
            n_text,
            (frames, height, width),
            n_clean_frames,
            rope_shifts,
        )
        positions = positions.unsqueeze(0).to(x.device)
        slots = [
            ("global", 1, n_reg),
            ("global", 1, n_anchor),
            ("global", 1, n_text),
            ("noised", frames, tpf),
        ]
        if n_clean:
            slots.append(("clean", n_clean_frames, tpf))
        mask = self._tf_masks.get(slots, chunk, True, x.device, _full_blocks(self.core))

        out, taps = self.core(x, positions, None, mask, self.tap_blocks)
        if self.positive_zero_residual_add:
            out = out + 0.0
        # The output projection runs once over the noised and the clean tokens.
        n_global = n_reg + n_anchor + n_text
        projected = self.output_projections[VIDEO](
            torch.cat(
                list(
                    out.narrow(1, n_global, n_noised + n_clean).split(
                        [n_noised, n_clean] if n_clean else [n_noised], dim=1
                    )
                ),
                dim=1,
            )
        )
        cond = projected.narrow(1, 0, n_noised).view(*noised_shape[:-1], -1)

        features = TapFeatures(
            taps=list(taps),
            embeddings=x,
            frames_per_chunk=chunk,
            tokens_per_frame=tpf,
            rope_ids=positions,
            global_spans={
                "register": (0, n_reg),
                "anchor": (n_reg, n_anchor),
                "text": (n_reg + n_anchor, n_text),
            },
            noised_span=(n_global, n_noised),
            clean_span=(n_global + n_noised, n_clean) if n_clean else None,
            n_chunks=kk,
        )
        return cond, features


class DecoderTower(nn.Module):
    """Decoder tower of the backbone.

    The tower moves the channel patch of the noised latent into space, runs
    the core on the full latent grid and patches the output again.
    ``positive_zero_residual_add``
    has the same function as in ``EncoderTower``.
    """

    def __init__(
        self,
        core: Transformer,
        dim: int = 3072,
        video_dim: int = 48,
        time_dim: int = 128,
        num_registers: int = 4,
        patch_size: tuple[int, int] = (2, 2),
        positive_zero_residual_add: bool = True,
        mask_block_size: int = 128,
    ) -> None:
        super().__init__()
        self.registers = nn.Parameter(torch.zeros(1, num_registers, dim))
        self.t_embedding = TimestepEmbedding(time_dim)
        self.t_proj = nn.ModuleDict({VIDEO: nn.Linear(time_dim, 2 * dim)})
        self.input_projections = nn.ModuleDict({VIDEO: VideoInputProjection(video_dim, dim)})
        self.output_projections = nn.ModuleDict({VIDEO: DecoderOutputProjection(dim, video_dim)})
        self.core = core
        self.patch_size = patch_size
        self.positive_zero_residual_add = positive_zero_residual_add
        self._tf_masks = _TeacherForcingMaskCache(mask_block_size)

    def forward_chunk_causal(
        self, noised: Tensor, cond: Tensor, timesteps: Tensor, rope_shift: int = 0
    ) -> Tensor:
        """Return the velocity of the K noised chunks of a chunk-causal window.

        The sequence is ``[registers | noised]`` at full latent resolution,
        with the rotary positions of the noised frames from ``rope_shift``
        (0 at inference). ``cond`` is the encoder output
        ``(1, K * F, h, w, dim)``. ``timesteps`` is ``(1, K)``. Noised chunk k
        attends the registers and itself.
        """
        ph, pw = self.patch_size
        kk = int(timesteps.shape[1])
        tok = self.input_projections[VIDEO](unpatch(noised, self.patch_size))
        shape = tok.shape
        _, frames, height, width, _ = shape
        if frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        chunk = frames // kk
        tpf = height * width
        shift, scale = self.t_proj[VIDEO](self.t_embedding(timesteps)).chunk(2, dim=-1)
        rows = [chunk * tpf] * kk
        flat = modulate(tok.flatten(1, -2), _chunk_rows(shift, rows), _chunk_rows(1 + scale, rows))
        x = torch.cat([self.registers, flat], dim=1)

        b, cf, h, w, c = cond.shape
        up = cond[:, :, :, None, :, None, :].expand(b, cf, h, ph, w, pw, c)
        up = up.reshape(b, cf * h * ph * w * pw, c)
        n_reg = self.registers.size(1)
        pad = torch.zeros(b, n_reg, c, device=up.device, dtype=up.dtype)
        modulation = torch.cat([pad, up], dim=1)

        pos = [
            torch.zeros(n_reg, dtype=torch.int64),
            video_positions(frames, height, width, rope_shift),
        ]
        positions = torch.cat(pos).unsqueeze(0).to(x.device)
        slots = [("global", 1, n_reg), ("noised", frames, tpf)]
        mask = self._tf_masks.get(slots, chunk, False, x.device, _full_blocks(self.core))

        out, _ = self.core(x, positions, modulation, mask)
        if self.positive_zero_residual_add:
            out = out + 0.0
        v = self.output_projections[VIDEO](out.narrow(1, n_reg, flat.size(1)))
        v = v.view(*shape[:-1], -1).movedim(-1, 1)
        return patch(v, self.patch_size)


class VideoBackbone(nn.Module):
    """Encoder tower and decoder tower for one chunk-causal denoise step."""

    def __init__(self, encoder: EncoderTower, decoder: DecoderTower) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward_chunk_causal(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        decode: bool = True,
        rope_shifts: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> BackboneOutput:
        """Run one denoise step of a chunk-causal window of K chunks.

        ``noised`` is ``(1, video_dim, K * F, h, w)``, ``clean`` the clean copy
        of chunks ``0 .. K - 2`` (None for K = 1), ``timesteps`` ``(1, K)``.
        All in the model dtype. The velocity covers the K noised chunks. With
        ``decode=False`` only the encoder runs and the velocity is None.
        ``rope_shifts`` holds the rotary offsets of training: ``(encoder
        anchor, encoder text, encoder video, decoder video)``. Inference uses
        no offset.
        """
        cond, taps = self.encoder.forward_chunk_causal(
            text, anchor, clean, noised, timesteps, rope_shifts[:3]
        )
        if not decode:
            return BackboneOutput(velocity=None, taps=taps)
        velocity = self.decoder.forward_chunk_causal(noised, cond, timesteps, rope_shifts[3])
        return BackboneOutput(velocity=velocity, taps=taps)


def _core_from_config(cfg: dict, tower: dict) -> Transformer:
    """Build the transformer stack of one tower from the config."""
    attn, rope, moe = cfg["attention"], cfg["rope"], cfg["moe"]
    return Transformer(
        depth=tower["depth"],
        num_dense=len(tower["dense_blocks"]),
        dim=cfg["dim"],
        head_dim=attn["head_dim"],
        num_kv_heads=attn["num_kv_heads"],
        dense_hidden=cfg["dense_ffn_hidden"],
        moe_hidden=moe["expert_hidden"],
        num_experts=moe["num_routed_experts"],
        top_k=moe["top_k_routed"],
        routed_scaling_factor=moe["routed_scaling_factor"],
        mod_dim=tower.get("modulation_dim"),
        modulation_dtype=cfg["modulation_dtype"],
        rope_effective_dim=rope["effective_dim"],
        rope_base=rope["base"],
        rope_table_size=rope["efficient_cache_max"],
        attention_backend=attn.get("backend", "triton"),
        moe_dispatch_chunk_size=moe.get("dispatch_chunk_size"),
    )


def build_backbone(config: dict) -> VideoBackbone:
    """Build the backbone from the ``backbone`` section of ``config.json``.

    The parameters are uninitialized. Load the checkpoint next, for example
    with ``praxis.checkpoint.load_checkpoint(model, path, prefix="backbone.")``.
    ``layout`` must be ``"chunk_causal"`` (the default).
    """
    layout = config.get("layout", "chunk_causal")
    if layout != "chunk_causal":
        raise ValueError(f"unknown backbone layout {layout!r}")
    enc_cfg, dec_cfg = config["encoder"], config["decoder"]
    dim, time_dim = config["dim"], config["timestep_dim"]
    block = config["attention"]["flex_block_size"]
    encoder = EncoderTower(
        core=_core_from_config(config, enc_cfg),
        dim=dim,
        text_dim=enc_cfg["input_projections"][TEXT]["c_in"],
        video_dim=enc_cfg["input_projections"][VIDEO]["c_in"],
        time_dim=time_dim,
        num_registers=config["num_registers"],
        tap_blocks=tuple(enc_cfg["tap_blocks"]),
        positive_zero_residual_add=enc_cfg["positive_zero_residual_add"],
        mask_block_size=block,
    )
    ph, pw = dec_cfg["unpatch_factor"][1:]
    decoder = DecoderTower(
        core=_core_from_config(config, dec_cfg),
        dim=dim,
        video_dim=dec_cfg["input_projections"][VIDEO]["c_in"],
        time_dim=time_dim,
        num_registers=config["num_registers"],
        patch_size=(ph, pw),
        positive_zero_residual_add=dec_cfg["positive_zero_residual_add"],
        mask_block_size=block,
    )
    return VideoBackbone(encoder, decoder)
