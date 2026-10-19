"""`architecture="temporal_decoder_grounded_grasp_v3"` only (see `configuration_safediff_vla.py`'s
architecture docstring). v2's held-out results (`temporal_decoder_grounded_grasp_v2`) improved
target-xyz localization error over the canonical baseline but did not reach the precision grasping
needs, and did not move task success: a single `TargetPointHead` MLP regressing straight from one
pooled `grounded_target_feature` to a final xyz has to do both "which object" (identity/grounding)
and "exactly where" (sub-centimeter localization) in one shot. This module splits that into two
stages instead:

    target-language query (`grounded_grasp_v2.py`'s `TargetQueryExtractor`, reused as-is)
        -> `SharpenedVisualGrounding`: explicit query-to-image-token similarity, sharpened via
           temperature-scaled softmax (+ optional hard top-k), replacing v2's 4-head
           `nn.MultiheadAttention` (`VisualGroundingCrossAttention`) whose per-head weights are
           never exposed and can stay diffuse even when individual heads are peaked
        -> coarse_grounded_feature [B, D] (+ loggable entropy/top1_mass/topk_mass selection stats)
        -> `CoarseTargetHead` -> coarse_xyz [B, 3]
        -> `FineResidualHead` (coarse_xyz + grounded feature + current state + target query)
           -> bounded delta_xyz [B, 3]
        -> final_target_xyz = coarse_xyz + delta_xyz

Only the grounding/localization head changes here -- `TemporalActionDecoder`'s
`nn.TransformerDecoder` core, `TargetQueryExtractor`, the reactive-close mechanism's threshold/
phase logic, and every other architecture are untouched (see `modeling_safediff_vla.py`'s
`temporal_decoder_grounded_grasp_v3` branch, which reuses `TemporalActionDecoder`'s existing
`use_grounded_grasp_v2` conditioning wiring unchanged -- the decoder cannot tell v2's and v3's
`grounded_target_feature`/`target_xyz` apart, by design).
"""

import torch
from torch import Tensor, nn

from .utils import IMAGE_MODALITY


class SharpenedVisualGrounding(nn.Module):
    """Coarse visual target selection: one explicit set of target-query -> image-token attention
    weights (single-head, manual scaled-dot-product -- NOT `nn.MultiheadAttention`, precisely so
    the weights themselves are directly available to sharpen and to log), instead of diffusing
    over the whole visual memory. Sharpening is controlled by `temperature` (divides the
    scaled-dot-product logits before softmax -- smaller is sharper) and an optional hard `top_k`
    restriction (keep only the k highest-scoring image tokens per sample, `-inf` out the rest,
    before the softmax). No hard-coded card/rank/suit parsing anywhere -- query/key/value
    projections are trained end-to-end against the target-xyz loss, exactly like
    `grounded_grasp_v2.py`'s `TargetQueryExtractor`/`VisualGroundingCrossAttention`.
    """

    def __init__(self, latent_dim: int, temperature: float = 0.5, top_k: int | None = None) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer or None")
        self.temperature = temperature
        self.top_k = top_k
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.scale = latent_dim**-0.5

    def forward(
        self, target_query: Tensor, latent_tokens: Tensor, latent_pad_mask: Tensor | None, latent_modality_ids: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """`target_query`: [B, D] (from `TargetQueryExtractor`). `latent_tokens`/`latent_pad_mask`/
        `latent_modality_ids`: same convention as `grounded_grasp_v2.py`'s modules.

        Returns `(coarse_grounded_feature [B, D], stats)`, where `stats` holds `[B]`-shaped
        `entropy` (nats), `top1_mass`, `topk_mass` (mass captured by the reporting top-k --
        `self.top_k` when the hard restriction is active, else a fixed diagnostic width capped at
        the token count), and the full `[B, N]` `attention_weights` themselves.
        """
        batch_size, n_tokens = latent_tokens.shape[0], latent_tokens.shape[1]
        if latent_pad_mask is None:
            latent_pad_mask = latent_tokens.new_ones((batch_size, n_tokens), dtype=torch.bool)
        valid = latent_pad_mask.to(torch.bool)

        image_valid = (latent_modality_ids == IMAGE_MODALITY) & valid
        # Same zero-valid-image-tokens fallback as `VisualGroundingCrossAttention`: never the case
        # with a real batch, but attend over every valid token for that row rather than mask every
        # key to `-inf` (which would otherwise propagate NaN through the softmax below).
        no_image_tokens = image_valid.sum(dim=1, keepdim=True) == 0
        image_valid = image_valid | (no_image_tokens & valid)

        query = self.query_proj(target_query)  # [B, D]
        keys = self.key_proj(latent_tokens)  # [B, N, D]
        values = self.value_proj(latent_tokens)  # [B, N, D]
        logits = torch.einsum("bd,bnd->bn", query, keys) * self.scale / self.temperature
        logits = logits.masked_fill(~image_valid, float("-inf"))

        effective_top_k = None
        if self.top_k is not None:
            effective_top_k = min(self.top_k, n_tokens)
            top_values, top_indices = logits.topk(effective_top_k, dim=1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            sparse_logits.scatter_(1, top_indices, top_values)
            logits = sparse_logits

        weights = torch.softmax(logits, dim=1)
        # Defensive, mirrors `TargetQueryExtractor`'s own guard: an all-`-inf` row would otherwise
        # soften into NaN. Only reachable if a row has literally zero `valid` positions of any
        # modality (never a real sample -- `no_image_tokens`'s fallback above already widens to
        # every valid position first).
        bad_rows = ~torch.isfinite(weights).all(dim=1)
        if bool(bad_rows.any()):
            uniform = valid.to(weights.dtype)
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1)
            weights = torch.where(bad_rows.unsqueeze(1), uniform, weights)

        coarse_feature = torch.einsum("bn,bnd->bd", weights, values)

        eps = 1e-12
        entropy = -(weights * (weights + eps).log()).sum(dim=1)
        top1_mass = weights.max(dim=1).values
        report_k = effective_top_k if effective_top_k is not None else min(5, n_tokens)
        topk_mass = weights.topk(report_k, dim=1).values.sum(dim=1)

        stats = {
            "entropy": entropy,
            "top1_mass": top1_mass,
            "topk_mass": topk_mass,
            "attention_weights": weights,
        }
        return coarse_feature, stats


class CoarseTargetHead(nn.Module):
    """Predicts a coarse grasp-target xyz `[B, 3]` from `SharpenedVisualGrounding`'s coarse
    grounded feature alone -- structurally identical to `target_point_head.TargetPointHead` (small
    SiLU MLP) but a separate, independently-parameterized module: v1/v2's `TargetPointHead` stays
    completely untouched (never imported here). This is the first of v3's two regression stages,
    refined by `FineResidualHead` below rather than used directly as the final target.
    """

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, grounded_feature: Tensor) -> Tensor:
        """`grounded_feature`: [B, latent_dim]. Returns coarse_xyz [B, 3]."""
        return self.net(grounded_feature)


class FineResidualHead(nn.Module):
    """Bounded local-correction residual on top of `CoarseTargetHead`'s estimate:
    `delta_xyz = max_refine_offset * tanh(raw_delta)`, where `raw_delta` comes from an MLP over
    `[grounded_feature, coarse_xyz, current_state, target_language_query]`. Conditioning on the
    coarse estimate and the current robot state (which `CoarseTargetHead` never sees) lets this
    head correct for state-relative offsets the coarse stage cannot; the `tanh`-bounded magnitude
    (config-exposed, see `SafeDiffVLAConfig.grounded_grasp_v3_max_refine_offset`) keeps this stage
    a strictly LOCAL correction, never an independent second regression that could override the
    coarse stage entirely.
    """

    def __init__(self, latent_dim: int, state_dim: int, hidden_dim: int, max_refine_offset: float) -> None:
        super().__init__()
        if max_refine_offset <= 0:
            raise ValueError("max_refine_offset must be positive")
        self.max_refine_offset = max_refine_offset
        input_dim = latent_dim + 3 + state_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, grounded_feature: Tensor, coarse_xyz: Tensor, current_state: Tensor, target_query: Tensor) -> Tensor:
        """All args batch-first ([B, *]); `coarse_xyz` is [B, 3]. Returns bounded delta_xyz [B, 3],
        `|delta_xyz| < max_refine_offset` element-wise (strict: `tanh` never saturates to exactly
        +/-1)."""
        x = torch.cat([grounded_feature, coarse_xyz, current_state, target_query], dim=-1)
        raw_delta = self.net(x)
        return self.max_refine_offset * torch.tanh(raw_delta)
