"""SafeDiff-VLA StagePlan v2.1: counterfactual referent grounding head.

Only the learned RGB+language -> (grasp stage variables) predictor changes; the v2 executor (`stageplan_v2.py`: PRE_GRASP chunk, arrival
CLOSE, decoder hand-off) is reused UNCHANGED. Outputs are key-compatible with `StagePlanV2Head`
(grasp_xyz, grasp_rot_sc, approach_dir, pre_dist, wp_rel) plus the explicit referent location `ref_xyz`.

Language authority is STRUCTURAL: the language query's ONLY path to any output is the referent attention
    alpha_{c,h,i} = softmax_i( q_h(lang) . k_h(vis_{c,i}) )          (per camera c, per head h, over that camera's 8x8 spatial tokens)
    referent_feature_{c,h} = sum_i alpha_{c,h,i} v_h(vis_{c,i}),   referent_coord_{c,h} = sum_i alpha_{c,h,i} (row_i, col_i)
The regressors consume ONLY [referent features, referent coords, proprio]. There is no global language vector, no global token pooling and
no decoder reading the raw tokens, so a vision-only prior cannot bypass the language-selected referent.
    ref_xyz   = referent location (identity),   grasp_xyz = ref_xyz + learned grasp offset (contact point vs. referent keypoint).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .stageplan_v2 import PRE_DIST_SCALE_M, WP_SCALE_M, StageVars, SafeDiffVLAStagePlanV2Policy  # noqa: F401
from .utils import IMAGE_MODALITY, TEXT_MODALITY

GRID = 8  # SmolVLA connector: 64 tokens per image = 8x8 raster
TOK_PER_CAM = GRID * GRID


class StagePlanV21Head(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int = 7, d_model: int = 256, n_heads: int = 4, n_cams: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_cams, self.d, self.nh = n_cams, d_model, n_heads
        assert d_model % n_heads == 0
        self.vis_proj, self.txt_proj = nn.Linear(latent_dim, d_model), nn.Linear(latent_dim, d_model)
        self.cam_embed, self.pos_embed = nn.Embedding(n_cams, d_model), nn.Embedding(TOK_PER_CAM, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.vis_ctx = nn.TransformerEncoder(enc, 1)  # language-FREE spatial context over the 3x64 tokens (no instruction enters here)
        self.lang_queries = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.q_ref, self.k_ref, self.v_ref = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        rows, cols = torch.meshgrid(torch.arange(GRID, dtype=torch.float32) + 0.5, torch.arange(GRID, dtype=torch.float32) + 0.5, indexing="ij")
        self.register_buffer("grid_uv", torch.stack([cols.reshape(-1), rows.reshape(-1)], -1), persistent=False)  # [64,2] (u=col, v=row), token units
        fuse_in = n_cams * d_model + n_cams * n_heads * 2 + d_model
        self.trunk = nn.Sequential(nn.Linear(fuse_in, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.ref_xyz, self.offset = nn.Linear(d_model, 3), nn.Linear(d_model, 3)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        self.grasp_rot, self.approach, self.pre_dist, self.wp = nn.Linear(d_model, 6), nn.Linear(d_model, 3), nn.Linear(d_model, 1), nn.Linear(d_model, 3)

    def camera_ids(self, is_img: Tensor) -> Tensor:
        rank = torch.cumsum(is_img.long(), dim=1) - 1
        per_cam = (is_img.sum(dim=1, keepdim=True) // self.n_cams).clamp_min(1)
        return (rank // per_cam).clamp(0, self.n_cams - 1)

    def forward(self, tokens: Tensor, pad_mask: Tensor | None, modality_ids: Tensor, state: Tensor, return_aux: bool = False) -> dict[str, Tensor]:
        B = tokens.shape[0]
        if pad_mask is None: pad_mask = tokens.new_ones(tokens.shape[:2], dtype=torch.bool)
        valid = pad_mask.to(torch.bool)
        is_img, is_txt = (modality_ids == IMAGE_MODALITY) & valid, (modality_ids == TEXT_MODALITY) & valid
        is_txt = is_txt | ((is_txt.sum(dim=1, keepdim=True) == 0) & valid)
        # gather the 3 x 64 image tokens in raster order (image tokens are contiguous per camera in the latent layout)
        img_idx = torch.nonzero(is_img[0], as_tuple=False).squeeze(-1)
        assert img_idx.numel() == self.n_cams * TOK_PER_CAM, f"expected {self.n_cams * TOK_PER_CAM} image tokens, got {img_idx.numel()}"
        vis_raw = tokens[:, img_idx]  # [B,192,L]
        pos = torch.arange(TOK_PER_CAM, device=tokens.device).repeat(self.n_cams); cam = torch.arange(self.n_cams, device=tokens.device).repeat_interleave(TOK_PER_CAM)
        vis = self.vis_proj(vis_raw) + self.cam_embed(cam)[None] + self.pos_embed(pos)[None]
        vis = self.vis_ctx(vis)  # [B,192,d]
        txt = self.txt_proj(tokens)
        lang, _ = self.lang_attn(self.lang_queries.expand(B, -1, -1), txt, txt, key_padding_mask=~is_txt, need_weights=False)  # [B,1,d]
        q = self.q_ref(lang.squeeze(1)).view(B, self.nh, -1)  # [B,H,dh]
        k = self.k_ref(vis).view(B, -1, self.nh, self.d // self.nh); v = self.v_ref(vis).view(B, -1, self.nh, self.d // self.nh)
        logits = torch.einsum("bhd,bnhd->bhn", q, k) / (self.d // self.nh) ** 0.5  # [B,H,192]
        logits = logits.view(B, self.nh, self.n_cams, TOK_PER_CAM)
        alpha = torch.softmax(logits.float(), dim=-1).to(v.dtype)  # per camera, per head, over its 64 tokens
        v = v.view(B, self.n_cams, TOK_PER_CAM, self.nh, -1)
        pooled = torch.einsum("bhcn,bcnhd->bchd", alpha, v).reshape(B, self.n_cams, self.d)  # referent feature per camera
        coords = torch.einsum("bhcn,nu->bchu", alpha.float(), self.grid_uv)  # [B,C,H,2] soft-argmax in token units
        h = self.trunk(torch.cat([pooled.reshape(B, -1), (coords.reshape(B, -1) / GRID).to(pooled.dtype), self.state_mlp(state)], -1))
        ref = self.ref_xyz(h); grasp = ref + self.offset(h)
        rot = self.grasp_rot(h).reshape(-1, 3, 2); rot = rot / rot.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ad = self.approach(h); ad = ad / ad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out = {"grasp_xyz": grasp, "ref_xyz": ref, "grasp_rot_sc": rot, "approach_dir": ad, "pre_dist": self.pre_dist(h).squeeze(-1), "wp_rel": self.wp(h), "ref_coords": coords}
        if return_aux:
            out.update({"alpha": alpha, "pooled": pooled, "h": h, "lang": lang.squeeze(1), "vis": vis, "logits": logits})
        return out


def make_learned_provider_v21(head: StagePlanV21Head):
    """INFERENCE provider (RGB + language + proprio -> StageVars), same conversion as the v2 provider."""
    import numpy as np

    @torch.no_grad()
    def provider(policy: SafeDiffVLAStagePlanV2Policy, batch: dict) -> StageVars:
        tokens, pad, mod = policy._encode_multimodal_latent(batch)
        out = head(tokens, pad, mod, policy._current_state(batch))
        g = (out["grasp_xyz"] * policy.action_pos_std + policy.action_pos_mean)[0].float().cpu().numpy().astype(np.float64)
        eul = torch.atan2(out["grasp_rot_sc"][0, :, 0], out["grasp_rot_sc"][0, :, 1]).float().cpu().numpy().astype(np.float64)
        r = float(np.clip(out["pre_dist"][0].item() * PRE_DIST_SCALE_M, 0.05, 0.15))
        w = g + out["wp_rel"][0].float().cpu().numpy().astype(np.float64) * WP_SCALE_M
        return StageVars(g, eul, out["approach_dir"][0].float().cpu().numpy().astype(np.float64), r, True, w)

    return provider
