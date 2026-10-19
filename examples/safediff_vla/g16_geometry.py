"""G16 prototype: SMOOTH, BOUNDED geometry conditioner for the frozen V2 temporal decoder.
e = target - EE (m) -> features [e/action_pos_std (3), unit_xy (2), r_xy/0.1, r_3d/0.1, r_descend], r_descend = sigmoid((d0 - r_xy)/tau), d0 = 5cm, tau = 2cm (fixed, no sweep; NOT a gripper signal).
queries += alpha_max * tanh(MLP(features))   (last layer zero-init => exactly V2 at init; `geo_on=False` reproduces V2). Everything else frozen.
Training-only side channel: `find_grasp_target` is wrapped so the demonstration grasp-transition anchor (and its validity mask) is available inside the decoder call (conditioning input where valid, else the passed target_xyz) and for L_progress. At inference these are never set: the conditioner reads the `target_xyz` the decoder is given, exactly like G11."""
from __future__ import annotations
import torch
from torch import nn
import lerobot.policies.safediff_vla.modeling_safediff_vla as MOD

D0, TAU, EPS = 0.05, 0.02, 1e-6


class GeometryConditioner(nn.Module):
    def __init__(self, alpha_max: float, hidden=256, out_dim=512, in_dim=8):
        super().__init__(); self.alpha_max = alpha_max; self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def features(e_m: torch.Tensor, pos_std: torch.Tensor) -> torch.Tensor:
        rxy = e_m[:, :2].norm(dim=-1, keepdim=True); r3 = e_m.norm(dim=-1, keepdim=True); return torch.cat([e_m / pos_std, e_m[:, :2] / (rxy + EPS * 100), rxy / 0.1, r3 / 0.1, torch.sigmoid((D0 - rxy) / TAU)], -1)

    def forward(self, e_m, pos_std): return self.alpha_max * torch.tanh(self.net(self.features(e_m, pos_std)))


def install(policy, alpha_max: float) -> GeometryConditioner:
    dec = policy.decoder
    if getattr(dec, "geo", None) is not None: return dec.geo
    dec.geo = GeometryConditioner(alpha_max, out_dim=dec.action_head.in_features).to(next(dec.parameters()).device); dec.geo_on = True; dec._gt = None; dec._last = {}
    dec._stats = {"s_mean": policy.state_pos_mean, "s_std": policy.state_pos_std, "a_mean": policy.action_pos_mean, "a_std": policy.action_pos_std}
    if not getattr(MOD, "_g16_wrapped", False):
        orig = MOD.find_grasp_target
        def wrapped(*a, **k):
            gt, valid = orig(*a, **k); policy.decoder._gt = (gt, valid); return gt, valid
        MOD.find_grasp_target = wrapped; MOD._g16_wrapped = True

    def forward(self, latent_tokens, latent_pad_mask, current_state, subgoal_state=None, target_xyz=None, instruction_embedding=None, latent_modality_ids=None, grounded_target_feature=None, grasp_phase=None):
        B = latent_tokens.shape[0]; memory = self.latent_projection(latent_tokens); mkpm = None if latent_pad_mask is None else ~latent_pad_mask
        queries = (self.action_queries + self.horizon_position_embedding)[None].expand(B, -1, -1); queries = queries + self.state_encoder(current_state)[:, None, :]
        if self.use_subgoal: queries = queries + self.subgoal_encoder(subgoal_state)[:, None, :]
        if self.use_target_xyz: queries = queries + self.target_xyz_encoder(target_xyz)[:, None, :]
        if self.use_instruction: queries = queries + self.instruction_encoder(instruction_embedding)[:, None, :]
        if self.use_grounded_grasp_v2: queries = queries + self.grounded_feature_encoder(grounded_target_feature)[:, None, :]; queries = queries + self.phase_encoder(grasp_phase)[:, None, :]
        gt = self._gt; self._gt = None; tgt_in = target_xyz
        if gt is not None: tgt_in = torch.where(gt[1][:, None], gt[0], target_xyz)
        if self.geo_on and target_xyz is not None:
            st = self._stats; ee = current_state[:, :3] * st["s_std"] + st["s_mean"]; tgt = tgt_in * st["a_std"] + st["a_mean"]; queries = queries + self.geo(tgt - ee, st["a_std"])[:, None, :]
        decoded = self.transformer(tgt=queries, memory=memory, memory_key_padding_mask=mkpm); out = self.action_head(decoded)
        self._last = {"pred": out, "phase": grasp_phase, "gt": None if gt is None else gt[0], "valid": None if gt is None else gt[1]}; return out

    dec.forward = forward.__get__(dec, type(dec)); return dec.geo


def progress_loss(dec) -> torch.Tensor:
    """L_progress = mean over PRE_GRASP & demonstration-anchor-valid samples of ReLU(d_30 - d_10), d_k = || chunk position k - anchor || in the normalised action-position space (same space as the position loss)."""
    L = dec._last; pred = L["pred"][..., :3]; d = (pred - L["gt"][:, None, :]).norm(dim=-1); m = (L["valid"] & (L["phase"] == 0)).float(); pen = torch.relu(d[:, 29] - d[:, 9])
    return (pen * m).sum() / m.sum().clamp(min=1.0)
