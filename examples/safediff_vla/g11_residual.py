"""G11-C prototype: target-RELATIVE residual conditioning for the frozen V2 temporal decoder.
ResidualConditioner(e) : e = (target_xyz - current_EE_xyz) / action_pos_std  (3-D, computed INSIDE the decoder call from the `target_xyz` and `current_state` the decoder already receives -> identical in training, fixed-observation tests and closed loop).
decoder queries += lambda_r * ResidualConditioner(e)    -- an extra additive branch next to the existing state / target_xyz / grounded-feature / phase terms (which are untouched and frozen).
The last layer is zero-initialised, so the augmented decoder starts EXACTLY equal to V2 (`residual_on=False` or an untrained conditioner reproduces V2 bit-for-bit). Gripper path, phase logic, VLM, decoder weights: frozen / unchanged."""
from __future__ import annotations
import torch
from torch import nn


class ResidualConditioner(nn.Module):
    def __init__(self, hidden=256, out_dim=512):
        super().__init__(); self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, e): return self.net(e)


def install(policy, lambda_r: float = 1.0) -> ResidualConditioner:
    """Attach a ResidualConditioner to policy.decoder and replace decoder.forward by a copy of TemporalActionDecoder.forward + the extra branch. Idempotent."""
    dec = policy.decoder
    if getattr(dec, "residual", None) is not None: return dec.residual
    dec.residual = ResidualConditioner(out_dim=dec.action_head.in_features).to(next(dec.parameters()).device); dec.residual_on = True; dec.lambda_r = lambda_r
    dec._stats = {"s_mean": policy.state_pos_mean, "s_std": policy.state_pos_std, "a_mean": policy.action_pos_mean, "a_std": policy.action_pos_std}

    def forward(self, latent_tokens, latent_pad_mask, current_state, subgoal_state=None, target_xyz=None, instruction_embedding=None, latent_modality_ids=None, grounded_target_feature=None, grasp_phase=None):
        B = latent_tokens.shape[0]; memory = self.latent_projection(latent_tokens); mkpm = None if latent_pad_mask is None else ~latent_pad_mask
        queries = (self.action_queries + self.horizon_position_embedding)[None].expand(B, -1, -1); queries = queries + self.state_encoder(current_state)[:, None, :]
        if self.use_subgoal: queries = queries + self.subgoal_encoder(subgoal_state)[:, None, :]
        if self.use_target_xyz: queries = queries + self.target_xyz_encoder(target_xyz)[:, None, :]
        if self.use_instruction: queries = queries + self.instruction_encoder(instruction_embedding)[:, None, :]
        if self.use_grounded_grasp_v2: queries = queries + self.grounded_feature_encoder(grounded_target_feature)[:, None, :]; queries = queries + self.phase_encoder(grasp_phase)[:, None, :]
        if self.residual_on and target_xyz is not None:
            st = self._stats; ee = current_state[:, :3] * st["s_std"] + st["s_mean"]; tgt = target_xyz * st["a_std"] + st["a_mean"]; e = (tgt - ee) / st["a_std"]; queries = queries + self.lambda_r * self.residual(e)[:, None, :]
        decoded = self.transformer(tgt=queries, memory=memory, memory_key_padding_mask=mkpm)
        return self.action_head(decoded)

    dec.forward = forward.__get__(dec, type(dec)); return dec.residual
