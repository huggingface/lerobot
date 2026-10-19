"""G17 helpers: a transparent re-implementation of TemporalActionDecoder.forward (post-norm nn.TransformerDecoderLayer x4, no final norm) with (i) recording at every sub-layer point, (ii) additive injection of a [B,512] vector at any location. Verified bit-close against dec.forward at import-use time (see `check`).
Layer l (0..3):  x1 = LN1(x + SA(x));  x2 = LN2(x1 + CA(x1, memory));  x3 = LN3(x2 + FFN(x2))   -> next layer input.  Head: action_head(x3 of layer 3), applied per horizon token.
Locations: I0 = input queries (the existing target branch), I1/I2/I3 = after layer 0/1/2 output (before the next layer), I4 = after layer 3 (= action-head input)."""
from __future__ import annotations
import torch

LOCS = ("I0", "I1", "I2", "I3", "I4")


def capture_install(policy):
    """Wrap decoder.forward so the last call's inputs are stored in dec._ctx (and the demonstration anchor via find_grasp_target when available)."""
    import lerobot.policies.safediff_vla.modeling_safediff_vla as MOD
    dec = policy.decoder; orig = dec.forward
    if not getattr(MOD, "_g17_wrapped", False):
        of = MOD.find_grasp_target
        def w(*a, **k): gt, v = of(*a, **k); policy.decoder._gt17 = (gt, v); return gt, v
        MOD.find_grasp_target = w; MOD._g17_wrapped = True
    def fwd(latent_tokens, latent_pad_mask, current_state, subgoal_state=None, target_xyz=None, instruction_embedding=None, latent_modality_ids=None, grounded_target_feature=None, grasp_phase=None):
        dec._ctx = {"tok": latent_tokens, "pad": latent_pad_mask, "state": current_state, "target": target_xyz, "feat": grounded_target_feature, "phase": grasp_phase, "mod": latent_modality_ids, "gt": getattr(dec, "_gt17", None)}; dec._gt17 = None
        return orig(latent_tokens, latent_pad_mask, current_state, subgoal_state, target_xyz, instruction_embedding, latent_modality_ids, grounded_target_feature, grasp_phase)
    dec.forward = fwd


def base_queries(dec, ctx):
    B = ctx["tok"].shape[0]; q = (dec.action_queries + dec.horizon_position_embedding)[None].expand(B, -1, -1); q = q + dec.state_encoder(ctx["state"])[:, None, :]
    return q + dec.grounded_feature_encoder(ctx["feat"])[:, None, :] + dec.phase_encoder(ctx["phase"])[:, None, :]


@torch.no_grad()
def run(dec, ctx, target, in_scale=1.0, extra_in=None, inj=None, record=False, geo_in=None):
    """target: [B,3] normalised anchor fed to target_xyz_encoder at the input with gain `in_scale` (0 = branch removed). extra_in: [B,512] added to the input queries. inj: {layer index: [B,512]} added after that layer's output."""
    memory = dec.latent_projection(ctx["tok"]); mkpm = None if ctx["pad"] is None else ~ctx["pad"]; x = base_queries(dec, ctx); rec = {}
    if in_scale != 0.0: x = x + in_scale * dec.target_xyz_encoder(target)[:, None, :]
    if extra_in is not None: x = x + extra_in[:, None, :]
    if record: rec["in"] = x
    for l, layer in enumerate(dec.transformer.layers):
        sa = layer.self_attn(x, x, x, need_weights=False)[0]; x1 = layer.norm1(x + sa); ca = layer.multihead_attn(x1, memory, memory, key_padding_mask=mkpm, need_weights=False)[0]; x2 = layer.norm2(x1 + ca); ff = layer.linear2(layer.activation(layer.linear1(x2))); x3 = layer.norm3(x2 + ff)
        if inj is not None and l in inj: x3 = x3 + inj[l][:, None, :]
        if record: rec.update({f"L{l}.sa": sa, f"L{l}.x1": x1, f"L{l}.ca": ca, f"L{l}.x2": x2, f"L{l}.ff": ff, f"L{l}.x3": x3})
        x = x3
    out = dec.action_head(x)
    if record: rec["out"] = out; rec["x_final"] = x
    return out, rec


def place(dec, ctx, target, loc, mode, vec=None, scale=1.0):
    """Decode with the target vector (default target_xyz_encoder(target)) at location `loc`. mode 'move': input branch removed, vector injected at loc (I0 == V2). mode 'extra': V2 unchanged + one extra copy of the vector at loc (I0 == gamma 2)."""
    v = dec.target_xyz_encoder(target) if vec is None else vec; v = scale * v
    if mode == "move":
        if loc == "I0": return run(dec, ctx, target, in_scale=1.0 if vec is None else 0.0, extra_in=None if vec is None else v)[0]
        return run(dec, ctx, target, in_scale=0.0, inj={int(loc[1]) - 1: v})[0]
    if loc == "I0": return run(dec, ctx, target, extra_in=v)[0]
    return run(dec, ctx, target, inj={int(loc[1]) - 1: v})[0]


def check(dec, ctx, ref_out):
    out, _ = run(dec, ctx, ctx["target"]); return float((out - ref_out).abs().max())
