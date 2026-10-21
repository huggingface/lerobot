"""SafeDiff-VLA StagePlan v2: historical-oracle-semantics-preserving integration.

Design (see the G19-G30 reference behaviours): an external *stage-variable provider* supplies ONLY the variables the historical
oracle interventions supplied -- grasp pose, approach direction, pre-grasp distance, optional global waypoint -- and this policy
executes exactly the validated structure:

    PRE_GRASP : the v2 planner's chunk is REPLACED by a geometric chunk (polyline ee -> [W] -> G - r*d -> G, orientation slerped to
                the grasp rotation), the same replacement the G19/G22/G28 harnesses monkeypatched in.
    CLOSE     : arrival-based (G29 semantics generalised): EE within CLOSE_TOL_M of the grasp pose, or stalled within
                STALL_RADIUS_M of it (subsumes the historical 3 cm reactive close and the G29 1 cm pose-completion close).
    POST_GRASP: the ORIGINAL learned v2 decoder resumes (phase flips to 1, executor queue cleared, target slot frozen at the last
                pre-grasp value). No geometric P2/P3.
    RELEASE   : decoder OPEN is the release proposal. `release_mode`: "latch" (historical G21, gripper held closed),
                "passthrough" (decoder gripper command executed), "oracle_gate" (historical G24 -- needs a PRIVILEGED
                `release_gate_fn`, audit/equivalence tests only).

Inference reads RGB/language/proprio through the provider and the controller's own state; a *privileged* provider is only ever
installed by the historical-equivalence harness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch import Tensor, nn

from .modeling_safediff_vla import SafeDiffVLAPolicy
from .rotation_encoding import GRIPPER_INDEX_RAW, ROTATION_SLICE_RAW

GRIPPER_OPEN = 1.0
# Global constants inherited from the validated diagnostics (NOT tuned here):
CHUNK_H = 50  # v2 action_horizon / execute_horizon
NEAR_G_M = 0.12  # G29/G30 replan rule: when within 12 cm of the grasp pose the chunk goes straight to it (no retreat)
N_STRAIGHT, N_HOLD = 20, 30  # ... 20 steps to the grasp pose, then hold (G29)
SEG_NO_WAYPOINT = (30, 20)  # G19/G22: -> pre (30), -> grasp (20); orientation slerp over the first 30
SEG_WAYPOINT = (20, 15, 15)  # G28 C: -> W (20), -> pre (15), -> grasp (15); orientation slerp over the first 35
CLOSE_TOL_M = 0.01  # G29 pose-completion tolerance (inherited, not claimed optimal)
STALL_RADIUS_M = 0.03  # historical reactive-close radius: a stalled EE inside it counts as arrived
STALL_SPEED_M = 0.002  # expert dwell speed (TRAIN statistic, stageplan_labels.DWELL_SPEED)
STALL_STEPS = 2


@dataclass
class StageVars:
    grasp_xyz: np.ndarray  # [3] robot-base frame (metres)
    grasp_euler: np.ndarray  # [3] scipy extrinsic 'xyz' (rad)
    approach_dir: np.ndarray  # [3] unit vector, pre-grasp -> grasp
    pre_dist: float  # metres
    wp_valid: bool = False
    wp_xyz: np.ndarray | None = None  # [3]


class _StageTargetHead(nn.Module):
    """Stands in for v2's `target_point_head`: returns the (normalised) planner-target slot from the stage variables, exactly the
    slot the historical `OverrideHead` filled with the oracle anchor."""

    def __init__(self, owner: SafeDiffVLAStagePlanV2Policy) -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    def forward(self, feature: Tensor) -> Tensor:
        o = self._owner
        g = torch.as_tensor(o._sv.grasp_xyz, dtype=feature.dtype, device=feature.device)
        return ((g - o.action_pos_mean) / o.action_pos_std).unsqueeze(0).expand(feature.shape[0], -1)


def build_pregrasp_chunk(ee: np.ndarray, euler_cur: np.ndarray, v: StageVars) -> tuple[np.ndarray, np.ndarray]:
    """Historical PRE_GRASP chunk (G19/G22/G28/G29): positions [H,3], eulers [H,3]."""
    G, d, r = np.asarray(v.grasp_xyz, float), np.asarray(v.approach_dir, float), float(v.pre_dist)
    d = d / (np.linalg.norm(d) + 1e-12)
    r1 = Rotation.from_euler("xyz", v.grasp_euler)
    if np.linalg.norm(ee - G) < NEAR_G_M:
        al = (np.arange(1, N_STRAIGHT + 1) / N_STRAIGHT)[:, None]
        pos = np.concatenate([ee + al * (G - ee), np.repeat(G[None], CHUNK_H - N_STRAIGHT, 0)], 0)
        rot = np.repeat(np.unwrap(np.array([euler_cur, r1.as_euler("xyz")]), axis=0)[1][None], CHUNK_H, 0)
        return pos, rot
    pre = G - r * d
    if v.wp_valid and v.wp_xyz is not None:
        segs, n_rot = [(np.asarray(v.wp_xyz, float), SEG_WAYPOINT[0]), (pre, SEG_WAYPOINT[1]), (G, SEG_WAYPOINT[2])], 35
    else:
        segs, n_rot = [(pre, SEG_NO_WAYPOINT[0]), (G, SEG_NO_WAYPOINT[1])], 30
    pts, cur = [], ee
    for tgt, n in segs:
        a = (np.arange(1, n + 1) / n)[:, None]
        pts.append(cur + a * (tgt - cur)); cur = tgt
    pos = np.concatenate(pts, 0)
    sl = Slerp([0.0, 1.0], Rotation.concatenate([Rotation.from_euler("xyz", euler_cur), r1]))
    fr = np.concatenate([np.arange(1, n_rot + 1) / n_rot, np.ones(len(pos) - n_rot)])
    eul = np.unwrap(np.concatenate([euler_cur[None], sl(fr).as_euler("xyz")], 0), axis=0)[1:]
    return pos, eul


class SafeDiffVLAStagePlanV2Policy(SafeDiffVLAPolicy):
    """v2 backbone/decoder UNCHANGED + stage-variable PRE_GRASP executor + arrival close + decoder-resumed post-grasp."""

    stage_var_provider: Callable[[SafeDiffVLAStagePlanV2Policy, dict], StageVars] | None = None
    release_mode: str = "passthrough"
    release_gate_fn: Callable[[], bool] | None = None  # PRIVILEGED (historical G24) -- audit harness only

    def install(self, provider: Callable, release_mode: str = "passthrough", release_gate_fn: Callable[[], bool] | None = None) -> None:
        self.stage_var_provider, self.release_mode, self.release_gate_fn = provider, release_mode, release_gate_fn
        self.target_point_head = _StageTargetHead(self)

    def reset(self) -> None:
        super().reset()
        self._sv: StageVars | None = None
        self._ee_hist: list[np.ndarray] = []
        self._released = False
        self.log: list[dict] = []

    # -- helpers -------------------------------------------------------------------------------------------------------------
    def _ee_physical(self, batch_or_state: Tensor) -> tuple[np.ndarray, np.ndarray]:
        s = batch_or_state[0]
        ee = (s[:3] * self.state_pos_std + self.state_pos_mean).float().cpu().numpy().astype(np.float64)
        eul = (s[3:6] * self.state_rot_std + self.state_rot_mean).float().cpu().numpy().astype(np.float64)
        return ee, eul

    def _ensure_vars(self, batch: dict) -> None:
        if self._sv is None:
            self._sv = self.stage_var_provider(self, batch)

    # -- PRE_GRASP: stage variables fill the planner-target slot, the geometric chunk replaces the decoder's chunk ----------------
    def _plan_temporal_decoder(self, batch: dict):
        self._ensure_vars(batch)
        return super()._plan_temporal_decoder(batch)

    def plan_action_chunk(self, batch: dict):
        actions, metrics = super().plan_action_chunk(batch)
        if self._v2_grasp_phase != 0 or self._sv is None:
            return actions, metrics  # POST_GRASP: the learned v2 decoder's own chunk, untouched
        ee, eul = self._ee_physical(self._current_state(batch))
        pos, rot = build_pregrasp_chunk(ee, eul, self._sv)
        x = actions.clone(); dt, dv = x.dtype, x.device
        x[0, :, :3] = (torch.as_tensor(pos, dtype=dt, device=dv) - self.action_pos_mean) / self.action_pos_std
        x[0, :, ROTATION_SLICE_RAW] = (torch.as_tensor(rot, dtype=dt, device=dv) - self.action_rot_mean) / self.action_rot_std
        x[0, :, GRIPPER_INDEX_RAW] = ((GRIPPER_OPEN - self.action_grip_mean) / self.action_grip_std).to(dt)
        return x, metrics

    # -- CLOSE (arrival) and RELEASE (decoder OPEN, optional gate) -------------------------------------------------------------
    def _grounded_grasp_v2_reactive_close(self, action: Tensor, current_state: Tensor) -> Tensor:
        if self._v2_last_target_xyz is None:
            return action
        ee = (current_state[..., :3] * self.state_pos_std + self.state_pos_mean)[0].float().cpu().numpy().astype(np.float64)
        if not self._v2_close_triggered:
            tgt = (self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean)[0].float().cpu().numpy().astype(np.float64)
            self._ee_hist.append(ee); dist = float(np.linalg.norm(ee - tgt))
            stalled = len(self._ee_hist) > STALL_STEPS and all(np.linalg.norm(self._ee_hist[-k] - self._ee_hist[-k - 1]) < STALL_SPEED_M for k in range(1, STALL_STEPS + 1))
            if dist <= CLOSE_TOL_M or (dist <= STALL_RADIUS_M and stalled):
                self._v2_close_triggered = True; self._v2_grasp_phase = 1; self._executor._action_queue.clear()
                self.log.append({"event": "close", "dist_m": dist, "stalled": bool(stalled)})
        if not self._v2_close_triggered:
            return action
        closed = ((0.0 - self.action_grip_mean) / self.action_grip_std).to(action.dtype)
        mode = self.release_mode
        if mode == "passthrough":
            return action
        if mode == "oracle_gate":  # historical G24: decoder's raw command only when the (privileged) gate is ready
            ready = bool(self.release_gate_fn()) if self.release_gate_fn is not None else False
            if ready or self._released:
                self._released = True
                return action
        action = action.clone(); action[..., GRIPPER_INDEX_RAW] = closed
        return action


# ------------------------------------------------------------------------------------------------ learned predictor
from .utils import IMAGE_MODALITY, TEXT_MODALITY  # noqa: E402

WP_SCALE_M = 0.25  # waypoint offset is regressed in units of the 25 cm radius
PRE_DIST_SCALE_M = 0.10  # pre-grasp distance regressed in units of 10 cm


class StagePlanV2Head(nn.Module):
    """Language-conditioned spatial fusion predicting ONLY the historical oracle variables.

    1. per-camera spatial image tokens (+ camera embedding), language tokens, proprio embedding;
    2. EXPLICIT referent gating: a language-pooled query scores every image token of each camera (softmax per camera) and the
       attention-weighted token features are pooled per camera -- no global pooling before the referent is selected;
    3. the three per-camera referent features + language vector + proprio are fused by an MLP;
    4. a small (2-layer) decoder lets 3 queries (grasp / approach / waypoint) read the fused vector and all tokens;
    5. heads: grasp_xyz (normalised action-xyz), grasp rotation (sin/cos Euler), approach_dir (unit), pre_dist, waypoint offset.
    """

    def __init__(self, latent_dim: int, state_dim: int = 7, d_model: int = 256, n_heads: int = 4, n_layers: int = 2, n_cams: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_cams, self.d = n_cams, d_model
        self.vis_proj, self.txt_proj = nn.Linear(latent_dim, d_model), nn.Linear(latent_dim, d_model)
        self.cam_embed = nn.Embedding(n_cams, d_model)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.lang_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.q_ref = nn.Linear(d_model, d_model); self.k_ref = nn.Linear(d_model, d_model)  # referent gating over image tokens
        self.fuse = nn.Sequential(nn.Linear(d_model * (n_cams + 2), d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.queries = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 2 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers); self.out_norm = nn.LayerNorm(d_model)
        self.grasp_xyz, self.grasp_rot = nn.Linear(d_model, 3), nn.Linear(d_model, 6)
        self.approach, self.pre_dist, self.wp = nn.Linear(d_model, 3), nn.Linear(d_model, 1), nn.Linear(d_model, 3)

    def camera_ids(self, is_img: Tensor) -> Tensor:
        rank = torch.cumsum(is_img.long(), dim=1) - 1
        per_cam = (is_img.sum(dim=1, keepdim=True) // self.n_cams).clamp_min(1)
        return (rank // per_cam).clamp(0, self.n_cams - 1)

    def forward(self, tokens: Tensor, pad_mask: Tensor | None, modality_ids: Tensor, state: Tensor) -> dict[str, Tensor]:
        if pad_mask is None: pad_mask = tokens.new_ones(tokens.shape[:2], dtype=torch.bool)
        valid = pad_mask.to(torch.bool)
        is_img, is_txt = (modality_ids == IMAGE_MODALITY) & valid, (modality_ids == TEXT_MODALITY) & valid
        is_txt = is_txt | ((is_txt.sum(dim=1, keepdim=True) == 0) & valid)
        cam = self.camera_ids(is_img)
        vis = self.vis_proj(tokens) + self.cam_embed(cam); txt = self.txt_proj(tokens)
        lang, _ = self.lang_attn(self.lang_query.expand(tokens.shape[0], -1, -1), txt, txt, key_padding_mask=~is_txt, need_weights=False)  # [B,1,d]
        qv = self.q_ref(lang.squeeze(1)); kv = self.k_ref(vis)
        logits = torch.einsum("bd,bnd->bn", qv, kv) / self.d ** 0.5
        pooled = []
        for c in range(self.n_cams):
            m = is_img & (cam == c)
            w = torch.softmax(logits.masked_fill(~m, float("-inf")).float(), dim=1).nan_to_num(0.0).to(vis.dtype)
            pooled.append(torch.einsum("bn,bnd->bd", w, vis))
        st = self.state_mlp(state)
        fused = self.fuse(torch.cat(pooled + [lang.squeeze(1), st], dim=-1))
        memory = torch.where(is_img.unsqueeze(-1), vis, txt); mem_valid = is_img | is_txt
        q = self.queries + fused.unsqueeze(1)
        h = self.out_norm(self.decoder(q, memory, memory_key_padding_mask=~mem_valid))
        rot = self.grasp_rot(h[:, 0]).reshape(-1, 3, 2); rot = rot / rot.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ad = self.approach(h[:, 1]); ad = ad / ad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return {"grasp_xyz": self.grasp_xyz(h[:, 0]), "grasp_rot_sc": rot, "approach_dir": ad, "pre_dist": self.pre_dist(h[:, 1]).squeeze(-1), "wp_rel": self.wp(h[:, 2]), "ref_logits": logits}


def make_learned_provider(head: StagePlanV2Head) -> Callable:
    """Provider used at INFERENCE: RGB + language + proprio -> StageVars. No privileged information."""

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
