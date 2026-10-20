"""SafeDiff-VLA StagePlan: a small language-conditioned head that predicts a generic 4-stage manipulation plan
(P0 pregrasp, P1 grasp, P2 postgrasp/extraction, P3 release/goal, each xyz + sin/cos Euler, plus a release_valid logit)
from the frozen SafeDiff v2 backbone's multimodal tokens, and an explicit stage-wise geometric controller that executes it.

Inference reads ONLY: RGB (3 cameras), the language instruction, proprio state, and the controller's own phase. No task
name, no simulator state, no oracle keypoint/waypoint/pose, no `is_grasped`. See `tests/policies/safediff_vla/test_stageplan.py`
(`test_policy_reads_only_allowed_inputs`).

Stage labels (training only) come from expert-trajectory EVENTS -- see `examples/safediff_vla/stageplan_labels.py`.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch import Tensor, nn

from .modeling_safediff_vla import SafeDiffVLAPolicy
from .rotation_encoding import GRIPPER_INDEX_RAW, ROTATION_SLICE_RAW
from .utils import IMAGE_MODALITY, TEXT_MODALITY

warnings.filterwarnings("ignore", message="Gimbal lock detected")  # euler<->quat at ry=-pi/2 is a benign branch choice
NUM_STAGES = 4  # P0 pregrasp, P1 grasp, P2 postgrasp, P3 release
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, 0.0  # raw gripper command convention of the dataset/env


class StagePlanHead(nn.Module):
    """Spatial-token head: per-camera image tokens (+camera embedding) and language tokens form the memory; 5 learned queries
    (P0..P3, release) -- each shifted by a language-pooled query and the proprio embedding -- cross-attend through a small
    (2-layer) transformer decoder. Stage-specific linear heads output normalized xyz [3], sin/cos Euler [3,2] and a release logit."""

    def __init__(self, latent_dim: int, state_dim: int = 7, d_model: int = 256, n_heads: int = 4, n_layers: int = 2, n_cams: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_cams = n_cams
        self.vis_proj = nn.Linear(latent_dim, d_model)
        self.txt_proj = nn.Linear(latent_dim, d_model)
        self.cam_embed = nn.Embedding(n_cams, d_model)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.lang_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lang_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.stage_queries = nn.Parameter(torch.randn(1, NUM_STAGES + 1, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 2 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.xyz_heads = nn.ModuleList([nn.Linear(d_model, 3) for _ in range(NUM_STAGES)])
        self.rot_heads = nn.ModuleList([nn.Linear(d_model, 6) for _ in range(NUM_STAGES)])
        self.release_head = nn.Linear(d_model, 1)

    def camera_ids(self, is_img: Tensor) -> Tensor:
        """[B,N] bool image-token mask -> [B,N] camera index: the image tokens of a row are consecutive equal-size per-camera blocks."""
        rank = torch.cumsum(is_img.long(), dim=1) - 1
        per_cam = (is_img.sum(dim=1, keepdim=True) // self.n_cams).clamp_min(1)
        return (rank // per_cam).clamp(0, self.n_cams - 1)

    def forward(self, tokens: Tensor, pad_mask: Tensor | None, modality_ids: Tensor, state: Tensor) -> dict[str, Tensor]:
        if pad_mask is None:
            pad_mask = tokens.new_ones(tokens.shape[:2], dtype=torch.bool)
        valid = pad_mask.to(torch.bool)
        is_img, is_txt = (modality_ids == IMAGE_MODALITY) & valid, (modality_ids == TEXT_MODALITY) & valid
        no_txt = is_txt.sum(dim=1, keepdim=True) == 0
        is_txt = is_txt | (no_txt & valid)
        vis = self.vis_proj(tokens) + self.cam_embed(self.camera_ids(is_img))
        txt = self.txt_proj(tokens)
        memory = torch.where(is_img.unsqueeze(-1), vis, txt)
        mem_valid = is_img | is_txt
        lang, _ = self.lang_attn(self.lang_query.expand(tokens.shape[0], -1, -1), txt, txt, key_padding_mask=~is_txt, need_weights=False)
        queries = self.stage_queries + lang + self.state_mlp(state).unsqueeze(1)
        h = self.out_norm(self.decoder(queries, memory, memory_key_padding_mask=~mem_valid))
        xyz = torch.stack([self.xyz_heads[i](h[:, i]) for i in range(NUM_STAGES)], dim=1)  # normalized action-xyz space
        rot = torch.stack([self.rot_heads[i](h[:, i]) for i in range(NUM_STAGES)], dim=1).reshape(-1, NUM_STAGES, 3, 2)
        rot = rot / rot.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # unit (sin, cos) per Euler axis
        return {"xyz": xyz, "rot_sc": rot, "release_logit": self.release_head(h[:, NUM_STAGES]).squeeze(-1)}


# ------------------------------------------------------------------------------------------------ controller


@dataclass
class StageConstants:
    """ONE global set measured on TRAIN expert trajectories (see `stageplan_labels.controller_constants`)."""

    v_xyz: float = 0.01017  # m / step
    v_rot: float = 0.0588  # rad / step
    hold_before_close: int = 3
    hold_after_close: int = 7
    hold_before_open: int = 2
    hold_after_open: int = 2
    seg_min: int = 2
    seg_max: int = 100

    @classmethod
    def from_json(cls, path: str | Path) -> StageConstants:
        raw = json.loads(Path(path).read_text())
        d = raw.get("constants", raw)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class _Op:
    kind: str  # "move" | "hold"
    pose: np.ndarray  # [6] target xyz+euler
    grip: float
    frames: int = 0  # for hold; <0 = forever


@dataclass
class StageController:
    """PREGRASP -> GRASP -> CLOSE -> POSTGRASP -> (TRANSPORT/RELEASE if release_valid else HOLD). Pure geometry: each `move`
    starts from the MEASURED EE pose, takes N = clip(ceil(max(|dxyz|/v_xyz, angle/v_rot)), seg_min, seg_max) steps, linear xyz +
    slerp rotation. CLOSE happens when the planned P1 stage completes (+ the expert's own global pre-close hold); no distance
    threshold. OPEN only through the planned release stage; no latch, no decoder gripper."""

    c: StageConstants = field(default_factory=StageConstants)
    ops: list[_Op] = field(default_factory=list)
    queue: list[tuple[np.ndarray, float]] = field(default_factory=list)
    cur_op: _Op | None = None
    hold_left: int = 0
    phase: str = "PLAN"
    last_cmd: np.ndarray | None = None
    last_grip: float = GRIPPER_OPEN
    _names: list[str] = field(default_factory=list)
    _i: int = -1

    def reset(self) -> None:
        self.ops, self.queue, self.cur_op, self.hold_left, self.phase, self.last_cmd, self.last_grip = [], [], None, 0, "PLAN", None, GRIPPER_OPEN

    def set_plan(self, poses: np.ndarray, release: bool) -> None:
        """poses: [4,6] (xyz + euler) for P0..P3."""
        c = self.c
        self.ops = [_Op("move", poses[0], GRIPPER_OPEN, 0), _Op("move", poses[1], GRIPPER_OPEN, 0), _Op("hold", poses[1], GRIPPER_OPEN, c.hold_before_close),
                    _Op("hold", poses[1], GRIPPER_CLOSED, c.hold_after_close), _Op("move", poses[2], GRIPPER_CLOSED, 0)]
        names = ["PREGRASP", "GRASP", "CLOSE", "CLOSE", "POSTGRASP"]
        if release:
            self.ops += [_Op("move", poses[3], GRIPPER_CLOSED, 0), _Op("hold", poses[3], GRIPPER_CLOSED, c.hold_before_open), _Op("hold", poses[3], GRIPPER_OPEN, c.hold_after_open), _Op("hold", poses[3], GRIPPER_OPEN, -1)]
            names += ["TRANSPORT", "RELEASE", "RELEASE", "DONE"]
        else:
            self.ops += [_Op("hold", poses[2], GRIPPER_CLOSED, -1)]
            names += ["HOLD"]
        self._names = names; self._i = -1; self.phase = "PLAN"

    def _segment(self, ee_xyz: np.ndarray, ee_euler: np.ndarray, target: np.ndarray) -> list[tuple[np.ndarray, float]]:
        c = self.c
        d = float(np.linalg.norm(target[:3] - ee_xyz))
        r0, r1 = Rotation.from_euler("xyz", ee_euler), Rotation.from_euler("xyz", target[3:])
        ang = float((r0.inv() * r1).magnitude())
        n = int(min(max(math.ceil(max(d / c.v_xyz, ang / c.v_rot)), c.seg_min), c.seg_max))
        sl = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        out = []
        for k in range(1, n + 1):
            a = k / n
            out.append((np.concatenate([ee_xyz + a * (target[:3] - ee_xyz), sl(a).as_euler("xyz")]), 0.0))
        return out

    def next_action(self, ee_xyz: np.ndarray, ee_euler: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (pose6 command, gripper raw command) for this step."""
        while not self.queue and self.hold_left == 0:
            self._i += 1
            if self._i >= len(self.ops):
                self._i = len(self.ops) - 1
            op = self.ops[self._i]; self.phase = self._names[self._i]; self.cur_op = op
            if op.kind == "move":
                self.queue = [(p, op.grip) for p, _ in self._segment(ee_xyz, ee_euler, op.pose)]
            else:
                self.hold_left = op.frames if op.frames >= 0 else 10**9
            if self._i == len(self.ops) - 1 and op.kind == "hold": break
        if self.queue:
            pose, grip = self.queue.pop(0)
        else:
            op = self.cur_op; pose, grip = op.pose, op.grip; self.hold_left -= 1
        self.last_cmd, self.last_grip = pose, grip
        return pose, grip


# ------------------------------------------------------------------------------------------------ policy


class SafeDiffVLAStagePlanPolicy(SafeDiffVLAPolicy):
    """Frozen SafeDiff v2 backbone/tokens + a trained `StagePlanHead` + `StageController`. Load the v2 checkpoint with
    `from_pretrained`, then `attach_stageplan(dir)`."""

    stageplan_head: StagePlanHead
    stage_controller: StageController

    def attach_stageplan(self, head_dir: str | Path, device: str | None = None) -> None:
        from safetensors.torch import load_file

        head_dir = Path(head_dir)
        cfg = json.loads((head_dir / "stageplan_config.json").read_text())
        self.stageplan_head = StagePlanHead(self._multimodal_latent_dim(), **cfg["head"])
        self.stageplan_head.load_state_dict(load_file(head_dir / "stageplan_head.safetensors"))
        dev = device or next(self.parameters()).device
        self.stageplan_head.to(dev).eval()
        self.stage_controller = StageController(StageConstants(**cfg["constants"]))
        self._sp_plan = None

    def reset(self) -> None:
        super().reset()
        self._sp_plan = None
        if hasattr(self, "stage_controller"):
            self.stage_controller.reset()

    @torch.no_grad()
    def predict_stageplan(self, batch: dict[str, Tensor]) -> dict[str, Tensor | np.ndarray]:
        """RGB + language + proprio -> stage plan in PHYSICAL units: poses [B,4,6] (xyz m, euler rad) and release probability [B]."""
        tokens, pad, mod = self._encode_multimodal_latent(batch)
        state = self._current_state(batch)
        out = self.stageplan_head(tokens, pad, mod, state)
        xyz = out["xyz"] * self.action_pos_std + self.action_pos_mean
        euler = torch.atan2(out["rot_sc"][..., 0], out["rot_sc"][..., 1])
        return {"poses": torch.cat([xyz, euler], dim=-1), "release_prob": torch.sigmoid(out["release_logit"])}

    def _state_to_physical(self, batch: dict[str, Tensor]) -> tuple[np.ndarray, np.ndarray]:
        s = self._current_state(batch)[0]
        ee = (s[:3] * self.state_pos_std + self.state_pos_mean).float().cpu().numpy().astype(np.float64)
        eul = (s[3:6] * self.state_rot_std + self.state_rot_mean).float().cpu().numpy().astype(np.float64)
        return ee, eul

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        ee, eul = self._state_to_physical(batch)
        if self._sp_plan is None:
            plan = self.predict_stageplan(batch)
            self._sp_plan = {"poses": plan["poses"][0].float().cpu().numpy().astype(np.float64), "release_prob": float(plan["release_prob"][0])}
            self.stage_controller.set_plan(self._sp_plan["poses"], self._sp_plan["release_prob"] > 0.5)
        pose, grip = self.stage_controller.next_action(ee, eul)
        dev = self.action_pos_mean.device
        act = torch.zeros(1, 7, device=dev, dtype=torch.float32)
        act[0, :3] = (torch.as_tensor(pose[:3], device=dev, dtype=torch.float32) - self.action_pos_mean) / self.action_pos_std
        act[0, ROTATION_SLICE_RAW] = (torch.as_tensor(pose[3:], device=dev, dtype=torch.float32) - self.action_rot_mean) / self.action_rot_std
        act[0, GRIPPER_INDEX_RAW] = ((torch.tensor(grip, device=dev, dtype=torch.float32) - self.action_grip_mean) / self.action_grip_std).squeeze()
        return act
