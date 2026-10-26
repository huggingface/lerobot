#!/usr/bin/env python
"""PHASE G0.5: clean grounding diagnostic -- EARLY-HORIZON data collection (first N env steps only).

Frozen `GroundedGrasp-v2` 20k checkpoint, policy behavior UNCHANGED: every diagnostic here is a
read-only side-channel call performed AFTER the original `select_action` has produced its action
(same pattern as `phase_g0_grounding_audit_rollout.py`). The episode is aborted after `--steps` env
steps by raising from the wrapper; rollout outcome is NOT an object of study in this phase.

Per env step (= per policy observation) it records, for the SAME observation the policy saw:
    - predicted target xyz (robot frame + world), language_query, grounded_target_feature
    - per-head cross-attention weights over ALL latent tokens (`[H, N]`) of both the
      TargetQueryExtractor (text) and VisualGroundingCrossAttention (image) modules
    - simulator GT (diagnostic label ONLY, never policy input): instructed object position,
      every same-type candidate's position and approximate size, EE position
    - per-camera segmentation-derived fractions of every 8x8 token cell covered by
      {instructed object, each distractor candidate, robot}; everything else is "background"
    - pinhole projection of the GT object origin into each camera and the seg-mask centroid of the
      object (cross-check of the projection model)
Token->pixel validation (section 1) runs once, at step 0 of the first episode: a localized gray
occlusion is applied to a COPY of the batch and the response of (a) `embed_prefix` output (pure
token ordering) and (b) the post-transformer `latent_tokens` the cross-attention actually reads is
measured per token.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/phase_g05_clean_grounding_collect.py \
        --tasks select_poker select_book --seeds 1000 1001 --steps 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, robot_base  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.utils import IMAGE_MODALITY, TEXT_MODALITY  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

OUT_ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g05_clean_grounding")
CAMERA_KEYS = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]
GRID = 8
RES = 480
CELL = RES // GRID  # 60 px; 480 -> 512 resize has NO padding (square), so cells map uniformly
SNAPSHOT_STEPS = (0, 15)
VALIDATION_CELLS = [(1, 1), (1, 6), (3, 5), (4, 4), (6, 1), (6, 6)]  # asymmetric: separates row/col-major


class StopEpisode(Exception):
    pass


class Holder:
    env_impl: Any = None


class StepStop:
    n_steps: int = 30


def _impl_env(env_impl):
    return env_impl._env


def geom_ids(physics, mjcf_model) -> np.ndarray:
    return np.asarray(physics.bind(mjcf_model.find_all("geom")).element_id, dtype=int)


def region_masks(env_impl) -> dict:
    """Per-camera boolean masks, built from simulator segmentation (geom ids)."""
    e = _impl_env(env_impl)
    ph, task = e.physics, e.task
    target_name = task.target_entity
    type_name = type(task.entities[target_name]).__name__
    cand_names = [n for n, ent in task.entities.items() if type(ent).__name__ == type_name]
    ids = {n: geom_ids(ph, task.entities[n].mjcf_model) for n in cand_names}
    robot_ids = geom_ids(ph, task.robot.mjcf_model)
    out = {"cand_names": cand_names, "target_name": target_name, "per_camera": []}
    for cam_id in env_impl._camera_order:
        seg = ph.render(height=RES, width=RES, camera_id=cam_id, segmentation=True)
        is_geom = seg[..., 1] == 5
        gid = seg[..., 0]
        masks = {n: is_geom & np.isin(gid, ids[n]) for n in cand_names}
        masks["__robot__"] = is_geom & np.isin(gid, robot_ids)
        out["per_camera"].append(masks)
    return out


def cell_fractions(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3)).reshape(-1)  # row-major [64]


def project(env_impl, cam_id: int, xyz: np.ndarray) -> list[float]:
    ph = _impl_env(env_impl).physics
    pos = ph.data.cam_xpos[cam_id]
    rot = ph.data.cam_xmat[cam_id].reshape(3, 3)  # cam->world
    p = rot.T @ (np.asarray(xyz) - pos)
    f = (RES / 2) / np.tan(np.deg2rad(ph.model.cam_fovy[cam_id]) / 2)
    if p[2] >= 0:
        return [float("nan"), float("nan")]
    return [float(f * p[0] / -p[2] + RES / 2), float(RES / 2 - f * p[1] / -p[2])]  # (x=col, y=row)


def object_extent(physics, mjcf_model) -> dict:
    ids = geom_ids(physics, mjcf_model)
    xp = physics.data.geom_xpos[ids]
    rb = physics.model.geom_rbound[ids]
    ok = rb > 0
    xp, rb = xp[ok], rb[ok]
    lo, hi = (xp - rb[:, None]).min(0), (xp + rb[:, None]).max(0)
    return {"extent_xyz_m": (hi - lo).tolist()}  # bounding-sphere based: UPPER bound for meshes


def entropy_of(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def occlusion_validation(policy: SafeDiffVLAPolicy, batch: dict) -> dict:
    """Section 1. Gray 60x60 block at known (cam, row, col) in a COPY of the batch."""
    results = []
    with torch.no_grad():
        base_batch = policy._prepare_backbone_batch(batch)
        model = policy.backbone.model

        def embed(b):
            images, img_masks = policy.backbone.prepare_images(b)
            state = policy.backbone.prepare_state(b)
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: PLC0415

            emb, pad, att = model.embed_prefix(
                images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], state=state
            )
            return emb.float()

        emb0 = embed(base_batch)
        lat0, _, mod_ids = policy._encode_multimodal_latent(batch)
        n_tok = emb0.shape[1]
        meta = {
            "n_tokens_total": int(n_tok),
            "n_image_tokens": int((mod_ids == IMAGE_MODALITY).sum()),
            "n_text_tokens_incl_pad": int((mod_ids == TEXT_MODALITY).sum()),
            "image_token_index_range": [int(torch.where(mod_ids[0] == IMAGE_MODALITY)[0].min()),
                                        int(torch.where(mod_ids[0] == IMAGE_MODALITY)[0].max())],
            "add_image_special_tokens": bool(model.add_image_special_tokens),
            "resize_imgs_with_padding": list(policy.backbone.config.resize_imgs_with_padding),
            "config_image_features_order": list(policy.backbone.config.image_features.keys())
            if hasattr(policy.backbone.config.image_features, "keys") else list(policy.backbone.config.image_features),
            "batch_image_shapes": {k: list(batch[k].shape) for k in CAMERA_KEYS if k in batch},
        }
        for cam in range(3):
            for r, c in VALIDATION_CELLS:
                b2 = dict(batch)
                img = batch[CAMERA_KEYS[cam]].clone()
                H = img.shape[-1]
                cs = H // GRID
                img[..., r * cs:(r + 1) * cs, c * cs:(c + 1) * cs] = 0.5
                b2[CAMERA_KEYS[cam]] = img
                bb = policy._prepare_backbone_batch(b2)
                d_emb = (embed(bb) - emb0).norm(dim=-1)[0, :192]
                lat1, _, _ = policy._encode_multimodal_latent(b2)
                d_lat = (lat1 - lat0).norm(dim=-1)[0, :192]
                exp = cam * 64 + r * 8 + c
                transp = cam * 64 + c * 8 + r
                nb = [cam * 64 + rr * 8 + cc for rr in range(max(0, r - 1), min(8, r + 2))
                      for cc in range(max(0, c - 1), min(8, c + 2))]
                results.append({
                    "camera": cam, "cell_rc": [r, c], "expected_token": exp, "transposed_token": transp,
                    "embed_argmax": int(d_emb.argmax()), "embed_expected_rank": int((d_emb > d_emb[exp]).sum()),
                    "embed_mass_in_3x3": float(d_emb[nb].sum() / d_emb.sum()),
                    "embed_mass_at_expected": float(d_emb[exp] / d_emb.sum()),
                    "latent_argmax": int(d_lat.argmax()), "latent_expected_rank": int((d_lat > d_lat[exp]).sum()),
                    "latent_mass_in_3x3": float(d_lat[nb].sum() / d_lat.sum()),
                    "latent_mass_at_expected": float(d_lat[exp] / d_lat.sum()),
                })
    return {"meta": meta, "cases": results}


def install_capture(policy: SafeDiffVLAPolicy, store: dict, flags: dict) -> None:
    orig = SafeDiffVLAPolicy.select_action

    def wrapped(self, batch):
        action = orig(self, batch)
        with torch.no_grad():
            tokens, pad, mod = self._encode_multimodal_latent(batch)
            tqe, vg = self.target_query_extractor, self.visual_grounding_cross_attn
            valid = pad.to(torch.bool)
            text_valid = (mod == TEXT_MODALITY) & valid
            text_valid |= (text_valid.sum(1, keepdim=True) == 0) & valid
            q = tqe.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            pooled, t_attn = tqe.attn(q, tokens, tokens, key_padding_mask=~text_valid,
                                      need_weights=True, average_attn_weights=False)
            lang_q = pooled.squeeze(1)
            img_valid = (mod == IMAGE_MODALITY) & valid
            img_valid |= (img_valid.sum(1, keepdim=True) == 0) & valid
            grounded, i_attn = vg.attn(lang_q.unsqueeze(1), tokens, tokens, key_padding_mask=~img_valid,
                                       need_weights=True, average_attn_weights=False)
            feat = grounded.squeeze(1)
            xyz = self.target_point_head(feat)
            xyz_m = (xyz * self.action_pos_std + self.action_pos_mean)[0].cpu().numpy()
            if flags.get("validate_pending"):
                flags["validate_pending"] = False
                flags["validation"] = occlusion_validation(self, batch)

        step = len(store["steps"])
        env_impl = Holder.env_impl
        e = _impl_env(env_impl)
        ph, task, robot = e.physics, e.task, e.task.robot
        masks = region_masks(env_impl)
        base = robot_base(env_impl)
        names = masks["cand_names"]
        pos = {n: np.asarray(task.entities[n].get_xpos(ph), dtype=float) for n in names}
        tname = masks["target_name"]
        if "extents" not in store:
            store["extents"] = {n: object_extent(ph, task.entities[n].mjcf_model) for n in names}
        rec = {
            "step": step, "target_name": tname, "cand_names": names,
            "cand_pos_world": {n: pos[n].tolist() for n in names},
            "ee_pos_world": np.asarray(robot.get_end_effector_pos(ph), dtype=float).tolist(),
            "pred_robot_frame": xyz_m.tolist(), "pred_world": (xyz_m + base).tolist(),
            "robot_base": np.asarray(base).tolist(),
        }
        i_np = i_attn[0, :, 0, :].cpu().numpy()  # [H, N]
        t_np = t_attn[0, :, 0, :].cpu().numpy()
        img_idx = torch.where(mod[0] == IMAGE_MODALITY)[0].cpu().numpy()
        txt_idx = torch.where(text_valid[0])[0].cpu().numpy()
        rec["image_token_idx_first_last"] = [int(img_idx[0]), int(img_idx[-1])]
        a = i_np.mean(0)[img_idx]  # [192] head-mean over image tokens (masked others = 0)
        rec["img_attn_total_on_image_tokens"] = float(i_np.mean(0)[img_idx].sum())
        rec["img_attn_entropy"] = entropy_of(a / max(a.sum(), 1e-12))
        rec["img_attn_max"] = float(a.max())
        ta = t_np.mean(0)[txt_idx]
        rec["text_attn_entropy"] = entropy_of(ta / max(ta.sum(), 1e-12))
        # region cell fractions [cam, cell] per region
        proj, cen = {}, {}
        for ci, cam_id in enumerate(env_impl._camera_order):
            proj[ci] = {n: project(env_impl, cam_id, pos[n]) for n in names}
            m = masks["per_camera"][ci][tname]
            ys, xs = np.nonzero(m)
            cen[ci] = [float(xs.mean()), float(ys.mean())] if len(xs) else None
        rec["proj_px"], rec["segcentroid_px"] = proj, cen
        rec["target_visible_px"] = [int(masks["per_camera"][ci][tname].sum()) for ci in range(3)]
        arrs = {
            "attn_img_heads": i_np[:, img_idx].astype(np.float32),  # [H,192]
            "attn_text_heads": t_np[:, txt_idx].astype(np.float32),
            "lang_query": lang_q[0].cpu().numpy().astype(np.float32),
            "feat": feat[0].cpu().numpy().astype(np.float32),
        }
        for n in names + ["__robot__"]:
            arrs[f"cellfrac::{n}"] = np.stack([cell_fractions(masks["per_camera"][ci][n]) for ci in range(3)])
        store["steps"].append(rec)
        store["arrays"].append(arrs)
        if step in SNAPSHOT_STEPS:
            frames = {k: (batch[k][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                      for k in CAMERA_KEYS if k in batch}
            labels = []
            for ci in range(3):
                lab = np.zeros((RES, RES), np.uint8)
                lab[masks["per_camera"][ci]["__robot__"]] = 3
                for n in names:
                    if n != tname:
                        lab[masks["per_camera"][ci][n]] = 2
                lab[masks["per_camera"][ci][tname]] = 1
                labels.append(lab)
            store["snapshots"][step] = {"frames": frames, "labels": np.stack(labels)}
        if len(store["steps"]) >= StepStop.n_steps:
            raise StopEpisode
        return action

    policy.select_action = wrapped.__get__(policy, SafeDiffVLAPolicy)


def install_env_hook() -> tuple:
    cls = VLABenchEnvImpl
    orig_reset = cls.reset

    def patched_reset(self, seed=None, **kw):
        out = orig_reset(self, seed=seed, **kw)
        Holder.env_impl = self
        return out

    cls.reset = patched_reset
    return cls, orig_reset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()
    StepStop.n_steps = args.steps
    out_root = Path(args.out)

    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to(args.device)
    policy.eval()
    flags: dict = {}
    cls, orig_reset = install_env_hook()
    try:
        for task_name in args.tasks:
            env_cfg = VLABenchEnv(task=task_name)
            envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
            env = envs[task_name][0]
            pre, post = make_pre_post_processors(
                policy_cfg=policy.config, pretrained_path=CHECKPOINT,
                preprocessor_overrides={"device_processor": {"device": args.device},
                                        "rename_observations_processor": {"rename_map": RENAME_MAP}},
            )
            env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
            flags["validate_pending"] = not (out_root / f"token_pixel_validation_{task_name}.json").exists()
            for seed in args.seeds:
                ep_dir = out_root / task_name / f"seed{seed}"
                if (ep_dir / "meta.json").exists():
                    continue
                store = {"steps": [], "arrays": [], "snapshots": {}}
                install_capture(policy, store, flags)
                set_seed(seed)
                try:
                    eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                                preprocessor=pre, postprocessor=post, n_episodes=1, max_episodes_rendered=0,
                                videos_dir=None, start_seed=seed, episode_callback=lambda *a, **k: None)
                except StopEpisode:
                    pass
                finally:
                    policy.__dict__.pop("select_action", None)
                ep_dir.mkdir(parents=True, exist_ok=True)
                if "validation" in flags:
                    out_root.mkdir(parents=True, exist_ok=True)
                    (out_root / f"token_pixel_validation_{task_name}.json").write_text(json.dumps(flags.pop("validation"), indent=2))
                keys = store["arrays"][0].keys()
                np.savez_compressed(ep_dir / "arrays.npz", **{k: np.stack([a[k] for a in store["arrays"]]) for k in keys})
                for st, snap in store["snapshots"].items():
                    for k, arr in snap["frames"].items():
                        Image.fromarray(arr).save(ep_dir / f"step{st:02d}_{k.split('.')[-1]}.png")
                    np.save(ep_dir / f"step{st:02d}_labels.npy", snap["labels"])
                (ep_dir / "meta.json").write_text(json.dumps(
                    {"task": task_name, "seed": seed, "extents": store["extents"], "steps": store["steps"]}))
                logger.info("[%s seed=%d] saved %d steps -> %s", task_name, seed, len(store["steps"]), ep_dir)
            env.close()
    finally:
        cls.reset = orig_reset


if __name__ == "__main__":
    main()
