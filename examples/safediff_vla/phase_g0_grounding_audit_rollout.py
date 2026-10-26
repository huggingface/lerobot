#!/usr/bin/env python
"""PHASE G0: object-identity grounding audit -- MINIMAL new rollout (justified: attention weights,
raw camera images, and full scene entity lists were never saved by any prior phase's logger, so
this information cannot be reconstructed from existing `episodes_raw.json`/videos alone). Frozen
`GroundedGrasp-v2` 20k checkpoint, policy behavior UNCHANGED -- every extra computation here is a
read-only diagnostic call against the SAME frozen submodules the policy already uses, exactly like
every prior "monitor-only forward" in this branch's history.

Per step (capped at `MAX_STEPS`, since R0 already established long/chaotic rollouts don't add
reliable signal beyond the first few chunks, and this keeps the audit's own compute/storage
bounded):
    - predicted_target_xyz (world frame) via v2's own frozen TargetQueryExtractor ->
      VisualGroundingCrossAttention -> TargetPointHead pipeline (same pattern as Phase 1A/2A/2C's
      monitor-only forward)
    - language_query [D] and grounded_target_feature [D] -- full vectors, for Phase G0 section 4's
      feature-separability probe
    - text cross-attention weights [num_heads, N_text] and image cross-attention weights
      [num_heads, N_image] -- obtained by calling the SAME frozen `nn.MultiheadAttention` modules
      (`policy.target_query_extractor.attn`, `policy.visual_grounding_cross_attn.attn`) a SECOND
      time with `need_weights=True, average_attn_weights=False` on the IDENTICAL query/key/value/
      mask tensors `TargetQueryExtractor.forward`/`VisualGroundingCrossAttention.forward` already
      computed internally -- this cannot change `action`/anything in the real forward path, since
      it is a separate call using already-frozen weights on tensors already computed read-only
      elsewhere in this same step.
    - attention entropy + max-weight (head-averaged) for section 5
    - nearest scene-object identity for predicted_target_xyz: `task.entities` filtered to entities
      sharing the INSTRUCTED object's own type (`type(task.entities[task.target_entity]).__name__`
      -- task-agnostic, no hardcoded per-task class name), nearest by Euclidean distance. GT/scene
      identity used ONLY for this diagnostic label, never fed to the policy.
    Every `STEP_IMAGE_EVERY`-th step: all 3 camera RGB frames saved as PNG (for the Phase G0
    section-2 attention-heatmap figure).

Image-token spatial layout (empirically verified via a throwaway probe, not reverse-engineered
from source): `latent_tokens` for this checkpoint/env is `[B, 198, 960]` = 192 IMAGE + 5 TEXT + 1
STATE tokens; IMAGE tokens are exactly 64 per camera (3 cameras x 64 = 192), contiguous in camera
order (camera1 idx 0-63, camera2 64-127, camera3 128-191), each reshapeable to an 8x8 grid
(row-major, standard ViT-style flattening -- not independently confirmed beyond this being the
only self-consistent factorization of 64, but stated as an assumption here, not fact).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/phase_g0_grounding_audit_rollout.py \
        --task select_poker --seed 1001
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
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

OUT_DIR_ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g0_grounding_audit")
MAX_STEPS = 150
STEP_IMAGE_EVERY = 10
PATCHES_PER_CAMERA = 64
GRID_SIDE = 8  # sqrt(64)
CAMERA_KEYS = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]


def entropy_of(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    return float(-(p * np.log(p)).sum())


def nearest_candidate_object(target_xyz: np.ndarray, candidates: dict[str, np.ndarray], instructed_name: str) -> dict:
    if not candidates:
        return {"nearest_name": None, "nearest_dist_m": None, "is_instructed": None}
    dists = {name: float(np.linalg.norm(target_xyz - pos)) for name, pos in candidates.items()}
    nearest_name = min(dists, key=dists.get)
    return {
        "nearest_name": nearest_name,
        "nearest_dist_m": dists[nearest_name],
        "is_instructed": nearest_name == instructed_name,
        "dist_to_instructed_m": dists.get(instructed_name),
    }


def install_grounding_capture(policy: SafeDiffVLAPolicy) -> dict:
    orig_select_action = SafeDiffVLAPolicy.select_action
    log: list[dict] = []
    image_cache: list[dict] = []

    def wrapped(self, batch):
        action = orig_select_action(self, batch)

        with torch.no_grad():
            latent_tokens, latent_pad_mask, latent_modality_ids = self._encode_multimodal_latent(batch)

            tqe = self.target_query_extractor
            valid = latent_pad_mask.to(torch.bool) if latent_pad_mask is not None else latent_tokens.new_ones(
                latent_tokens.shape[:2], dtype=torch.bool
            )
            text_valid = (latent_modality_ids == TEXT_MODALITY) & valid
            no_text = text_valid.sum(dim=1, keepdim=True) == 0
            text_valid = text_valid | (no_text & valid)
            query = tqe.query.unsqueeze(0).expand(latent_tokens.shape[0], -1, -1)
            pooled, text_attn = tqe.attn(
                query=query, key=latent_tokens, value=latent_tokens, key_padding_mask=~text_valid,
                need_weights=True, average_attn_weights=False,
            )
            language_query = pooled.squeeze(1)

            vgca = self.visual_grounding_cross_attn
            image_valid = (latent_modality_ids == IMAGE_MODALITY) & valid
            no_image = image_valid.sum(dim=1, keepdim=True) == 0
            image_valid = image_valid | (no_image & valid)
            q2 = language_query.unsqueeze(1)
            grounded, image_attn = vgca.attn(
                query=q2, key=latent_tokens, value=latent_tokens, key_padding_mask=~image_valid,
                need_weights=True, average_attn_weights=False,
            )
            grounded_target_feature = grounded.squeeze(1)
            predicted_target_xyz = self.target_point_head(grounded_target_feature)

        target_pos_m = (predicted_target_xyz * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy()
        text_attn_np = text_attn[0, :, 0, :].detach().cpu().numpy()  # [heads, N] (query dim squeezed)
        image_attn_np = image_attn[0, :, 0, :].detach().cpu().numpy()  # [heads, N]
        image_attn_mean = image_attn_np.mean(axis=0)
        text_attn_mean = text_attn_np.mean(axis=0)

        entry = {
            "predicted_target_robot_frame": target_pos_m.tolist(),
            "language_query": language_query[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "grounded_target_feature": grounded_target_feature[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "text_attn_mean_head": text_attn_mean.tolist(),
            "image_attn_mean_head": image_attn_mean.tolist(),
            "image_attn_entropy": entropy_of(image_attn_mean),
            "image_attn_max": float(image_attn_mean.max()),
            "text_attn_entropy": entropy_of(text_attn_mean),
            "text_attn_max": float(text_attn_mean.max()),
            "n_text_tokens": int(text_attn_np.shape[-1]),
            "n_image_tokens": int(image_attn_np.shape[-1]),
        }
        log.append(entry)

        if len(log) % STEP_IMAGE_EVERY == 1:
            frame = {}
            for k in CAMERA_KEYS:
                if k in batch:
                    img = batch[k][0].detach().cpu().clamp(0, 1).numpy()  # [3,H,W]
                    frame[k] = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
            image_cache.append({"step": len(log) - 1, "frames": frame, "image_attn_mean_head": image_attn_mean.tolist()})

        return action

    policy.select_action = wrapped.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log, "image_cache": image_cache}


def uninstall_grounding_capture(policy: SafeDiffVLAPolicy) -> None:
    if "select_action" in policy.__dict__:
        del policy.__dict__["select_action"]


def install_env_instrumentation(capture: dict, instructed_type_holder: dict) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def candidates_and_record(env, step_ix: int) -> dict:
        physics = env._env.physics
        task = env._env.task
        robot = task.robot
        target_name = task.target_entity
        ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
        try:
            target_pos = np.asarray(task.entities[target_name].get_xpos(physics), dtype=float)
        except Exception:  # noqa: BLE001
            target_pos = np.array([np.nan, np.nan, np.nan])
        if instructed_type_holder.get("type_name") is None and target_name in task.entities:
            instructed_type_holder["type_name"] = type(task.entities[target_name]).__name__
        type_name = instructed_type_holder.get("type_name")
        candidates = {}
        if type_name is not None:
            for name, ent in task.entities.items():
                if type(ent).__name__ != type_name:
                    continue
                try:
                    candidates[name] = np.asarray(ent.get_xpos(physics), dtype=float)
                except Exception:  # noqa: BLE001
                    pass
        try:
            is_grasped = bool(task.entities[target_name].is_grasped(physics, robot))
        except Exception:  # noqa: BLE001
            is_grasped = False
        record = {
            "step_ix": step_ix, "ee_pos": ee_pos.tolist(), "target_entity": target_name,
            "target_pos": target_pos.tolist(), "is_grasped": is_grasped,
            "ee_to_target_dist": float(np.linalg.norm(ee_pos - target_pos)),
            "candidate_type": type_name, "candidate_names": list(candidates.keys()),
        }
        base = robot_base(env)
        if capture["log"]:
            p = capture["log"][-1]
            world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
            record["predicted_target_world"] = world.tolist()
            record["predicted_target_dist_to_gt"] = float(np.linalg.norm(world - target_pos))
            nearest = nearest_candidate_object(world, candidates, target_name)
            record["nearest_candidate"] = nearest
            record["image_attn_entropy"] = p["image_attn_entropy"]
            record["image_attn_max"] = p["image_attn_max"]
            record["text_attn_entropy"] = p["text_attn_entropy"]
            record["text_attn_max"] = p["text_attn_max"]
            record["language_query"] = p["language_query"]
            record["grounded_target_feature"] = p["grounded_target_feature"]
        return record

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            state["current_episode_steps"] = [candidates_and_record(self, 0)]
        return result

    def patched_step(self, action):
        result = orig_step(self, action)
        terminated = result[2]
        if not terminated and len(state["current_episode_steps"]) < MAX_STEPS:
            state["current_episode_steps"].append(candidates_and_record(self, len(state["current_episode_steps"])))
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_env_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR_ROOT / args.task

    logger.info("=== loading checkpoint: %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    env_cfg = VLABenchEnv(task=args.task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[args.task][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    capture = install_grounding_capture(policy)
    instructed_type_holder: dict[str, Any] = {}
    env_state = install_env_instrumentation(capture, instructed_type_holder)

    try:
        def episode_callback(episode_ix, rollout_data, env_idx, done_index):
            pass

        set_seed(args.seed)
        info = eval_policy(
            env=env, policy=policy, env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=1,
            max_episodes_rendered=0, videos_dir=None, start_seed=args.seed, episode_callback=episode_callback,
        )
        success = bool(info["per_episode"][0]["success"])
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        uninstall_grounding_capture(policy)

    steps = env_state["current_episode_steps"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "steps.json").write_text(json.dumps({"task": args.task, "seed": args.seed, "success": success, "steps": steps}, indent=2))

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for entry in capture["image_cache"]:
        step = entry["step"]
        for cam_key, arr in entry["frames"].items():
            cam_short = cam_key.split(".")[-1]
            Image.fromarray(arr).save(images_dir / f"step{step:04d}_{cam_short}.png")
        np.save(images_dir / f"step{step:04d}_image_attn.npy", np.array(entry["image_attn_mean_head"]))

    logger.info(
        "[%s seed=%d] success=%s n_steps=%d n_image_snapshots=%d -> %s",
        args.task, args.seed, success, len(steps), len(capture["image_cache"]), out_dir,
    )


if __name__ == "__main__":
    main()
