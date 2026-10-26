#!/usr/bin/env python
"""Controlled counterfactual audit: with visual observation and robot state FROZEN (a single
`env.reset()` per scene, never `env.step()` -- no rollout, no training, no checkpoint changes),
vary ONLY the language instruction across the different real cards present in that scene, and
record the policy's own pooled VLM latent, and its temporal-decoder action-chunk output (first-10
xyz, predicted first-close xyz), per instruction.

Checkpoint: the canonical `temporal_decoder` checkpoint (padfix_20k/020000) -- the one actually
observed in rollout videos, and the one every closed-loop diagnostic this session has used. No
`temporal_decoder_grounded_grasp` checkpoint exists on disk (that architecture has never been
trained -- see prior audit turns), so there is no separate TargetPointHead output to read; the
decoder's own predicted-first-close xyz (first step in its own 50-step output chunk where the
gripper channel crosses closed) is used as the load-bearing "where is it aiming" signal instead,
exactly the quantity the observed "always same spot" video behavior is about. This is stated
explicitly rather than silently substituted.

No model/execution/env code is modified -- reads the policy's own internal methods
(_encode_multimodal_latent, _pooled_latent, _encode_state, decoder, _decode_action_prediction) for
introspection only, the same non-invasive pattern this session's other diagnostics use for env
internals.

Usage:
    uv run python examples/safediff_vla/instruction_counterfactual_audit.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_INDEX = 6
GRIPPER_OPEN_THRESHOLD = 0.5
SEEDS = [1000, 1001, 1002, 1003, 1004]


def pretty_identity(identity: str) -> str:
    rank, _, suit = identity.partition("_of_")
    return f"{rank} of {suit}"


def get_scene_cards(env_impl: VLABenchEnvImpl) -> dict[str, np.ndarray]:
    physics = env_impl._env.physics
    task = env_impl._env.task
    base = env_impl._robot_base_xyz if env_impl._robot_base_xyz is not None else np.zeros(3, dtype=float)
    cards: dict[str, np.ndarray] = {}
    for name, ent in task.entities.items():
        if type(ent).__name__ != "Poker":
            continue
        pos_world = np.asarray(ent.get_xpos(physics), dtype=float)
        cards[name] = pos_world - base
    return cards


def capture_scene(seed: int, device: str) -> dict[str, Any]:
    """One reset, zero steps. Returns the frozen observation (as a plain dict, CPU tensors) plus
    ground-truth card identities/positions (robot frame) for that exact reset."""
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_impl: VLABenchEnvImpl = env.envs[0]
    try:
        observation, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        target_name = env_impl._env.task.target_entity
        cards = get_scene_cards(env_impl)
        obs_t = preprocess_observation(observation)
        obs_t = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs_t.items()}
    finally:
        env.close()
    return {"seed": seed, "obs_t": obs_t, "cards": cards, "target_name": target_name}


def run_instruction(
    policy: SafeDiffVLAPolicy,
    env_preprocessor,
    preprocessor,
    postprocessor,
    obs_t_base: dict[str, Any],
    instruction: str,
    device: str,
) -> dict[str, Any]:
    obs_t = dict(obs_t_base)
    obs_t["task"] = [instruction]
    obs_t = env_preprocessor(obs_t)
    obs_norm = preprocessor(obs_t)

    with torch.no_grad():
        latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(obs_norm)
        pooled = policy._pooled_latent(latent_tokens, latent_pad_mask)
        current_state = policy._encode_state(policy._current_state(obs_norm))
        actions_encoded = policy.decoder(latent_tokens, latent_pad_mask, current_state, None, target_xyz=None)
        actions_norm = policy._decode_action_prediction(actions_encoded)

    actions_raw = postprocessor(actions_norm[0].clone()).cpu().numpy()  # [H, 7]
    gripper = actions_raw[:, GRIPPER_INDEX]
    closed = gripper <= GRIPPER_OPEN_THRESHOLD
    close_idx = int(np.argmax(closed)) if closed.any() else None

    return {
        "instruction": instruction,
        "pooled_latent": pooled[0].float().cpu().numpy(),
        "first10_xyz": actions_raw[:10, :3].tolist(),
        "predicted_close_step": close_idx,
        "predicted_close_xyz": actions_raw[close_idx, :3].tolist() if close_idx is not None else None,
        "last_step_xyz_fallback": actions_raw[-1, :3].tolist(),
    }


def nearest_card(point: np.ndarray, cards: dict[str, np.ndarray]) -> tuple[str, float]:
    dists = {name: float(np.linalg.norm(point - pos)) for name, pos in cards.items()}
    nearest_name = min(dists, key=dists.get)
    return nearest_name, dists[nearest_name]


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_instruction_counterfactual_audit")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seeds[0])
    logger.info("=== loading policy from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"

    env_cfg = VLABenchEnv(task=TASK)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    scene_results = []
    for seed in args.seeds:
        logger.info("=== scene seed=%d: single reset, zero steps ===", seed)
        scene = capture_scene(seed, args.device)
        cards = scene["cards"]
        if len(cards) < 2:
            logger.info("seed=%d: fewer than 2 cards in scene, skipping", seed)
            continue

        instructions = {name: f"primitive: Please pick the poker {pretty_identity(name)}" for name in cards}
        # Calibration control: a maximally different instruction (wrong task domain entirely,
        # same image/state) -- to tell apart "language barely moves the latent at all" (bucket 1)
        # from "the latent registers coarse task changes fine, but is blind to which-card
        # specifically" (a finer-grained version of bucket 1, still a grounding problem but a
        # different one). Not a real card name -- excluded from nearest-card/same-card bookkeeping,
        # reported separately as __unrelated_control__.
        instructions["__unrelated_control__"] = "primitive: Pick up the mahjong of 1 pin"  # calibration only, not a real select_poker instruction
        per_instruction = {}
        for name, instr in instructions.items():
            res = run_instruction(policy, env_preprocessor, preprocessor, postprocessor, scene["obs_t"], instr, args.device)
            point = np.array(res["predicted_close_xyz"] if res["predicted_close_xyz"] is not None else res["last_step_xyz_fallback"])
            nearest_name, nearest_dist = nearest_card(point, cards)
            res["nearest_card_identity"] = nearest_name
            res["nearest_card_dist"] = nearest_dist
            res["used_fallback_point"] = res["predicted_close_xyz"] is None
            per_instruction[name] = res
            logger.info(
                "  instruction card=%s -> predicted_close_step=%s nearest_card=%s(%.3fm) fallback=%s",
                name, res["predicted_close_step"], nearest_name, nearest_dist, res["used_fallback_point"],
            )

        # Pairwise comparisons across every instruction pair in this scene.
        pair_stats = []
        names = list(instructions.keys())
        for a, b in itertools.combinations(names, 2):
            ra, rb = per_instruction[a], per_instruction[b]
            latent_l2 = float(np.linalg.norm(ra["pooled_latent"] - rb["pooled_latent"]))
            latent_cos = float(
                np.dot(ra["pooled_latent"], rb["pooled_latent"])
                / (np.linalg.norm(ra["pooled_latent"]) * np.linalg.norm(rb["pooled_latent"]) + 1e-8)
            )
            pa = np.array(ra["predicted_close_xyz"] if ra["predicted_close_xyz"] is not None else ra["last_step_xyz_fallback"])
            pb = np.array(rb["predicted_close_xyz"] if rb["predicted_close_xyz"] is not None else rb["last_step_xyz_fallback"])
            target_l2 = float(np.linalg.norm(pa - pb))
            traj_a = np.array(ra["first10_xyz"])
            traj_b = np.array(rb["first10_xyz"])
            traj_l2_mean = float(np.linalg.norm(traj_a - traj_b, axis=1).mean())
            same_nearest_card = ra["nearest_card_identity"] == rb["nearest_card_identity"]
            pair_type = (
                "card_vs_unrelated_control"
                if "__unrelated_control__" in (a, b)
                else "card_vs_card"
            )
            pair_stats.append(
                {
                    "instruction_a_card": a,
                    "instruction_b_card": b,
                    "pair_type": pair_type,
                    "pooled_latent_l2": latent_l2,
                    "pooled_latent_cosine_sim": latent_cos,
                    "predicted_target_xyz_l2": target_l2,
                    "first10_traj_mean_l2": traj_l2_mean,
                    "same_nearest_card_despite_different_instruction": same_nearest_card,
                }
            )

        scene_results.append(
            {
                "seed": seed,
                "target_entity_env_default": scene["target_name"],
                "cards_in_scene": {name: pos.tolist() for name, pos in cards.items()},
                "per_instruction": {
                    name: {k: v for k, v in res.items() if k != "pooled_latent"} for name, res in per_instruction.items()
                },
                "pooled_latent_norms": {name: float(np.linalg.norm(res["pooled_latent"])) for name, res in per_instruction.items()},
                "pairwise": pair_stats,
            }
        )

    # ---- aggregate across all scenes, split by pair type ----
    all_pairs = [p for s in scene_results for p in s["pairwise"]]
    card_pairs = [p for p in all_pairs if p["pair_type"] == "card_vs_card"]
    control_pairs = [p for p in all_pairs if p["pair_type"] == "card_vs_unrelated_control"]

    def agg(pairs: list[dict], key: str) -> dict | None:
        if not pairs:
            return None
        vals = [p[key] for p in pairs]
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}

    def same_card_rate(pairs: list[dict]) -> float | None:
        return float(np.mean([p["same_nearest_card_despite_different_instruction"] for p in pairs])) if pairs else None

    aggregate = {
        "n_scenes": len(scene_results),
        "card_vs_card_pairs": {
            "n": len(card_pairs),
            "pooled_latent_l2": agg(card_pairs, "pooled_latent_l2"),
            "pooled_latent_cosine_sim": agg(card_pairs, "pooled_latent_cosine_sim"),
            "predicted_target_xyz_l2": agg(card_pairs, "predicted_target_xyz_l2"),
            "first10_traj_mean_l2": agg(card_pairs, "first10_traj_mean_l2"),
            "frac_same_nearest_card_despite_different_instruction": same_card_rate(card_pairs),
        },
        "card_vs_unrelated_control_pairs": {
            "n": len(control_pairs),
            "description": "same image/state, instruction swapped for a COMPLETELY unrelated task (different task domain entirely) -- calibration baseline for how much latent/action CAN move under language change alone.",
            "pooled_latent_l2": agg(control_pairs, "pooled_latent_l2"),
            "pooled_latent_cosine_sim": agg(control_pairs, "pooled_latent_cosine_sim"),
            "predicted_target_xyz_l2": agg(control_pairs, "predicted_target_xyz_l2"),
            "first10_traj_mean_l2": agg(control_pairs, "first10_traj_mean_l2"),
        },
    }

    result = {
        "checkpoint": CHECKPOINT,
        "note": (
            "No temporal_decoder_grounded_grasp checkpoint exists on disk -- this audits the "
            "canonical temporal_decoder checkpoint's own decoder-chunk output as the "
            "'predicted target' signal (predicted_close_xyz: first step in its own 50-step output "
            "where the gripper crosses closed; falls back to the chunk's last xyz, flagged, if the "
            "chunk never closes)."
        ),
        "scenes": scene_results,
        "aggregate": aggregate,
    }
    (out_dir / "instruction_counterfactual_audit.json").write_text(json.dumps(result, indent=2))
    logger.info("=== aggregate ===")
    logger.info(json.dumps(aggregate, indent=2))
    logger.info("wrote %s", out_dir / "instruction_counterfactual_audit.json")


if __name__ == "__main__":
    main()
