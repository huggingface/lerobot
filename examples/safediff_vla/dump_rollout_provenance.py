#!/usr/bin/env python
"""Dump this machine's rollout provenance for the clean-v2 20k `temporal_decoder` checkpoint,
for the exact same task/seed as `eval_baseline_rollout.py`'s baseline condition (select_poker,
seed=1000), captured right after `env.reset()` and *before* any action is executed.

Read-only diagnostic: builds env + policy exactly like `eval_baseline_rollout.py`, resets once,
runs a single `plan_action_chunk` forward pass, and dumps everything -- never calls `env.step()`,
never modifies any model/processor code.

Usage:
    uv run python examples/safediff_vla/dump_rollout_provenance.py
"""

import argparse
import hashlib
import json
import logging
import platform
import socket
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open

from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import VLABenchEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_v2_baseline_20k/checkpoints/020000/pretrained_model"
TASK = "select_poker"
SEED = 1000
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_stats(safetensors_path: Path, feature: str) -> dict[str, list[float]] | None:
    if not safetensors_path.exists():
        return None
    with safe_open(str(safetensors_path), framework="pt") as f:
        keys = set(f.keys())
        mean_key, std_key = f"{feature}.mean", f"{feature}.std"
        if mean_key not in keys or std_key not in keys:
            return None
        return {
            "mean": f.get_tensor(mean_key).flatten().tolist(),
            "std": f.get_tensor(std_key).flatten().tolist(),
        }


def jsonable(obj):
    """Best-effort conversion of a policy config dataclass (enums, Path, etc.) to plain JSON."""
    return json.loads(json.dumps(obj, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-prefix", default=None, help="Defaults to outputs/eval/debug_machine_<hostname>")
    args = parser.parse_args()

    hostname = socket.gethostname()
    output_prefix = Path(args.output_prefix or f"outputs/eval/debug_machine_{hostname}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(CHECKPOINT).resolve()
    config_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "model.safetensors"
    preproc_stats_path = ckpt_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    postproc_stats_path = ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    logger.info("=== hashing checkpoint files ===")
    checkpoint_provenance = {
        "absolute_path": str(ckpt_dir),
        "config_json_sha256": sha256_file(config_path),
        "model_safetensors_sha256": sha256_file(weights_path),
    }
    logger.info("%s", checkpoint_provenance)

    import dm_control
    import importlib.metadata as importlib_metadata

    import mujoco
    import VLABench

    import lerobot

    try:
        dm_control_version = dm_control.__version__
    except AttributeError:
        dm_control_version = importlib_metadata.version("dm_control")

    environment_provenance = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "lerobot_source_path": str(Path(lerobot.__file__).parent),
        "vlabench_source_path": str(Path(VLABench.__file__).parent),
        "mujoco_version": mujoco.__version__,
        "dm_control_version": dm_control_version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": hostname,
        "platform": platform.platform(),
    }
    logger.info("=== environment ===")
    logger.info("%s", environment_provenance)

    logger.info("=== loading policy from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    policy.reset()

    policy_config = jsonable(asdict(policy.config))

    action_norm = dump_stats(preproc_stats_path, ACTION)
    state_norm = dump_stats(preproc_stats_path, OBS_STATE)
    action_unnorm = dump_stats(postproc_stats_path, ACTION)
    state_unnorm = dump_stats(postproc_stats_path, OBS_STATE)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    processor_provenance = {
        "checkpoint_dir": str(ckpt_dir),
        "preprocessor_config_json": str(ckpt_dir / "policy_preprocessor.json"),
        "postprocessor_config_json": str(ckpt_dir / "policy_postprocessor.json"),
        "normalizer_stats_safetensors": str(preproc_stats_path),
        "unnormalizer_stats_safetensors": str(postproc_stats_path),
    }

    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    try:
        logger.info("=== env.reset(seed=[%d]) -- no action executed yet ===", SEED)
        raw_observation, _ = env.reset(seed=[SEED])

        observation = preprocess_observation(raw_observation)
        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            observation["task"] = list(env.call("task"))
        observation = env_preprocessor(observation)
        batch = preprocessor(observation)

        camera_keys = {
            k: tuple(v.shape) for k, v in batch.items() if k.startswith("observation.images.")
        }
        current_state = batch[OBS_STATE].detach().cpu().flatten().tolist()

        with torch.no_grad():
            pred_actions_norm, _ = policy.plan_action_chunk(batch)
            postprocessed = postprocessor(pred_actions_norm)
            action_transition = env_postprocessor({ACTION: postprocessed})
            final_env_input_action = action_transition[ACTION]

        raw_first10 = pred_actions_norm[0, :10].detach().cpu().tolist()
        final_first10 = final_env_input_action[0, :10].detach().cpu().tolist()
    finally:
        env.close()

    report = {
        "task": TASK,
        "seed": SEED,
        "checkpoint": checkpoint_provenance,
        "environment": environment_provenance,
        "policy_config": policy_config,
        "processors": processor_provenance,
        "action_normalization": {
            "preprocessor_normalizer_mean_std": action_norm,
            "postprocessor_unnormalizer_mean_std": action_unnorm,
        },
        "state_normalization": {
            "preprocessor_normalizer_mean_std": state_norm,
            "postprocessor_unnormalizer_mean_std": state_unnorm,
        },
        "camera_keys_and_shapes": camera_keys,
        "current_state_at_reset": current_state,
        "model_raw_output_first_10_steps_normalized_space": raw_first10,
        "postprocessed_final_env_input_action_first_10_steps": final_first10,
        "rotation_dims_3_6": {
            "model_raw_output": [step[3:6] for step in raw_first10],
            "postprocessed_final_env_input": [step[3:6] for step in final_first10],
        },
    }

    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    logger.info("=== wrote %s ===", json_path)

    txt_path = output_prefix.with_suffix(".txt")
    lines = [f"=== rollout provenance: {TASK} seed={SEED} on {hostname} ===", ""]

    def add_section(title: str, d: dict) -> None:
        lines.append(f"--- {title} ---")
        lines.append(json.dumps(d, indent=2))
        lines.append("")

    add_section("checkpoint", checkpoint_provenance)
    add_section("environment", environment_provenance)
    add_section("processors", processor_provenance)
    add_section("action_normalization", report["action_normalization"])
    add_section("state_normalization", report["state_normalization"])
    add_section("camera_keys_and_shapes", camera_keys)
    lines.append("--- current_state_at_reset ---")
    lines.append(str(current_state))
    lines.append("")
    lines.append("--- model_raw_output_first_10_steps (normalized space) ---")
    for i, step in enumerate(raw_first10):
        lines.append(f"  t={i}: {step}")
    lines.append("")
    lines.append("--- postprocessed_final_env_input_action_first_10_steps ---")
    for i, step in enumerate(final_first10):
        lines.append(f"  t={i}: {step}")
    lines.append("")
    lines.append("--- rotation dims [3:6] only ---")
    lines.append("  raw model output:")
    for i, step in enumerate(raw_first10):
        lines.append(f"    t={i}: {step[3:6]}")
    lines.append("  postprocessed final env-input:")
    for i, step in enumerate(final_first10):
        lines.append(f"    t={i}: {step[3:6]}")
    lines.append("")
    lines.append("--- policy_config ---")
    lines.append(json.dumps(policy_config, indent=2))

    txt_path.write_text("\n".join(lines))
    logger.info("=== wrote %s ===", txt_path)


if __name__ == "__main__":
    main()
