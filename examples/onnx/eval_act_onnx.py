#!/usr/bin/env python
"""Evaluate an ACT policy in sim with either the PyTorch or ONNX network.

The ONNX backend swaps only ``policy.model`` (ResNet + transformer + action head)
with an onnxruntime session. Everything else - the LeRobot processor pipeline
(normalization), the action queue, and the gym env - is identical, so any
difference in success rate is attributable to the network backend alone.

Run both backends with the same seed to compare:

    python examples/onnx/eval_act_onnx.py \
        --policy-path=lerobot/act_aloha_sim_transfer_cube_human \
        --task=AlohaTransferCube-v0 \
        --backend=torch --n-episodes=50 --batch-size=10 --device=cuda

    python examples/onnx/eval_act_onnx.py \
        --policy-path=lerobot/act_aloha_sim_transfer_cube_human \
        --task=AlohaTransferCube-v0 \
        --onnx=outputs/onnx/act_transfer_cube.onnx \
        --backend=onnx --n-episodes=50 --batch-size=10 --device=cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.random_utils import set_seed


class ONNXACTModel(nn.Module):
    """Drop-in replacement for ``ACTPolicy.model`` backed by onnxruntime."""

    def __init__(
        self, onnx_path: str, image_keys: list[str], has_state: bool, has_env_state: bool, device: str
    ):
        super().__init__()
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if str(device).startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.image_keys = image_keys
        self.has_state = has_state
        self.has_env_state = has_env_state
        print(f"[onnx] providers in use: {self.sess.get_providers()}")

    def forward(self, batch: dict):
        state = batch[OBS_STATE] if self.has_state else batch[OBS_ENV_STATE]
        ref = state
        ort_inputs = {"state": state.detach().cpu().numpy().astype(np.float32)}
        images = batch[OBS_IMAGES]
        for i, img in enumerate(images):
            ort_inputs[f"image_{i}"] = img.detach().cpu().numpy().astype(np.float32)
        out = self.sess.run(None, ort_inputs)[0]
        actions = torch.from_numpy(out).to(ref.device, dtype=ref.dtype)
        return actions, None


def load_stats_from_checkpoint(policy_path: str, input_features, output_features) -> dict:
    """Recover MEAN_STD stats baked into a legacy ACT checkpoint's safetensors buffers.

    Legacy checkpoints store normalization as buffers like
    ``normalize_inputs.buffer_observation_state.{mean,std}``. We map those back to
    feature names so we can rebuild the processor pipeline without the dataset.
    """
    from safetensors.torch import load_file

    p = Path(policy_path)
    if p.is_dir():
        st_path = p / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        st_path = Path(hf_hub_download(policy_path, "model.safetensors"))

    sd = load_file(str(st_path))
    stats: dict = {}
    for feat in list(input_features) + list(output_features):
        buf = "buffer_" + feat.replace(".", "_")
        for prefix in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
            mkey, skey = f"{prefix}.{buf}.mean", f"{prefix}.{buf}.std"
            if mkey in sd and skey in sd:
                stats[feat] = {"mean": sd[mkey].numpy(), "std": sd[skey].numpy()}
                break
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--task", required=True, help="e.g. AlohaTransferCube-v0")
    parser.add_argument("--env-type", default="aloha")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--onnx", default=None, help="Path to .onnx (required for --backend=onnx)")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    if args.backend == "onnx" and not args.onnx:
        raise SystemExit("--backend=onnx requires --onnx=<path>")

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    set_seed(args.seed)

    print(f"[1/4] Loading ACT policy from '{args.policy_path}'...")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.config.device = device
    policy.eval()
    policy.to(device)
    cfg = policy.config

    if args.backend == "onnx":
        image_keys = list(cfg.image_features)
        has_state = cfg.robot_state_feature is not None
        has_env_state = cfg.env_state_feature is not None
        print(f"[2/4] Swapping policy.model with ONNX backend ({args.onnx})")
        policy.model = ONNXACTModel(args.onnx, image_keys, has_state, has_env_state, device)
        policy.to(device)
    else:
        print("[2/4] Using PyTorch backend")

    print("[3/4] Building processors and environment...")
    stats = load_stats_from_checkpoint(args.policy_path, cfg.input_features, cfg.output_features)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    env_cfg = make_env_config(args.env_type, task=args.task)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    env_groups = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False)
    # make_env returns {task_group: {idx: VectorEnv}}; grab the single env.
    first_group = next(iter(env_groups.values()))
    env = next(iter(first_group.values()))

    print(f"[4/4] Evaluating backend='{args.backend}' for {args.n_episodes} episodes (seed={args.seed})...")
    with torch.no_grad():
        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            start_seed=args.seed,
        )

    agg = info["aggregated"]
    print("\n==== RESULT ====")
    print(f"backend       : {args.backend}")
    print(f"task          : {args.task}")
    print(f"n_episodes    : {args.n_episodes}")
    print(f"pc_success    : {agg['pc_success']:.1f}%")
    print(f"avg_max_reward: {agg['avg_max_reward']:.4f}")
    print(f"eval_ep_s     : {agg['eval_ep_s']:.2f}s")

    env.close()


if __name__ == "__main__":
    main()
