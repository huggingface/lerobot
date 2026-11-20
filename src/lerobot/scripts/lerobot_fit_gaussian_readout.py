#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fit an opt-in Gaussian readout from a frozen checkpoint and offline observations.

Run ``python -m lerobot.scripts.lerobot_fit_gaussian_readout --help``.
This command samples the teacher; it does not interact with a robot or simulator.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common import gaussian_readout as gar
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import AbsoluteActionsProcessorStep, DeviceProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.import_utils import _datasets_available, require_package

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _action_affine(postprocessor: Any, action_dim: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Derive the affine inverse from the actual postprocessor, including quantile modes."""
    for step in postprocessor.steps:
        if isinstance(step, AbsoluteActionsProcessorStep) and not step.enabled:
            continue
        if not isinstance(step, (UnnormalizerProcessorStep, DeviceProcessorStep)):
            raise ValueError(
                "fitting requires a standard affine action postprocessor without relative actions"
            )
    zero = torch.zeros(1, 1, action_dim, device=device)
    mean = postprocessor(zero).detach().float().cpu().reshape(action_dim)
    scale = postprocessor(torch.ones_like(zero)).detach().float().cpu().reshape(action_dim) - mean
    for value in (-1.0, 2.0):
        decoded = postprocessor(torch.full_like(zero, value)).detach().float().cpu().reshape(action_dim)
        if not torch.allclose(decoded, mean + value * scale, rtol=1e-5, atol=1e-6):
            raise ValueError("action postprocessor must be affine")
    return mean, scale


def _collect_targets(
    policy: Any,
    preprocessor: Any,
    contexts: Iterable[dict[str, Any]],
    draws: int,
    seed: int,
    mean: Tensor,
    scale: Tensor,
    threshold: float,
    close_command: float,
) -> dict[str, Tensor]:
    model, config = policy.model, policy.config
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probe = gar.make_epsilon_star(config.chunk_size, config.max_action_dim).to(device)
    rows: dict[str, list[Tensor]] = {key: [] for key in ("features", "raw", "mean", "variance", "frequency")}
    for index, observation in enumerate(contexts):
        captured: list[Tensor] = []

        def capture(module: nn.Module, args: tuple[Tensor, ...], captured: list[Tensor] = captured) -> None:
            captured.append(args[0].detach().float().cpu().clone())

        hook = model.action_out_proj.register_forward_pre_hook(capture)
        previous_steps = config.num_inference_steps
        try:
            config.num_inference_steps = 1
            policy.reset()
            with torch.no_grad():
                raw = policy.predict_action_chunk(preprocessor(dict(observation)), noise=probe).float().cpu()
        finally:
            config.num_inference_steps = previous_steps
            hook.remove()
        if len(captured) != 1 or raw.shape[:2] != (1, config.chunk_size):
            raise ValueError("fitting requires one observation per context and one full action chunk")
        features = captured[0]
        expected_raw = gar.flow_raw_action(
            probe.cpu(), gar.project_velocity(model, features.to(device)).cpu()
        )
        if not torch.allclose(raw, expected_raw[..., : raw.shape[-1]], rtol=1e-3, atol=1e-3):
            raise ValueError("public one-step actions do not match the supported readout convention")
        samples = []
        for _ in range(draws):
            noise = torch.randn(probe.shape, generator=generator).to(device)
            policy.reset()
            with torch.no_grad():
                sample = policy.predict_action_chunk(preprocessor(dict(observation)), noise=noise)
            samples.append(sample.detach().float().cpu())
        samples = torch.stack(samples)
        gripper = samples[..., -1] * scale[-1] + mean[-1]
        closed = gripper >= threshold if close_command > threshold else gripper <= threshold
        rows["features"].append(features)
        rows["raw"].append(raw[..., :-1])
        rows["mean"].append(samples[..., :-1].mean(0))
        rows["variance"].append(samples[..., :-1].var(0, unbiased=False).clamp_min(gar.VARIANCE_FLOOR))
        rows["frequency"].append(closed.float().mean(0))
        logger.info("Collected context %s", index + 1)
    if not rows["features"]:
        raise ValueError("fitting and tuning each require at least one context")
    return {key: torch.cat(values) for key, values in rows.items()}


def fit_readout(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    fit_contexts: Iterable[dict[str, Any]],
    tune_contexts: Iterable[dict[str, Any]],
    output: Path,
    *,
    draws: int = 32,
    updates: int = 2000,
    batch_size: int = 32,
    seed: int = 17,
    open_command: float = -1.0,
    close_command: float = 1.0,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Collect teacher moments and fit a readout. Context iterables must be disjoint.

    Supports single-observation, affine action processing and one final binary gripper.
    Fitting uses the configured full-chunk teacher, not demonstration actions.
    """
    config, model = policy.config, policy.model
    if config.type not in {"pi0", "pi05", "smolvla", "fineart_vla"}:
        raise ValueError("unsupported Gaussian readout policy")
    if config.gaussian_readout_path is not None or config.n_obs_steps != 1:
        raise ValueError("fit an Euler teacher with n_obs_steps=1 and no existing readout")
    if getattr(config, "adapt_to_pi_aloha", False) or getattr(config, "use_relative_actions", False):
        raise ValueError("fitting does not support ALOHA adaptation or relative actions")
    if draws < 2 or updates < 1 or batch_size < 1:
        raise ValueError("draws must be at least two; updates and batch size must be positive")
    gar._reject_unsupported(model, None, None, {})
    policy.eval()
    gar.prepare_gaussian_readout_teacher(policy)
    before = gar.state_dict_sha256(policy)
    digest = gar.processor_sha256(preprocessor, postprocessor)
    device = next(model.parameters()).device
    action_dim = config.action_feature.shape[0]
    mean, scale = _action_affine(postprocessor, action_dim, device)
    # Initialize on CPU inside a local RNG scope; leave all caller generators intact.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        readout = gar.GaussianReadout(
            model.action_out_proj.in_features,
            chunk_size=config.chunk_size,
            motion_dim=action_dim - 1,
            gripper_index=action_dim - 1,
            max_action_dim=config.max_action_dim,
            action_dim=action_dim,
            open_normalized=float(gar.normalized_command(open_command, mean[-1], scale[-1])),
            close_normalized=float(gar.normalized_command(close_command, mean[-1], scale[-1])),
            action_mean=mean,
            action_std=scale,
            gripper_threshold=threshold,
            open_command=open_command,
            close_command=close_command,
        )
    fit = _collect_targets(
        policy, preprocessor, fit_contexts, draws, seed, mean, scale, threshold, close_command
    )
    tune = _collect_targets(
        policy, preprocessor, tune_contexts, draws, seed + 1, mean, scale, threshold, close_command
    )
    weight, bias = gar.solve_affine_residual(fit["features"], fit["raw"], fit["mean"])
    with torch.no_grad():
        readout.affine_weight.copy_(weight)
        readout.affine_bias.copy_(bias)
    variance_update = gar.fit_variance_head(
        readout,
        fit["features"],
        fit["variance"],
        tune["features"],
        tune["variance"],
        updates=updates,
        batch_size=batch_size,
        seed=seed,
    )
    gripper_update = gar.fit_gripper_head(
        readout,
        fit["features"],
        fit["frequency"],
        tune["features"],
        tune["frequency"],
        updates=updates,
        batch_size=batch_size,
        seed=seed,
    )
    if gar.state_dict_sha256(policy) != before or gar.processor_sha256(preprocessor, postprocessor) != digest:
        raise ValueError("teacher or processors changed during fitting")
    gar.save_readout(readout, output, teacher_sha256=before, processor_sha256=digest, teacher=policy)
    with torch.no_grad():
        prediction = readout.continuous_mean(tune["features"], tune["raw"])
        result = {
            "label": "Offline",
            "teacher_sha256": before,
            "processor_sha256": digest,
            "teacher_steps": config.num_inference_steps,
            "draws_per_context": draws,
            "fit_contexts": len(fit["features"]),
            "tune_contexts": len(tune["features"]),
            "variance_selected_update": variance_update,
            "gripper_selected_update": gripper_update,
            "tune_motion_mean_mse": float((prediction - tune["mean"]).square().mean()),
            "one_step_tune_motion_mean_mse": float((tune["raw"] - tune["mean"]).square().mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="checkpoint directory or Hub repository")
    parser.add_argument("--policy-revision", help="pin a Hub commit")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--fit-episodes", required=True, type=int, nargs="+")
    parser.add_argument("--tune-episodes", required=True, type=int, nargs="+")
    parser.add_argument("--fit-contexts", type=int, default=64)
    parser.add_argument("--tune-contexts", type=int, default=16)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--open-command", type=float, default=-1)
    parser.add_argument("--close-command", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if set(args.fit_episodes) & set(args.tune_episodes):
        parser.error("fitting and tuning episodes must be disjoint")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("output artifact or report already exists")
    require_package("datasets", extra="dataset")
    logging.basicConfig(level=logging.INFO)
    config = PreTrainedConfig.from_pretrained(args.policy, revision=args.policy_revision)
    config.device, config.gaussian_readout_path = args.device, None
    policy = get_policy_class(config.type).from_pretrained(
        args.policy, config=config, revision=args.policy_revision
    )
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=args.policy,
        pretrained_revision=args.policy_revision,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    def contexts(episodes: list[int], count: int) -> Iterable[dict[str, Any]]:
        dataset = LeRobotDataset(
            args.dataset, root=args.dataset_root, episodes=episodes, revision=args.dataset_revision
        )
        if count < 1 or count > len(dataset):
            raise ValueError("context count must be positive and no larger than the selected dataset")
        generator = torch.Generator().manual_seed(args.seed)
        for index in torch.randperm(len(dataset), generator=generator)[:count].tolist():
            row = dataset[index]
            yield {
                key: value for key, value in row.items() if key.startswith("observation.") or key == "task"
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = fit_readout(
        policy,
        pre,
        post,
        contexts(args.fit_episodes, args.fit_contexts),
        contexts(args.tune_episodes, args.tune_contexts),
        args.output,
        draws=args.draws,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed,
        open_command=args.open_command,
        close_command=args.close_command,
        threshold=args.threshold,
    )
    report["configuration"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
