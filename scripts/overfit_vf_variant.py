"""Overfit any distributional VF architecture on a small real-data batch."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.rewards.distributional_value_function.configuration_distributional_value_function import (
    DistributionalVFConfig,
)
from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
    IMAGE_MASK_SUFFIX,
)
from lerobot.rewards.factory import make_reward_model, make_reward_pre_post_processors
from lerobot.rewards.nanovlm_value_function.configuration_nanovlm_value_function import (
    NanoVLMVFConfig,
)
from lerobot.rewards.temporal_siglip_value_function.configuration_temporal_siglip_value_function import (
    TemporalSiglipVFConfig,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--reward_type",
        choices=(
            "distributional_value_function",
            "temporal_siglip_value_function",
            "nanovlm_value_function",
        ),
        required=True,
    )
    parser.add_argument("--vlm_pretrained_path", default=None)
    parser.add_argument("--nanovlm_pretrained_path", default="lusxvr/nanoVLM-460M-8k")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--history_steps", type=int, default=6)
    parser.add_argument("--history_frame_gap", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.root)
    input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(metadata.features[key]["shape"]))
        for key in metadata.camera_keys
    }
    if OBS_STATE in metadata.features:
        input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=tuple(metadata.features[OBS_STATE]["shape"]),
        )
    common = {"input_features": input_features, "device": str(device), "target_method": "dirac_delta"}
    if args.reward_type == "distributional_value_function":
        config = DistributionalVFConfig(
            **common,
            vlm_pretrained_path=args.vlm_pretrained_path,
            freeze_vision_encoder=True,
        )
    elif args.reward_type == "temporal_siglip_value_function":
        config = TemporalSiglipVFConfig(
            **common,
            history_steps=args.history_steps,
            frame_gap=args.history_frame_gap,
        )
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
        }
    else:
        config = NanoVLMVFConfig(
            **common,
            nanovlm_pretrained_path=args.nanovlm_pretrained_path,
        )

    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    indices = np.linspace(0, len(dataset) - 1, args.num_samples, dtype=int).tolist()
    preprocessor, _ = make_reward_pre_post_processors(
        config,
        dataset_stats=metadata.stats,
    )

    processed_samples = []
    returns = []
    terminals = []
    for index in indices:
        sample = dataset[index]
        returns.append(torch.as_tensor(sample["mc_return"]).reshape(-1)[0])
        terminals.append(torch.as_tensor(sample["is_terminal"]).reshape(-1)[0])
        processed_samples.append(preprocessor(sample))
    batch = _collate_processed(processed_samples)
    batch["mc_return"] = torch.stack(returns).to(device)
    batch["is_terminal"] = torch.stack(terminals).bool().to(device)

    model = make_reward_model(config).to(device)
    _run_stage(
        model,
        batch,
        steps=args.steps // 2,
        learning_rate=args.lr_head,
        head_only=True,
        log_every=args.log_every,
        label="head probe",
    )
    _run_stage(
        model,
        batch,
        steps=args.steps - args.steps // 2,
        learning_rate=args.lr_backbone,
        head_only=False,
        log_every=args.log_every,
        label="fine-tune",
    )
    _image_shuffle_diagnostic(model, batch, metadata.camera_keys)


def _collate_processed(samples):
    keys = {
        *[key for key in samples[0] if key.startswith("observation.images.")],
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_ATTENTION_MASK,
    }
    if OBS_STATE in samples[0]:
        keys.add(OBS_STATE)
    for key in list(keys):
        if key.startswith("observation.images.") and not key.endswith(IMAGE_MASK_SUFFIX):
            keys.add(key + IMAGE_MASK_SUFFIX)
    return {key: torch.cat([sample[key] for sample in samples], dim=0) for key in keys}


def _set_trainable(model, *, head_only: bool):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.value_head.parameters():
        param.requires_grad = True
    if hasattr(model, "value_query"):
        for param in model.value_query.parameters():
            param.requires_grad = True
    if head_only:
        return

    if hasattr(model, "multi_modal_projector"):
        model.multi_modal_projector.requires_grad_(True)
        model.language_model.requires_grad_(True)
    elif hasattr(model, "temporal_transformer"):
        for name, param in model.named_parameters():
            if not name.startswith("siglip."):
                param.requires_grad = True
    else:
        model.nanovlm.MP.requires_grad_(True)
        model.nanovlm.decoder.requires_grad_(True)


def _run_stage(model, batch, *, steps, learning_rate, head_only, log_every, label):
    _set_trainable(model, head_only=head_only)
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    print(f"\n{label}: {sum(param.numel() for param in params):,} trainable parameters")
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model(batch)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps:
            print(
                f"step={step:04d} loss={metrics['loss']:.4f} "
                f"mae={metrics['mae']:.4f} acc={metrics['acc_neighbor']:.3f}"
            )


@torch.no_grad()
def _image_shuffle_diagnostic(model, batch, camera_keys):
    model.eval()
    matched_loss, _ = model(batch)
    shuffled = dict(batch)
    permutation = torch.roll(torch.arange(batch["mc_return"].shape[0], device=matched_loss.device), 1)
    for key in camera_keys:
        shuffled[key] = batch[key][permutation]
    shuffled_loss, _ = model(shuffled)
    print(
        f"\nvisual dependence: matched_loss={matched_loss.item():.4f} "
        f"shuffled_loss={shuffled_loss.item():.4f} "
        f"gap={shuffled_loss.item() - matched_loss.item():+.4f}"
    )


if __name__ == "__main__":
    main()
