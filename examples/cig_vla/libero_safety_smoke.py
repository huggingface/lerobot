#!/usr/bin/env python
"""Real LIBERO-Safety batch smoke for CIG-VLA; optionally loads actual Qwen3-VL 2B."""

import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.adapters.libero_safety_v21 import LiberoSafetyV21Dataset
from lerobot.policies.cig_vla.configuration_cig_vla import CIGVLAConfig
from lerobot.policies.cig_vla.modeling_interaction_cig_vla import CIGVLAPolicy
from lerobot.policies.cig_vla.processor_cig_vla import make_cig_vla_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


def build_config(device, chunk_size, actual_qwen, model_path=None, local_response=False):
    return CIGVLAConfig(
        device=device,
        qwen_model_name=model_path or "Qwen/Qwen3-VL-2B-Instruct",
        input_features={
            "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        grounding_hidden_dim=128 if actual_qwen else 16,
        grounding_num_heads=4,
        grounding_num_layers=1,
        controller_hidden_dim=128 if actual_qwen else 16,
        controller_num_layers=2 if actual_qwen else 1,
        controller_num_heads=4,
        num_inference_steps=2,
        lora_rank=4,
        lora_alpha=8,
        detach_bottleneck_for_main_action=True,
        enable_causal_intervention=local_response,
        causal_loss_weight=0.01,
    )


def load_batch(dataset, config, device):
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=lerobot_collate_fn)))
    for key in dataset.meta.camera_keys:
        batch[key] = batch[key].float() / 255.0
    preprocessor, _ = make_cig_vla_pre_post_processors(config, dataset.meta.stats)
    return preprocessor(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-qwen", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--local-response", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    from mock_components import MockBackbone

    device = "cuda" if args.actual_qwen else "cpu"
    dataset = LiberoSafetyV21Dataset(episodes=[0], chunk_size=args.chunk_size)
    config = build_config(device, args.chunk_size, args.actual_qwen, args.model_path, args.local_response)
    backbone = None if args.actual_qwen else MockBackbone()
    policy = CIGVLAPolicy(config, backbone=backbone, dataset_stats=dataset.meta.stats).to(device)
    batch = load_batch(dataset, config, device)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)

    if args.actual_qwen:
        torch.cuda.reset_peak_memory_stats()
    initial = {
        name: value.detach().clone() for name, value in policy.named_parameters() if value.requires_grad
    }
    history = []
    gradient_norm = 0.0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        gradients = [
            value.grad for value in policy.parameters() if value.requires_grad and value.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"missing or non-finite gradients at step {step}")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0))
        optimizer.step()
        history.append({key: float(value.detach()) for key, value in metrics.items()})

    parameter_update = any(
        not torch.equal(initial[name], value.detach())
        for name, value in policy.named_parameters()
        if name in initial
    )
    if not parameter_update:
        raise RuntimeError("no trainable parameter changed")

    output = Path(args.output_dir or tempfile.mkdtemp(prefix="cig_libero_safety_smoke_"))
    policy.save_pretrained(output)
    policy.eval()
    torch.manual_seed(123)
    fixed_prediction = policy.predict_action_chunk(batch).detach().cpu()
    diagnostics = {
        "steps": args.steps,
        "actual_qwen": args.actual_qwen,
        "local_response": args.local_response,
        "losses": history,
        "gradient_norm": gradient_norm,
        "parameter_update": parameter_update,
        "action_chunk_shape": list(batch["action"].shape),
        "bottleneck_shape": [1, 13],
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in policy.parameters()),
        "checkpoint": str(output),
    }
    if args.actual_qwen:
        diagnostics.update(policy.backbone.last_input_diagnostics)
        diagnostics["qwen_hidden_shape"] = list(policy.backbone.last_hidden_shape)
        diagnostics["lora_parameters"] = sum(
            p.numel() for name, p in policy.named_parameters() if "lora_" in name and p.requires_grad
        )
        diagnostics["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        diagnostics["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()

    del optimizer, policy
    gc.collect()
    if args.actual_qwen:
        torch.cuda.empty_cache()
    if args.actual_qwen:
        reloaded = CIGVLAPolicy.from_pretrained(output, strict=True).to(device)
    else:
        from unittest.mock import patch

        with patch(
            "lerobot.policies.cig_vla.modeling_interaction_cig_vla.Qwen3VLGroundingBackbone",
            MockBackbone,
        ):
            reloaded = CIGVLAPolicy.from_pretrained(output, strict=True).to(device)
    reloaded.eval()
    torch.manual_seed(123)
    reloaded_prediction = reloaded.predict_action_chunk(batch).detach().cpu()
    diagnostics["reload_finite"] = bool(torch.isfinite(reloaded_prediction).all())
    diagnostics["fixed_inference_shape_match"] = fixed_prediction.shape == reloaded_prediction.shape
    diagnostics["fixed_inference_reload_match"] = bool(
        torch.allclose(fixed_prediction, reloaded_prediction, atol=1e-5, rtol=1e-4)
    )
    (output / "smoke_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
