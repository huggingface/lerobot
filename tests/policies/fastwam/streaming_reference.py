# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
"""Load pinned Streaming-WAM definitions for an optional, offline parity proof.

The original eager sampler and transformer definitions run unchanged. AST
selection avoids importing unrelated training, CUDA, and benchmark dependencies.
Frozen encoders are injected separately so reference/native runs share fixtures.
Streaming-WAM: Xuyao Huang et al.; copyright Shaohua Pan, Apache-2.0.
Revision ff36d6e66514afd65bf957a06a0a2410f182b0e3.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def load_reference(directory: Path) -> types.SimpleNamespace:
    """Read previously downloaded source files; make no network requests."""
    for name in ("streamingwam", "streamingwam.modules", "streamingwam.inference", "streamingwam.training"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    def load(name: str) -> types.ModuleType:
        path = directory / (name.replace(".", "_") + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    block = load("streamingwam.modules.wan_block")
    load("streamingwam.modules.mot")
    inference = load("streamingwam.inference.ac_stream")
    consistency = load("streamingwam.inference.consistency")
    streams = load("streamingwam.modules.ac_stream")
    action = load("streamingwam.modules.action_dit")
    load("streamingwam.modules.scheduler")
    flow = load("streamingwam.training.flow")

    def select(filename: str, names: set[str], namespace: dict[str, Any]) -> None:
        path = directory / filename
        parsed = ast.parse(path.read_text())
        nodes = [node for node in parsed.body if getattr(node, "name", None) in names]
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    video_namespace = {**vars(block), "torch": torch, "nn": nn}
    select("streamingwam_backbone_wan22.py", {"Wan22Head", "Wan22Dit"}, video_namespace)
    base_namespace = {"torch": torch, "Tensor": Tensor}
    parsed = ast.parse((directory / "streamingwam_wam_mot_wam.py").read_text())
    base = next(node for node in parsed.body if getattr(node, "name", None) == "MoTWAM")
    method = next(node for node in base.body if getattr(node, "name", None) == "_append_proprio_to_context")
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), "reference_proprio", "exec"), base_namespace)
    base_type = type("MoTWAM", (nn.Module,), {"_append_proprio_to_context": base_namespace[method.name]})
    wam_namespace = {
        **vars(inference),
        **vars(consistency),
        **vars(streams),
        "MoTWAM": base_type,
        "build_inference_schedule": flow.build_inference_schedule,
        "AC_STREAM_SLOT_ENCODER_NAME": "rtc_1step_selfatt_z1_slot_encoder",
        "FASTWAM_AC_STREAM_ARCHITECTURE": "fastwam_stage2_selfatt_z1",
        "STARWAM_RTC_ARCHITECTURE": "starwam_rtc_h32_s16_d8_z1_method3_v2",
    }
    select("streamingwam_wam_ac_stream_wam.py", {"ACStreamingWAM"}, wam_namespace)
    return types.SimpleNamespace(
        Video=video_namespace["Wan22Dit"],
        Action=action.ActionDiT,
        MoT=streams.ACStreamMoT,
        WAM=wam_namespace["ACStreamingWAM"],
        Slot=streams.ACStreamSlotEncoder,
        inference=inference,
        consistency=consistency,
    )


def reference_model(
    reference: types.SimpleNamespace, config: Any, payload: dict[str, Any], first_latents: Tensor
) -> nn.Module:
    """Install mmap release weights into the original reference modules on meta."""
    video_config = config.video_dit_config
    info = types.SimpleNamespace(
        **{
            key: video_config[key]
            for key in (
                "hidden_dim",
                "num_heads",
                "attn_head_dim",
                "num_layers",
                "freq_dim",
                "text_dim",
                "patch_size",
                "eps",
                "ffn_dim",
            )
        },
        in_channels=video_config["in_dim"],
    )
    action_config = {key: value for key, value in config.action_dit_config.items() if key != "fp32_attention"}
    # The pinned reference always includes text cross-attention. Upstream now
    # exposes that choice for Faster-WAM, but the release architecture is fixed.
    if not action_config.pop("use_text_cross_attention", True):
        raise ValueError("The reference requires action text cross-attention.")
    with torch.device("meta"):
        video = reference.Video(info).to(dtype=config.dtype)
        action = reference.Action(**action_config).to(dtype=config.dtype)
        video.rtc_1step_selfatt_z1_slot_encoder = reference.Slot(action.hidden_dim).to(dtype=config.dtype)
        proprio = nn.Linear(config.proprio_dim, video.text_dim).to(dtype=config.dtype)
    for name, expert in (("video", video), ("action", action)):
        prefix = f"mixtures.{name}."
        expert.load_state_dict(
            {key[len(prefix) :]: value for key, value in payload["mot"].items() if key.startswith(prefix)},
            strict=True,
            assign=True,
        )
    proprio.load_state_dict(payload["proprio_encoder"], strict=True, assign=True)
    model = reference.WAM.__new__(reference.WAM)
    nn.Module.__init__(model)
    model.action_expert = action
    model.mot = reference.MoT({"video": video, "action": action}, checkpoint_mixed_attn=False)
    model.proprio_encoder = proprio
    model.proprio_dim = config.proprio_dim
    scheduler = types.SimpleNamespace(num_train_timesteps=1000, infer_shift=5.0)
    model.video_scheduler = scheduler
    model.action_scheduler = scheduler
    model.config = types.SimpleNamespace(
        action_dim=config.action_dim, chunk_size=32, video_scheduler=scheduler, action_scheduler=scheduler
    )
    model.ac_stream_architecture = "fastwam_stage2_selfatt_z1"
    model.backbone = types.SimpleNamespace(
        get_dit=lambda: video,
        get_vae=lambda: types.SimpleNamespace(temporal_compress=4),
        encode_video=lambda image: first_latents,
    )
    model._ac_stream_acceleration = None
    return model.eval()
