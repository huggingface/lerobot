# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: The DiffSynth-Studio contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/models/wan22/helpers/state_dict_converters.py
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Source: https://github.com/modelscope/DiffSynth-Studio @ 974cfa37f27ac55eba3b6d10efa21f876900572d :: diffsynth/utils/state_dict_converters/wan_video_dit.py
# Source: https://github.com/modelscope/DiffSynth-Studio @ 974cfa37f27ac55eba3b6d10efa21f876900572d :: diffsynth/utils/state_dict_converters/wan_video_vae.py
# Changes: Nested DiffSynth-Studio/Wan implementations incorporated through FastWAM retain their Apache-2.0 terms.
# End Long-WAM attribution.


def wan_video_vae_state_dict_converter(state_dict):
    converted = {}
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    for name, value in state_dict.items():
        converted[f"model.{name}"] = value
    return converted


def wan_video_dit_from_diffusers(state_dict):
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.attn2.add_k_proj.bias": "blocks.0.cross_attn.k_img.bias",
        "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
        "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.v_img.bias",
        "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
        "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
        "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
        "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
        "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
        "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
        "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
        "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
        "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }
    converted = {}
    for name, value in state_dict.items():
        if name in rename_dict:
            converted[rename_dict[name]] = value
        else:
            probe_name = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
            if probe_name in rename_dict:
                mapped = rename_dict[probe_name]
                mapped = ".".join(mapped.split(".")[:1] + [name.split(".")[1]] + mapped.split(".")[2:])
                converted[mapped] = value
    return converted


def wan_video_dit_state_dict_converter(state_dict):
    converted = {}
    for name, value in state_dict.items():
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]:
            continue
        key = name[6:] if name.startswith("model.") else name
        converted[key] = value
    return converted
