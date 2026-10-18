# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: The DiffSynth-Studio contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/models/wan22/helpers/gradient.py
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Source: https://github.com/modelscope/DiffSynth-Studio @ 974cfa37f27ac55eba3b6d10efa21f876900572d :: diffsynth/core/gradient/gradient_checkpoint.py
# Changes: Nested DiffSynth-Studio/Wan implementations incorporated through FastWAM retain their Apache-2.0 terms.
# End Long-WAM attribution.

import torch


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)

    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
