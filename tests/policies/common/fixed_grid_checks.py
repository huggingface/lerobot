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
"""Checks shared by the fixed-grid Qwen-VL vision tests.

Their exported programs must not depend on tensor values for any shape.
"""

import io

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols


def save_and_load(program: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    buffer = io.BytesIO()
    torch.export.save(program, buffer)
    buffer.seek(0)
    return torch.export.load(buffer)


def export_prediction(policy, batch, noise, grid):
    keys = sorted(key for key, value in batch.items() if isinstance(value, torch.Tensor))

    class PredictActionChunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # A submodule: the program holds its buffers, and export warns if tracing sets a tensor attribute.
            self.policy = policy

        def forward(self, noise, *tensors):
            inputs = {**batch, **dict(zip(keys, tensors, strict=True))}
            return self.policy.predict_action_chunk(inputs, noise=noise, vision_grid=grid)

    # With grad enabled, export wraps the policy's `no_grad` region, which `torch.export.load` refuses.
    with torch.no_grad():
        program = torch.export.export(
            PredictActionChunk(), (noise.clone(), *(batch[key] for key in keys)), strict=False
        )

    def run(inputs, noise=noise):
        return program.module()(noise.clone(), *(inputs[key] for key in keys))

    return program, run


def data_dependent_nodes(program: torch.export.ExportedProgram) -> list[str]:
    """Nodes that depend on tensor values for a shape: `nonzero`, `masked_select`, boolean-mask
    indexing and other unbacked sizes.

    Runtime asserts (`check_input`) read values too, but only to raise, so they are not listed.
    """
    found = []
    # Autocast and grad-mode switches inside the exported function become submodules with their own graphs.
    nodes = [
        node
        for module in program.graph_module.modules()
        if hasattr(module, "graph")
        for node in module.graph.nodes
    ]
    for node in nodes:
        if node.op != "call_function":
            continue
        name = str(node.target)
        indices = node.args[1] if name.startswith(("aten.index.", "aten.index_put")) else []
        values = [getattr(index, "meta", {}).get("val") for index in indices]
        if any(isinstance(v, torch.Tensor) and v.dtype == torch.bool for v in values):
            found.append(f"{name} with a boolean mask")
        elif "nonzero" in name or "masked_select" in name:
            found.append(name)
        else:
            value = node.meta.get("val")
            for tensor in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(tensor, torch.Tensor) and free_unbacked_symbols(tensor.shape):
                    found.append(f"{name} with a data-dependent shape")
    return found
