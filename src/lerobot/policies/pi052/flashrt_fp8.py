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

"""Optional FlashRT FP8 MLP kernels for PI052.

The opt-in swap calibrates once on a real observation and falls back to BF16 when unavailable.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0


def _roundtrip_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize->dequantize an activation through FP8 E4M3 at ``scale`` (f32)."""
    q = torch.clamp(x.float() / scale.float(), -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.float() * scale.float()


_SWIGLU_REPO = "flashrt/flashrt-fp8-swiglu-ffn"
_GELU_REPO = "flashrt/flashrt-fp8-ffn"
_GEMM_REPO = "flashrt/flashrt-gemm-epilogues"


def _get_kernel(repo: str):
    """Load a FlashRT Hub package (cached). Returns None if unavailable."""
    from kernels import get_kernel

    return get_kernel(repo, version=1)


def _quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(weight.detach().float().abs().max().item(), 1e-12) / _FP8_MAX
    fp8 = torch.clamp(weight.float() / scale, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return fp8.contiguous(), torch.tensor([scale], dtype=torch.float32)


def _static_scale(amax: float, safety: float) -> torch.Tensor:
    return torch.tensor([max(amax, 1e-12) / _FP8_MAX * safety], dtype=torch.float32)


class _FlashRTGeGLU(nn.Module):
    """FP8 drop-in for a Gemma GeGLU MLP (gate/up/down, gelu_pytorch_tanh, no bias)."""

    def __init__(self, mlp, in_amax, hid_amax, ffn_ops, quant_ops, safety, fuse_weight=None):
        super().__init__()
        self.ffn_ops = ffn_ops
        self.quant_ops = quant_ops
        self.in_features = mlp.gate_proj.weight.shape[1]
        device = mlp.gate_proj.weight.device
        gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).float()
        # Fold fixed RMSNorm weights into the GEMM to avoid FP8 activation outliers.
        # Adaptive RMSNorm instead uses an identity channel scale.
        if fuse_weight is not None:
            f = 1.0 + fuse_weight.detach().float()
            gate_up = gate_up * f[None, :]
            channel_scale = (1.0 / f).to(torch.bfloat16)
        else:
            channel_scale = torch.ones(self.in_features, dtype=torch.bfloat16)
        gate_up_fp8, gate_up_scale = _quantize_fp8(gate_up)
        down_fp8, down_scale = _quantize_fp8(mlp.down_proj.weight)
        self.register_buffer("gate_up_fp8", gate_up_fp8.to(device))
        self.register_buffer("down_fp8", down_fp8.to(device))
        self.register_buffer("gate_up_scale", gate_up_scale.to(device))
        self.register_buffer("down_scale", down_scale.to(device))
        self.register_buffer("input_scale", _static_scale(in_amax, safety).to(device))
        self.register_buffer("hidden_scale", _static_scale(hid_amax, safety).to(device))
        self.register_buffer("channel_scale", channel_scale.to(device))
        self.safety = safety
        self.calibrating = False
        self._ia = 0.0
        self._ha = 0.0

    def _calibrate_step(self, x):
        # FP8-propagated calibration (FlashRT contract): measure input/hidden amax
        # on the live (already-FP8-upstream) activation, running-max across steps.
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        xq = flat.float() * self.channel_scale.float()
        self._ia = max(self._ia, xq.abs().max().item())
        self.input_scale.copy_(_static_scale(self._ia, self.safety).to(self.input_scale.device))
        xdq = _roundtrip_fp8(xq, self.input_scale)
        wdq = self.gate_up_fp8.float() * self.gate_up_scale.float()
        gate, up = (xdq @ wdq.t()).chunk(2, dim=-1)
        hidden = F.gelu(gate, approximate="tanh") * up
        self._ha = max(self._ha, hidden.abs().max().item())
        self.hidden_scale.copy_(_static_scale(self._ha, self.safety).to(self.hidden_scale.device))

    def forward(self, x):
        if self.calibrating:
            self._calibrate_step(x)
        shape = x.shape
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        x_fp8 = self.quant_ops.channel_scale_quantize_fp8_static_bf16(
            flat, self.channel_scale, self.input_scale
        )
        out = self.ffn_ops.fp8_geglu_mlp_bf16(
            x_fp8,
            self.gate_up_fp8,
            self.down_fp8,
            self.input_scale,
            self.gate_up_scale,
            self.hidden_scale,
            self.down_scale,
        )
        return out.reshape(shape)


class _FlashRTGeluMLP(nn.Module):
    """FP8 drop-in for a SigLIP MLP (fc1 -> gelu_tanh -> fc2, with bias)."""

    def __init__(self, mlp, in_amax, hid_amax, ffn_ops, quant_ops, safety):
        super().__init__()
        self.ffn_ops = ffn_ops
        self.quant_ops = quant_ops
        self.in_features = mlp.fc1.weight.shape[1]
        self.out_features = mlp.fc2.weight.shape[0]
        device = mlp.fc1.weight.device
        up_fp8, up_scale = _quantize_fp8(mlp.fc1.weight)
        down_fp8, down_scale = _quantize_fp8(mlp.fc2.weight)
        self.register_buffer("up_fp8", up_fp8.to(device))
        self.register_buffer("down_fp8", down_fp8.to(device))
        self.register_buffer("up_scale", up_scale.to(device))
        self.register_buffer("down_scale", down_scale.to(device))
        self.register_buffer("up_bias", mlp.fc1.bias.detach().to(torch.bfloat16))
        self.register_buffer("down_bias", mlp.fc2.bias.detach().to(torch.bfloat16))
        self.register_buffer("input_scale", _static_scale(in_amax, safety).to(device))
        self.register_buffer("hidden_scale", _static_scale(hid_amax, safety).to(device))
        self.register_buffer(
            "channel_scale", torch.ones(self.in_features, device=device, dtype=torch.bfloat16)
        )
        self.safety = safety
        self.calibrating = False
        self._ia = 0.0
        self._ha = 0.0

    def _calibrate_step(self, x):
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        self._ia = max(self._ia, flat.float().abs().max().item())
        self.input_scale.copy_(_static_scale(self._ia, self.safety).to(self.input_scale.device))
        xdq = _roundtrip_fp8(flat.float(), self.input_scale)
        hid = (xdq @ (self.up_fp8.float() * self.up_scale.float()).t()) + self.up_bias.float()
        hid = F.gelu(hid, approximate="tanh")
        self._ha = max(self._ha, hid.abs().max().item())
        self.hidden_scale.copy_(_static_scale(self._ha, self.safety).to(self.hidden_scale.device))

    def forward(self, x):
        if self.calibrating:
            self._calibrate_step(x)
        shape = x.shape
        dtype = x.dtype
        flat = x.reshape(-1, self.in_features).to(torch.bfloat16)
        x_fp8 = self.quant_ops.channel_scale_quantize_fp8_static_bf16(
            flat, self.channel_scale, self.input_scale
        )
        out = self.ffn_ops.fp8_gelu_mlp_bf16(
            x_fp8,
            self.up_fp8,
            self.up_bias,
            self.down_fp8,
            self.down_bias,
            self.input_scale,
            self.up_scale,
            self.hidden_scale,
            self.down_scale,
        )
        return out.reshape(*shape[:-1], self.out_features).to(dtype)


def _siglip_mlps(model) -> list:
    tower = model.paligemma_with_expert.paligemma.model.vision_tower
    return [m for _, m in tower.named_modules() if type(m).__name__ == "SiglipMLP"]


def _run_forward(policy, batches) -> None:
    """Run predict_action_chunk in eager mode (a compiled ``sample_actions``
    bypasses the Python module forwards, so drop it for the calibration pass)."""
    model = policy.model
    saved = {name: vars(model).pop(name) for name in ("sample_actions", "forward") if name in vars(model)}
    with torch.inference_mode():
        for batch in batches:
            policy.predict_action_chunk(
                {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            )
    torch.cuda.synchronize()
    vars(model).update(saved)


def _fixed_norm_weight(norm):
    """RMSNorm (1+w) fold weight if ``norm`` is fixed-weight; None if adaptive."""
    return norm.weight if getattr(norm, "dense", None) is None else None


def _fp8_supported(device) -> bool:
    """FP8 E4M3 tensor cores require CUDA SM >= 8.9 (Ada / Hopper / Blackwell).
    Older GPUs (e.g. A100 SM 8.0) and CPU have no FP8 path, so the kernels would
    fail at runtime — gate here and keep BF16."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def apply_fp8_mlp(policy, batch, *, safety: float = 1.05) -> bool:
    """Swap every Gemma GeGLU MLP and SigLIP GELU MLP to the FlashRT FP8 kernels.

    Calibration is FP8-propagated (the FlashRT contract): the FP8 modules are
    swapped in first, then a single forward on one representative frame measures
    each GEMM's input/hidden amax on the *already-quantized* activations it sees
    at runtime (not on a clean BF16 forward), running-max across denoise steps.
    The preceding fixed RMSNorm weight (1+w) is folded into the GEMM so the
    quantized activation is the uniform rms(x); adaptive-RMSNorm inputs (action
    expert) do not fold. ``scale = amax/448 * safety``. A single frame is enough.
    Returns True on success, False (no-op, BF16 kept) if the device lacks FP8
    support or the kernels are unavailable.
    """
    device = next(policy.parameters()).device
    if not _fp8_supported(device):
        logger.warning(
            "PI052: device %s has no FP8 (E4M3) support (needs CUDA SM>=8.9); keeping BF16.",
            device,
        )
        return False
    batches = batch if isinstance(batch, (list, tuple)) else [batch]
    try:
        ffn_ops = _get_kernel(_SWIGLU_REPO)
        gelu_ops = _get_kernel(_GELU_REPO)
        quant_ops = _get_kernel(_GEMM_REPO)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PI052: FlashRT FP8 kernels unavailable (%s); keeping BF16.", exc)
        return False

    model = policy.model
    calibrating = []

    gemma_layers = list(model.paligemma_with_expert.gemma_expert.model.layers) + list(
        model.paligemma_with_expert.paligemma.model.language_model.layers
    )
    for layer in gemma_layers:
        fw = _fixed_norm_weight(layer.post_attention_layernorm)
        layer.mlp = _FlashRTGeGLU(layer.mlp, 1.0, 1.0, ffn_ops, quant_ops, safety, fuse_weight=fw).to(device)
        calibrating.append(layer.mlp)

    siglip = _siglip_mlps(model)
    for mlp_parent in model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers:
        mlp_parent.mlp = _FlashRTGeluMLP(mlp_parent.mlp, 1.0, 1.0, gelu_ops, quant_ops, safety).to(device)
        calibrating.append(mlp_parent.mlp)

    # FP8-propagated calibration: one forward with every module in calibrate mode.
    for m in calibrating:
        m.calibrating = True
    _run_forward(policy, batches)
    for m in calibrating:
        m.calibrating = False

    logger.info(
        "PI052: FlashRT FP8 enabled (%d Gemma + %d SigLIP MLPs).",
        len(gemma_layers),
        len(siglip),
    )
    return True
