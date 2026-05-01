#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Export a trained LeRobot policy to ONNX or TensorRT for edge deployment.

Requires: pip install 'lerobot[export]'

Usage examples:

Export an ACT policy to ONNX (CPU, FP32):
```
lerobot-export --policy.path=lerobot/act_pusht_image --output-path=./exports/act
```

Export a Diffusion policy with full DDIM loop unrolled to 10 steps:
```
lerobot-export --policy.path=lerobot/diffusion_pusht --diffusion-mode=ddim-10
```

Export ACT to ONNX in FP16:
```
lerobot-export --policy.path=lerobot/act_pusht_image --precision=fp16
```

Export ACT to TensorRT FP16 (requires CUDA + trtexec on PATH):
```
lerobot-export \\
    --policy.path=lerobot/act_pusht_image \\
    --format=tensorrt \\
    --precision=fp16 \\
    --device=cuda
```

Fold normalization stats into the ONNX graph (clients do not need to normalize inputs):
```
lerobot-export --policy.path=lerobot/act_pusht_image --fold-normalization
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for the lerobot-export CLI.

    All fields can be set from the command line, e.g.:
        lerobot-export --policy.path=lerobot/act_pusht_image --format=tensorrt --precision=fp16
    """

    # Either a Hub repo ID (e.g. "lerobot/act_pusht_image") or a local directory containing
    # config.json + model.safetensors.  Set via --policy.path on the CLI.
    policy: PreTrainedConfig | None = None

    # Where to write exported artifacts (directory).
    output_path: Path = field(default_factory=lambda: Path("exported_model"))

    # Export format: "onnx" or "tensorrt".
    format: str = "onnx"

    # Numerical precision: "fp32", "fp16", or "int8" (int8 requires --calibration-data).
    precision: str = "fp32"

    # ONNX opset version (18 = native LayerNorm + better attention coverage).
    opset_version: int = 18

    # ONNX exporter backend:
    #   "auto"   — try the dynamo path, fall back to legacy with a warning on failure.
    #   "dynamo" — use torch.onnx.export(..., dynamo=True). Required for ACT batch_size > 1.
    #   "legacy" — use the TorchScript-based tracer. Only supports batch_size=1 for ACT.
    exporter: Literal["auto", "dynamo", "legacy"] = "auto"

    # Run numerical parity validation after ONNX export.
    validate: bool = True

    # Relative tolerance for torch.allclose during validation.
    rtol: float = 1e-3

    # Absolute tolerance for torch.allclose during validation.
    atol: float = 1e-5

    # Device to use for tracing ("cpu" or "cuda").
    device: str = "cpu"

    # Batch size used for ONNX sample inputs.
    batch_size: int = 1

    # TensorRT workspace memory limit in GB.
    trt_workspace_gb: int = 4

    # Force TensorRT engine rebuild even if a cached .engine file exists.
    force_rebuild: bool = False

    # Path to calibration data directory (required when precision="int8").
    calibration_data: Path | None = None

    # Fold normalization stats into the ONNX graph so clients receive raw outputs.
    fold_normalization: bool = False

    # Diffusion-only: "unet-only" (single denoising step) or "ddim-N" (full N-step loop).
    diffusion_mode: str = "unet-only"

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        else:
            raise ValueError(
                "A pretrained policy path is required. Provide it with --policy.path=<hub_id_or_local_dir>"
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enables the parser to load config via --policy.path=<path>."""
        return ["policy"]


@parser.wrap()
def export_policy(cfg: ExportConfig) -> None:
    from lerobot.export.core import make_export_wrapper
    from lerobot.export.normalization import build_normalized_wrapper, save_normalization_stats
    from lerobot.export.onnx_export import export_to_onnx
    from lerobot.export.validation import validate_onnx

    logger.info(
        f"Export configuration: format={cfg.format}, precision={cfg.precision}, "
        f"exporter={cfg.exporter}, opset={cfg.opset_version}, device={cfg.device}"
    )

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_safe_torch_device(cfg.device, log=True)

    # ── Load policy ───────────────────────────────────────────────────────────
    logger.info(f"Loading policy '{cfg.policy.type}' from {cfg.policy.pretrained_path}")
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy = policy.to(device)
    policy.eval()

    # ── Load processor pipelines (separate from policy weights) ────────────────
    # `from_pretrained` only loads model weights — preprocessors must be loaded
    # explicitly. They carry the normalization stats we need for the export.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(cfg.policy.pretrained_path),
    )

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build export wrapper ───────────────────────────────────────────────────
    wrapper, spec = make_export_wrapper(policy, cfg)
    wrapper = wrapper.to(device)
    wrapper.eval()

    # ── Save processor artifacts and stats ────────────────────────────────────
    # Always save canonical processor artifacts (policy_preprocessor.json + safetensors)
    # so clients can use lerobot.processor.hotswap_stats() at the edge.
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    if cfg.fold_normalization:
        wrapper = build_normalized_wrapper(wrapper, preprocessor, postprocessor)
        wrapper = wrapper.to(device)
        wrapper.eval()
    else:
        save_normalization_stats(preprocessor, postprocessor, output_dir)

    # ── ONNX export ────────────────────────────────────────────────────────────
    onnx_stem = f"{cfg.policy.type}_{cfg.precision}"
    onnx_path = export_to_onnx(
        wrapper=wrapper,
        spec=spec,
        output_path=output_dir / onnx_stem,
        opset_version=cfg.opset_version,
        precision=cfg.precision,
        exporter=cfg.exporter,
    )

    # ── Numerical parity validation ────────────────────────────────────────────
    if cfg.validate:
        results = validate_onnx(
            wrapper=wrapper,
            sample_inputs=spec.sample_inputs,
            onnx_path=onnx_path,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )
        if not results["allclose"]:
            logger.warning(
                f"Validation failed — max_abs_error={results['max_abs_error']:.2e}, "
                f"cos_sim={results['cos_sim']:.6f}. "
                "The exported model may produce different outputs than PyTorch."
            )

    # ── Optional: TensorRT export ──────────────────────────────────────────────
    engine_path: Path | None = None
    if cfg.format == "tensorrt":
        from lerobot.export.tensorrt_export import export_to_tensorrt

        engine_path = export_to_tensorrt(
            onnx_path=onnx_path,
            output_path=output_dir,
            precision=cfg.precision,
            workspace_gb=cfg.trt_workspace_gb,
            force_rebuild=cfg.force_rebuild,
            calibration_data=cfg.calibration_data,
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info(f"  Policy type : {cfg.policy.type}")
    logger.info(f"  Format      : {cfg.format}")
    logger.info(f"  Precision   : {cfg.precision}")
    logger.info(f"  ONNX model  : {onnx_path}")
    if engine_path is not None:
        logger.info(f"  TRT engine  : {engine_path}")
    if not cfg.fold_normalization:
        logger.info(f"  Norm stats  : {output_dir / 'normalization_stats.json'}")
    logger.info("=" * 60)


def main() -> None:
    init_logging()
    register_third_party_plugins()
    export_policy()


if __name__ == "__main__":
    main()
