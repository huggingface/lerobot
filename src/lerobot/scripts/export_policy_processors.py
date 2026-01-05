#!/usr/bin/env python
"""Export `policy_preprocessor.json` and `policy_postprocessor.json` for an existing policy.

This is useful for async-inference servers which expect these files to exist in the same
directory (or Hugging Face Hub repo) as the policy weights.

Examples:

```bash
# Export into a local model directory (containing config.json + model.safetensors)
uv run python -m lerobot.scripts.export_policy_processors \
  --model jackvial/so101_smolvla_pickplaceorangecube_0_e50_5000_v2 \
  --output-dir /tmp/exported_processors

# Export directly into the existing local checkpoint folder
uv run python -m lerobot.scripts.export_policy_processors \
  --model /path/to/pretrained_model \
  --output-dir /path/to/pretrained_model
```
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export policy processor pipelines (pre/post) next to a policy checkpoint."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local directory path or Hugging Face model id (passed to Policy.from_pretrained).",
    )
    parser.add_argument(
        "--policy-type",
        default=None,
        help=(
            "Optional explicit policy type (e.g., 'smolvla'). If omitted, inferred from loaded policy config."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write `policy_preprocessor.json` / `policy_postprocessor.json` into.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to construct the preprocessor for (default: cpu).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    model_id_or_path = args.model
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.policy_type is None:
        # Infer policy type from config.json (local dir or Hub repo).
        if Path(model_id_or_path).is_dir():
            config_path = Path(model_id_or_path) / CONFIG_NAME
        else:
            config_path = Path(hf_hub_download(repo_id=model_id_or_path, filename=CONFIG_NAME))

        with open(config_path) as f:
            cfg_json = json.load(f)
        args.policy_type = str(cfg_json["type"])

    policy_cls = get_policy_class(args.policy_type)
    policy = policy_cls.from_pretrained(model_id_or_path)

    policy_cfg = policy.config
    policy_cfg.device = args.device

    # Build processors from config (not from pretrained_path) so we can export even if the JSONs are missing.
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=None)

    # Ensure filenames are exactly what async-inference expects.
    preprocessor.save_pretrained(out_dir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(out_dir, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")

    # Small sanity check: make sure the produced json files exist.
    expected = [
        out_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        out_dir / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Expected processor files not created: {missing}")

    logging.info("Wrote processor configs to %s", str(out_dir))
    logging.info(" - %s", expected[0].name)
    logging.info(" - %s", expected[1].name)


if __name__ == "__main__":
    main()


