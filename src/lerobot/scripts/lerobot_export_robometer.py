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

"""Re-save a Robometer checkpoint in LeRobot HF format.

LeRobot's reward model format is ``config.json`` (a draccus-encoded
:class:`~lerobot.rewards.robometer.RobometerConfig`) plus a single
``model.safetensors`` containing the merged base + heads weights. The
released checkpoint at ``lilkm/robometer-4b`` already follows this layout;
this script is for converting other Robometer variants (e.g. a future
upstream release or a local training run) into the same format.

Example:

.. code-block:: shell

   lerobot-export-robometer \\
       --src robometer/Robometer-4B \\
       --dst ./robometer-4b-lerobot
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer._upstream_loader import apply_upstream_checkpoint
from lerobot.utils.utils import init_logging


def export_robometer_to_lerobot(
    src: str,
    dst: str | Path,
    *,
    device: str = "cpu",
    dataset_repo_id: str = "",
    write_model_card: bool = True,
) -> Path:
    """Load Robometer from ``src`` and re-save it under ``dst`` in LeRobot HF format.

    Produces ``config.json``, ``model.safetensors``, and (optionally) ``README.md``.

    Args:
        src: Upstream source. Hugging Face repo id (``"robometer/Robometer-4B"``,
            optionally ``"...@revision"``) or a local snapshot directory.
        dst: Output directory. ``config.json`` and ``model.safetensors`` are
            written here.
        device: Where to place the model during loading. Defaults to CPU; use
            ``"cuda"`` if you want to verify on GPU before saving.
        dataset_repo_id: Hugging Face dataset id the model was trained on
            (e.g. ``"robometer/RBM-1M"``). Written into the model card's
            ``datasets:`` metadata. Leave empty if not applicable.
        write_model_card: Generate a ``README.md`` using LeRobot's reward
            model card template. Disable if you want to write the README
            yourself.

    Returns:
        The resolved output directory.
    """
    # A fresh ``RobometerConfig`` has ``vlm_config=None``, which routes
    # ``__init__`` through the upstream-matching path: download base Qwen,
    # resize embeddings per ``ROBOMETER_SPECIAL_TOKENS``. ``apply_upstream_checkpoint``
    # then resizes again (if needed) to match the upstream checkpoint's vocab
    # and overlays its weights. ``_save_pretrained`` snapshots the resulting
    # post-resize architecture into ``vlm_config`` for fast future loads.
    cfg = RobometerConfig(pretrained_path=src, device=device)
    model = RobometerRewardModel(cfg)
    apply_upstream_checkpoint(model, src)
    model.to(device)
    model.eval()

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst))

    if write_model_card:
        card = model.generate_model_card(
            dataset_repo_id=dataset_repo_id,
            model_type=model.config.type,
            license=model.config.license,
            tags=model.config.tags,
        )
        card.save(str(dst / "README.md"))

    return dst


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        default="robometer/Robometer-4B",
        help="Upstream Robometer source (HF repo id or local directory).",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Output directory for the LeRobot-format checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to load the model on (default: cpu). Conversion only "
        "needs CPU; use cuda if you also want to smoke-test inference.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional Hugging Face dataset id used for training "
        "(e.g. `robometer/RBM-1M`). Written into the auto-generated model card's "
        "`datasets:` metadata.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip writing README.md. Use if you want to author the model card by hand.",
    )
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()
    out = export_robometer_to_lerobot(
        src=args.src,
        dst=args.dst,
        device=args.device,
        dataset_repo_id=args.dataset,
        write_model_card=not args.no_readme,
    )
    logging.info("Saved LeRobot-format Robometer checkpoint to %s", out)


if __name__ == "__main__":
    main()
