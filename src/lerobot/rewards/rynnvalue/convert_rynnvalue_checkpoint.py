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

"""Convert an official RynnValue checkpoint into a self-contained LeRobot checkpoint.

Example:
    uv run python -m lerobot.rewards.rynnvalue.convert_rynnvalue_checkpoint \
        --output-dir outputs/rynnvalue-4b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .configuration_rynnvalue import RynnValueConfig
from .modeling_rynnvalue import RynnValueRewardModel, _torch_dtype
from .rynn_value_lang.configuration_rynn_value_lang import RynnValueLangConfig
from .rynn_value_lang.modeling_rynn_value_lang import RynnValueLangModel
from .rynn_value_lang.processing_rynn_value_lang import RynnValueLangProcessor

DEFAULT_SOURCE_MODEL_ID = "Alibaba-DAMO-Academy/RynnValue-4B"


def convert_rynnvalue_checkpoint(
    output_dir: str | Path,
    *,
    source_model_id: str = DEFAULT_SOURCE_MODEL_ID,
    revision: str | None = None,
    torch_dtype: str = "bfloat16",
) -> Path:
    """Download and convert an official checkpoint and its processor assets."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    model_config = RynnValueLangConfig.from_pretrained(source_model_id, revision=revision)
    model_config._attn_implementation = "pred_slot_isolated_eager"
    model = RynnValueLangModel.from_pretrained(
        source_model_id,
        revision=revision,
        config=model_config,
        dtype=_torch_dtype(torch_dtype),
    )
    processor = RynnValueLangProcessor.from_pretrained(source_model_id, revision=revision)

    lerobot_config = RynnValueConfig(
        device="cpu",
        model_id=source_model_id,
        model_revision=revision,
        torch_dtype=torch_dtype,
        model_config=model_config.to_dict(),
    )
    reward_model = RynnValueRewardModel(lerobot_config, model=model)

    # Processor assets and LeRobot files coexist at the checkpoint root.
    processor.save_pretrained(output_path)
    reward_model.save_pretrained(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model-id", default=DEFAULT_SOURCE_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()
    output_path = convert_rynnvalue_checkpoint(
        args.output_dir,
        source_model_id=args.source_model_id,
        revision=args.revision,
        torch_dtype=args.torch_dtype,
    )
    print(f"Converted RynnValue checkpoint saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
