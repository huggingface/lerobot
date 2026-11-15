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

"""Merge a LoRA adapter trained with ``lerobot-train --peft.*`` into a full praxis checkpoint.

The result is a LeRobot policy directory (the config, the sharded weights with the adapters
folded into the base weights, the processors) that ``PraxisPolicy.from_pretrained``,
``lerobot-eval`` and the praxis sim load like any other checkpoint, as ``merged_state_dict`` of
the praxis repository does for its own LoRA saves.

    python examples/praxis/merge_praxis_adapter.py \\
        --base outputs/praxis-1-droid --adapter outputs/train/praxis_lora/checkpoints/last/pretrained_model \\
        --out outputs/praxis-1-droid-lora-merged
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.praxis.configuration_praxis import PraxisConfig
from lerobot.policies.praxis.modeling_praxis import PraxisPolicy
from lerobot.utils.import_utils import require_package

__all__ = ["merge_adapter"]

logger = logging.getLogger(__name__)


def _local_dir(spec: str | Path) -> Path:
    """A local directory for ``spec``: the path itself, or the processor files of a Hub repo."""
    path = Path(spec)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(spec), allow_patterns=["policy_*processor*"]))


def merge_adapter(base: str | Path, adapter: str | Path, out: str | Path, device: str = "cpu") -> Path:
    """Load ``base``, apply the PEFT adapter of ``adapter``, fold it into the weights, save to ``out``.

    Args:
        base: The base policy directory or Hub repo (a converted praxis checkpoint).
        adapter: The adapter directory (``adapter_config.json`` + ``adapter_model.safetensors``).
        out: The output policy directory.
        device: Where the merge runs (``cpu`` needs about 110 GB of RAM for the 27B model; a GPU
            with 80 GB holds the weights in bfloat16).

    Returns:
        The output directory.
    """
    require_package("peft", extra="peft")
    from peft import PeftModel

    out = Path(out)
    # The weights load straight onto the requested device (the base config may name another).
    config = PreTrainedConfig.from_pretrained(base)
    if not isinstance(config, PraxisConfig):
        raise TypeError(f"{base} holds a {type(config).__name__}, expected a PraxisConfig")
    config.device = str(torch.device(device))
    policy = PraxisPolicy.from_pretrained(base, config=config, strict=True, load_frozen_parts=False)
    peft_model = PeftModel.from_pretrained(policy, str(adapter))
    merged = peft_model.merge_and_unload()
    if not isinstance(merged, PraxisPolicy):
        raise TypeError(f"merge_and_unload returned {type(merged).__name__}, expected PraxisPolicy")
    merged.config.pretrained_path = None
    merged.config.use_peft = False
    merged.save_pretrained(out)
    # The processors of the adapter checkpoint first (training can change settings such as the
    # camera keys), those of the base when the adapter has none.
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for source_dir in (_local_dir(adapter), _local_dir(base)):
            source = source_dir / name
            if source.is_file():
                shutil.copy2(source, out / name)
                for extra in source_dir.glob(name.replace(".json", "_step_*")):
                    shutil.copy2(extra, out / extra.name)
                break
    logger.info("merged adapter %s into %s -> %s", adapter, base, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    merge_adapter(args.base, args.adapter, args.out, device=args.device)


if __name__ == "__main__":
    main()
