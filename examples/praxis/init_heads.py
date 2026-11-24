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

"""Make a single-arm LeRobot policy of your rig from the Praxis-1 base checkpoint.

The base checkpoint (``"embodiment": "base"``: the video backbone and the action-expert trunk,
no heads of any rig) does not load as a policy. This script adds the six head tensors of a
single-arm rig (new random tensors with ``--seed``, or ``--heads`` from an earlier fine-tune of
the same rig), the rig fields of the config (the tile canvas of the two cameras, the action
space, the flange pose frame) and the normalization statistics of your data, and writes a
LeRobot policy directory that ``PraxisPolicy.from_pretrained``, ``lerobot-train`` and
``lerobot-eval`` load. It is the LeRobot form of ``scripts/init_heads.py`` of the praxis
repository: the same config (``praxis.rig.single_arm_config``) and the same head init
(``praxis.rig.init_head_tensors``), with the base shards linked instead of copied.

    python examples/praxis/compute_norm_stats.py --repo-id <your dataset> ... --out stats.json
    python examples/praxis/init_heads.py --base runwayml/praxis-1-base-lerobot --out outputs/praxis-1-myrig \\
        --stats stats.json --tile-height 224 --tile-width 416 --antialias --link

``--base`` is a LeRobot base directory (``convert_checkpoint.py`` of the base checkpoint, or the
Hub repo) or the praxis base checkpoint directory itself. ``--stats-from`` copies the
normalization of an existing single-arm checkpoint instead of a stats file (the DROID rig).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from lerobot.policies.praxis.praxis_core import rig
from lerobot.policies.praxis.praxis_core.checkpoint import (
    BASE_HEAD_TENSORS,
    CONFIG_NAME,
    INDEX_NAME,
    CheckpointError,
    validate_base_config,
    validate_config,
    weight_map,
)
from lerobot.policies.praxis.praxis_core.model import PraxisModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_checkpoint import build_config, read_checksums, write_checksums  # noqa: E402

logger = logging.getLogger(__name__)

HEADS_SHARD = "model-heads.safetensors"


def _local_dir(spec: str | Path) -> Path:
    path = Path(spec)
    if path.is_dir():
        return path
    return Path(snapshot_download(str(spec)))


def base_config_of(base: Path) -> dict:
    """The praxis base config of a praxis base directory or a LeRobot base directory."""
    config = json.loads((base / CONFIG_NAME).read_text())
    if "model_config" in config:  # a LeRobot directory
        config = config["model_config"]
    validate_base_config(config)
    return config


def normalization_of(stats: str | None, stats_from: str | None, domain: str) -> dict:
    if (stats is None) == (stats_from is None):
        raise SystemExit(
            "give exactly one of --stats (a stats JSON) or --stats-from (a single-arm checkpoint)"
        )
    if stats is not None:
        return rig.normalization_from_stats(json.loads(Path(stats).read_text()), domain=domain)
    assert stats_from is not None
    source = json.loads((_local_dir(stats_from) / CONFIG_NAME).read_text())
    if "model_config" in source:
        source = source["model_config"]
    norm = source.get("normalization")
    if not norm:
        raise SystemExit(f"{stats_from} has no normalization section")
    return dict(norm)


def init_heads(
    base: str | Path,
    out: str | Path,
    *,
    tile_height: int,
    tile_width: int,
    antialias: bool,
    normalization: dict,
    seed: int = 0,
    heads: dict[str, torch.Tensor] | None = None,
    link: bool = False,
    **config_overrides,
) -> Path:
    """Write the LeRobot single-arm policy directory ``out`` from the base checkpoint ``base``."""
    src, dst = _local_dir(base), Path(out)
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} exists and is not empty")
    base_cfg = base_config_of(src)
    cfg = rig.single_arm_config(
        base_cfg,
        tile_height=tile_height,
        tile_width=tile_width,
        antialias=antialias,
        normalization=normalization,
    )
    validate_config(cfg)
    names = weight_map(src)
    if found := sorted(set(BASE_HEAD_TENSORS) & set(names)):
        raise CheckpointError(f"{src} is not a base checkpoint: it holds {found}")
    new = dict(heads) if heads is not None else rig.init_head_tensors(cfg, seed)
    if sorted(new) != sorted(BASE_HEAD_TENSORS):
        raise CheckpointError(f"heads must hold exactly {list(BASE_HEAD_TENSORS)}, got {sorted(new)}")
    new = {k: v.to(torch.bfloat16).contiguous() for k, v in new.items()}
    # Every tensor of the model must come from the base or the heads, with the heads' shapes right.
    with torch.device("meta"):
        model = PraxisModel.from_config(cfg)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(names) - set(new))
    unexpected = sorted((set(names) | set(new)) - set(expected))
    if missing or unexpected:
        raise CheckpointError(
            f"the base and the heads do not match the model: missing {missing[:4]}, unexpected {unexpected[:4]}"
        )
    for name, tensor in new.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise CheckpointError(f"{name}: shape {tuple(tensor.shape)} != {tuple(expected[name].shape)}")
    dst.mkdir(parents=True, exist_ok=True)
    for shard in sorted(set(names.values())):
        target = dst / shard
        if link:
            target.symlink_to((src / shard).resolve())
        else:
            shutil.copy2(src / shard, target)
    save_file(new, str(dst / HEADS_SHARD), metadata={"format": "pt"})
    index = {"metadata": {"format": "pt"}, "weight_map": {**names, **dict.fromkeys(new, HEADS_SHARD)}}
    (dst / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    config = build_config(cfg, **config_overrides)
    config._save_pretrained(dst)
    # The directory's own SHA256SUMS: the base digests of the base shards, the heads shard, the
    # index and the config computed here.
    write_checksums(dst, read_checksums(src))
    from lerobot.policies.praxis.processor_praxis import make_praxis_pre_post_processors

    pre, post = make_praxis_pre_post_processors(config)
    pre.save_pretrained(dst)
    post.save_pretrained(dst)
    logger.info("wrote %s: %d tensors from the base, %d new head tensors", dst, len(names), len(new))
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base", required=True, help="the base checkpoint: a LeRobot or praxis directory, or a Hub repo"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats", default=None, help="normalization statistics JSON (compute_norm_stats.py)")
    parser.add_argument(
        "--stats-from", default=None, help="copy the normalization of this single-arm checkpoint"
    )
    parser.add_argument(
        "--domain", default="custom", help="the name of the data in the normalization section"
    )
    parser.add_argument("--tile-height", type=int, default=224)
    parser.add_argument("--tile-width", type=int, default=416)
    parser.add_argument("--antialias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0, help="the seed of the new head tensors")
    parser.add_argument("--heads", default=None, help="a safetensors file with the six head tensors instead")
    parser.add_argument("--link", action="store_true", help="symlink the base shards instead of copying them")
    parser.add_argument("--device", default="cuda", help="the device written into the config")
    parser.add_argument("--exterior-image-key", default="observation.images.exterior")
    parser.add_argument("--wrist-image-key", default="observation.images.wrist")
    parser.add_argument("--attention-backend", default=None, choices=("auto", "flash", "triton", "eager"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {
        "device": args.device,
        "exterior_image_key": args.exterior_image_key,
        "wrist_image_key": args.wrist_image_key,
    }
    if args.attention_backend:
        overrides["attention_backend"] = args.attention_backend
    heads = load_file(args.heads) if args.heads else None
    out = init_heads(
        args.base,
        args.out,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        antialias=args.antialias,
        normalization=normalization_of(args.stats, args.stats_from, args.domain),
        seed=args.seed,
        heads=heads,
        link=args.link,
        **overrides,
    )
    print(f"{out}: load with PraxisPolicy.from_pretrained or --policy.path; fine-tune with lerobot-train")


if __name__ == "__main__":
    main()
