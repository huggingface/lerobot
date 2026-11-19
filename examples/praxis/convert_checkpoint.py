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

"""Convert a praxis checkpoint directory to a LeRobot pretrained policy directory.

A praxis checkpoint holds ``config.json`` (the architecture), ``model-XXXXX-of-NNNNN.safetensors``
shards, ``model.safetensors.index.json`` and ``SHA256SUMS``. The LeRobot directory holds the
LeRobot ``config.json`` (a ``PraxisConfig`` with the praxis config under ``model_config``), the
same shards and index, and the serialized pre- and post-processors.

    python examples/praxis/convert_checkpoint.py \\
        --checkpoint checkpoints/praxis-1-droid --out outputs/praxis-1-droid-lerobot --link

``--link`` symlinks the 54 GB of shards instead of copying them. The script verifies
``SHA256SUMS`` of the source first unless ``--skip-checksums`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.praxis.configuration_praxis import (
    BIMANUAL_CAMERA_KEYS,
    EMBODIMENT_ARMS,
    PRAXIS_ACTION_DIM,
    PraxisConfig,
)
from lerobot.policies.praxis.praxis_core.checkpoint import (
    BASE_EMBODIMENT,
    CHECKSUM_NAME,
    CONFIG_NAME,
    INDEX_NAME,
    SINGLE_NAME,
    validate_base_config,
    validate_config,
    verify_checksums,
    weight_map,
)
from lerobot.policies.praxis.processor_praxis import make_praxis_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


def read_checksums(path: Path) -> dict[str, str]:
    """The ``name -> digest`` entries of ``SHA256SUMS`` in ``path`` (empty without the file)."""
    listed: dict[str, str] = {}
    if (path / CHECKSUM_NAME).is_file():
        for line in (path / CHECKSUM_NAME).read_text().splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                listed[name.strip().lstrip("*")] = digest
    return listed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def write_checksums(dst: Path, known: Mapping[str, str] | None = None) -> Path:
    """Write the ``SHA256SUMS`` of a policy directory: every shard of its index, the index and the
    LeRobot ``config.json``, so ``sha256sum -c SHA256SUMS`` and ``verify_checksums`` pass on it.

    ``known`` holds the digests of the source list (verified before) for the shards that were copied
    or linked unchanged; the digests of the other files are computed here.
    """
    known = dict(known or {})
    shards = sorted(set(weight_map(dst).values()))
    lines = []
    for name in [*shards, INDEX_NAME, CONFIG_NAME]:
        digest = known.get(name) if name in shards else None
        lines.append(f"{digest or _sha256(dst / name)}  {name}\n")
    (dst / CHECKSUM_NAME).write_text("".join(lines))
    return dst / CHECKSUM_NAME


DEFAULT_IMAGE_SHAPE = (3, 288, 512)  # the DROID rig cameras of the praxis sim; any size works
# The serve settings of each rig's checkpoint (the praxis repository's `configs/sim_droid.yaml`, `configs/serve_yam.yaml`);
# the single-arm values are the PraxisConfig defaults.
RIG_SAMPLER_DEFAULTS: dict[str, dict[str, float | int]] = {
    "single_arm": {},
    "bimanual": {"num_steps": 2, "t_shift": 1.0, "cfg_scale": 1.0, "cfg_video_scale": 1.0},
}


def build_config(
    praxis_config: dict,
    *,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    exterior_image_key: str = "observation.images.exterior",
    wrist_image_key: str = "observation.images.wrist",
    camera_keys: list[str] | None = None,
    device: str = "cuda",
    **overrides,
) -> PraxisConfig:
    """Return the LeRobot config of a praxis checkpoint config.

    The rig comes from ``embodiment``: the single-arm checkpoint takes the exterior and wrist keys,
    the bimanual one ``camera_keys`` (default `BIMANUAL_CAMERA_KEYS`: top, left wrist, right wrist)
    and the serve settings of its checkpoint (`RIG_SAMPLER_DEFAULTS`, unless overridden). A base
    checkpoint (no action heads) gets a config that only `examples/praxis/init_heads.py` completes.
    """
    embodiment = str(praxis_config.get("embodiment", "single_arm"))
    chunk = int(praxis_config["chunking"]["action_chunk"])
    if embodiment == BASE_EMBODIMENT:
        validate_base_config(praxis_config)
        return PraxisConfig(
            model_config=praxis_config, chunk_size=chunk, n_action_steps=chunk, device=device, **overrides
        )
    validate_config(praxis_config)
    if embodiment not in EMBODIMENT_ARMS:
        raise ValueError(f"unknown embodiment {embodiment!r}")
    n_arms = EMBODIMENT_ARMS[embodiment]
    if embodiment == "bimanual":
        keys = list(camera_keys or BIMANUAL_CAMERA_KEYS)
    else:
        keys = list(camera_keys or [exterior_image_key, wrist_image_key])
        exterior_image_key, wrist_image_key = keys
    settings = {**RIG_SAMPLER_DEFAULTS.get(embodiment, {}), **overrides}
    config = PraxisConfig(
        model_config=praxis_config,
        exterior_image_key=exterior_image_key,
        wrist_image_key=wrist_image_key,
        camera_keys=keys if embodiment == "bimanual" else [],
        chunk_size=chunk,
        n_action_steps=chunk,
        input_features={
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(image_shape)) for key in keys},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(PRAXIS_ACTION_DIM * n_arms,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(PRAXIS_ACTION_DIM * n_arms,))},
        device=device,
        **settings,
    )
    config.validate_features()
    return config


def convert(
    checkpoint: str | Path,
    out: str | Path,
    *,
    link: bool = False,
    skip_checksums: bool = False,
    **config_overrides,
) -> Path:
    """Write the LeRobot policy directory ``out`` for the praxis checkpoint ``checkpoint``."""
    src, dst = Path(checkpoint), Path(out)
    praxis_config = json.loads((src / CONFIG_NAME).read_text())
    if not skip_checksums and (src / CHECKSUM_NAME).is_file():
        logger.info("verifying %s", src / CHECKSUM_NAME)
        verify_checksums(src)
    config = build_config(praxis_config, **config_overrides)
    dst.mkdir(parents=True, exist_ok=True)
    config._save_pretrained(dst)
    files = sorted(set(weight_map(src).values()))
    if (src / INDEX_NAME).is_file():
        files.append(INDEX_NAME)
    for name in files:
        target = dst / name
        if target.is_symlink() or target.exists():
            target.unlink()
        if link and name != INDEX_NAME:
            target.symlink_to((src / name).resolve())
        else:
            shutil.copy2(src / name, target)
    # The directory's own list: the source digests of the shards, the LeRobot config and the index.
    write_checksums(dst, read_checksums(src))
    if config.embodiment != BASE_EMBODIMENT:  # a base checkpoint gets its processors from init_heads
        pre, post = make_praxis_pre_post_processors(config)
        pre.save_pretrained(dst)
        post.save_pretrained(dst)
    n_weights = sum(1 for name in files if name != INDEX_NAME)
    logger.info("wrote %s (%d weight files, %s)", dst, n_weights, "linked" if link else "copied")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="the praxis checkpoint directory")
    parser.add_argument("--out", required=True, help="the LeRobot policy directory to write")
    parser.add_argument("--link", action="store_true", help="symlink the shards instead of copying them")
    parser.add_argument("--skip-checksums", action="store_true", help="do not verify SHA256SUMS")
    parser.add_argument("--device", default="cuda", help="the device written into the config")
    parser.add_argument("--exterior-image-key", default="observation.images.exterior")
    parser.add_argument("--wrist-image-key", default="observation.images.wrist")
    parser.add_argument("--vae-id", default=None, help="Wan2.2 VAE weights (path or repo_id:filename)")
    parser.add_argument("--text-encoder-id", default=None, help="Qwen3.5-2B (path or repo id)")
    parser.add_argument("--attention-backend", default=None, choices=("auto", "flash", "triton", "eager"))
    parser.add_argument(
        "--camera-keys",
        default=None,
        help="comma list of the camera keys in canvas order (bimanual default: top_left, wrist_left, wrist_right)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=None, help="sampler steps (default: the rig's serve setting)"
    )
    parser.add_argument(
        "--t-shift", type=float, default=None, help="flow shift (default: the rig's serve setting)"
    )
    parser.add_argument("--cfg-scale", type=float, default=None, help="action guidance (default: the rig's)")
    parser.add_argument(
        "--cfg-video-scale", type=float, default=None, help="video guidance (default: the rig's)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = {}
    if args.vae_id:
        overrides["vae_id"] = args.vae_id
    if args.text_encoder_id:
        overrides["text_encoder_id"] = args.text_encoder_id
    if args.attention_backend:
        overrides["attention_backend"] = args.attention_backend
    for name in ("num_steps", "t_shift", "cfg_scale", "cfg_video_scale"):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    if args.camera_keys:
        overrides["camera_keys"] = [k.strip() for k in args.camera_keys.split(",") if k.strip()]
    convert(
        args.checkpoint,
        args.out,
        link=args.link,
        skip_checksums=args.skip_checksums,
        device=args.device,
        exterior_image_key=args.exterior_image_key,
        wrist_image_key=args.wrist_image_key,
        **overrides,
    )
    if not (Path(args.out) / SINGLE_NAME).exists():
        print(f"{args.out}: sharded weights; load with PraxisPolicy.from_pretrained or --policy.path")


if __name__ == "__main__":
    main()
