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

"""Upstream/legacy Robometer checkpoint loader.

This module is **only** used by the one-time conversion tooling
(:mod:`lerobot.scripts.lerobot_export_robometer` and
``scripts/verify_robometer_export.py``). It supports:

- Sharded upstream checkpoints (``model-0000X-of-Y.safetensors`` + index).
- PEFT/LoRA adapter checkpoints (``adapter_config.json`` + adapter weights).
- Local snapshot directories or Hugging Face Hub repo ids.

Once :class:`~lerobot.rewards.robometer.RobometerRewardModel` is loaded
through this module, calling ``save_pretrained`` writes the canonical
LeRobot-native layout (single ``model.safetensors`` + ``config.json``) that
the base loader understands.

The runtime path
(:meth:`~lerobot.rewards.pretrained.PreTrainedRewardModel.from_pretrained`)
does **not** import this file. It is safe to delete once you no longer need
the conversion tooling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)


def _download_robometer_snapshot(
    pretrained_path: str,
    *,
    hub_token: str | None = None,
) -> Path:
    """Resolve a Robometer snapshot directory.

    - If ``pretrained_path`` is an existing local directory, return it directly.
    - Otherwise treat ``pretrained_path`` as a Hugging Face repo id (optionally
      with ``@revision``) and download it via ``snapshot_download``.
    """
    local_candidate = Path(pretrained_path)
    if local_candidate.is_dir():
        return local_candidate

    if "@" in pretrained_path:
        repo_id, revision = pretrained_path.split("@", 1)
    else:
        repo_id, revision = pretrained_path, None

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=hub_token,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.txt",
                "*.model",
                "tokenizer*",
                "special_tokens_map.json",
            ],
        )
    )


def _maybe_apply_peft(base_model: Any, snapshot_dir: Path) -> Any:
    adapter_config = snapshot_dir / "adapter_config.json"
    if not adapter_config.exists():
        return base_model

    require_package("peft", extra="peft-dep")
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, str(snapshot_dir))


def _remap_state_dict_keys(state_dict: dict[str, Tensor], model: nn.Module) -> dict[str, Tensor]:
    """Try a few common prefix swaps so PEFT-wrapped checkpoints load cleanly."""
    model_keys = set(model.state_dict().keys())
    remapped: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if key in model_keys:
            remapped[key] = value
            continue

        candidates: list[str] = []
        if key.startswith("model.model."):
            candidates.append(key.replace("model.model.", "model.base_model.model.model.", 1))
            candidates.append(key.replace("model.model.", "model.", 1))
        if key.startswith("model."):
            candidates.append(f"model.{key}")
            candidates.append(key.replace("model.", "", 1))
        else:
            candidates.append(f"model.{key}")
        if key.startswith("model.") and not key.startswith("model.base_model."):
            parts = key.split(".", 1)
            if len(parts) == 2:
                candidates.append(f"model.base_model.{parts[1]}")

        for candidate in candidates:
            if candidate in model_keys:
                remapped[candidate] = value
                break
        else:
            remapped[key] = value

    return remapped


def _resolve_checkpoint_safetensors_files(snapshot_dir: Path) -> list[Path]:
    """Pick the safetensors files that hold the full model weights.

    When ``model.safetensors.index.json`` is present, only the files it lists are
    loaded. Otherwise any ``model*.safetensors`` shards are preferred over
    sidecar files. Falls back to every ``*.safetensors`` in the snapshot.
    """
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            weight_map = json.load(f).get("weight_map", {})
        indexed = sorted(
            {snapshot_dir / name for name in weight_map.values() if (snapshot_dir / name).exists()}
        )
        if indexed:
            return indexed

    model_shards = sorted(snapshot_dir.glob("model*.safetensors"))
    if model_shards:
        return model_shards

    return sorted(snapshot_dir.glob("*.safetensors"))


def apply_upstream_checkpoint(
    model: nn.Module,
    pretrained_path: str,
    *,
    hub_token: str | None = None,
) -> None:
    """Load an upstream (sharded / PEFT) Robometer checkpoint into ``model``.

    Downloads the snapshot, optionally applies PEFT wrapping, merges sharded
    ``.safetensors`` files in memory, remaps PEFT-prefixed keys, and loads them
    into ``model`` non-strictly. ``model`` must already be constructed with the
    matching Robometer architecture (e.g. via
    :class:`~lerobot.rewards.robometer.RobometerRewardModel` ``__init__``).
    """
    snapshot_dir = _download_robometer_snapshot(pretrained_path, hub_token=hub_token)

    # PEFT adapter checkpoints wrap the base model before weight loading so the
    # remapper can place adapter tensors at the right prefix.
    base_model = getattr(model, "model", None)
    if base_model is not None:
        wrapped = _maybe_apply_peft(base_model, snapshot_dir)
        if wrapped is not base_model:
            model.model = wrapped

    files = _resolve_checkpoint_safetensors_files(snapshot_dir)
    if not files:
        logger.warning("No *.safetensors files in %s; using freshly initialised heads", snapshot_dir)
        return

    merged: dict[str, Tensor] = {}
    for path in files:
        merged.update(load_file(str(path)))

    remapped = _remap_state_dict_keys(merged, model)

    # Defensive vocab-match. With the corrected resize logic
    # (``_resize_embeddings_for_robometer`` uses ``len(tokenizer) + 5``),
    # a freshly built ``RobometerRewardModel`` should already share the same
    # vocabulary as the upstream checkpoint (e.g. 151,674 for
    # ``robometer/Robometer-4B``). This block stays in place as a safety net
    # in case a future upstream variant uses a different vocab — we never
    # want ``load_state_dict`` to trip on a silent shape mismatch.
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "get_input_embeddings"):
        for key in (
            "model.model.language_model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ):
            tensor = remapped.get(key)
            if tensor is None:
                continue
            ckpt_vocab = int(tensor.shape[0])
            current_vocab = int(base_model.get_input_embeddings().num_embeddings)
            if ckpt_vocab != current_vocab:
                logger.info(
                    "Resizing model embed table %d -> %d to match upstream checkpoint vocab "
                    "(upstream was trained against a different Qwen revision).",
                    current_vocab,
                    ckpt_vocab,
                )
                base_model.resize_token_embeddings(ckpt_vocab)
            break

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.debug("Robometer checkpoint missing %d keys (sample: %s)", len(missing), missing[:5])
    if unexpected:
        logger.debug(
            "Robometer checkpoint had %d unexpected keys (sample: %s)", len(unexpected), unexpected[:5]
        )
