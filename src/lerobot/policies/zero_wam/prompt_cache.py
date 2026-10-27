# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable demonstration caches with optional shared, content-addressed caption tensors."""

import hashlib
import json
import uuid
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor


def resolve_prompt_path(path: str | Path, relative_to: str | Path) -> str:
    """Resolve a cache reference relative to its manifest or latent file; keep absolute paths valid."""
    path = Path(path)
    # Hub snapshots may symlink files into a blob store. References belong to the snapshot's
    # directory, not the resolved blob's parent.
    return str(path if path.is_absolute() else Path(relative_to).absolute().parent / path)


def atomic_torch_save(value: object, path: Path) -> None:
    """Publish a complete cache entry so interrupted writes remain safe to resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_shared_prompt_text(embedding: Tensor, output_dir: Path) -> str:
    """Store the full padded caption once and return a path relative to the cache directory."""
    embedding = embedding.detach().cpu().contiguous().clone()
    identity = json.dumps({"shape": list(embedding.shape), "dtype": str(embedding.dtype)}, sort_keys=True)
    checksum = hashlib.sha256(identity.encode())
    checksum.update(embedding.view(torch.uint8).numpy().tobytes())
    relative = Path("text_embeddings") / f"{checksum.hexdigest()}.pth"
    destination = output_dir / relative
    if not destination.exists():
        atomic_torch_save(embedding, destination)
    return relative.as_posix()


@lru_cache(maxsize=16)
def _load_shared_prompt_text(path: str) -> Tensor:
    # Shared entries have immutable content-addressed names. Keep a bounded CPU cache, outside
    # model weights; a blank-caption dataset needs just one tensor per process.
    embedding = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(embedding, Tensor) or embedding.ndim != 2:
        raise ValueError("Shared prompt text must be a [tokens, hidden] tensor")
    return embedding


def load_prompt_text(item: dict, latent_path: str | Path) -> Tensor | None:
    """Read inline legacy captions or a shared caption relative to its owning latent file."""
    if "text_emb_path" not in item:
        return item.get("text_emb")
    if "text_emb" in item:
        raise ValueError("A prompt must use only one of text_emb and text_emb_path")
    reference = item["text_emb_path"]
    if not isinstance(reference, str) or not reference:
        raise ValueError("Shared prompt text requires a nonempty text_emb_path")
    # A missing reference must fail instead of silently substituting the target instruction.
    return _load_shared_prompt_text(resolve_prompt_path(reference, latent_path))
