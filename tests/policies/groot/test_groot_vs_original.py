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

"""Parity tests: original NVIDIA GR00T N1.7 vs the GR00T N1.7 integration in LeRobot.

Two comparisons run per embodiment tag, against per-tag ``.npz`` artifacts produced
once in the original ``gr00t`` env by the companion script
``utils/dump_original_n1_7.py`` (in the ``utils`` package next to this file):

1. **Model parity** -- the self-contained LeRobot reimplementation of the GR00T N1.7
   action head + Qwen3-VL backbone must produce the SAME raw model output
   (``action_pred``, the normalized flow-matching prediction before any action
   decoding) as NVIDIA's original ``gr00t`` package, given byte-identical
   pre-processed inputs and the flow-matching seed recorded in the artifact.
2. **Preprocessor parity** -- LeRobot's own preprocessor pipeline (real Qwen3-VL chat
   template / tokenizer / image packing + state normalization, no mocks) must produce
   the SAME collated model inputs (``input_ids``, ``pixel_values``, ``state``, ...)
   as the original package's processor, given the identical raw observations
   (images, state, language) recorded in the artifact. Artifacts written by older
   versions of the dump script carry no raw observations; this case then SKIPS with
   a regeneration hint.

These tests are LOCAL-only and skip on CI, when ``gr00t``-side prerequisites are not
present, or when no artifact has been generated. By default they look for artifacts in
``<this dir>/artifacts/``; override with ``GROOT_N1_7_PARITY_DIR``. See the
"Original-vs-LeRobot parity test" section of ``src/lerobot/policies/groot/README.md``
for the full run procedure.
"""

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Requires a local GR00T N1.7 checkpoint + pre-generated artifacts; not for CI.",
)

from lerobot.policies.groot.configuration_groot import GROOT_N1_7  # noqa: E402,F401
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

# Fallback flow-matching seed for artifacts predating the recorded ``seed`` field.
SEED = 42
DEVICE = os.environ.get("GROOT_PARITY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ATOL = float(os.environ.get("GROOT_PARITY_ATOL", "1e-3"))
RTOL = float(os.environ.get("GROOT_PARITY_RTOL", "1e-3"))

# Artifact filenames are original_n1_7_<embodiment_tag>.npz
_ARTIFACT_PREFIX = "original_n1_7_"
_ARTIFACT_SUFFIX = ".npz"

# Collated keys compared by the preprocessor parity case: integer/id tensors must
# match exactly; float tensors within ATOL/RTOL.
_COLLATED_EXACT_KEYS = ("input_ids", "attention_mask", "image_grid_thw", "embodiment_id")
_COLLATED_CLOSE_KEYS = ("pixel_values", "state")


def _artifact_dir() -> Path:
    """Directory holding the per-embodiment .npz artifacts.

    Self-contained by default: a sibling ``artifacts/`` directory next to this test.
    Override with ``GROOT_N1_7_PARITY_DIR`` (e.g. to point at a scratch location).
    The directory is read-only here -- it is populated by ``utils/dump_original_n1_7.py``
    run in the original gr00t environment; the test never creates it.
    """
    env = os.environ.get("GROOT_N1_7_PARITY_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "artifacts"


def _discover_artifacts() -> list[tuple[str, Path]]:
    """Return [(embodiment_tag, npz_path), ...] for every dumped artifact."""
    d = _artifact_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(f"{_ARTIFACT_PREFIX}*{_ARTIFACT_SUFFIX}")):
        tag = p.name[len(_ARTIFACT_PREFIX) : -len(_ARTIFACT_SUFFIX)]
        out.append((tag, p))
    return out


def _resolve_checkpoint() -> str:
    env = os.environ.get("GROOT_N1_7_LIBERO_CKPT")
    if env:
        if not Path(env).exists():
            pytest.skip(f"GROOT_N1_7_LIBERO_CKPT={env} does not exist")
        return env
    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            "nvidia/GR00T-N1.7-LIBERO",
            local_files_only=True,
            allow_patterns=["libero_10/*"],
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"GR00T N1.7 LIBERO checkpoint not available locally: {exc}")
    ckpt = Path(root) / "libero_10"
    if not (ckpt / "config.json").exists():
        pytest.skip(f"GR00T N1.7 LIBERO checkpoint incomplete at {ckpt}")
    return str(ckpt)


def _load_artifact(path: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Return (original action_pred, collated model inputs, flow-matching seed)."""
    data = np.load(path, allow_pickle=True)
    original_action = torch.from_numpy(data["action_pred"]).float()
    if "seed" in data.files:
        seed = int(data["seed"])
    else:
        warnings.warn(
            f"Artifact '{path.name}' does not record the producer seed (it predates the current "
            f"dump_original_n1_7.py); falling back to seed={SEED}. If the parity comparison fails, "
            "regenerate the artifact with the current dump script.",
            stacklevel=2,
        )
        seed = SEED
    dtypes = dict(zip(data["meta_keys"].tolist(), data["meta_dtypes"].tolist(), strict=False))
    inputs = {}
    for key in data.files:
        if not key.startswith("in::"):
            continue
        name = key[4:]
        arr = data[key]
        t = torch.from_numpy(np.asarray(arr))
        declared = dtypes.get(key, "")
        if "int" in declared or "long" in declared:
            t = t.long()
        inputs[name] = t
    return original_action, inputs, seed


def _load_raw_observation(path: Path) -> dict[str, Any] | None:
    """Return the raw observation recorded in the artifact, or None for old artifacts.

    Artifacts produced by the current ``dump_original_n1_7.py`` additionally store the
    exact raw observation the producer fed to the original processor: per-camera uint8
    frames (``raw::video.<key>``, (B, T, H, W, C)), per-key state vectors
    (``raw::state.<key>``, (B, T, dim)) and the language instruction
    (``raw::language``, one string per batch element). ``raw_video_keys`` /
    ``raw_state_keys`` record the checkpoint modality-key order.
    """
    data = np.load(path, allow_pickle=True)
    markers = ("raw_video_keys", "raw_state_keys", "raw::language")
    if any(marker not in data.files for marker in markers):
        return None
    video_keys = [str(k) for k in data["raw_video_keys"].tolist()]
    state_keys = [str(k) for k in data["raw_state_keys"].tolist()]
    return {
        "video": {k: data[f"raw::video.{k}"] for k in video_keys},
        "state": {k: data[f"raw::state.{k}"] for k in state_keys},
        "language": [str(t) for t in data["raw::language"].tolist()],
    }


def _raw_observation_to_lerobot_batch(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert the producer's raw observation into a LeRobot policy batch."""
    batch: dict[str, Any] = {}
    for key, frames in raw["video"].items():
        # (B, T, H, W, C) uint8 -> (B, T, C, H, W); the pack step converts back losslessly.
        batch[f"{OBS_IMAGES}.{key}"] = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).contiguous()
    # observation.state is the per-key state vectors (latest frame) concatenated in
    # checkpoint modality-key order -- the layout the LeRobot pack step and the
    # flattened checkpoint statistics expect.
    state_parts = [torch.from_numpy(np.asarray(arr)[:, -1, :]).float() for arr in raw["state"].values()]
    batch[OBS_STATE] = torch.cat(state_parts, dim=-1)
    batch["task"] = list(raw["language"])
    return batch


def _unflatten(inputs: dict[str, torch.Tensor]) -> dict:
    """Rebuild the nested model-input dict from dot-prefixed flat keys."""
    nested: dict = {}
    for dotted, value in inputs.items():
        parts = dotted.split(".")
        cur = nested
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return nested.get("inputs", nested)


def _assert_collated_parity(
    embodiment_tag: str, name: str, lerobot_value: Any, original_value: torch.Tensor, *, exact: bool
) -> None:
    """Compare one collated tensor produced by LeRobot against the original's."""
    assert isinstance(lerobot_value, torch.Tensor), (
        f"[{embodiment_tag}] LeRobot preprocessor output '{name}' is "
        f"{type(lerobot_value).__name__}, expected a tensor."
    )
    lerobot_t = lerobot_value.detach().cpu()
    original_t = original_value.detach().cpu()
    assert lerobot_t.shape == original_t.shape, (
        f"[{embodiment_tag}] collated '{name}' shape mismatch: lerobot={tuple(lerobot_t.shape)} vs "
        f"original={tuple(original_t.shape)}."
    )
    if exact:
        mismatched = int((lerobot_t.long() != original_t.long()).sum())
        assert mismatched == 0, (
            f"[{embodiment_tag}] collated '{name}' differs from the original processor output: "
            f"{mismatched}/{original_t.numel()} elements mismatch."
        )
    else:
        lerobot_f, original_f = lerobot_t.float(), original_t.float()
        max_diff = (lerobot_f - original_f).abs().max().item()
        print(f"[{embodiment_tag}] {name}: shape {tuple(lerobot_t.shape)} max|diff|={max_diff:.6e}")
        assert torch.allclose(lerobot_f, original_f, atol=ATOL, rtol=RTOL), (
            f"[{embodiment_tag}] collated '{name}' differs from the original processor output beyond "
            f"atol={ATOL}, rtol={RTOL}: max|diff|={max_diff:.6e}."
        )


@pytest.fixture(scope="module")
def lerobot_model():
    """Load the LeRobot GR00T N1.7 model once (fp32 + SDPA) and reuse across tags."""
    ckpt = _resolve_checkpoint()
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model = GR00TN17.from_pretrained(
        ckpt,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        transformers_loading_kwargs={"trust_remote_code": True},
    )
    # fp32 + SDPA on both sides: bf16 + differing attention kernels otherwise introduce
    # ~1e-2 numerical noise unrelated to the implementations.
    model.compute_dtype = "float32"
    model.config.compute_dtype = model.compute_dtype
    model.to(device=DEVICE, dtype=torch.float32)
    model.eval()
    return model


_ARTIFACTS = _discover_artifacts()

_requires_artifacts = pytest.mark.skipif(
    not _ARTIFACTS,
    reason=(
        "No GR00T N1.7 parity artifacts found. Generate them first in the original gr00t "
        "env:\n  .venv-original/bin/python tests/policies/groot/utils/dump_original_n1_7.py "
        "--ckpt <ckpt> --out-dir tests/policies/groot/artifacts --device cuda"
    ),
)


@_requires_artifacts
@pytest.mark.parametrize("embodiment_tag,artifact", _ARTIFACTS, ids=[t for t, _ in _ARTIFACTS])
def test_groot_get_action_parity(embodiment_tag, artifact, lerobot_model):
    """Raw model.get_action(action_pred) parity per embodiment: original vs LeRobot."""
    original_action, flat_inputs, seed = _load_artifact(artifact)
    model_inputs = _unflatten(flat_inputs)

    # Align the flow-matching RNG exactly as the producer did (seed right before sampling).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        out = lerobot_model.get_action(model_inputs)
    lerobot_action = out["action_pred"].float().cpu()

    assert lerobot_action.shape == original_action.shape, (
        f"GR00T N1.7 action_pred shape mismatch for embodiment '{embodiment_tag}': "
        f"lerobot={tuple(lerobot_action.shape)} vs original={tuple(original_action.shape)}. "
        "The same checkpoint and inputs must produce identical shapes; this indicates an "
        "action-horizon or action-dim regression (or a stale artifact -- regenerate it with "
        "utils/dump_original_n1_7.py)."
    )

    diff = torch.abs(lerobot_action - original_action)
    max_diff = diff.max().item()
    print(
        f"\n[{embodiment_tag}] shapes lerobot={tuple(lerobot_action.shape)} "
        f"original={tuple(original_action.shape)}  "
        f"max|diff|={max_diff:.6e}  mean|diff|={diff.mean().item():.6e}"
    )

    assert torch.allclose(lerobot_action, original_action, atol=ATOL, rtol=RTOL), (
        f"GR00T N1.7 raw action_pred differs for embodiment '{embodiment_tag}' beyond "
        f"atol={ATOL}, rtol={RTOL}: max|diff|={max_diff:.6e}"
    )


@_requires_artifacts
@pytest.mark.parametrize("embodiment_tag,artifact", _ARTIFACTS, ids=[t for t, _ in _ARTIFACTS])
def test_groot_preprocessor_parity(embodiment_tag, artifact):
    """LeRobot's real preprocessor vs the original's collated tensors, from identical raw obs.

    Runs LeRobot's full preprocessor pipeline -- including the real Qwen3-VL chat
    template, tokenizer and image packing plus the checkpoint-driven state
    normalization (no mocks) -- on the raw observations recorded in the artifact, and
    compares every collated model input against the ones the original ``gr00t``
    processor produced from the same raw observations.
    """
    raw = _load_raw_observation(artifact)
    if raw is None:
        pytest.skip(
            f"Artifact '{artifact.name}' was produced by an older dump_original_n1_7.py that does "
            "not record raw observations; regenerate it with the current dump script to run the "
            "preprocessor parity case."
        )
    _, flat_inputs, _ = _load_artifact(artifact)
    original_inputs = _unflatten(flat_inputs)

    ckpt = _resolve_checkpoint()
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

    # CPU keeps this case runnable without a GPU; the preprocessor is deterministic.
    config = GrootConfig(base_model_path=ckpt, embodiment_tag=embodiment_tag, device="cpu")
    preprocessor, _ = make_groot_pre_post_processors(config)

    processed = preprocessor(_raw_observation_to_lerobot_batch(raw))

    compared_keys = (*_COLLATED_EXACT_KEYS, *_COLLATED_CLOSE_KEYS)
    missing_original = [k for k in compared_keys if k not in original_inputs]
    missing_lerobot = [k for k in compared_keys if k not in processed]
    assert not missing_original, (
        f"[{embodiment_tag}] artifact collated inputs miss {missing_original} "
        f"(available: {sorted(original_inputs)}); regenerate the artifact with the current dump script."
    )
    assert not missing_lerobot, (
        f"[{embodiment_tag}] LeRobot preprocessor output misses {missing_lerobot} (tensor keys "
        f"available: {sorted(k for k, v in processed.items() if isinstance(v, torch.Tensor))})."
    )

    for name in compared_keys:
        _assert_collated_parity(
            embodiment_tag,
            name,
            processed[name],
            original_inputs[name],
            exact=name in _COLLATED_EXACT_KEYS,
        )
