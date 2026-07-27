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

"""Parity test: original NVIDIA GR00T N1.7 vs the GR00T N1.7 integration in LeRobot.

Verifies that the self-contained LeRobot reimplementation of the GR00T N1.7 action
head + Qwen3-VL backbone produces the SAME raw model output (``action_pred``, the
normalized flow-matching prediction before any action decoding) as NVIDIA's original
``gr00t`` package, given byte-identical pre-processed inputs and the same
flow-matching seed. The comparison is parametrized over every embodiment tag present
in the checkpoint.

To keep the comparison fair, the original outputs + the exact collated inputs are
produced once per embodiment in the original ``gr00t`` env via the companion script
``utils/dump_original_n1_7.py`` (in the ``utils`` package next to this file) and saved
to per-tag ``.npz`` files.
This test discovers those artifacts, replays the identical inputs through the LeRobot
model, and compares.

This test is LOCAL-only and skips on CI, when ``gr00t``-side prerequisites are not
present, or when no artifact has been generated. By default it looks for artifacts in
``<this dir>/artifacts/``; override with ``GROOT_N1_7_PARITY_DIR``. See the
"Original-vs-LeRobot parity test" section of ``src/lerobot/policies/groot/README.md``
for the full run procedure.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Requires a local GR00T N1.7 checkpoint + pre-generated artifacts; not for CI.",
)

from lerobot.policies.groot.configuration_groot import GROOT_N1_7  # noqa: E402,F401

SEED = 42
DEVICE = os.environ.get("GROOT_PARITY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ATOL = float(os.environ.get("GROOT_PARITY_ATOL", "1e-3"))
RTOL = float(os.environ.get("GROOT_PARITY_RTOL", "1e-3"))

# Artifact filenames are original_n1_7_<embodiment_tag>.npz
_ARTIFACT_PREFIX = "original_n1_7_"
_ARTIFACT_SUFFIX = ".npz"


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


def _load_artifact(path: Path):
    data = np.load(path, allow_pickle=True)
    original_action = torch.from_numpy(data["action_pred"]).float()
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
    return original_action, inputs


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


@pytest.mark.skipif(
    not _ARTIFACTS,
    reason=(
        "No GR00T N1.7 parity artifacts found. Generate them first in the original gr00t "
        "env:\n  .venv-original/bin/python tests/policies/groot/utils/dump_original_n1_7.py "
        "--ckpt <ckpt> --out-dir tests/policies/groot/artifacts --device cuda"
    ),
)
@pytest.mark.parametrize("embodiment_tag,artifact", _ARTIFACTS, ids=[t for t, _ in _ARTIFACTS])
def test_groot_get_action_parity(embodiment_tag, artifact, lerobot_model):
    """Raw model.get_action(action_pred) parity per embodiment: original vs LeRobot."""
    original_action, flat_inputs = _load_artifact(artifact)
    model_inputs = _unflatten(flat_inputs)

    # Align the flow-matching RNG exactly as the producer did (seed right before sampling).
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        out = lerobot_model.get_action(model_inputs)
    lerobot_action = out["action_pred"].float().cpu()

    t = min(original_action.shape[1], lerobot_action.shape[1])
    d = min(original_action.shape[2], lerobot_action.shape[2])
    original_action = original_action[:, :t, :d]
    lerobot_action = lerobot_action[:, :t, :d]

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
