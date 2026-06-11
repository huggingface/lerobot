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

This is the N1.7 analogue of ``test_groot_vs_original.py`` (which covers N1.5/GR1).
It verifies that the *self-contained* LeRobot reimplementation of the GR00T N1.7
action head + Qwen3-VL backbone produces the SAME raw model output (``action_pred``,
the normalized flow-matching prediction before any action decoding) as NVIDIA's
original ``gr00t`` package, given byte-identical pre-processed inputs and the same
flow-matching seed.

WHY TWO ENVIRONMENTS
--------------------
The original ``gr00t`` package pins ``transformers==4.57.3`` (Python 3.10) and its
model-config dataclasses subclass ``PretrainedConfig``. Under the transformers 5.x
that the LeRobot GR00T N1.7 integration requires, ``PretrainedConfig`` is itself a
defaulted dataclass, so the original config classes fail to import ("non-default
argument follows default argument"). The two implementations therefore CANNOT be
imported in the same Python process.

To keep the comparison fair, the original outputs + the exact collated inputs are
produced once in the original ``gr00t`` env via
``groot_vs_lerobot/scripts/dump_original_n1_7.py`` and saved to an ``.npz``. This
test loads that artifact, replays the identical inputs through the LeRobot model,
and compares.

This test is LOCAL-only and skips on CI, when ``gr00t``-side prerequisites are not
present, or when the artifact has not been generated. No hardcoded paths: the
artifact location comes from ``GROOT_N1_7_PARITY_NPZ`` (default:
``groot_vs_lerobot/artifacts/original_n1_7_libero.npz`` relative to the repo root).
See ``groot_vs_lerobot/README.md`` for the full run procedure.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Requires a local GR00T N1.7 checkpoint + a pre-generated artifact; not for CI.",
)

from lerobot.policies.groot.configuration_groot import GROOT_N1_7  # noqa: E402,F401

EMBODIMENT_TAG = "libero_sim"
SEED = 42
DEVICE = os.environ.get("GROOT_PARITY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ATOL = float(os.environ.get("GROOT_PARITY_ATOL", "1e-3"))
RTOL = float(os.environ.get("GROOT_PARITY_RTOL", "1e-3"))


def _artifact_path() -> Path:
    env = os.environ.get("GROOT_N1_7_PARITY_NPZ")
    if env:
        return Path(env)
    # repo_root/tests/policies/groot/<this file>  ->  repo_root parent holds groot_vs_lerobot/
    repo_root = Path(__file__).resolve().parents[3]
    # The companion workspace lives alongside the repo, not inside it.
    return repo_root.parent / "groot_vs_lerobot" / "artifacts" / "original_n1_7_libero.npz"


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
    if not path.exists():
        pytest.skip(
            f"Parity artifact not found at {path}. Generate it first in the original gr00t "
            f"env:\n  .venv-original/bin/python groot_vs_lerobot/scripts/dump_original_n1_7.py "
            f"--ckpt <ckpt> --out {path} --device cuda --seed {SEED}"
        )
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
        # Restore integer dtypes that np may have widened.
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
    # The producer flattened the top-level collated dict; "inputs" is its only branch.
    return nested.get("inputs", nested)


def test_groot_n1_7_get_action_parity():
    """Raw model.get_action(action_pred) parity: original gr00t vs LeRobot integration."""
    ckpt = _resolve_checkpoint()
    original_action, flat_inputs = _load_artifact(_artifact_path())

    # Load the underlying GR00T N1.7 model directly (mirrors the original side, which
    # calls ``policy.model.get_action``). This bypasses the LeRobot policy feature
    # pipeline so the comparison is strictly between the two model reimplementations
    # on identical pre-processed inputs.
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    # Run fp32 + SDPA on the LeRobot side to match the producer exactly (the original
    # artifact is dumped fp32 + SDPA). bf16 + differing attention kernels otherwise
    # introduce ~1e-2 numerical noise unrelated to the implementations.
    dtype = torch.float32
    model = GR00TN17.from_pretrained(
        ckpt,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        transformers_loading_kwargs={"trust_remote_code": True},
    )
    model.compute_dtype = "float32"
    model.config.compute_dtype = model.compute_dtype
    model.to(device=DEVICE, dtype=dtype)
    model.eval()

    model_inputs = _unflatten(flat_inputs)

    # Align the flow-matching RNG exactly as the producer did (seed right before sampling).
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        out = model.get_action(model_inputs)
    lerobot_action = out["action_pred"].float().cpu()

    t = min(original_action.shape[1], lerobot_action.shape[1])
    d = min(original_action.shape[2], lerobot_action.shape[2])
    original_action = original_action[:, :t, :d]
    lerobot_action = lerobot_action[:, :t, :d]

    diff = torch.abs(lerobot_action - original_action)
    print(f"\nShapes: lerobot={tuple(lerobot_action.shape)} original={tuple(original_action.shape)}")
    print(f"{'idx':<5}{'LeRobot':>14}{'Original':>14}{'|diff|':>14}")
    for di in range(min(8, lerobot_action.shape[-1])):
        lr = lerobot_action[0, 0, di].item()
        og = original_action[0, 0, di].item()
        print(f"{di:<5}{lr:>14.6f}{og:>14.6f}{abs(lr - og):>14.6f}")
    max_diff = diff.max().item()
    print(f"\nmax|diff| = {max_diff:.6e}   mean|diff| = {diff.mean().item():.6e}")

    assert torch.allclose(lerobot_action, original_action, atol=ATOL, rtol=RTOL), (
        f"GR00T N1.7 raw action_pred differs beyond atol={ATOL}, rtol={RTOL}: "
        f"max|diff|={max_diff:.6e}"
    )
    print(f"\nSUCCESS: GR00T N1.7 raw outputs match (max|diff|={max_diff:.6e}, atol={ATOL})")
