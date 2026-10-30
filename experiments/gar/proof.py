#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""One approved Gaussian Action Readout proof for the public pi0.5 LIBERO checkpoint.

``--prepare-only`` selects and decodes contexts on CPU. The GPU job loads the manifest
produced by that step and does not download the dataset or the tokenizer again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import torch

BASE_SHA = "724b1b62ff41ed6ada55d5157b51101f52011cd6"
MODEL_ID = "lerobot/pi05_libero_finetuned_v044"
MODEL_REVISION = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
DATASET_ID = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
TOKENIZER_ID = "google/paligemma-3b-pt-224"
TASK_LANGUAGE = "put both the alphabet soup and the tomato sauce in the basket"
SIM_TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
BDDL_URL = (
    "https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/"
    "libero/libero/bddl_files/libero_10/"
    f"{SIM_TASK}.bddl"
)
SOUP = "alphabet_soup_1"
SAUCE = "tomato_sauce_1"
BASKET_REGION = "basket_1_contain_region"
IMAGE_KEY = "observation.images.image"
IMAGE2_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
CHUNK_SIZE = 50
EXECUTED_ACTIONS = 10
MOTION_DIM = 6
GRIPPER_INDEX = 6
ACTION_DIM = 7
TEACHER_DRAWS = 32
FIT_CONTEXTS = 64
FRAMES_PER_DEMO = 8
DEMO_COUNT = 10
FIT_UPDATES = 2000
FIT_BATCH = 32
FIT_SEED = 17
CONTROL_STEPS = 520
SETTLE_STEPS = 10
LIFT_METERS = 0.02
LIFT_FRAMES = 3
POLICY_SEEDS = (17, 18, 19, 20)
INIT_STATES = (0, 1)
NOISE_SEED_BASE = 17_000_003
ORDERS = ("soup_then_sauce", "sauce_then_soup")


class BudgetExhaustedError(RuntimeError):
    """The in-container wall clock is spent. Partial files stay on disk."""


class StopProofError(RuntimeError):
    """Loading, memory, or a required check failed. Partial files stay on disk."""


def require_bfloat16(dtype: object) -> None:
    """Accept the checkpoint dtype as ``torch.bfloat16`` or the config string."""
    if dtype in (torch.bfloat16, "bfloat16"):
        return
    raise StopProofError(f"checkpoint dtype is {dtype}, not bfloat16")


def libero_config_file() -> Path:
    """Return the config file LIBERO reads before it asks for a dataset path."""
    directory = os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))
    return Path(directory) / "config.yaml"


def find_libero_benchmark_root(search_path: list[str] | None = None) -> Path:
    """Locate ``libero/libero`` by scanning path entries. This does not import it."""
    for entry in sys.path if search_path is None else search_path:
        init_file = Path(entry) / "libero" / "libero" / "__init__.py"
        if init_file.is_file():
            return init_file.parent.resolve()
    raise StopProofError("libero/libero/__init__.py was not found on sys.path")


def _package_directory(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate.resolve()
    raise StopProofError(f"{root} has none of these directories: {', '.join(names)}")


def write_libero_config(search_path: list[str] | None = None, datasets_dir: Path | None = None) -> Path:
    """Write the five LIBERO paths before any import of the package."""
    root = find_libero_benchmark_root(search_path)
    datasets = (libero_config_file().parent / "datasets") if datasets_dir is None else datasets_dir
    datasets.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": root,
        "bddl_files": _package_directory(root, "bddl_files"),
        "init_states": _package_directory(root, "init_files", "init_states"),
        "datasets": datasets.resolve(),
        "assets": _package_directory(root, "assets"),
    }
    missing = [key for key, path in paths.items() if not path.is_dir()]
    if missing:
        raise StopProofError(f"LIBERO config paths do not exist: {', '.join(missing)}")
    config_path = libero_config_file()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "".join(f"{key}: {json.dumps(str(path))}\n" for key, path in paths.items()),
        encoding="utf-8",
    )
    return config_path


class Clock:
    def __init__(self, seconds: float) -> None:
        self.started = time.monotonic()
        self.limit = float(seconds)

    def remaining(self) -> float:
        return self.limit - (time.monotonic() - self.started)

    def ensure(self, stage: str) -> None:
        if self.remaining() <= 0:
            raise BudgetExhaustedError(f"{stage} exceeded the {self.limit:.0f}s wall clock")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def emit(output: Path, stage: str, **fields: Any) -> None:
    """Print one stage and append it before the next expensive step."""
    output.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "at_unix": time.time(), **_json_ready(fields)}
    print(json.dumps(record), flush=True)
    with (output / "partial.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    (output / "latest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


def interior_indices(
    length: int, count: int = FRAMES_PER_DEMO, horizon: int = CHUNK_SIZE
) -> list[int] | None:
    """Evenly spaced observation indices that still have a full action suffix.

    Endpoints of the valid range are excluded. The episode is skipped when that
    interior does not contain ``count`` distinct indices.
    """
    candidates = list(range(1, length - horizon))
    if len(candidates) < count:
        return None
    if len(candidates) == count:
        return candidates
    picks = [candidates[round(i * (len(candidates) - 1) / (count - 1))] for i in range(count)]
    if len(set(picks)) != count:
        return None
    return picks


def _git_head() -> str | None:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _find_task_index(tasks_path: Path) -> int:
    import pyarrow.parquet as pq

    frame = pq.read_table(tasks_path).to_pandas().reset_index()
    language = frame.iloc[:, 0].astype(str)
    matches = frame.loc[language == TASK_LANGUAGE, "task_index"]
    if matches.empty:
        raise StopProofError(f"dataset tasks do not contain {TASK_LANGUAGE!r}")
    return int(matches.iloc[0])


def _episode_rows(meta_root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in sorted((meta_root / "episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _episodes_for_task(data_root: Path, task_index: int) -> set[int]:
    import pyarrow.parquet as pq

    seen: dict[int, set[int]] = {}
    for path in sorted(data_root.rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        for episode, task in zip(
            table.column("episode_index").to_pylist(),
            table.column("task_index").to_pylist(),
            strict=True,
        ):
            seen.setdefault(int(episode), set()).add(int(task))
    return {episode for episode, tasks in seen.items() if tasks == {task_index}}


def _load_states(data_file: Path, episode_index: int, frames: list[int]) -> dict[int, list[float]]:
    import pyarrow.parquet as pq

    table = pq.read_table(data_file, columns=["episode_index", "frame_index", STATE_KEY])
    wanted = set(frames)
    states: dict[int, list[float]] = {}
    for episode, frame, state in zip(
        table.column("episode_index").to_pylist(),
        table.column("frame_index").to_pylist(),
        table.column(STATE_KEY).to_pylist(),
        strict=True,
    ):
        if int(episode) != episode_index or int(frame) not in wanted:
            continue
        values = state.tolist() if hasattr(state, "tolist") else list(state)
        states[int(frame)] = [float(item) for item in values]
    missing = wanted - set(states)
    if missing:
        raise StopProofError(f"episode {episode_index} is missing states for frames {sorted(missing)}")
    return states


def _select_demos(rows: list[dict[str, Any]], task_episodes: set[int]) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        episode = int(row["episode_index"])
        if episode not in task_episodes:
            continue
        frames = interior_indices(int(row["length"]))
        if frames is None:
            continue
        eligible.append({**row, "frames": frames})
    eligible.sort(key=lambda row: int(row["episode_index"]))
    if len(eligible) < DEMO_COUNT:
        raise StopProofError(f"only {len(eligible)} eligible demonstrations; need {DEMO_COUNT}")
    pool = [int(row["episode_index"]) for row in eligible]
    chosen = random.Random(FIT_SEED).sample(pool, DEMO_COUNT)
    by_episode = {int(row["episode_index"]): row for row in eligible}
    return [by_episode[episode] for episode in chosen]


def _decode_episode(
    dataset_root: Path, row: dict[str, Any], frames: list[int], fps: float
) -> dict[str, torch.Tensor]:
    from lerobot.datasets.video_utils import decode_video_frames

    decoded = {}
    timestamps = [frame / fps for frame in frames]
    for key in (IMAGE_KEY, IMAGE2_KEY):
        video_path = dataset_root / (
            f"videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/"
            f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
        )
        shifted = [stamp + float(row[f"videos/{key}/from_timestamp"]) for stamp in timestamps]
        frames_tensor = decode_video_frames(video_path, shifted, 1e-4, backend="pyav", return_uint8=True)
        if tuple(frames_tensor.shape) != (len(frames), 3, 256, 256):
            raise StopProofError(
                f"{key} frames for episode {row['episode_index']} have shape {tuple(frames_tensor.shape)}"
            )
        decoded[key] = frames_tensor
    return decoded


def _download_videos(dataset_root: Path, rows: list[dict[str, Any]]) -> None:
    from huggingface_hub import hf_hub_download

    paths: set[str] = set()
    for row in rows:
        for key in (IMAGE_KEY, IMAGE2_KEY):
            paths.add(
                f"videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/"
                f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
            )
    for relative in sorted(paths):
        destination = dataset_root / relative
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        hf_hub_download(
            DATASET_ID,
            relative,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=dataset_root,
        )


def _copy_policy_assets(cache_policy: Path, assets: Path) -> None:
    from huggingface_hub import hf_hub_download

    names = [
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    assets.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = cache_policy / name
        if not source.is_file():
            hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=cache_policy)
            source = cache_policy / name
        target = assets / name
        target.write_bytes(source.read_bytes())


def _download_tokenizer(destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        TOKENIZER_ID,
        local_dir=destination,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "preprocessor_config.json",
        ],
    )


def _gripper_stats(normalizer: Path) -> dict[str, float]:
    from safetensors import safe_open

    with safe_open(normalizer, framework="pt", device="cpu") as handle:
        mean = handle.get_tensor("action.mean")
        std = handle.get_tensor("action.std")
        action_min = handle.get_tensor("action.min")
        action_max = handle.get_tensor("action.max")
    return {
        "mean": [float(item) for item in mean],
        "std": [float(item) for item in std],
        "min": [float(item) for item in action_min],
        "max": [float(item) for item in action_max],
        "gripper_mean": float(mean[GRIPPER_INDEX]),
        "gripper_std": float(std[GRIPPER_INDEX]),
        "threshold": 0.0,
        "open_command": -1.0,
        "close_command": 1.0,
        "encode": "(command - mean) / std",
        "decode": "normalized * std + mean",
        "close_if_decoded_gt": 0.0,
    }


def _bddl_text() -> str:
    with urllib.request.urlopen(BDDL_URL, timeout=60) as response:
        return response.read().decode("utf-8")


def prepare(output: Path, cache: Path) -> None:
    """Decode the approved contexts and write the manifest. No teacher fit and no simulation."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    head = _git_head()
    if head is not None and head != BASE_SHA:
        raise StopProofError(f"worktree HEAD {head} is not the approved pin {BASE_SHA}")
    dataset_root = cache / "libero"
    snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=dataset_root,
        allow_patterns=["meta/**", "data/**"],
    )
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    task_index = _find_task_index(dataset_root / "meta" / "tasks.parquet")
    if task_index != 5:
        raise StopProofError(f"expected task_index 5 for the soup and sauce language, found {task_index}")
    task_episodes = _episodes_for_task(dataset_root / "data", task_index)
    selected = _select_demos(_episode_rows(dataset_root / "meta"), task_episodes)
    _download_videos(dataset_root, selected)
    assets = output / "assets"
    policy_cache = cache / "policy"
    _copy_policy_assets(policy_cache, assets / "policy")
    _download_tokenizer(assets / "tokenizer")
    bddl = _bddl_text()
    if f"(In {SOUP} {BASKET_REGION})" not in bddl or f"(In {SAUCE} {BASKET_REGION})" not in bddl:
        raise StopProofError("BDDL goal does not require both objects in the basket")
    (assets / f"{SIM_TASK}.bddl").write_text(bddl, encoding="utf-8")
    gripper = _gripper_stats(
        assets / "policy" / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    )
    records = []
    images = []
    wrists = []
    states = []
    split_for = ["fit"] * 8 + ["tune"] + ["hold"]
    for split, row in zip(split_for, selected, strict=True):
        frames = list(row["frames"])
        decoded = _decode_episode(dataset_root, row, frames, fps)
        data_file = dataset_root / (
            f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
        )
        state_by_frame = _load_states(data_file, int(row["episode_index"]), frames)
        for frame in frames:
            records.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(frame),
                    "length": int(row["length"]),
                    "split": split,
                }
            )
            images.append(decoded[IMAGE_KEY][frames.index(frame)])
            wrists.append(decoded[IMAGE2_KEY][frames.index(frame)])
            states.append(torch.tensor(state_by_frame[frame], dtype=torch.float32))
    if [row["split"] for row in records].count("fit") != FIT_CONTEXTS:
        raise StopProofError("context split is not 64 fit, 8 tune, and 8 hold")
    save_file(
        {
            "image": torch.stack(images),
            "image2": torch.stack(wrists),
            "state": torch.stack(states),
        },
        str(output / "contexts.safetensors"),
    )
    _write_json(output / "contexts.json", {"contexts": records, "task": TASK_LANGUAGE})
    processor_files = [
        assets / "policy" / "policy_preprocessor.json",
        assets / "policy" / "policy_postprocessor.json",
        assets / "policy" / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        assets / "policy" / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    manifest = {
        "push_to_hub": False,
        "base_sha": BASE_SHA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "task_language": TASK_LANGUAGE,
        "task_index": task_index,
        "sim_task": SIM_TASK,
        "objects": {"soup": SOUP, "sauce": SAUCE, "region": BASKET_REGION},
        "seed": FIT_SEED,
        "demo_episode_indices": [int(row["episode_index"]) for row in selected],
        "splits": {"fit": 64, "tune": 8, "hold": 8},
        "frames_per_demo": FRAMES_PER_DEMO,
        "chunk_size": CHUNK_SIZE,
        "executed_actions": EXECUTED_ACTIONS,
        "teacher_draws": TEACHER_DRAWS,
        "teacher_steps": 10,
        "noise_seed": f"{NOISE_SEED_BASE} + context_index * 100 + draw",
        "fit_updates": FIT_UPDATES,
        "fit_batch": FIT_BATCH,
        "readout_time": 1.0,
        "contexts": "contexts.safetensors",
        "context_index": "contexts.json",
        "assets": "assets",
        "processor_sha256": _sha256_tree(processor_files),
        "contexts_sha256": _sha256_file(output / "contexts.safetensors"),
        "bddl_sha256": _sha256_file(assets / f"{SIM_TASK}.bddl"),
        "gripper": gripper,
        "licenses": {"model": "gemma", "dataset": "apache-2.0"},
        "simulation": {
            "init_states": list(INIT_STATES),
            "policy_seeds": list(POLICY_SEEDS),
            "control_steps": CONTROL_STEPS,
            "settle_steps": SETTLE_STEPS,
            "lift_meters": LIFT_METERS,
            "lift_frames": LIFT_FRAMES,
        },
    }
    _write_json(output / "manifest.json", manifest)
    emit(output, "prepare", contexts=len(records), episodes=manifest["demo_episode_indices"])


def _load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    if manifest.get("base_sha") != BASE_SHA or manifest.get("push_to_hub") is not False:
        raise StopProofError("manifest pin or push_to_hub flag does not match the approved proof")
    contexts = root / manifest["contexts"]
    if _sha256_file(contexts) != manifest["contexts_sha256"]:
        raise StopProofError("context file hash does not match the manifest")
    return manifest, root


def _load_contexts(
    root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    from safetensors.torch import load_file

    tensors = load_file(str(root / manifest["contexts"]))
    records = json.loads((root / manifest["context_index"]).read_text(encoding="utf-8"))["contexts"]
    if len(records) != tensors["image"].shape[0]:
        raise StopProofError("context index length does not match the tensor batch")
    return tensors, records


def _observation(tensors: dict[str, torch.Tensor], index: int, task: str) -> dict[str, Any]:
    return {
        IMAGE_KEY: tensors["image"][index].float() / 255.0,
        IMAGE2_KEY: tensors["image2"][index].float() / 255.0,
        STATE_KEY: tensors["state"][index].float(),
        "task": task,
    }


def _load_policy(assets: Path, manifest: dict[str, Any]):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.gaussian_readout import bind_processor_sha256
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor.factory import load_pretrained_policy_processors

    config = PI05Config.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=False)
    config.compile_model = False
    config.gradient_checkpointing = False
    config.n_action_steps = EXECUTED_ACTIONS
    config.push_to_hub = False
    config.device = "cuda"
    require_bfloat16(config.dtype)
    policy = PI05Policy(config)
    weight_file = hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    state = load_file(weight_file)
    remapped = policy._convert_openpi_state_dict(state)  # noqa: SLF001
    missing, unexpected = policy.load_state_dict(remapped, strict=True)
    del state, remapped
    if missing or unexpected:
        raise StopProofError(
            f"checkpoint load was not strict: missing={len(missing)} unexpected={len(unexpected)}"
        )
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    processors = load_pretrained_policy_processors(
        assets / "policy",
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(assets / "tokenizer")}},
    )
    bind_processor_sha256(policy.model, manifest["processor_sha256"])
    return policy, processors


def _model_batch(policy, batch: dict[str, Any]):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
    states, state_masks = policy._prepare_memory_states(batch)  # noqa: SLF001
    return (
        images,
        img_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        states,
        state_masks,
    )


def _teacher_noise(context_index: int, draw: int, max_action_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(NOISE_SEED_BASE + context_index * 100 + draw)
    noise = torch.randn((1, CHUNK_SIZE, max_action_dim), generator=generator, dtype=torch.float32)
    return noise.to(device)


def _decode_gripper(normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return normalized * std[GRIPPER_INDEX] + mean[GRIPPER_INDEX]


def _cache_teacher(
    policy, preprocessor, tensors, records, manifest, output: Path, clock: Clock
) -> dict[str, Any]:
    from lerobot.policies.pi05.gaussian_readout import (
        encode_prefix,
        flow_raw_action,
        make_epsilon_star,
        project_velocity,
        sample_euler_from_cache,
    )

    device = torch.device("cuda")
    mean = torch.tensor(manifest["gripper"]["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(manifest["gripper"]["std"], dtype=torch.float32, device=device)
    count = len(records)
    width = policy.model.action_out_proj.in_features
    features = torch.empty(count, CHUNK_SIZE, width, dtype=torch.float32)
    raw_motion = torch.empty(count, CHUNK_SIZE, MOTION_DIM, dtype=torch.float32)
    teacher_mean = torch.empty(count, CHUNK_SIZE, MOTION_DIM, dtype=torch.float32)
    teacher_var = torch.empty(count, CHUNK_SIZE, MOTION_DIM, dtype=torch.float32)
    gripper_frequency = torch.empty(count, CHUNK_SIZE, dtype=torch.float32)
    heldout_motion = []
    probe = make_epsilon_star(CHUNK_SIZE, policy.config.max_action_dim)
    verified = False
    for index, record in enumerate(records):
        clock.ensure(f"teacher cache context {index}")
        batch = preprocessor(_observation(tensors, index, manifest["task_language"]))
        images, img_masks, tokens, masks, states, state_masks = _model_batch(policy, batch)
        with torch.inference_mode():
            prefix_pad, past = encode_prefix(
                policy.model, images, img_masks, tokens, masks, states, state_masks
            )
            if index == 0:
                noise = _teacher_noise(0, 0, policy.config.max_action_dim, device)
                cached = sample_euler_from_cache(
                    policy.model, prefix_pad, past, noise, manifest["teacher_steps"]
                )
                ordinary = policy.model.sample_actions(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    states=states,
                    state_masks=state_masks,
                    noise=noise,
                    num_steps=10,
                )
                difference = (cached - ordinary).abs().max().item()
                verified = difference <= 1e-3
                emit(output, "prefix-cache-check", max_abs_diff=difference, passed=verified)
                if not verified:
                    raise StopProofError(f"cached prefix disagrees with the ordinary sampler by {difference}")
            motion_sum = torch.zeros(CHUNK_SIZE, MOTION_DIM, device=device)
            motion_sq = torch.zeros(CHUNK_SIZE, MOTION_DIM, device=device)
            close_count = torch.zeros(CHUNK_SIZE, device=device)
            stored = []
            for draw in range(TEACHER_DRAWS):
                noise = _teacher_noise(index, draw, policy.config.max_action_dim, device)
                action = sample_euler_from_cache(
                    policy.model, prefix_pad, past, noise, manifest["teacher_steps"]
                )
                motion = action[0, :, :MOTION_DIM].float()
                motion_sum += motion
                motion_sq += motion.square()
                decoded = _decode_gripper(action[0, :, GRIPPER_INDEX].float(), mean, std)
                close_count += (decoded > manifest["gripper"]["close_if_decoded_gt"]).float()
                if record["split"] == "hold":
                    stored.append(motion.cpu())
            timestep = torch.ones(tokens.shape[0], dtype=torch.float32, device=device)
            expanded = probe.to(device).expand(tokens.shape[0], -1, -1)
            feature = policy.model.suffix_features(prefix_pad, past, expanded, timestep)
            raw = flow_raw_action(expanded, project_velocity(policy.model, feature))
        features[index] = feature[0].cpu()
        raw_motion[index] = raw[0, :, :MOTION_DIM].cpu()
        teacher_mean[index] = (motion_sum / TEACHER_DRAWS).cpu()
        population = motion_sq / TEACHER_DRAWS - (motion_sum / TEACHER_DRAWS).square()
        teacher_var[index] = population.clamp_min(1e-6).cpu()
        gripper_frequency[index] = (close_count / TEACHER_DRAWS).cpu()
        if stored:
            heldout_motion.append(torch.stack(stored))
        del past
        if index % 8 == 7:
            emit(output, "teacher-cache-progress", contexts=index + 1, of=count)
    return {
        "features": features,
        "raw_motion": raw_motion,
        "teacher_mean": teacher_mean,
        "teacher_var": teacher_var,
        "gripper_frequency": gripper_frequency,
        "heldout_teacher_motion": torch.stack(heldout_motion) if heldout_motion else torch.empty(0),
        "width": width,
    }


def _split_index(records: list[dict[str, Any]], name: str) -> list[int]:
    return [index for index, record in enumerate(records) if record["split"] == name]


def _fit_readout(policy, cache: dict[str, Any], records, manifest, output: Path) -> Path:
    from lerobot.policies.pi05.gaussian_readout import (
        GaussianReadout,
        fit_gripper_head,
        fit_variance_head,
        gripper_loss,
        normalized_command,
        save_readout,
        solve_affine_residual,
        state_dict_sha256,
        variance_loss,
    )

    device = torch.device("cuda")
    before = state_dict_sha256(policy.model)
    fit_rows = _split_index(records, "fit")
    tune_rows = _split_index(records, "tune")
    features = cache["features"].to(device)
    raw = cache["raw_motion"].to(device)
    teacher_mean = cache["teacher_mean"].to(device)
    weight, bias = solve_affine_residual(features[fit_rows], raw[fit_rows], teacher_mean[fit_rows])
    gripper = manifest["gripper"]
    mean = torch.tensor(gripper["mean"], dtype=torch.float32)
    std = torch.tensor(gripper["std"], dtype=torch.float32)
    readout = GaussianReadout(
        cache["width"],
        chunk_size=CHUNK_SIZE,
        motion_dim=MOTION_DIM,
        gripper_index=GRIPPER_INDEX,
        max_action_dim=policy.config.max_action_dim,
        action_dim=ACTION_DIM,
        open_normalized=float(
            normalized_command(gripper["open_command"], mean[GRIPPER_INDEX], std[GRIPPER_INDEX])
        ),
        close_normalized=float(
            normalized_command(gripper["close_command"], mean[GRIPPER_INDEX], std[GRIPPER_INDEX])
        ),
        action_mean=mean[:ACTION_DIM],
        action_std=std[:ACTION_DIM],
    ).to(device)
    with torch.no_grad():
        readout.affine_weight.copy_(weight)
        readout.affine_bias.copy_(bias)
    variance_selected = fit_variance_head(
        readout,
        features[fit_rows],
        cache["teacher_var"].to(device)[fit_rows],
        features[tune_rows],
        cache["teacher_var"].to(device)[tune_rows],
        updates=FIT_UPDATES,
        batch_size=FIT_BATCH,
        seed=FIT_SEED,
    )
    gripper_selected = fit_gripper_head(
        readout,
        features[fit_rows],
        cache["gripper_frequency"].to(device)[fit_rows],
        features[tune_rows],
        cache["gripper_frequency"].to(device)[tune_rows],
        updates=FIT_UPDATES,
        batch_size=FIT_BATCH,
        seed=FIT_SEED,
    )
    after = state_dict_sha256(policy.model)
    if before != after:
        raise StopProofError("teacher weights changed while the readout was fit")
    path = output / "readout.safetensors"
    save_readout(readout.cpu(), path, teacher_sha256=before, processor_sha256=manifest["processor_sha256"])
    scale = cache["teacher_var"][fit_rows].mean().pow(2).clamp_min(1e-6)
    with torch.no_grad():
        tune_variance = float(
            variance_loss(
                readout.variance(features[tune_rows].cpu()),
                cache["teacher_var"][tune_rows],
                scale,
            )
        )
        tune_gripper = float(
            gripper_loss(
                readout.gripper_latent(features[tune_rows].cpu()), cache["gripper_frequency"][tune_rows]
            )
        )
    emit(
        output,
        "fit",
        teacher_sha256=before,
        variance_update=variance_selected,
        gripper_update=gripper_selected,
        tune_variance_loss=tune_variance,
        tune_gripper_bce=tune_gripper,
    )
    return path


def _heldout_report(policy, readout_path: Path, cache, records, manifest, output: Path) -> dict[str, Any]:
    from lerobot.policies.pi05.gaussian_readout import (
        gripper_loss,
        load_gaussian_readout,
        variance_loss,
    )

    rows = _split_index(records, "hold")
    device = torch.device("cuda")
    readout = load_gaussian_readout(readout_path, policy.model, manifest["processor_sha256"]).to(device)
    features = cache["features"][rows].to(device)
    raw = cache["raw_motion"][rows].to(device)
    teacher_mean = cache["teacher_mean"][rows].to(device)
    teacher_var = cache["teacher_var"][rows].to(device)
    frequency = cache["gripper_frequency"][rows].to(device)
    scale = cache["teacher_var"][_split_index(records, "fit")].mean().pow(2).clamp_min(1e-6).to(device)
    with torch.no_grad():
        predicted_mean = readout.continuous_mean(features, raw)
        mean_mse = float((predicted_mean - teacher_mean).pow(2).mean())
        predicted_var = readout.variance(features)
        variance_mse = float(variance_loss(predicted_var, teacher_var, scale))
        sd_mse = float((predicted_var.sqrt() - teacher_var.sqrt()).pow(2).mean())
        bce = float(gripper_loss(readout.gripper_latent(features), frequency))
        torch.manual_seed(FIT_SEED)
        samples = torch.stack(
            [readout.sample_action(features, raw)[:, :, :MOTION_DIM].cpu() for _ in range(TEACHER_DRAWS)]
        )
    action_min = torch.tensor(manifest["gripper"]["min"][:MOTION_DIM])
    action_max = torch.tensor(manifest["gripper"]["max"][:MOTION_DIM])
    mean = torch.tensor(manifest["gripper"]["mean"][:MOTION_DIM])
    std = torch.tensor(manifest["gripper"]["std"][:MOTION_DIM])
    decoded = samples * std + mean
    outside = (decoded < action_min) | (decoded > action_max)
    clipping = float(outside.float().mean())
    teacher_samples = cache["heldout_teacher_motion"].float()
    _write_pca_svg(
        teacher_samples.reshape(-1, CHUNK_SIZE * MOTION_DIM),
        samples.permute(1, 0, 2, 3).reshape(-1, CHUNK_SIZE * MOTION_DIM),
        output / "heldout_pca.svg",
    )
    report = {
        "label": "Measured",
        "heldout_mean_mse": mean_mse,
        "heldout_variance_loss": variance_mse,
        "heldout_sd_mse": sd_mse,
        "heldout_gripper_bce": bce,
        "clipping_rate": clipping,
        "plot": "heldout_pca.svg",
    }
    emit(output, "heldout", **report)
    return report


def _write_pca_svg(teacher: torch.Tensor, student: torch.Tensor, path: Path) -> None:
    both = torch.cat([teacher, student], dim=0).float()
    centered = both - both.mean(dim=0)
    _, _, axes = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    span = projected.abs().max().clamp_min(1e-6)
    points = projected / span

    def circles(block: torch.Tensor, color: str) -> str:
        markup = []
        for x_value, y_value in block.tolist():
            markup.append(
                f'<circle cx="{400 + 180 * x_value:.2f}" cy="{300 - 180 * y_value:.2f}" r="2.2" fill="{color}"/>'
            )
        return "\n".join(markup)

    teacher_points = points[: teacher.shape[0]]
    student_points = points[teacher.shape[0] :]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
<rect width="800" height="600" fill="white"/>
<text x="24" y="32" font-family="sans-serif" font-size="16">Held-out motion, first two principal components</text>
<text x="24" y="56" font-family="sans-serif" font-size="14" fill="#1d4e89">Teacher</text>
<text x="110" y="56" font-family="sans-serif" font-size="14" fill="#c44900">GAR</text>
{circles(teacher_points, "#1d4e89")}
{circles(student_points, "#c44900")}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _sync() -> None:
    torch.cuda.synchronize()


def _time_paths(
    policy, processors, tensors, records, manifest, readout_path: Path, output: Path, clock: Clock
):
    preprocessor, postprocessor = processors
    rows = _split_index(records, "hold")
    device = torch.device("cuda")
    policy.config.gaussian_readout_path = None
    for _ in range(5):
        for index in rows:
            clock.ensure("timing warmup")
            _time_one(policy, preprocessor, postprocessor, tensors, index, manifest, "teacher", device)
            policy.config.gaussian_readout_path = str(readout_path)
            _time_one(policy, preprocessor, postprocessor, tensors, index, manifest, "gar", device)
            policy.config.gaussian_readout_path = None
    walls = {"teacher": [], "gar": [], "teacher_one_step": []}
    prefixes = {"teacher": [], "gar": []}
    actions = {"teacher": [], "gar": []}
    torch.cuda.reset_peak_memory_stats()
    for measurement in range(20):
        order = ("teacher", "gar") if measurement % 2 == 0 else ("gar", "teacher")
        for name in order:
            for index in rows:
                clock.ensure("timing")
                policy.config.gaussian_readout_path = None if name == "teacher" else str(readout_path)
                sample = _time_one(
                    policy, preprocessor, postprocessor, tensors, index, manifest, name, device
                )
                walls[name].append(sample["wall"])
                prefixes[name].append(sample["prefix"])
                actions[name].append(sample["action"])
    policy.config.gaussian_readout_path = None
    for index in rows:
        for _ in range(5):
            _time_one(
                policy, preprocessor, postprocessor, tensors, index, manifest, "teacher", device, num_steps=1
            )
        for _ in range(20):
            clock.ensure("one-step timing")
            sample = _time_one(
                policy, preprocessor, postprocessor, tensors, index, manifest, "teacher", device, num_steps=1
            )
            walls["teacher_one_step"].append(sample["wall"])
    teacher_median = _median(walls["teacher"])
    gar_median = _median(walls["gar"])
    report = {
        "label": "Measured",
        "samples_per_path": len(walls["teacher"]),
        "teacher_median_s": teacher_median,
        "teacher_p95_s": _percentile(walls["teacher"], 0.95),
        "gar_median_s": gar_median,
        "gar_p95_s": _percentile(walls["gar"], 0.95),
        "speed_ratio_teacher_over_gar": teacher_median / gar_median,
        "teacher_prefix_median_s": _median(prefixes["teacher"]),
        "teacher_action_median_s": _median(actions["teacher"]),
        "gar_prefix_median_s": _median(prefixes["gar"]),
        "gar_action_median_s": _median(actions["gar"]),
        "teacher_one_step_median_s": _median(walls["teacher_one_step"]),
        "teacher_one_step_p95_s": _percentile(walls["teacher_one_step"], 0.95),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "percentile": "ceil(p * n) - 1 on the sorted sample",
    }
    emit(output, "timing", **report)
    return report


def _time_one(policy, preprocessor, postprocessor, tensors, index, manifest, name, device, num_steps=None):
    from lerobot.policies.pi05.gaussian_readout import (
        encode_prefix,
        flow_raw_action,
        make_epsilon_star,
        project_velocity,
        sample_euler_from_cache,
    )

    observation = _observation(tensors, index, manifest["task_language"])
    _sync()
    started = time.perf_counter()
    batch = preprocessor(observation)
    images, img_masks, tokens, masks, states, state_masks = _model_batch(policy, batch)
    _sync()
    prefix_started = time.perf_counter()
    prefix_pad, past = encode_prefix(policy.model, images, img_masks, tokens, masks, states, state_masks)
    _sync()
    action_started = time.perf_counter()
    with torch.inference_mode():
        if name == "gar":
            probe = make_epsilon_star(CHUNK_SIZE, policy.config.max_action_dim).to(device)
            timestep = torch.ones(tokens.shape[0], dtype=torch.float32, device=device)
            features = policy.model.suffix_features(
                prefix_pad, past, probe.expand(tokens.shape[0], -1, -1), timestep
            )
            raw = flow_raw_action(
                probe.expand(tokens.shape[0], -1, -1), project_velocity(policy.model, features)
            )
            from lerobot.policies.pi05.gaussian_readout import _cached_readout

            action = (
                _cached_readout(policy.model, policy.config.gaussian_readout_path)
                .to(device)
                .sample_action(features, raw)
            )
        else:
            steps = policy.config.num_inference_steps if num_steps is None else num_steps
            noise = policy.model.sample_noise(
                (tokens.shape[0], CHUNK_SIZE, policy.config.max_action_dim), device
            )
            action = sample_euler_from_cache(policy.model, prefix_pad, past, noise, steps)
    action = action[:, :EXECUTED_ACTIONS, :ACTION_DIM].reshape(-1, ACTION_DIM)
    _sync()
    action_ended = time.perf_counter()
    decoded = postprocessor(action)
    _sync()
    ended = time.perf_counter()
    if decoded.shape[-1] != ACTION_DIM:
        raise StopProofError(f"decoded action dim is {decoded.shape[-1]}")
    return {
        "wall": ended - started,
        "prefix": action_started - prefix_started,
        "action": action_ended - action_started,
    }


def _batch_tree(value):
    if isinstance(value, dict):
        return {key: _batch_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim == 1:
        return value.unsqueeze(0)
    return value


def _problem_env(env):
    current = getattr(env, "_env", None)
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_eval_predicate") and hasattr(current, "obj_body_id"):
            return current
        current = getattr(current, "env", None)
    raise StopProofError("LIBERO problem environment was not found")


def _body_z(problem, name: str) -> float:
    body_id = problem.obj_body_id[name]
    return float(problem.sim.data.body_xpos[body_id][2])


def _in_basket(problem, name: str) -> bool:
    return bool(problem._eval_predicate(["In", name, BASKET_REGION]))  # noqa: SLF001


def _write_video(path: Path, frames: list) -> None:
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=20)
    height, width = frames[0].shape[:2]
    stream.height = height
    stream.width = width
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        array = np.ascontiguousarray(frame)
        video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _rollout(
    env, policy, processors, manifest, seed: int, init_state: int, use_gar: bool, readout_path: Path
):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep

    preprocessor, postprocessor = processors
    env_processor = LiberoProcessorStep()
    policy.config.gaussian_readout_path = str(readout_path) if use_gar else None
    policy.reset()
    torch.manual_seed(seed)
    env.init_state_id = init_state
    observation, _info = env.reset(seed=seed)
    problem = _problem_env(env)
    reset_z = {SOUP: _body_z(problem, SOUP), SAUCE: _body_z(problem, SAUCE)}
    lift_run = {SOUP: 0, SAUCE: 0}
    first_lift = {SOUP: None, SAUCE: None}
    first_in = {SOUP: None, SAUCE: None}
    identity = "none"
    switches = 0
    frames = []
    success = False
    success_step = None
    task = env.task_description
    for step in range(1, CONTROL_STEPS + 1):
        frames.append(observation["pixels"]["image"].copy())
        processed = preprocess_observation(observation)
        processed["observation.robot_state"] = _batch_tree(processed["observation.robot_state"])
        processed["task"] = task
        processed = env_processor.observation(processed)
        batch = preprocessor(processed)
        with torch.inference_mode():
            action = policy.select_action(batch)
        decoded = postprocessor(action)
        numpy_action = decoded.detach().cpu().numpy()
        if numpy_action.ndim == 2:
            numpy_action = numpy_action[0]
        observation, _reward, terminated, _truncated, info = env.step(numpy_action)
        soup_up = _body_z(problem, SOUP) - reset_z[SOUP] >= LIFT_METERS
        sauce_up = _body_z(problem, SAUCE) - reset_z[SAUCE] >= LIFT_METERS
        for name, up in ((SOUP, soup_up), (SAUCE, sauce_up)):
            lift_run[name] = lift_run[name] + 1 if up else 0
            if first_lift[name] is None and lift_run[name] >= LIFT_FRAMES:
                first_lift[name] = step
            if first_in[name] is None and _in_basket(problem, name):
                first_in[name] = step
        current = "both" if soup_up and sauce_up else "soup" if soup_up else "sauce" if sauce_up else "none"
        if identity == "none" and current != "none":
            identity = current
        elif current != identity and identity != "none":
            switches += 1
            identity = current
        if info.get("is_success"):
            success = True
            success_step = step
            frames.append(observation["pixels"]["image"].copy())
            break
        if terminated:
            break
    if first_in[SOUP] is not None and first_in[SAUCE] is not None:
        outcome = "soup_then_sauce" if first_in[SOUP] <= first_in[SAUCE] else "sauce_then_soup"
    elif success:
        outcome = "failed"
    else:
        outcome = "neither"
    return {
        "success": success,
        "completion_length": success_step if success_step is not None else step,
        "first_lift": first_lift,
        "first_in_basket": first_in,
        "switches": switches,
        "outcome": outcome,
        "frames": frames,
        "task_description": task,
    }


def _simulate(policy, processors, manifest, readout_path: Path, output: Path, clock: Clock) -> dict[str, Any]:
    write_libero_config()
    import libero.libero.benchmark as benchmark

    from lerobot.envs.libero import LiberoEnv
    from lerobot.policies.pi05.gaussian_readout import collapse_label

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_id = None
    for index in range(suite.n_tasks):
        if suite.get_task(index).name == SIM_TASK:
            task_id = index
            break
    if task_id is None:
        raise StopProofError(f"LIBERO-Long does not contain {SIM_TASK}")
    bddl_name = suite.get_task(task_id).bddl_file
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name="libero_10",
        episode_length=CONTROL_STEPS,
        camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=True,
        episode_index=0,
        n_envs=1,
        num_steps_wait=SETTLE_STEPS,
        control_mode="relative",
        hard_reset=True,
    )
    if env.task != SIM_TASK:
        raise StopProofError(f"resolved simulator task {env.task} is not {SIM_TASK}")
    rows = []
    pair_index = 0
    try:
        for init_state in INIT_STATES:
            for seed in POLICY_SEEDS:
                clock.ensure(f"simulation pair {pair_index}")
                order = ("teacher", "gar") if pair_index % 2 == 0 else ("gar", "teacher")
                pair = {"init_state": init_state, "seed": seed, "order": list(order), "episodes": []}
                for name in order:
                    episode = _rollout(
                        env,
                        policy,
                        processors,
                        manifest,
                        seed,
                        init_state,
                        use_gar=name == "gar",
                        readout_path=readout_path,
                    )
                    video = None
                    if episode["success"]:
                        video_path = (
                            output
                            / "videos"
                            / f"{name}_state{init_state}_seed{seed}_{episode['outcome']}.mp4"
                        )
                        _write_video(video_path, episode["frames"])
                        video = str(video_path.name)
                    stored = {key: value for key, value in episode.items() if key != "frames"}
                    stored.update({"policy": name, "init_state": init_state, "seed": seed, "video": video})
                    rows.append(stored)
                    pair["episodes"].append(stored)
                _write_json(output / "pairs.json", {"pairs_complete": pair_index + 1, "episodes": rows})
                emit(
                    output,
                    "sim-pair",
                    pair=pair_index,
                    init_state=init_state,
                    seed=seed,
                    episodes=pair["episodes"],
                )
                pair_index += 1
    finally:
        env.close()
    per_state = {}
    labels = []
    for init_state in INIT_STATES:
        teacher_orders = {
            row["outcome"]
            for row in rows
            if row["policy"] == "teacher"
            and row["init_state"] == init_state
            and row["success"]
            and row["outcome"] in ORDERS
        }
        gar_orders = {
            row["outcome"]
            for row in rows
            if row["policy"] == "gar"
            and row["init_state"] == init_state
            and row["success"]
            and row["outcome"] in ORDERS
        }
        label = collapse_label(teacher_orders, gar_orders)
        labels.append(label)
        per_state[str(init_state)] = {
            "teacher_orders": sorted(teacher_orders),
            "gar_orders": sorted(gar_orders),
            "label": label,
        }
    if "collapse-observed" in labels:
        overall = "collapse-observed"
    elif "both-covered" in labels:
        overall = "both-covered"
    else:
        overall = "inconclusive"
    paired = []
    for init_state in INIT_STATES:
        for seed in POLICY_SEEDS:
            teacher = next(
                row
                for row in rows
                if row["policy"] == "teacher" and row["init_state"] == init_state and row["seed"] == seed
            )
            gar = next(
                row
                for row in rows
                if row["policy"] == "gar" and row["init_state"] == init_state and row["seed"] == seed
            )
            paired.append(
                {
                    "init_state": init_state,
                    "seed": seed,
                    "teacher_success": teacher["success"],
                    "gar_success": gar["success"],
                    "success_difference": int(teacher["success"]) - int(gar["success"]),
                }
            )
    report = {
        "label": "Simulation",
        "bddl_file": bddl_name,
        "task_description": rows[0]["task_description"] if rows else None,
        "episodes": rows,
        "per_init_state": per_state,
        "two_mode": overall,
        "paired_success": paired,
        "failures": [row for row in rows if not row["success"] or row["outcome"] == "failed"],
    }
    emit(output, "simulation", two_mode=overall, episodes=len(rows))
    return report


def run_job(manifest_path: Path, output: Path, wall_seconds: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise StopProofError("the approved proof requires CUDA")
    clock = Clock(wall_seconds)
    manifest, root = _load_manifest(manifest_path)
    tensors, records = _load_contexts(root, manifest)
    emit(output, "load-start", wall_seconds=wall_seconds, contexts=len(records))
    clock.ensure("policy load")
    policy, processors = _load_policy(root / manifest["assets"], manifest)
    torch.cuda.reset_peak_memory_stats()
    emit(
        output,
        "policy-loaded",
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    cache = _cache_teacher(policy, processors[0], tensors, records, manifest, output, clock)
    from safetensors.torch import save_file

    save_file(
        {key: value.contiguous() for key, value in cache.items() if isinstance(value, torch.Tensor)},
        str(output / "teacher_cache.safetensors"),
    )
    emit(output, "teacher-cache", contexts=len(records), width=cache["width"])
    clock.ensure("fit")
    readout_path = _fit_readout(policy, cache, records, manifest, output)
    heldout = _heldout_report(policy, readout_path, cache, records, manifest, output)
    clock.ensure("timing")
    timing = _time_paths(policy, processors, tensors, records, manifest, readout_path, output, clock)
    clock.ensure("simulation")
    simulation = _simulate(policy, processors, manifest, readout_path, output, clock)
    results = {
        "push_to_hub": False,
        "base_sha": BASE_SHA,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "heldout": heldout,
        "timing": timing,
        "simulation": simulation,
        "two_mode": simulation["two_mode"],
    }
    _write_json(output / "results.json", results)
    emit(output, "finished", two_mode=simulation["two_mode"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--wall-seconds", type=int, default=3300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.prepare_only:
            cache = args.cache or (args.output / "cache")
            prepare(args.output, cache)
            return
        if args.manifest is None:
            raise StopProofError("--manifest is required for the GPU proof")
        run_job(args.manifest, args.output, args.wall_seconds)
    except BudgetExhaustedError as exc:
        emit(args.output, "budget-exhausted", error=str(exc))
        raise SystemExit(3) from exc
    except StopProofError as exc:
        emit(args.output, "stopped", error=str(exc))
        raise SystemExit(2) from exc
    except torch.cuda.OutOfMemoryError as exc:
        emit(args.output, "oom", error=str(exc))
        raise SystemExit(2) from exc
    except Exception as exc:
        emit(args.output, "error", error=str(exc), traceback=traceback.format_exc())
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
