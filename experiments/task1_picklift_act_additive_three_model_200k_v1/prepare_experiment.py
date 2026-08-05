from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from lerobot.datasets import LeRobotDataset, MatchedTwoStreamSampler
from lerobot.datasets.dataset_tools import merge_datasets, split_dataset

EXP = "task1_picklift_act_additive_three_model_200k_v1"
REPO = Path("/home/ubuntu24/Teleop/lerobot")
HERE = REPO / "experiments" / EXP
ART = Path("/home/ubuntu24/Teleop/artifacts")
REAL24 = ART / "derived/task1_picklift_real24_budget_extension_v1/accepted"
REAL48 = ART / "derived/task1_picklift_real48_accepted_v1/accepted"
SIM24 = Path("/home/ubuntu24/SO101QuestLocalSim-data/postcollection/task1-localsim48-gridphase0-v2-s01-finalization-v1/derived/sim24_gap/dataset")
OLD_SIM_COMBINED = ART / "datasets/task1_picklift_real24_localsim48_gap_recovery_act_v1/real24_sim24_gap/combined48_v1"
DATA_ROOT = ART / f"datasets/{EXP}"
EVIDENCE = ART / f"evidence/{EXP}/software_gate_v1"
TRAIN_ROOT = ART / f"training/{EXP}"

EXPECTED = {
    "real24": "c01c45f9dcaee557248bff997f3c244a9fdba2b6c13211821ee335d4bfee0712",
    "real48": "c4534befc536c10217638da91f5cbbaff59b0795ec91f0633e53e8a6d99507b9",
    "sim24": "7f3b0ada0525fa1c358179f9c8823c877f1d81f88569253eaaecd81322531b22",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def legacy_tree(root: Path) -> str:
    rows = [f"{sha(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(root.rglob("*")) if p.is_file()]
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def lp_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(8, "big")); h.update(rel)
        h.update(p.stat().st_size.to_bytes(8, "big")); h.update(bytes.fromhex(sha(p)))
    return h.hexdigest()


def derived_tree(root: Path) -> str:
    rows = [f"{sha(p)}  {p.stat().st_size}  {p.relative_to(root).as_posix()}\n" for p in sorted(root.rglob("*")) if p.is_file()]
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def loader(root: Path, repo_id: str, episodes: int, frames: int) -> LeRobotDataset:
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    assert ds.meta.total_episodes == episodes and len(ds) == frames and ds.meta.fps == 20
    for i in (0, frames - 1):
        s = ds[i]
        assert s["observation.state"].shape == (6,) and s["action"].shape == (6,)
        assert s["observation.images.front"].shape == (3, 480, 640)
        assert all(torch.isfinite(s[k]).all() for k in ("observation.state", "action", "observation.images.front"))
    return ds


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    if legacy_tree(REAL24) != EXPECTED["real24"] or legacy_tree(REAL48) != EXPECTED["real48"]:
        raise RuntimeError("Real source tree mismatch")
    if lp_tree(SIM24) != EXPECTED["sim24"]:
        raise RuntimeError("LocalSim source tree mismatch")
    r24 = loader(REAL24, "local/task1_picklift_real24_budget_extension_v1_accepted", 24, 4263)
    r48 = loader(REAL48, "local/task1_picklift_real48_accepted_v1", 48, 8955)
    loader(SIM24, "local/task1_picklift_localsim24_gap_gridphase0_v0420", 24, 1933)

    gap_manifest = json.loads((HERE / "research_control/task1-picklift-real48-minus-real24-gap24-v1.json").read_text())
    frozen_ids = gap_manifest["ordered_plan_item_ids"]
    r48_rows = read_jsonl(REAL48 / "provenance/source_episode_map.jsonl")
    id_to_episode = {row["plan_item_id"]: episode for episode, row in enumerate(r48_rows)}
    # The provenance ledger is in accepted Dataset order. Its embedded source
    # and derived indices refer to earlier source/session trees and are sparse.
    gap_indices = [id_to_episode[item] for item in frozen_ids]
    if len(set(gap_indices)) != 24:
        raise RuntimeError("Frozen Real-gap membership does not map to 24 unique episodes")
    r24_ids = {row["plan_item_id"] for row in read_jsonl(REAL24 / "provenance/source_episode_map.jsonl")}
    if r24_ids & set(frozen_ids) or r24_ids | set(frozen_ids) != set(id_to_episode):
        raise RuntimeError("Real24/Real-gap disjoint-union identity failed")

    real_combined = DATA_ROOT / "real24_realgap24/combined48_v1"
    sim_combined = DATA_ROOT / "real24_localsim24gap/combined48_v1"
    if not real_combined.exists():
        split_root = DATA_ROOT / "_real48_gap_split"
        gap = split_dataset(r48, {"gap24": gap_indices}, output_dir=split_root)["gap24"]
        merge_datasets([r24, gap], "local/task1_picklift_real24_realgap24_additive_v1", real_combined,
                       concatenate_videos=False, concatenate_data=False)
        shutil.copy2(REAL24 / "meta/stats.json", real_combined / "meta/stats.json")
        shutil.rmtree(split_root)
    if not sim_combined.exists():
        shutil.copytree(OLD_SIM_COMBINED, sim_combined)
        shutil.copy2(REAL24 / "meta/stats.json", sim_combined / "meta/stats.json")

    real_gap_frames = sum(int(row.length) for row in pd.concat([
        pd.read_parquet(p, columns=["episode_index", "length"])
        for p in sorted((REAL48 / "meta/episodes").glob("*/*.parquet"))
    ]).itertuples() if int(row.episode_index) in set(gap_indices))
    real_ds = loader(real_combined, "local/task1_picklift_real24_realgap24_additive_v1", 48, 4263 + real_gap_frames)
    sim_ds = loader(sim_combined, "local/task1_picklift_real24_localsim24gap_additive_v1", 48, 6196)
    stats_sha = sha(REAL24 / "meta/stats.json")
    assert sha(real_combined / "meta/stats.json") == stats_sha == sha(sim_combined / "meta/stats.json")

    template = json.loads((REPO / "experiments/task1_picklift_real24_budget_extension_act_v1/train_config_full.json").read_text())
    conditions = {
        "r24_repeat": (REAL24, "local/task1_picklift_real24_budget_extension_v1_accepted", [*range(24)], [*range(24)]),
        "r24_realgap24": (real_combined, "local/task1_picklift_real24_realgap24_additive_v1", [*range(24)], [*range(24, 48)]),
        "r24_localsim24gap": (sim_combined, "local/task1_picklift_real24_localsim24gap_additive_v1", [*range(24)], [*range(24, 48)]),
    }
    configs = {}
    stream0_sequences = []
    for name, (root, repo_id, a, b) in conditions.items():
        ds = r24 if name == "r24_repeat" else (real_ds if name == "r24_realgap24" else sim_ds)
        base = json.loads(json.dumps(template))
        base["dataset"].update({
            "repo_id": repo_id, "root": str(root), "episodes": sorted(set(a + b)),
            "matched_two_stream_episode_groups": {"real24": a, "source_b": b},
            "matched_two_stream_batches_per_epoch": 1000,
        })
        base["steps"] = 200000; base["save_freq"] = 20000
        base["output_dir"] = str(TRAIN_ROOT / name / "full_200k")
        base["job_name"] = f"{EXP}_{name}_seed1000_step200000"
        for kind, steps in (("smoke", 500), ("full", 200000)):
            cfg = json.loads(json.dumps(base)); cfg["steps"] = steps
            cfg["save_freq"] = 500 if kind == "smoke" else 20000
            cfg["output_dir"] = str(TRAIN_ROOT / name / ("smoke_500" if kind == "smoke" else "full_200k"))
            path = HERE / "configs" / f"{name}_{kind}.json"; write(path, cfg)
            configs[f"{name}_{kind}"] = {"path": str(path), "sha256": sha(path)}
        sampler = MatchedTwoStreamSampler(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"],
            ds.meta.episodes["episode_index"], {"real24": a, "source_b": b}, 8, 1000,
            episode_indices_to_use=ds.episodes, seed=1000, absolute_to_relative_idx=ds.absolute_to_relative_idx)
        order = list(sampler)
        assert all(len(order[i:i+4]) == 4 and len(order[i+4:i+8]) == 4 for i in range(0, len(order), 8))
        stream0_sequences.append([order[i:i+4] for i in range(0, len(order), 8)])
    assert stream0_sequences[0] == stream0_sequences[1] == stream0_sequences[2]

    result = {
        "status": "software_gate_pass_ready_for_smokes",
        "source_trees": EXPECTED,
        "real_gap_episode_indices": gap_indices,
        "real_gap_plan_item_ids": frozen_ids,
        "real_gap_frames": real_gap_frames,
        "real24_stats_sha256": stats_sha,
        "derived": {
            "real_additive": {"root": str(real_combined), "tree_sha256": derived_tree(real_combined), "episodes": 48, "frames": 4263 + real_gap_frames},
            "sim_additive": {"root": str(sim_combined), "tree_sha256": derived_tree(sim_combined), "episodes": 48, "frames": 6196},
        },
        "sampler": {"batches_per_epoch": 1000, "batch": "4 real24 + 4 source_b", "stream0_shared": True,
                    "formal_sample_slots_each_stream": 800000},
        "configs": configs,
        "hardware_accessed": False,
    }
    write(EVIDENCE / "preflight.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
