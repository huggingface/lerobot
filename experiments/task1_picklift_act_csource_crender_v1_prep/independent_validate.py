from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from lerobot.datasets import LeRobotDataset

HERE = Path(__file__).resolve().parent
ART = Path("/home/ubuntu24/Teleop/artifacts")
BINDING = ART / "evidence/task1_picklift_act_csource_crender_v1/binding_v1/binding_result.json"
TRAINING = HERE / "training_result_v1.json"
PLAN = HERE / "bound_evaluation_plan.json"
GATE = ART / "evaluation/task1_picklift_act_csource_vs_crender_eval24_v1/software_preparation_v1"
DRY = GATE / "dry_run.json"
MANIFEST = GATE / "manifest.json"
DATASET = ART / "datasets/task1_picklift_act_csource_crender_v1/combined48_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_tree(root: Path) -> tuple[str, int, int]:
    rows = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total += size
        rows.append(f"{sha(path)}  {size}  {path.relative_to(root).as_posix()}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest(), len(rows), total


def main() -> None:
    output = GATE / "independent_validation.json"
    if output.exists():
        raise FileExistsError(output)
    binding = json.loads(BINDING.read_text())
    training = json.loads(TRAINING.read_text())
    plan = json.loads(PLAN.read_text())
    dry = json.loads(DRY.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if binding["status"] != "bound_ready_for_authorized_smoke_then_fresh_200k":
        raise RuntimeError("binding status mismatch")
    expected_streams = {
        "full": "2251e609746b8d317366b07e0bab3aa636ea4fadd197590196ce1c8b3c5367d2",
        "real24": "f392d7b148905d90467a2565229df92d33e7805e8037a48eeca02c6d31730c53",
        "sim": "909fdfa48f61f11c09e773c954aa1ff843dac08ea3fda336a54ce84573166c28",
    }
    if binding["sampling_digest"] != expected_streams:
        raise RuntimeError("sampling stream identity mismatch")
    if training["selected_step"] != 200000 or training["model"]["model_sha256"] != "4ea775e8cab32d7959ba32e1dde6072185227dd6eab392b29acd139f40a67b53":
        raise RuntimeError("fixed model identity mismatch")
    if training["formal"]["sampling_counts"]["actual_samples_seen_by_main_process"] != {"real24": 800000, "source_b": 800000}:
        raise RuntimeError("sample count mismatch")
    dataset = LeRobotDataset("local/task1_picklift_real24_localsim24gap_rerender_additive_v1", root=DATASET, video_backend="pyav")
    if dataset.meta.total_episodes != 48 or len(dataset) != 6196 or dataset.meta.fps != 20:
        raise RuntimeError("combined Dataset mismatch")
    tree, files, size = derived_tree(DATASET)
    trials = plan["trials"]
    if len(trials) != 48 or dry["trials"] != 48 or manifest["status"] != "pass_hardware_not_authorized":
        raise RuntimeError("software gate mismatch")
    if [tuple(row["model_key"] for row in trials[i:i + 2]) for i in range(0, 48, 2)] != [
        ("S", "R") if pose % 2 else ("R", "S") for pose in range(1, 25)
    ]:
        raise RuntimeError("paired order mismatch")
    result = {
        "status": "pass",
        "dataset": {"root": str(DATASET), "episodes": 48, "frames": 6196, "tree_sha256": tree, "file_count": files, "bytes": size},
        "source_nonvisual_exact": binding["nonvisual_exact_comparison"],
        "sampling_streams": expected_streams,
        "training_result_sha256": sha(TRAINING),
        "model_sha256": training["model"]["model_sha256"],
        "processor_stats_sha256": training["model"]["processor_stats_sha256"],
        "offline_inference": training["offline_validation"],
        "evaluation_plan_sha256": sha(PLAN),
        "evaluation_trial_count": 48,
        "model_first_counts": {"S": 12, "R": 12},
        "early_stop_symmetric": dry["success_early_stop_symmetric"],
        "hardware_accessed": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    files_to_hash = [BINDING, TRAINING, PLAN, DRY, MANIFEST, output]
    (GATE / "hashes_all.sha256").write_text("".join(f"{sha(path)}  {path}\n" for path in files_to_hash))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
