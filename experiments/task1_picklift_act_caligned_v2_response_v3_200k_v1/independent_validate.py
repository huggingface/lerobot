from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lerobot.datasets import LeRobotDataset

HERE = Path(__file__).resolve().parent
ART = Path("/home/ubuntu24/Teleop/artifacts")
EXP = "task1_picklift_act_caligned_v2_response_v3_200k_v1"
EVAL_ID = "task1_picklift_csource_vs_response_v3_simseen6_paired_eval_v1"
BINDING = ART / f"evidence/{EXP}/binding_v1/binding_result.json"
TRAINING = HERE / "training_result_v1.json"
PLAN = HERE / "bound_simseen6_evaluation_plan.json"
GATE = ART / f"evaluation/{EVAL_ID}/software_preparation_v1"
DRY = GATE / "dry_run.json"
MANIFEST = GATE / "manifest.json"
DATASET = ART / f"datasets/{EXP}/combined48_v1"
REPO_ID = "local/task1_picklift_real24_localsim24gap_response_v3_real_appearance_additive_v1"
EXPECTED_REAL_STREAM = "f392d7b148905d90467a2565229df92d33e7805e8037a48eeca02c6d31730c53"
EXPECTED_SOURCE_MODEL = "dd5a7002d850da8ea45dc8097a14de89e51e98432fab05dc898b35e2cc34811f"
EXPECTED_RESEARCH_CONTRACT = "f488689e72e2f51e580f8c26a2cecffd539fcd98dca7bc7e3f4d930eac2aeaad"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_tree(root: Path) -> tuple[str, int, int]:
    rows = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
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
    if binding["sim24"]["episodes"] != 24 or binding["sim24"]["frames"] != 3023:
        raise RuntimeError("new response-v3 Sim identity mismatch")
    if binding["historical_sim_state_action_comparison_performed"] is not False:
        raise RuntimeError("historical Sim must not be the row comparator")
    if binding["sampling"]["real24_index_stream_sha256_int64le"] != EXPECTED_REAL_STREAM:
        raise RuntimeError("Real24 sampling stream mismatch")
    if binding["sampling"]["sample_slots"] != {"real24": 800000, "source_b": 800000}:
        raise RuntimeError("binding sample slots mismatch")
    if training["status"] != "offline_training_complete_ready_for_simseen6_software_gate":
        raise RuntimeError("training status mismatch")
    if training["selected_step"] != 200000:
        raise RuntimeError("fixed checkpoint mismatch")
    if training["formal"]["sampling_counts"]["actual_samples_seen_by_main_process"] != {
        "real24": 800000,
        "source_b": 800000,
    }:
        raise RuntimeError("formal sample count mismatch")
    dataset = LeRobotDataset(REPO_ID, root=DATASET, video_backend="pyav")
    if dataset.meta.total_episodes != 48 or len(dataset) != 7286 or dataset.meta.fps != 20:
        raise RuntimeError("combined Dataset mismatch")
    tree, files, size = derived_tree(DATASET)
    trials = plan["trials"]
    if len(trials) != 12 or [row["order"] for row in trials] != list(range(1, 13)):
        raise RuntimeError("Sim-seen6 paired12 plan mismatch")
    pairs = [trials[index : index + 2] for index in range(0, 12, 2)]
    if any(len({row["eval_pose_id"] for row in pair}) != 1 for pair in pairs):
        raise RuntimeError("Sim-seen6 same-pose pairing mismatch")
    expected_order = [("S", "A") if pose % 2 else ("A", "S") for pose in range(1, 7)]
    if [tuple(row["model_key"] for row in pair) for pair in pairs] != expected_order:
        raise RuntimeError("Sim-seen6 model order mismatch")
    if len({row["eval_pose_id"] for row in trials}) != 6:
        raise RuntimeError("Sim-seen6 pose count mismatch")
    if any(row["restore_nominal_cube_pose_before_this_trial"] is not True for row in trials):
        raise RuntimeError("independent per-trial placement restore missing")
    if plan["pose_bank"]["automatic_full_eval24_fallback"] is not False:
        raise RuntimeError("full Eval24 fallback is not disabled")
    if plan["source_identities"]["research_eval_contract_sha256"] != EXPECTED_RESEARCH_CONTRACT:
        raise RuntimeError("research Sim-seen6 contract mismatch")
    if plan["models"]["S"]["model_sha256"] != EXPECTED_SOURCE_MODEL:
        raise RuntimeError("C-source model mismatch")
    if plan["models"]["A"]["model_sha256"] != training["model"]["model_sha256"]:
        raise RuntimeError("C-aligned-v2 model mismatch")
    if dry["status"] != "software_dry_run_pass_hardware_not_accessed" or dry["trials"] != 12:
        raise RuntimeError("Sim-seen6 dry-run mismatch")
    if dry["model_reset_calls"] != {"A": 6, "S": 6}:
        raise RuntimeError("dry-run policy-reset counts mismatch")
    if dry["success_early_stop_symmetric"] is not True:
        raise RuntimeError("success early-stop is not symmetric")
    if manifest["status"] != "pass_hardware_not_authorized" or manifest["trials"] != 12:
        raise RuntimeError("software gate manifest mismatch")
    if any(plan.get("hardware_authorized") is not False for _ in (0,)):
        raise RuntimeError("hardware must remain unauthorized")
    result = {
        "status": "pass",
        "dataset": {
            "root": str(DATASET),
            "episodes": 48,
            "frames": 7286,
            "tree_sha256": tree,
            "file_count": files,
            "bytes": size,
        },
        "new_response_v3_source_to_derived_exactness": binding["new_source_to_derived_exactness"],
        "sampling": binding["sampling"],
        "training_result_sha256": sha(TRAINING),
        "model_sha256": training["model"]["model_sha256"],
        "processor_stats_sha256": training["model"]["processor_stats_sha256"],
        "offline_inference": training["offline_validation"],
        "evaluation_plan_sha256": sha(PLAN),
        "evaluation_trial_count": 12,
        "evaluation_pose_count": 6,
        "model_first_counts": {"S": 3, "A": 3},
        "early_stop_symmetric": True,
        "automatic_full_eval24_fallback": False,
        "hardware_accessed": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    files_to_hash = [BINDING, TRAINING, PLAN, DRY, MANIFEST, output]
    (GATE / "hashes_all.sha256").write_text("".join(f"{sha(path)}  {path}\n" for path in files_to_hash))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
