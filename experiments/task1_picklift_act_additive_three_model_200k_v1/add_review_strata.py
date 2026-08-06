from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path("/home/ubuntu24/Teleop/lerobot")
EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
FINAL = Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1/canonical_video_review_v1/final_review_v1")
OUT = FINAL / "stratified_addendum_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat(rows: list[dict]) -> dict:
    success = sum(row["final_success"] for row in rows)
    return {"success": success, "total": len(rows), "rate": success / len(rows)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(line) for line in (FINAL / "trials.jsonl").read_text().splitlines()]
    models = ("A", "B", "C")
    tiers = ("seen_by_real24", "added_by_real48", "added_by_real96", "unseen_by_both")
    cells = sorted({row["cell"] for row in rows})
    result = {
        "schema": "task1_additive_eval24_canonical_review_stratified_addendum_v1",
        "source_final_trials_sha256": sha256(FINAL / "trials.jsonl"),
        "by_model_and_coverage_tier": {model: {tier: stat([row for row in rows if row["model_key"] == model and row["coverage_tier"] == tier]) for tier in tiers} for model in models},
        "by_model_and_yaw": {model: {str(yaw): stat([row for row in rows if row["model_key"] == model and row["yaw_degrees"] == yaw]) for yaw in (0, 45)} for model in models},
        "by_model_and_cell": {model: {cell: stat([row for row in rows if row["model_key"] == model and row["cell"] == cell]) for cell in cells} for model in models},
        "failure_categories_by_model": {model: dict(sorted(Counter(row["final_failure_category"] for row in rows if row["model_key"] == model and not row["final_success"]).items())) for model in models},
    }
    for model in models:
        expected = {"seen_by_real24": 6, "added_by_real48": 6, "added_by_real96": 9, "unseen_by_both": 3}
        actual = {tier: result["by_model_and_coverage_tier"][model][tier]["total"] for tier in tiers}
        if actual != expected:
            raise RuntimeError(f"tier denominator mismatch {model}:{actual}")
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    files = [FINAL / "trials.jsonl", FINAL / "summary.json", FINAL / "manifest.json", FINAL / "independent_validation.json", OUT]
    hashes = FINAL / "hashes_v3.sha256"
    hashes.write_text("".join(f"{sha256(path)}  {path}\n" for path in files))
    index = {"schema": result["schema"], "status": "pass", "evidence_path": str(OUT), "sha256": sha256(OUT), "hashes_v3_sha256": sha256(hashes), **result}
    (EXP / "canonical_review_stratified_addendum_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
