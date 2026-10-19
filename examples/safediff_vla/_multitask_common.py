"""Shared task-category definitions for the B-track multi-task `LocalGraspRefiner` adaptation
(`train_grounded_grasp_v2_5_multitask_stage1_5k.py`, `multitask_grasp_supervision_audit.py`,
`eval_grounded_grasp_v2_5_multitask_stage1_5k.py`). NOT imported by anything under `src/` --
example-script-only, same status as every other constant duplicated across scripts on this branch,
just factored into one place because these three scripts must agree EXACTLY on which dataset
episodes belong to which of the 7 requested tasks.

`lerobot/vlabench_unified`'s `meta/tasks.parquet` has 295 distinct task strings; VLABench's own
`select_fruit`/`select_toy`/.../`select_poker` env task registries were cross-checked against
every "primitive: ..." string in that file (see the corresponding `select_*_series.py` files under
`VLABench/tasks/hierarchical_tasks/primitive/`) to build the substring rule below. Verified
mutually exclusive and exhaustive for these 7 categories against the real dataset: exactly 500
episodes per task, zero overlap between rules, 3500/10977 episodes matched in total. Explicitly
NOT included here (the "upstream-invalid" / out-of-scope tasks the B-track instructions say to
leave out): select_painting (9 "Please select the painting of style X." strings), insert_flower
(vase_seen), add_condiment (dish), and every "composite: ..." multi-step task -- none of these were
requested and none is a single-object pick/select primitive comparable to the other 7.
"""

from __future__ import annotations

TASK_SUBSTRINGS: dict[str, str] = {
    "select_fruit": "plate_seen",  # "Put the {apple,banana,orange,pear} into the plate_seen"
    "select_toy": "giftbox_seen",  # "Put the {18 disney/marvel figures} into the giftbox_seen"
    "select_drink": "fridge_open",  # "Take out the {cola,juice_seen,jumpstart,monster} from the fridge_open"
    "select_book": "take the book",  # "Please take the book {17 titles}"
    "select_ingredient": "solution",  # "Take out the {CuCl2,CuSO4,FeCl3,KMnO4} solution"
    "select_mahjong": "mahjong",  # "Pick up the mahjong of {value} {man,pin,sou}"
    "select_poker": "poker",  # "Please pick the poker {rank} of {suit}"
}

TASK_NAMES: list[str] = list(TASK_SUBSTRINGS)

# `select_ingredient` excluded from Stage 2 onward: `dataset_camera_provenance_check.py` found a
# confirmed train/eval TASK-VARIANT mismatch (not a camera-role/scene-diversity issue like the
# other 6 tasks) -- every `lerobot/vlabench_unified` "select_ingredient" episode (4/4 sampled
# episodes, all 4 distinct instructions: FeCl3/KMnO4x2/CuSO4) shows a chemistry-solution/reagent-
# rack scene, while the LIVE VLABenchEnv "select_ingredient" task (4/4 seeds: 1000-1003) always
# renders a refrigerator/vegetable scene instead -- zero overlap across 8 independent samples. This
# is a real train/eval distribution mismatch for this task specifically, not a performance issue --
# Stage 1's own select_ingredient results should be read as "evaluated out-of-training-distribution"
# rather than as a measure of this architecture's grounding quality. See
# `outputs/eval/safediff_vla_camera_mapping_diagnostic/dataset_camera_provenance_report.json`.
STAGE2_TASK_NAMES: list[str] = [t for t in TASK_NAMES if t != "select_ingredient"]

REPO_ID = "lerobot/vlabench_unified"

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}


def classify_task(task_str: str) -> str | None:
    """First-match against `TASK_SUBSTRINGS` -- order matters only in principle; empirically the
    7 substrings never co-occur in the same task string (see module docstring)."""
    for name, substring in TASK_SUBSTRINGS.items():
        if substring in task_str:
            return name
    return None


def build_episode_task_map(repo_id: str = REPO_ID) -> dict[int, str]:
    """`{episode_index: task_name}` for every episode in `repo_id` matching one of the 7 tasks
    above (episodes matching none of them, e.g. select_painting/composite episodes, are simply
    absent from the returned dict). Metadata-only -- no data/video download."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    episode_tasks = meta.episodes["tasks"]
    out: dict[int, str] = {}
    for ep_idx in range(meta.total_episodes):
        tasks = episode_tasks[ep_idx]
        t0 = tasks[0] if tasks else ""
        category = classify_task(t0)
        if category is not None:
            out[ep_idx] = category
    return out
