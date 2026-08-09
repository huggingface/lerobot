from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset, MatchedTwoStreamSampler

REPO = Path("/home/ubuntu24/Teleop/lerobot")
ACT_EXP = REPO / "experiments/task1_picklift_act_additive_three_model_200k_v1"
HERE = REPO / "experiments/task1_picklift_act_csource_crender_v1_prep"
OLD_COMBINED = Path("/home/ubuntu24/Teleop/artifacts/datasets/task1_picklift_act_additive_three_model_200k_v1/real24_localsim24gap/combined48_v1")
OUTPUT = HERE / "old_c_sampling_reference.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config_path = ACT_EXP / "configs/r24_localsim24gap_full.json"
    result_path = ACT_EXP / "training_result_v1.json"
    plan_path = ACT_EXP / "evaluation_plan.json"
    if sha256(config_path) != "4fffd12cefb1d01ccede56d0be9bff0fad06abf70489e2cbb72aa1665724a10f":
        raise RuntimeError("old C config drift")
    result = json.loads(result_path.read_text())
    if result["models"]["C"]["model_sha256"] != "dd5a7002d850da8ea45dc8097a14de89e51e98432fab05dc898b35e2cc34811f":
        raise RuntimeError("old C model drift")
    config = json.loads(config_path.read_text())
    ds = LeRobotDataset(config["dataset"]["repo_id"], root=OLD_COMBINED, video_backend="pyav")
    if ds.meta.total_episodes != 48 or len(ds) != 6196 or ds.meta.fps != 20:
        raise RuntimeError("old C combined Dataset identity mismatch")
    for index in (0, 4262, 4263, 6195):
        sample = ds[index]
        if sample["observation.state"].shape != (6,) or sample["action"].shape != (6,):
            raise RuntimeError("state/action shape mismatch")
        if sample["observation.images.front"].shape != (3, 480, 640):
            raise RuntimeError("front image shape mismatch")
        if not all(torch.isfinite(sample[key]).all() for key in ("observation.state", "action", "observation.images.front")):
            raise RuntimeError("non-finite sample")
    sampler = MatchedTwoStreamSampler(
        ds.meta.episodes["dataset_from_index"],
        ds.meta.episodes["dataset_to_index"],
        ds.meta.episodes["episode_index"],
        config["dataset"]["matched_two_stream_episode_groups"],
        8,
        1000,
        episode_indices_to_use=ds.episodes,
        drop_n_last_frames=0,
        seed=1000,
        absolute_to_relative_idx=ds.absolute_to_relative_idx,
    )
    full = hashlib.sha256()
    real = hashlib.sha256()
    sim = hashlib.sha256()
    first_indices = None
    for epoch in range(200):
        order = list(sampler)
        if len(order) != 8000:
            raise RuntimeError("sampler epoch length mismatch")
        if first_indices is None:
            first_indices = order[:32]
        for offset in range(0, len(order), 8):
            batch = order[offset : offset + 8]
            if len(batch) != 8:
                raise RuntimeError("partial batch")
            for index in batch:
                full.update(struct.pack("<q", index))
            for index in batch[:4]:
                real.update(struct.pack("<q", index))
            for index in batch[4:]:
                sim.update(struct.pack("<q", index))
    output = {
        "schema": "task1_old_c_matched_sampler_reference_v1",
        "status": "readonly_reference_complete",
        "old_c_config_sha256": sha256(config_path),
        "old_c_training_result_sha256": sha256(result_path),
        "old_c_eval24_plan_sha256": sha256(plan_path),
        "old_c_model_sha256": result["models"]["C"]["model_sha256"],
        "dataset": {"root": str(OLD_COMBINED), "episodes": 48, "frames": 6196, "fps": 20},
        "sampler": {
            "seed": 1000,
            "epochs": 200,
            "batches_per_epoch": 1000,
            "batch_size": 8,
            "total_indices": 1600000,
            "real24_indices": 800000,
            "sim_indices": 800000,
            "full_index_stream_sha256_int64le": full.hexdigest(),
            "real24_index_stream_sha256_int64le": real.hexdigest(),
            "sim_index_stream_sha256_int64le": sim.hexdigest(),
            "first_32_indices": first_indices,
        },
        "rerender_dataset_read": False,
        "training_started": False,
        "hardware_accessed": False,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
