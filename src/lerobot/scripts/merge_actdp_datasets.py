"""Build the combined ACT/DP training dataset.

Step 1: filter `local/sim_assemble_sarm_merged_v1` down to its 40 success-only
episodes (per `outputs/merged_v1_success_filter.json`) → tmp dataset.
Step 2: merge that with `domrachev03/sim_assemble_sarm_multistage_two_stages_success`
(36 success episodes already) → `local/sim_assemble_actdp_combined`.

Output: 76 success episodes for ACT/DP BC training.

Usage:
    uv run python -m lerobot.scripts.merge_actdp_datasets
"""

import json
import logging
import shutil
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE  # noqa: F401

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


SRC_V1 = "local/sim_assemble_sarm_merged_v1"
SRC_NEW = "domrachev03/sim_assemble_sarm_multistage_two_stages_success"
TMP_REPO = "local/sim_assemble_v1_success40"
DST_REPO = "local/sim_assemble_actdp_combined"
SUCCESS_FILTER = Path("outputs/merged_v1_success_filter.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    success_eps = sorted(json.loads(SUCCESS_FILTER.read_text())["success_episodes"])
    logging.info("Success eps from filter: %d", len(success_eps))

    src_v1 = LeRobotDataset(SRC_V1, download_videos=False)
    failures = [i for i in range(src_v1.meta.total_episodes) if i not in success_eps]
    logging.info("merged_v1 total=%d, dropping %d failures", src_v1.meta.total_episodes, len(failures))

    tmp_root = HF_LEROBOT_HOME / TMP_REPO.split("/")[-1]
    if tmp_root.exists():
        logging.info("Removing stale tmp %s", tmp_root)
        shutil.rmtree(tmp_root)
    v1_success = delete_episodes(
        dataset=src_v1, episode_indices=failures, output_dir=tmp_root, repo_id=TMP_REPO
    )
    logging.info("v1_success40 built: %d eps, %d frames", v1_success.meta.total_episodes, v1_success.meta.total_frames)

    src_new = LeRobotDataset(SRC_NEW)
    logging.info("two_stages_success: %d eps, %d frames", src_new.meta.total_episodes, src_new.meta.total_frames)

    dst_root = HF_LEROBOT_HOME / DST_REPO.split("/")[-1]
    if dst_root.exists():
        logging.info("Removing stale dst %s", dst_root)
        shutil.rmtree(dst_root)

    merged = merge_datasets([v1_success, src_new], output_repo_id=DST_REPO, output_dir=dst_root)
    logging.info(
        "Merged dataset: %d eps, %d frames at %s",
        merged.meta.total_episodes,
        merged.meta.total_frames,
        dst_root,
    )


if __name__ == "__main__":
    main()
