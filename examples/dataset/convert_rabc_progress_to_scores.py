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

"""Convert an RA-BC progress file to named scores, without scoring the dataset again.

Before ``lerobot-score``, ``python -m lerobot.rewards.<model>.compute_rabc_weights`` wrote one
progress file at the dataset root: ``sarm_progress.parquet``, ``robometer_progress.parquet`` or
``topreward_progress.parquet``. This script saves its progress columns as the named scores that
``lerobot-score`` writes and RA-BC reads. The model is inferred from the file name, and the file is
downloaded from the dataset repository when the local dataset folder does not have it. Videos are
not downloaded.

Example:

    uv run examples/dataset/convert_rabc_progress_to_scores.py \\
        --repo-id lerobot/high_quality_folding \\
        --push-to-hub

Then train with ``--sample_weighting.type=rabc --sample_weighting.score_name=sarm
--sample_weighting.signal_name=sarm.sparse_progress``.
"""

import argparse
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.rewards.robometer import ROBOMETER_PROGRESS_SIGNAL, ROBOMETER_SIGNAL_DESCRIPTORS
from lerobot.rewards.sarm import (
    SARM_DENSE_PROGRESS_SIGNAL,
    SARM_SIGNAL_DESCRIPTORS,
    SARM_SPARSE_PROGRESS_SIGNAL,
)
from lerobot.rewards.topreward import TOPREWARD_PROGRESS_SIGNAL, TOPREWARD_SIGNAL_DESCRIPTORS
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import _av_available, _datasets_available, require_package

if TYPE_CHECKING or (_datasets_available and _av_available):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.scores import add_scores, push_scores_to_hub

# The signal each model's progress columns become.
_SIGNALS = {
    "sarm": {"progress_sparse": SARM_SPARSE_PROGRESS_SIGNAL, "progress_dense": SARM_DENSE_PROGRESS_SIGNAL},
    "robometer": {"progress_sparse": ROBOMETER_PROGRESS_SIGNAL},
    "topreward": {"progress_sparse": TOPREWARD_PROGRESS_SIGNAL},
}
_DESCRIPTORS = {**SARM_SIGNAL_DESCRIPTORS, **ROBOMETER_SIGNAL_DESCRIPTORS, **TOPREWARD_SIGNAL_DESCRIPTORS}


def _get_progress_file(repo_id: str, root: Path, filename: str | None) -> Path:
    """Return the local progress file, downloading it from the dataset repository if needed."""
    candidates = [filename] if filename else [f"{model}_progress.parquet" for model in _SIGNALS]
    for candidate in candidates:
        if (root / candidate).is_file():
            return root / candidate
    api = HfApi()
    for candidate in candidates:
        if api.file_exists(repo_id, candidate, repo_type="dataset"):
            return Path(hf_hub_download(repo_id, candidate, repo_type="dataset", local_dir=root))
    raise FileNotFoundError(f"No progress file {candidates} in {root} or hf://datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repository, such as lerobot/full_folding.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset folder. Defaults to $HF_LEROBOT_HOME/<repo-id>.",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Progress file name. Defaults to the <model>_progress.parquet the dataset has.",
    )
    parser.add_argument("--name", default=None, help="Score name. Defaults to the model, such as sarm.")
    parser.add_argument("--overwrite", action="store_true", help="Replace scores with the same name.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the scores to the repository.")
    args = parser.parse_args()
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")

    root = args.root or HF_LEROBOT_HOME / args.repo_id
    path = _get_progress_file(args.repo_id, root, args.progress_file)
    model = path.name.removesuffix("_progress.parquet")
    if model not in _SIGNALS:
        raise ValueError(f"Cannot infer the model from {path.name}; expected one of {sorted(_SIGNALS)}")
    table = pq.read_table(path)
    columns = {column: signal for column, signal in _SIGNALS[model].items() if column in table.column_names}
    if not columns:
        raise ValueError(f"{path.name} has none of the progress columns {sorted(_SIGNALS[model])}")

    # Rows grouped by episode, in frame order.
    episodes = table.column("episode_index").to_numpy()
    frames = table.column("frame_index").to_numpy()
    order = np.lexsort((frames, episodes))
    episodes, frames = episodes[order], frames[order]
    indices = table.column("index").to_numpy()[order]
    values = {
        signal: table.column(column).to_numpy()[order].astype(np.float32)
        for column, signal in columns.items()
    }
    scored, starts = np.unique(episodes, return_index=True)
    rows = {
        int(episode): slice(start, end)
        for episode, start, end in zip(scored, starts, [*starts[1:], len(episodes)], strict=True)
    }

    dataset = LeRobotDataset(args.repo_id, root=root, download_videos=False)

    def episode_scores(episode_index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        episode_rows = rows[episode_index]
        start = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
        if np.any(indices[episode_rows] != start + frames[episode_rows]):
            raise ValueError(
                f"{path.name} no longer matches episode {episode_index} of {args.repo_id}; "
                "score the dataset again with lerobot-score"
            )
        return frames[episode_rows], {signal: value[episode_rows] for signal, value in values.items()}

    # Old progress files mark frames without a prediction with NaN.
    descriptors = {
        signal: dataclasses.replace(_DESCRIPTORS[signal], allow_nan=True)
        if np.isnan(value).any()
        else _DESCRIPTORS[signal]
        for signal, value in values.items()
    }
    metadata = table.schema.metadata or {}
    model_id = metadata.get(b"reward_model_path", metadata.get(b"vlm_name"))
    info = {
        "provenance": {
            "model": {"type": model, "id": model_id.decode() if model_id else None, "revision": None},
            "scorer": {"converted_from": path.name},
        },
        "signals": {signal: dataclasses.asdict(descriptor) for signal, descriptor in descriptors.items()},
    }
    name = args.name or model
    add_scores(dataset, name, episode_scores, info=info, episodes=sorted(rows), overwrite=args.overwrite)
    print(f"Saved {len(episodes)} frames of {path.name} as scores {name!r} with signals {sorted(values)}")
    if args.push_to_hub:
        push_scores_to_hub(dataset, name, overwrite=args.overwrite)
        print(f"Uploaded scores {name!r} to hf://datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
