"""Episode-index -> dataset-native task_index map for `lerobot/libero` / `lerobot/libero_plus`.

Neither dataset's own `meta/episodes/*.parquet` carries a usable per-episode task column for our
purposes here (LIBERO's has none at all; LIBERO-Plus's own `tasks` column is redundant with this),
so this derives the map directly from the per-FRAME `task_index` column in the data parquet files
(column-pushdown read -- only `episode_index`/`task_index` are fetched, not full rows), same
method used throughout the dataset/action-semantics audit this experiment builds on. Cached to a
local parquet next to this file so repeated runs (train + eval + a re-run) don't re-scan the
Hub every time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_IDS = {"libero": "lerobot/libero", "libero_plus": "lerobot/libero_plus"}
_CACHE_DIR = Path(__file__).parent / ".episode_task_cache"


def build_episode_task_map(dataset: str) -> dict[int, int]:
    """Returns {episode_index: task_index} for every episode in the dataset."""
    cache_path = _CACHE_DIR / f"{dataset}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        from huggingface_hub import HfFileSystem
        import pyarrow.dataset as ds

        fs = HfFileSystem()
        repo_id = REPO_IDS[dataset]
        paths = sorted(fs.glob(f"datasets/{repo_id}/data/**/*.parquet"))
        dset = ds.dataset(paths, filesystem=fs, format="parquet")
        table = dset.to_table(columns=["episode_index", "task_index"])
        df = table.to_pandas()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    ep2task = df.groupby("episode_index")["task_index"].first()
    return {int(k): int(v) for k, v in ep2task.items()}


def per_task_episode_counts(dataset: str) -> pd.Series:
    m = build_episode_task_map(dataset)
    return pd.Series(list(m.values())).value_counts().sort_index()
