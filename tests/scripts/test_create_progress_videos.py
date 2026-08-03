import numpy as np
import pandas as pd
import pytest

from examples.dataset.create_progress_videos import load_progress_data


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_load_progress_data_accepts_local_dense_exports(tmp_path, suffix):
    path = tmp_path / f"progress{suffix}"
    frame = pd.DataFrame(
        {
            "episode_index": [1, 0, 0],
            "frame_index": [0, 1, 0],
            "progress": [0.9, 0.6, 0.2],
            "progress_sparse": [0.1, 0.1, 0.1],
        }
    )
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)

    progress = load_progress_data(tmp_path, episode=0, progress_path=path)

    np.testing.assert_allclose(progress, [[0, 0.2], [1, 0.6]])


def test_load_progress_data_returns_none_for_missing_episode(tmp_path):
    path = tmp_path / "progress.csv"
    pd.DataFrame(
        {
            "episode_index": [0],
            "frame_index": [0],
            "progress": [0.2],
        }
    ).to_csv(path, index=False)

    assert load_progress_data(tmp_path, episode=1, progress_path=path) is None
