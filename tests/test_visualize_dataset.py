import pytest

from lerobot.scripts.visualize_dataset import visualize_dataset


@pytest.mark.parametrize(
    "repo_id",
    ["lerobot/pusht"],
)
def test_visualize_dataset(tmpdir, repo_id):
    rrd_path = visualize_dataset(
        repo_id,
        episode_index=0,
        batch_size=32,
        save=True,
        output_dir=tmpdir,
    )
    assert rrd_path.exists()
