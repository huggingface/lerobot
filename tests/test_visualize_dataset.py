import pytest

from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.visualize_dataset import visualize_dataset

from .utils import DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "dataset_id",
    [
        "aloha_sim_insertion_human",
    ],
)
def test_visualize_dataset(tmpdir, dataset_id):
    # TODO(rcadene): this test might fail with other datasets/policies/envs, since visualization_dataset
    # doesnt support multiple timesteps which requires delta_timestamps to None for images.
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            "policy=act",
            "env=aloha",
            f"dataset_id={dataset_id}",
        ],
    )
    video_paths = visualize_dataset(cfg, out_dir=tmpdir)

    assert len(video_paths) > 0

    for video_path in video_paths:
        assert video_path.exists()
